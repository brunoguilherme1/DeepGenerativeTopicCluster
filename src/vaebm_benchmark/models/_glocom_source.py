"""Vendored, verbatim GloCOM model + ECR loss, from the official repository
https://github.com/qducnguyen/GloCOM (main branch, checked 2026-08), files
`models/GloCOM/GloCOM.py` and `models/GloCOM/ECR.py`. GloCOM is a plain
research repo, not a pip-installable package - there is no PyPI package to
depend on, so vendoring the exact upstream source is the closest available
equivalent to "use the official implementation."

Nguyen, Q. D., Nguyen, T., Nguyen, D. A., Ngo Van, L., Dinh, S., & Nguyen,
T. H. (2025). "GloCOM: A Short Text Neural Topic Model via Global
Clustering Context." NAACL 2025. https://aclanthology.org/2025.naacl-long.51/
The ECR loss is Wu, X., Dong, X., Nguyen, T. T., & Luu, A. T. (2023).
"Effective Neural Topic Modeling with Embedding Clustering Regularization."
ICML 2023 (the same loss topmost.ECRTM uses internally; GloCOM's own repo
credits TopMost as a basis for parts of its implementation).

ONLY the two nn.Module classes below are reproduced (byte-identical to the
official repo, save for this docstring) - the surrounding data pipeline
(models/glocom_adapter.py in this repo) is this project's own, because the
official repo's own `dataloader.py`/`Trainer.py`/`run.py` assume a fixed
on-disk dataset layout (precomputed `bow.npz`/`global_bow.npz`/
`global_maps.txt` per dataset) this project does not use, and because -
important for anyone reading protocol.json later - inspecting that
dataloader.py shows `self.test_bow = scipy.sparse.load_npz(f'{path}/bow.npz')`,
i.e. the official repo's own reference run never evaluates on a genuinely
held-out split at all; see protocols/glocom_protocol.py and
docs/methodological_notes.md for how this project's protocol reproduces
that same full-corpus/transductive evaluation rather than inventing a
train/test split the paper itself never used.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class ECR(nn.Module):
    """Effective Neural Topic Modeling with Embedding Clustering
    Regularization. ICML 2023. Xiaobao Wu, Xinshuai Dong, Thong Thanh
    Nguyen, Anh Tuan Luu. Vendored verbatim from
    github.com/qducnguyen/GloCOM's models/GloCOM/ECR.py."""

    def __init__(self, weight_loss_ECR, sinkhorn_alpha, OT_max_iter=5000, stopThr=0.5e-2):
        super().__init__()

        self.sinkhorn_alpha = sinkhorn_alpha
        self.OT_max_iter = OT_max_iter
        self.weight_loss_ECR = weight_loss_ECR
        self.stopThr = stopThr
        self.epsilon = 1e-16

    def forward(self, M):
        # M: KxV
        # a: Kx1
        # b: Vx1
        if self.weight_loss_ECR <= 1e-6:
            return 0.0

        device = M.device

        # Sinkhorn's algorithm
        a = (torch.ones(M.shape[0]) / M.shape[0]).unsqueeze(1).to(device)
        b = (torch.ones(M.shape[1]) / M.shape[1]).unsqueeze(1).to(device)

        u = (torch.ones_like(a) / a.size()[0]).to(device)  # Kx1

        K = torch.exp(-M * self.sinkhorn_alpha)
        err = 1
        cpt = 0
        while err > self.stopThr and cpt < self.OT_max_iter:
            v = torch.div(b, torch.matmul(K.t(), u) + self.epsilon)
            u = torch.div(a, torch.matmul(K, v) + self.epsilon)
            cpt += 1
            if cpt % 50 == 1:
                bb = torch.mul(v, torch.matmul(K.t(), u))
                err = torch.norm(torch.sum(torch.abs(bb - b), dim=0), p=float("inf"))

        transp = u * (K * v.T)

        loss_ECR = torch.sum(transp * M)
        loss_ECR *= self.weight_loss_ECR

        return loss_ECR


class GloCOM(nn.Module):
    """GloCOM: A Short Text Neural Topic Model via Global Clustering
    Context. Quang Duc Nguyen et al., NAACL 2025. Vendored verbatim from
    github.com/qducnguyen/GloCOM's models/GloCOM/GloCOM.py.

    `forward`/`get_theta` both expect `input` to be the horizontal
    concatenation `[local_bow | global_bow]`, shape (batch, 2*vocab_size)
    - the split is internal (`input[:, :self.vocab_size]` /
    `input[:, self.vocab_size:]`), which is exactly what lets this run,
    unmodified, through `topmost.BasicTrainer` (whose train()/test() only
    ever pass a single batch tensor straight to `model(batch)` /
    `model.get_theta(batch)`) - see glocom_adapter.py for how the
    concatenated input and the "global" BOW per document are built."""

    def __init__(
        self,
        vocab_size,
        num_topics=50,
        en_units=200,
        dropout=0.0,
        pretrained_WE=None,
        embed_size=200,
        sinkhorn_alpha=20.0,
        sinkhorn_max_iter=100,
        beta_temp=0.2,
        aug_coef=0.5,
        prior_var=0.01,
        weight_loss_ECR=30.0,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.num_topics = num_topics
        self.beta_temp = beta_temp
        self.prior_var = prior_var
        self.aug_coef = aug_coef

        # Prior
        # global document
        self.a = 1 * np.ones((1, num_topics)).astype(np.float32)
        self.mu2 = nn.Parameter(torch.as_tensor((np.log(self.a).T - np.mean(np.log(self.a), 1)).T))
        self.var2 = nn.Parameter(
            torch.as_tensor(
                (((1.0 / self.a) * (1 - (2.0 / num_topics))).T + (1.0 / (num_topics * num_topics)) * np.sum(1.0 / self.a, 1)).T
            )
        )
        self.mu2.requires_grad = False
        self.var2.requires_grad = False

        # local adapter
        self.local_adapter_mu = nn.Parameter(torch.ones_like(self.mu2), requires_grad=False)
        self.local_adapter_var = nn.Parameter(torch.ones_like(self.var2) * self.prior_var, requires_grad=False)

        # global document encoder
        self.fc11 = nn.Linear(vocab_size, en_units)
        self.fc12 = nn.Linear(en_units, en_units)
        self.fc21 = nn.Linear(en_units, num_topics)
        self.fc22 = nn.Linear(en_units, num_topics)
        self.fc1_dropout = nn.Dropout(dropout)

        self.mean_bn = nn.BatchNorm1d(num_topics)
        self.mean_bn.weight.requires_grad = False
        self.logvar_bn = nn.BatchNorm1d(num_topics)
        self.logvar_bn.weight.requires_grad = False

        # local adapter encoder
        self.fc11_adapter = nn.Linear(vocab_size, en_units)
        self.fc12_adapter = nn.Linear(en_units, en_units)
        self.fc21_adapter = nn.Linear(en_units, num_topics)
        self.fc22_adapter = nn.Linear(en_units, num_topics)
        self.fc1_adapter_dropout = nn.Dropout(dropout)

        self.adapter_mean_bn = nn.BatchNorm1d(num_topics)
        self.adapter_mean_bn.weight.requires_grad = False
        self.adapter_logvar_bn = nn.BatchNorm1d(num_topics)
        self.adapter_logvar_bn.weight.requires_grad = False

        # decoder
        self.decoder_bn = nn.BatchNorm1d(vocab_size)
        self.decoder_bn.weight.requires_grad = False

        # word embedding
        if pretrained_WE is not None:
            self.word_embeddings = torch.from_numpy(pretrained_WE).float()
        else:
            self.word_embeddings = nn.init.trunc_normal_(torch.empty(vocab_size, embed_size))
        self.word_embeddings = nn.Parameter(F.normalize(self.word_embeddings))

        self.topic_embeddings = torch.empty((num_topics, self.word_embeddings.shape[1]))
        nn.init.trunc_normal_(self.topic_embeddings, std=0.1)
        self.topic_embeddings = nn.Parameter(F.normalize(self.topic_embeddings))

        self.ECR = ECR(weight_loss_ECR, sinkhorn_alpha, sinkhorn_max_iter)

    def global_encode(self, input):
        e1 = F.softplus(self.fc11(input))
        e1 = F.softplus(self.fc12(e1))
        e1 = self.fc1_dropout(e1)
        mu = self.mean_bn(self.fc21(e1))
        logvar = self.logvar_bn(self.fc22(e1))
        z = self.global_reparameterize(mu, logvar)
        theta = F.softmax(z, dim=1)

        loss_KL = self.compute_loss_global_KL(mu, logvar)

        return theta, loss_KL

    def global_reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + (eps * std)
        else:
            return mu

    def compute_loss_global_KL(self, mu, logvar):
        var = logvar.exp()
        var_division = var / self.var2
        diff = mu - self.mu2
        diff_term = diff * diff / self.var2
        logvar_division = self.var2.log() - logvar
        # KLD: N*K
        KLD = 0.5 * ((var_division + diff_term + logvar_division).sum(axis=1) - self.num_topics)
        KLD = KLD.mean()
        return KLD

    def local_adapter_encode(self, input):
        e1 = F.softplus(self.fc11_adapter(input))
        e1 = F.softplus(self.fc12_adapter(e1))
        e1 = self.fc1_adapter_dropout(e1)
        mu = self.adapter_mean_bn(self.fc21_adapter(e1))
        logvar = self.adapter_logvar_bn(self.fc22_adapter(e1))
        z = self.local_adapter_reparameterize(mu, logvar)

        loss_KL = self.compute_loss_local_KL(mu, logvar)

        return z, loss_KL

    def local_adapter_reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + (eps * std)
        else:
            return mu

    def compute_loss_local_KL(self, mu, logvar):
        var = logvar.exp()
        var_division = var / self.local_adapter_var
        diff = mu - self.local_adapter_mu
        diff_term = diff * diff / self.local_adapter_var
        logvar_division = self.local_adapter_var.log() - logvar
        # KLD: N*K
        KLD = 0.5 * ((var_division + diff_term + logvar_division).sum(axis=1) - self.num_topics)
        KLD = KLD.mean()
        return KLD

    def get_beta(self):
        dist = self.pairwise_euclidean_distance(self.topic_embeddings, self.word_embeddings)
        beta = F.softmax(-dist / self.beta_temp, dim=0)
        return beta

    def get_theta(self, input):
        local_x = input[:, : self.vocab_size]
        global_x = input[:, self.vocab_size :]
        local_adapter_theta, local_adapter_loss_KL = self.local_adapter_encode(local_x)
        global_theta, global_loss_KL = self.global_encode(global_x)

        if self.training:
            return F.softmax(global_theta * local_adapter_theta, dim=1), global_loss_KL, local_adapter_loss_KL
        else:
            return F.softmax(global_theta * local_adapter_theta, dim=1)

    def get_loss_ECR(self):
        cost = self.pairwise_euclidean_distance(self.topic_embeddings, self.word_embeddings)
        loss_ECR = self.ECR(cost)
        return loss_ECR

    def pairwise_euclidean_distance(self, x, y):
        cost = torch.sum(x**2, axis=1, keepdim=True) + torch.sum(y**2, dim=1) - 2 * torch.matmul(x, y.t())
        return cost

    def forward(self, input, is_ECR=True):
        local_x = input[:, : self.vocab_size]
        global_x = input[:, self.vocab_size :]
        local_adapter_theta, local_adapter_loss_KL = self.local_adapter_encode(local_x)
        global_theta, global_loss_KL = self.global_encode(global_x)
        beta = self.get_beta()

        recon = F.softmax(self.decoder_bn(torch.matmul(F.softmax(global_theta * local_adapter_theta, dim=1), beta)), dim=-1)
        recon_loss = -((local_x + self.aug_coef * global_x) * recon.log()).sum(axis=1).mean()

        loss_TM = recon_loss + global_loss_KL + local_adapter_loss_KL

        if is_ECR:
            loss_ECR = self.get_loss_ECR()
        else:
            loss_ECR = 0

        loss = loss_TM + loss_ECR

        rst_dict = {"loss": loss, "loss_TM": loss_TM, "loss_ECR": loss_ECR}

        return rst_dict
