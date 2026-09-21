"""Vendored WAE (Wasserstein Autoencoder) model from the official S2WTM
repo - https://github.com/AdhyaSuman/S2WTM, file
`octis/models/spherical_SWTM/models/wae_sp.py`. S2WTM (Adhya & Sanyal,
ACL 2025) is a plain research repo built on the OCTIS framework, not a
pip-installable package, so this adapter (like GloCOM's) vendors the
model itself rather than depending on it.

Trimmed from the original file: the `dist='vmf'`/`'mixture_vmf'`
branches of `sample()` (and the `hyperbolic.py`/`power_spherical.py`/
`von_mises_fisher.py` helper modules they depend on) are NOT vendored
here. This adapter only ever constructs WAE with `dist='unif_sphere'`
and `loss_type='sph_sw'` - the paper's own named default in the official
wrapper (`octis/models/S2WTM.py`'s own `__init__` defaults to exactly
these two values) - under which those branches are unreachable dead
code. Everything actually exercised by that configuration (`encode`,
`decode`, `forward`, the `unif_sphere` branch of `sample`, and the
sliced-Wasserstein-on-sphere loss `sp_swd_loss`/`sp_sliced_cost`/
`binary_search_circle`/`Cost`/`dCost`/`emd1D_circle`/`w2_unif_circle`,
i.e. the paper's actual contribution) is vendored verbatim.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class WAE(nn.Module):
    def __init__(
        self,
        encode_dims=[2000, 1024, 512, 20],
        decode_dims=[20, 1024, 2000],
        dropout=0.0,
        nonlin="relu",
        dist="unif_sphere",
        batch_size=256,
        temperature=0.7,
    ):
        super().__init__()

        self.dist = dist
        self.batch_size = batch_size

        self.encoder = nn.ModuleDict(
            {f"enc_{i}": nn.Linear(encode_dims[i], encode_dims[i + 1]) for i in range(len(encode_dims) - 1)}
        )

        self.decoder = nn.ModuleDict(
            {f"dec_{i}": nn.Linear(decode_dims[i], decode_dims[i + 1]) for i in range(len(decode_dims) - 1)}
        )
        self.latent_dim = encode_dims[-1]
        self.dropout = nn.Dropout(p=dropout)
        self.nonlin = {"relu": F.relu, "sigmoid": torch.sigmoid}[nonlin]
        self.z_dim = encode_dims[-1]

        self.temperature = nn.Parameter(torch.tensor(temperature))

        self.proj = nn.Sequential(
            nn.Linear(self.z_dim, self.z_dim),
            nn.ReLU(),
            nn.Linear(self.z_dim, self.z_dim),
            nn.LayerNorm(self.z_dim),
        )

    def encode(self, x):
        hid = x
        n_layers = len(self.encoder)
        for i, (_, layer) in enumerate(self.encoder.items()):
            if i < n_layers - 1:
                hid = self.dropout(layer(hid))
                hid = self.nonlin(hid)
            else:
                hid = layer(hid)
        return hid

    def decode(self, z):
        hid = z
        for i, (_, layer) in enumerate(self.decoder.items()):
            hid = layer(hid)
            if i < len(self.decoder) - 1:
                hid = self.nonlin(self.dropout(hid))
        return hid

    def forward(self, x):
        z = self.encode(x)
        z_norm = F.normalize(z, p=2, dim=1)
        z_proj = self.proj(z_norm)
        theta = F.softmax(self.dropout(z_proj) / self.temperature, dim=1)
        x_reconst = self.decode(theta)
        return x_reconst, z_norm

    def sample(self, dirichlet_alpha=0.1, ori_data=None):
        if self.dist == "dirichlet":
            z_true = np.random.dirichlet(np.ones(self.z_dim) * dirichlet_alpha, size=self.batch_size)
            return torch.from_numpy(z_true).float()

        if self.dist == "gaussian":
            z_true = np.random.randn(self.batch_size, self.z_dim)
            return torch.softmax(torch.from_numpy(z_true), dim=1).float()

        if self.dist == "gmm_std":
            odes = np.eye(self.z_dim) * 20
            ides = np.random.randint(low=0, high=self.z_dim, size=self.batch_size)
            mus = odes[ides]
            sigmas = np.ones((self.batch_size, self.z_dim)) * 0.2 * 20
            z_true = np.random.normal(mus, sigmas)
            return F.softmax(torch.from_numpy(z_true).float(), dim=1)

        if self.dist == "unif_sphere":
            target_latent = torch.randn(self.batch_size, self.z_dim)
            return F.normalize(target_latent, p=2, dim=-1)

        # dist='vmf'/'mixture_vmf'/'gmm_ctm' from the official module are not
        # vendored here (see module docstring) - this adapter never selects them.
        raise NotImplementedError(f"dist='{self.dist}' is not supported by this vendored subset of S2WTM's WAE")

    # -- sliced-Wasserstein-on-sphere loss (the paper's actual contribution) --

    def binary_search_circle(
        self, u_values, v_values, u_weights=None, v_weights=None, p=1, Lm=10, Lp=10, tm=-1, tp=1, eps=1e-6, require_sort=True
    ):
        n = u_values.shape[-1]
        m = v_values.shape[-1]
        device = u_values.device
        dtype = u_values.dtype

        if u_weights is None:
            u_weights = torch.full((n,), 1 / n, dtype=dtype, device=device)
        if v_weights is None:
            v_weights = torch.full((m,), 1 / m, dtype=dtype, device=device)

        if require_sort:
            u_values, u_sorter = torch.sort(u_values, -1)
            v_values, v_sorter = torch.sort(v_values, -1)
            u_weights = u_weights[..., u_sorter]
            v_weights = v_weights[..., v_sorter]

        u_cdf = torch.cumsum(u_weights, -1)
        v_cdf = torch.cumsum(v_weights, -1)

        L = max(Lm, Lp)

        tm = tm * torch.ones((u_values.shape[0],), dtype=dtype, device=device).view(-1, 1)
        tm = tm.repeat(1, m)
        tp = tp * torch.ones((u_values.shape[0],), dtype=dtype, device=device).view(-1, 1)
        tp = tp.repeat(1, m)
        tc = (tm + tp) / 2

        done = torch.zeros((u_values.shape[0], m))

        while torch.any(1 - done):
            dCp, dCm = self.dCost(tc, u_values, v_values, u_cdf, v_cdf, p)
            done = ((dCp * dCm) <= 0) * 1

            mask = ((tp - tm) < eps / L) * (1 - done)

            if torch.any(mask):
                dCptp, dCmtp = self.dCost(tp, u_values, v_values, u_cdf, v_cdf, p)
                dCptm, dCmtm = self.dCost(tm, u_values, v_values, u_cdf, v_cdf, p)
                Ctm = self.Cost(tm, u_values, v_values, u_cdf, v_cdf, p).reshape(-1, 1)
                Ctp = self.Cost(tp, u_values, v_values, u_cdf, v_cdf, p).reshape(-1, 1)

                mask_end = mask * (torch.abs(dCptm - dCmtp) > 0.001)
                tc[mask_end > 0] = ((Ctp - Ctm + tm * dCptm - tp * dCmtp) / (dCptm - dCmtp))[mask_end > 0]
                done[torch.prod(mask, dim=-1) > 0] = 1
            elif torch.any(1 - done):
                tm[((1 - mask) * (dCp < 0)) > 0] = tc[((1 - mask) * (dCp < 0)) > 0]
                tp[((1 - mask) * (dCp >= 0)) > 0] = tc[((1 - mask) * (dCp >= 0)) > 0]
                tc[((1 - mask) * (1 - done)) > 0] = (tm[((1 - mask) * (1 - done)) > 0] + tp[((1 - mask) * (1 - done)) > 0]) / 2

        return self.Cost(tc.detach(), u_values, v_values, u_cdf, v_cdf, p)

    def dCost(self, theta, u_values, v_values, u_cdf, v_cdf, p):
        v_values = v_values.clone()
        n = u_values.shape[-1]

        v_cdf_theta = v_cdf - (theta - torch.floor(theta))
        mask_p = v_cdf_theta >= 0
        mask_n = v_cdf_theta < 0

        v_values[mask_n] += torch.floor(theta)[mask_n] + 1
        v_values[mask_p] += torch.floor(theta)[mask_p]

        if torch.any(mask_n) and torch.any(mask_p):
            v_cdf_theta[mask_n] += 1

        v_cdf_theta2 = v_cdf_theta.clone()
        v_cdf_theta2[mask_n] = np.inf
        shift = -torch.argmin(v_cdf_theta2, axis=-1)

        v_cdf_theta = self.roll_by_gather(v_cdf_theta, 1, shift.view(-1, 1))
        v_values = self.roll_by_gather(v_values, 1, shift.view(-1, 1))
        v_values = torch.cat([v_values, v_values[:, 0].view(-1, 1) + 1], dim=1)

        u_index = torch.searchsorted(u_cdf, v_cdf_theta)
        u_icdf_theta = torch.gather(u_values, -1, u_index.clip(0, n - 1))

        u_cdfm = torch.cat([u_cdf, u_cdf[:, 0].view(-1, 1) + 1], dim=1)
        u_valuesm = torch.cat([u_values, u_values[:, 0].view(-1, 1) + 1], dim=1)
        u_indexm = torch.searchsorted(u_cdfm, v_cdf_theta, right=True)
        u_icdfm_theta = torch.gather(u_valuesm, -1, u_indexm.clip(0, n))

        dCp = torch.sum(
            torch.pow(torch.abs(u_icdf_theta - v_values[:, 1:]), p) - torch.pow(torch.abs(u_icdf_theta - v_values[:, :-1]), p),
            axis=-1,
        )
        dCm = torch.sum(
            torch.pow(torch.abs(u_icdfm_theta - v_values[:, 1:]), p) - torch.pow(torch.abs(u_icdfm_theta - v_values[:, :-1]), p),
            axis=-1,
        )
        return dCp.reshape(-1, 1), dCm.reshape(-1, 1)

    def Cost(self, theta, u_values, v_values, u_cdf, v_cdf, p):
        v_values = v_values.clone()
        m_batch, m = v_values.shape
        n_batch, n = u_values.shape

        v_cdf_theta = v_cdf - (theta - torch.floor(theta))
        mask_p = v_cdf_theta >= 0
        mask_n = v_cdf_theta < 0

        v_values[mask_n] += torch.floor(theta)[mask_n] + 1
        v_values[mask_p] += torch.floor(theta)[mask_p]

        if torch.any(mask_n) and torch.any(mask_p):
            v_cdf_theta[mask_n] += 1

        v_cdf_theta2 = v_cdf_theta.clone()
        v_cdf_theta2[mask_n] = np.inf
        shift = -torch.argmin(v_cdf_theta2, axis=-1)

        v_cdf_theta = self.roll_by_gather(v_cdf_theta, 1, shift.view(-1, 1))
        v_values = self.roll_by_gather(v_values, 1, shift.view(-1, 1))
        v_values = torch.cat([v_values, v_values[:, 0].view(-1, 1) + 1], dim=1)

        cdf_axis, cdf_axis_sorter = torch.sort(torch.cat((u_cdf, v_cdf_theta), -1), -1)
        cdf_axis_pad = torch.nn.functional.pad(cdf_axis, (1, 0))
        delta = cdf_axis_pad[..., 1:] - cdf_axis_pad[..., :-1]

        u_index = torch.searchsorted(u_cdf, cdf_axis)
        u_icdf = torch.gather(u_values, -1, u_index.clip(0, n - 1))

        v_values = torch.cat([v_values, v_values[:, 0].view(-1, 1) + 1], dim=1)
        v_index = torch.searchsorted(v_cdf_theta, cdf_axis)
        v_icdf = torch.gather(v_values, -1, v_index.clip(0, m))

        if p == 1:
            return torch.sum(delta * torch.abs(u_icdf - v_icdf), axis=-1)
        if p == 2:
            return torch.sum(delta * torch.square(u_icdf - v_icdf), axis=-1)
        return torch.sum(delta * torch.pow(torch.abs(u_icdf - v_icdf), p), axis=-1)

    def roll_by_gather(self, mat, dim, shifts: torch.LongTensor):
        n_rows, n_cols = mat.shape
        if dim == 0:
            arange1 = torch.arange(n_rows, device=mat.device).view((n_rows, 1)).repeat((1, n_cols))
            arange2 = (arange1 - shifts) % n_rows
            return torch.gather(mat, 0, arange2)
        arange1 = torch.arange(n_cols, device=mat.device).view((1, n_cols)).repeat((n_rows, 1))
        arange2 = (arange1 - shifts) % n_cols
        return torch.gather(mat, 1, arange2)

    def emd1D_circle(self, u_values, v_values, u_weights=None, v_weights=None, p=1, require_sort=True):
        n = u_values.shape[-1]
        m = v_values.shape[-1]
        device = u_values.device
        dtype = u_values.dtype

        if u_weights is None:
            u_weights = torch.full((n,), 1 / n, dtype=dtype, device=device)
        if v_weights is None:
            v_weights = torch.full((m,), 1 / m, dtype=dtype, device=device)

        if require_sort:
            u_values, u_sorter = torch.sort(u_values, -1)
            v_values, v_sorter = torch.sort(v_values, -1)
            u_weights = u_weights[..., u_sorter]
            v_weights = v_weights[..., v_sorter]

        if p == 1:
            values_sorted, values_sorter = torch.sort(torch.cat((u_values, v_values), -1), -1)
            cdf_diff = torch.cumsum(torch.gather(torch.cat((u_weights, -v_weights), -1), -1, values_sorter), -1)
            cdf_diff_sorted, cdf_diff_sorter = torch.sort(cdf_diff, axis=-1)

            values_sorted = torch.nn.functional.pad(values_sorted, (0, 1), value=1)
            delta = values_sorted[..., 1:] - values_sorted[..., :-1]
            weight_sorted = torch.gather(delta, -1, cdf_diff_sorter)

            sum_weights = torch.cumsum(weight_sorted, axis=-1) - 0.5
            sum_weights[sum_weights < 0] = np.inf
            inds = torch.argmin(sum_weights, axis=-1)

            levMed = torch.gather(cdf_diff_sorted, -1, inds.view(-1, 1))

            return torch.sum(delta * torch.abs(cdf_diff - levMed), axis=-1)
        raise NotImplementedError("emd1D_circle only implements p=1, matching the official module")

    def sp_sliced_cost(self, Xs, Xt, Us, p=2, u_weights=None, v_weights=None):
        n_projs, d, k = Us.shape
        n, _ = Xs.shape
        m, _ = Xt.shape

        Xps = torch.matmul(torch.transpose(Us, 1, 2)[:, None], Xs[:, :, None]).reshape(n_projs, n, 2)
        Xpt = torch.matmul(torch.transpose(Us, 1, 2)[:, None], Xt[:, :, None]).reshape(n_projs, m, 2)

        Xps = F.normalize(Xps, p=2, dim=-1)
        Xpt = F.normalize(Xpt, p=2, dim=-1)

        Xps = (torch.atan2(-Xps[:, :, 1], -Xps[:, :, 0]) + np.pi) / (2 * np.pi)
        Xpt = (torch.atan2(-Xpt[:, :, 1], -Xpt[:, :, 0]) + np.pi) / (2 * np.pi)

        if p == 1:
            w1 = self.emd1D_circle(Xps, Xpt, u_weights=u_weights, v_weights=v_weights)
        else:
            w1 = self.binary_search_circle(Xps, Xpt, p=p, u_weights=u_weights, v_weights=v_weights)

        return torch.mean(w1)

    def sp_swd_loss(self, Xs, Xt, num_projections, device, u_weights=None, v_weights=None, p=2):
        d = Xs.shape[1]
        Z = torch.randn((num_projections, d, 2), device=device)
        U, _ = torch.linalg.qr(Z)
        return self.sp_sliced_cost(Xs, Xt, U, p=p, u_weights=u_weights, v_weights=v_weights)
