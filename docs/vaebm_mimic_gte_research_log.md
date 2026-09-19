# VAE-BM "beat HiCOT via GTE-mimicry" research log

2026-09-19, user-authorized autonomous research pass on FutureLab.
Goal: modify VAE-BM (any variant) so its topic-experiment K=50 numbers on
Cv/Purity/NMI beat HiCOT's own reported K=50 numbers on at least 4/5
datasets, while remaining a VAE-BM-derived model (not literally raw
embedding+KMeans). TD is not optimized for this task.

## HiCOT K=50 reference (target)

| Dataset (hicot_* id) | Cv | Purity | NMI |
|---|---:|---:|---:|
| SearchSnippets (`hicot_search_snippets`) | 0.460 | 0.818 | 0.478 |
| GoogleNews (`hicot_google_news`) | 0.454 | 0.465 | 0.657 |
| 20NG (`hicot_20ng`) | 0.451 | 0.626 | 0.583 |
| AGNews (`hicot_agnews`) | 0.446 | 0.857 | 0.412 |
| IMDB (`hicot_imdb`) | 0.404 | 0.737 | 0.082 |

## Round 0/1 - baseline "mimic-GTE" config (job 1612, cancelled after 21/80
combos to redirect FutureLab to this research task - checkpoint preserved
at `results/vaebm_mimic_gte_topic_1612/checkpoint.json`)

Config: `VAEBM_EMBEDDER=thenlper/gte-large VAEBM_UNITS=1024 VAEBM_DIM_EMB=""
VAEBM_ALPHA=0.0 VAEBM_LR=1e-4 VAEBM_EPOCHS=1` (kl_weight/freeze not yet
added at this point - effectively kl_weight=1.0, freeze=False).
Commits: b84723d (units/lr), ebff885 (Keras 3 empty-Sequential fix).

Model=vaebm, K=50, hicot_* variant:

| Dataset | Cv | Purity | NMI | Beats (Cv/Purity/NMI) |
|---|---:|---:|---:|---|
| hicot_20ng | 0.775 | 0.547 | 0.486 | 1/3 (Cv only) |
| hicot_search_snippets | 0.438 | 0.845 | 0.492 | 2/3 (Purity, NMI) |
| hicot_google_news | 0.586 | 0.619 | 0.803 | **3/3** |
| hicot_agnews | 0.587 | 0.836 | 0.343 | 1/3 (Cv only) |
| hicot_imdb | 0.358 | 0.787 | 0.102 | 2/3 (Purity, NMI) |

**1/5 datasets at 3/3.** Blockers: 20ng/agnews weak on Purity+NMI;
search_snippets/imdb weak on Cv only (close).

Also got one `vaebm_poe|hicot_20ng|50` data point: Cv=0.785, Purity=0.553,
NMI=0.486 - same pattern as plain vaebm on this dataset (1/3).

## Round 2 - isolate kl_weight vs freeze_embedding_branch (in progress)

New knobs added this round (commit 58358ae): `VAEBM_KL_WEIGHT` (beta-VAE
style weight on the KL term, default 1.0 unchanged) and
`VAEBM_FREEZE_EMB` (freezes mlp_emb/mu_emb/log_sigma_emb right after
construction, so mu_emb stays at its identity-init value - a near-exact
copy of the raw embedding - for the entire run). Both required fixing a
Keras 3 bug first (zero-layer Sequential rejected - commits ebff885,
7b918cd) and required knowing epochs=0 crashes vaebm_poe/vaebm_dec/
vaebm_ckpt's custom per-epoch loop (kept epochs=1).

Base config held fixed at Round 1's values (gte-large, units=1024,
dim_emb="", alpha=0, lr=1e-4, epochs=1), varying only:
  - A: freeze=1, kl_weight=1.0 (isolate freeze alone)
  - B: freeze=0, kl_weight=0.1 (isolate low-KL alone)
  - C: freeze=1, kl_weight=0.1 (combined)

Model=vaebm only (cleanest/fastest case), K=50, all 5 hicot_* target
datasets. Driver: `scripts/run_vaebm_gte_research_round2.py`.

### Results (15/15 combos successful)

| Dataset | Config | Cv | Purity | NMI | Beats |
|---|---|---:|---:|---:|---|
| hicot_20ng | A_freeze | 0.796 | 0.664 | 0.580 | 2/3 (NMI misses by 0.003) |
| hicot_20ng | B_lowkl | 0.771 | 0.586 | 0.502 | 1/3 |
| hicot_20ng | C_freeze_lowkl | 0.796 | 0.664 | 0.580 | 2/3 (= A_freeze, bit-identical) |
| hicot_search_snippets | A_freeze | 0.382 | 0.856 | 0.503 | 2/3 (Cv weak) |
| hicot_search_snippets | B_lowkl | 0.359 | 0.831 | 0.479 | 2/3 (Cv weaker) |
| hicot_search_snippets | C_freeze_lowkl | 0.382 | 0.856 | 0.503 | 2/3 (= A_freeze) |
| hicot_google_news | A_freeze | 0.546 | 0.614 | 0.820 | **3/3** |
| hicot_google_news | B_lowkl | 0.533 | 0.622 | 0.823 | **3/3** |
| hicot_google_news | C_freeze_lowkl | 0.546 | 0.614 | 0.820 | **3/3** (= A_freeze) |
| hicot_agnews | A_freeze | 0.520 | 0.859 | 0.371 | 2/3 (NMI weak) |
| hicot_agnews | B_lowkl | 0.500 | 0.868 | 0.371 | 2/3 (NMI weak, same ceiling) |
| hicot_agnews | C_freeze_lowkl | 0.520 | 0.859 | 0.371 | 2/3 (= A_freeze) |
| hicot_imdb | A_freeze | 0.358 | 0.803 | 0.115 | 2/3 (Cv weak) |
| hicot_imdb | B_lowkl | 0.360 | 0.799 | 0.115 | 2/3 (Cv weak, same ceiling) |
| hicot_imdb | C_freeze_lowkl | 0.358 | 0.803 | 0.115 | 2/3 (= A_freeze) |

**Key finding**: `C_freeze_lowkl` is bit-for-bit identical to `A_freeze` on
every single dataset. This is expected, not a bug: once
`freeze_embedding_branch=True`, `mu`/`log_sigma` come entirely from frozen
weights (given `alpha=0`), so the KL term's gradient w.r.t. every
trainable weight is exactly zero regardless of `kl_weight`'s value -
`kl_weight` is a no-op whenever freeze is on. Corollary: under freeze,
Purity/NMI are fixed at their epoch-0 value forever (KMeans always
clusters the same frozen `mu`) - only Cv (decoder/topic-word-driven) can
still change with more training. This means more epochs/higher LR under
freeze is a **zero-risk lever for Cv only**.

**Best lever so far**: `A_freeze` (freeze=1, kl_weight irrelevant).
Beats count: baseline 9/15 -> A_freeze 11/15 -> B_lowkl 10/15.
1/5 datasets at 3/3 (google_news). `hicot_20ng` misses only by NMI=0.003.
Remaining blockers: search_snippets/imdb Cv (decoder-fixable, safe under
freeze), agnews NMI (clustering-fixed under full freeze - needs a
different lever, e.g. tiny non-zero alpha to let a bit of BoW signal
back into mu without giving up most of freeze's benefit).

## Round 3 - decoder-only epoch boost + tiny-alpha partial freeze (in progress)

Two independent ideas tested together (freeze=1, alpha=0 for D; freeze=1,
alpha=0.02 for E/F - alpha>0 re-enables mu_bow's gradient path even
though the embedding branch stays frozen, since mu = alpha*mu_bow +
(1-alpha)*mu_emb only zeroes mu_bow's contribution when alpha is exactly
0):
  - D_freeze_epochs15: freeze=1, alpha=0, epochs=15, lr=1e-3 (zero risk to
    Purity/NMI per the finding above - only tests whether more decoder
    training improves Cv on search_snippets/imdb)
  - E_freeze_alpha002: freeze=1, alpha=0.02, epochs=1, lr=1e-4 (tests
    whether a tiny BoW blend into mu, while embedding stays frozen, moves
    agnews/20ng's NMI)
  - F_freeze_alpha002_epochs15: combines both

Model=vaebm, K=50, all 5 hicot_* datasets. Driver:
`scripts/run_vaebm_gte_research_round3.py`.

(results filled in as they land)
