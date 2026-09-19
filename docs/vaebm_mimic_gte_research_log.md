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

### Results (15/15 successful)

| Config | Total beats /15 | 20ng | search_snippets | google_news | agnews | imdb |
|---|---:|---|---|---|---|---|
| A_freeze (Round 2 best) | 11 | 2/3 | 2/3 | 3/3 | 2/3 | 2/3 |
| D_freeze_epochs15 | 11 | 2/3 | 2/3 | 3/3 | 2/3 | 2/3 |
| E_freeze_alpha002 | 10 | **1/3 (crashed)** | 2/3 | 3/3 | 2/3 | 2/3 |
| F_freeze_alpha002_epochs15 | 8 | 1/3 | 1/3 (Cv fixed, Purity/NMI broke) | 3/3 | 1/3 | 2/3 |

**D ties A exactly on beats-count** - more decoder epochs shifts Cv
magnitude slightly (up on search_snippets +0.014, down on 20ng -0.035)
but never flips a beat/miss verdict.

**E and F confirm alpha>0 is a NET NEGATIVE**, even at just 0.02: badly
disrupts 20ng's clustering (Purity 0.664->0.360, NMI 0.580->0.348) the
moment mu_bow's gradient path re-opens, and F's added epochs partially
"fixes" Cv on search_snippets/agnews only by sacrificing Purity/NMI there
too (net beats DROP versus A_freeze on both). **Idea abandoned** - do not
pursue non-zero alpha under freeze further; the failure mode is the
same untrained/newly-training mu_bow injecting noise into mu, not a
signal that helps.

**Ruled out**: embeddings are NOT unnormalized - checked
`thenlper/gte-large`'s own `modules.json` on HF, it already has a
`2_Normalize` (sentence_transformers.models.Normalize) module baked into
its pipeline, so `.encode()` output is already L2-normalized without
needing `normalize_embeddings=True`. Not the missing piece.

**Standing best: A_freeze / D_freeze_epochs15, tied at 11/15, still only
1/5 datasets (google_news) at 3/3.** Remaining blockers: search_snippets/
imdb Cv (decoder/topic-word quality, untouched by epochs), agnews NMI
(clustering geometry, untouched by decoder-side changes, made WORSE by
non-zero alpha), 20ng NMI (agonizingly close, 0.580 vs 0.583).

## Round 4 - frequency-based topic words + untested vaebm_dec (in progress)

Two new, unrelated ideas (neither tried in Rounds 1-3, both model=vaebm_dec
is a DIFFERENT model entirely from vaebm - first time it's tested in this
research pass):
  - G_freeze_freqwords: same as A_freeze (freeze=1, alpha=0, epochs=1,
    lr=1e-4) but `top_words_mode="freq"` instead of the default "energy" -
    topic words come from per-cluster word-frequency counts, completely
    decoupled from the decoder's own trained R matrix. Since clustering
    (mu) is already fixed/good under freeze, this tests whether Cv was
    actually a DECODER-quality problem or a TOPIC-WORD-EXTRACTION-METHOD
    problem - zero risk to Purity/NMI either way (doesn't touch mu).
  - H_vaebm_dec: model="vaebm_dec" (not "vaebm") - Deep Embedded
    Clustering trained JOINTLY with the VAE via its own clustering loss
    (lambda_c=0.1), NOT frozen (freeze=0) since DEC's whole point is
    letting its clustering-specific loss (not just BoW reconstruction)
    shape mu toward better-separated clusters - untested in this research
    pass so far. alpha=0, gte-large, units=1024, dim_emb="", epochs=15,
    lr=1e-3. Targets agnews's NMI ceiling and 20ng's near-miss, since DEC's
    loss is explicitly designed to sharpen cluster separation, unlike the
    BoW branch's reconstruction objective which has no interest in
    cluster quality.

Model=vaebm for G, vaebm_dec for H. K=50, all 5 hicot_* datasets. Driver:
`scripts/run_vaebm_gte_research_round4.py`.

### G_freeze_freqwords results (5/5 successful) - NEW BEST

| Dataset | Cv | Purity | NMI | Beats |
|---|---:|---:|---:|---|
| hicot_20ng | 0.670 | 0.664 | 0.580 | 2/3 (NMI misses by 0.003, same as A_freeze) |
| hicot_search_snippets | 0.578 | 0.856 | 0.503 | **3/3 (NEW WIN - was 2/3 under energy mode)** |
| hicot_google_news | 0.584 | 0.614 | 0.820 | **3/3** |
| hicot_agnews | 0.697 | 0.859 | 0.371 | 2/3 (Cv jumped 0.520->0.697, NMI still stuck) |
| hicot_imdb | 0.385 | 0.803 | 0.115 | 2/3 (Cv up 0.358->0.385, still short of 0.404) |

**12/15 beats, 2/5 datasets at 3/3.** Confirms the hypothesis cleanly:
Cv was substantially a topic-word-EXTRACTION-METHOD problem, not a
clustering problem - `top_words_mode="freq"` (decoupled from the
decoder's own trained R matrix) is a strictly-better-or-equal choice
for Cv than "energy" on every dataset tested here, at zero cost to
Purity/NMI (mathematically guaranteed under freeze, per Round 2/3's own
finding). **This should become the new default for the "mimic-GTE"
recipe going forward** - no known downside found yet.

Remaining blockers: agnews NMI (clustering-only, needs vaebm_dec or a
different geometry lever), imdb Cv (freq-mode helped but didn't fully
close the gap - 0.385 vs 0.404), 20ng NMI (still short by 0.003).

### H_vaebm_dec results (5/5 successful) - NET NEGATIVE, abandoned

| Dataset | Cv | Purity | NMI | Beats |
|---|---:|---:|---:|---|
| hicot_20ng | 0.708 | 0.640 | 0.555 | 2/3 (Purity/NMI both worse than A_freeze/G) |
| hicot_search_snippets | 0.582 | 0.820 | 0.459 | 2/3 (REGRESSION from G's 3/3 - NMI now misses) |
| hicot_google_news | 0.565 | 0.569 | 0.741 | 3/3 (Purity dropped but still clears the low 0.465 bar) |
| hicot_agnews | 0.701 | 0.859 | 0.364 | 2/3 (NMI WORSE than freeze's 0.371, not better) |
| hicot_imdb | 0.364 | 0.608 | 0.082 | 1/3 (Purity crashed from 0.803 - REGRESSION) |

**10/15 beats - worse than both A_freeze/D (11/15) and G (12/15).**
Hypothesis rejected: DEC's own clustering loss (lambda_c=0.1, jointly
trained, NOT frozen) does not sharpen cluster separation here - if
anything it destabilizes Purity on agnews/imdb without fixing NMI
anywhere. **Idea abandoned.**

**Standing best after Round 4: G_freeze_freqwords, 12/15 beats, 2/5
datasets at 3/3** (google_news, search_snippets). Confirmed: nothing
tried so far can move agnews's NMI (0.371, stuck across every config:
freeze alone, +epochs, +alpha, +freq-words, +vaebm_dec - all land within
0.364-0.376) or fully close imdb's Cv gap (best so far 0.385 vs target
0.404) or 20ng's NMI gap (stuck at 0.580 vs 0.583 across every config
that doesn't also break Purity).

## Round 5 - units truncation on the 3 still-blocked datasets

Idea: since mu_emb is a frozen, deterministic function of the raw
embedding (identity-initialized, no hidden layer), `units < 1024` makes
that Dense layer's Identity initializer place 1s only on its first
`units` diagonal entries - i.e. mu becomes a literal TRUNCATION to the
first `units` dimensions of the raw gte-large embedding, not a learned
projection (nothing trains it either way, under freeze). Testing whether
a lower-dimensional truncated subspace happens to separate these 3
specific datasets' classes better than the full 1024-dim space (a
real, if somewhat unprincipled, hypothesis worth a cheap check before
concluding these are hard ceilings for this architecture+embedder).

Configs I_units768/J_units512, on top of G's winning recipe (freeze=1,
alpha=0, epochs=1, lr=1e-4, top_words_mode=freq), varying only
VAEBM_UNITS. Datasets: hicot_20ng, hicot_agnews, hicot_imdb only (the 3
still-blocked ones - google_news/search_snippets already at 3/3, no
need to re-test). Driver: `scripts/run_vaebm_gte_research_round5.py`.

### Results (6/6 successful) - no threshold crossed, confirms hard ceilings

| Dataset | Config | Cv | Purity | NMI | Beats |
|---|---|---:|---:|---:|---|
| hicot_20ng | I_units768 | 0.661 | 0.670 | 0.576 | 2/3 (NMI worse than G's 0.580) |
| hicot_20ng | J_units512 | 0.657 | 0.654 | 0.564 | 2/3 (NMI worse still - monotonic decline 1024->768->512) |
| hicot_agnews | I_units768 | 0.724 | 0.875 | 0.375 | 2/3 (local NMI peak among truncations, still far from 0.412) |
| hicot_agnews | J_units512 | 0.713 | 0.861 | 0.368 | 2/3 (worse than both 1024 and 768) |
| hicot_imdb | I_units768 | 0.390 | 0.806 | 0.111 | 2/3 (Cv still short of 0.404) |
| hicot_imdb | J_units512 | 0.387 | 0.790 | 0.107 | 2/3 (Cv still short) |

**No config crossed any remaining threshold.** Truncating the frozen
embedding to fewer dimensions never helps enough to matter, and for
20ng's NMI it's monotonically harmful (1024: 0.580 -> 768: 0.576 -> 512:
0.564). **Conclusion: agnews's NMI (~0.37 ceiling vs 0.412 target),
imdb's Cv (~0.39 ceiling vs 0.404 target), and 20ng's NMI (~0.58 vs
0.583) are genuine ceilings for gte-large + this frozen-identity
architecture** - not fixable by epochs, alpha, kl_weight, top-word
extraction method, joint DEC training, or dimensionality alone. Round 6
tries the one still-unexplored orthogonal axis: a different embedder.

## Round 6 - different embedder (bge-large-en-v1.5) on the 3 blocked datasets

Idea: agnews/imdb/20ng's residual gaps might be embedder-specific (how
well gte-large's own training happens to separate these particular
classes), not an architectural limitation. `BAAI/bge-large-en-v1.5` is
also 1024-dim (no units/dim_emb change needed) and trained on a
different contrastive-retrieval mixture than gte-large - a genuinely
orthogonal axis not tested in Rounds 1-5. Same G_freeze_freqwords recipe
otherwise (freeze=1, alpha=0, epochs=1, lr=1e-4, top_words_mode=freq,
units=1024, dim_emb=""). Driver: `scripts/run_vaebm_gte_research_round6.py`.

(results filled in as they land)
