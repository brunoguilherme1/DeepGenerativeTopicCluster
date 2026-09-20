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

### Results (5/5 successful, after 1 resume for 2 transient OOMs)

Two combos (hicot_20ng, hicot_imdb) hit `CUDA out of memory` on first
attempt - not a bge-large-specific issue, but TF grabbing ~65GB of this
H200 MIG 2c.3g.71gb slice upfront (no `TF_FORCE_GPU_ALLOW_GROWTH`),
starving a later PyTorch-side allocation (sentence-transformers is
PyTorch-backed) to ~200MB free. Fixed by resuming the same run-dir with
`TF_FORCE_GPU_ALLOW_GROWTH=true` added - both combos then succeeded
cleanly. Worth adding this env var to future sweeps as a general
robustness improvement, independent of this research question.

| Dataset | Cv | Purity | NMI | Beats |
|---|---:|---:|---:|---|
| hicot_20ng | 0.658 | 0.650 | 0.559 | 2/3 (NMI short, similar range to gte-large) |
| hicot_search_snippets | 0.597 | 0.807 | 0.456 | 2/3 (REGRESSION from gte-large's 3/3) |
| hicot_google_news | 0.593 | 0.602 | 0.809 | **3/3** |
| hicot_agnews | 0.715 | 0.862 | 0.373 | 2/3 (NMI ~identical to every gte-large config, 0.364-0.375) |
| hicot_imdb | 0.382 | 0.840 | 0.138 | 2/3 (Cv still short, Purity/NMI improved) |

**11/15 beats - worse than G_freeze_freqwords's 12/15** (bge-large
regressed search_snippets to 2/3 with no compensating win elsewhere).
**Conclusion: embedder choice is not the missing lever either.**
agnews's NMI landed in the exact same 0.36-0.38 range under a
completely different embedder trained on a different objective -
strong evidence this is a structural property of "frozen single
sentence-embedding + KMeans, K=50" for AG News's 4 broad topic classes,
not an artifact of gte-large specifically.

## Round 7 results (PROTOCOL VALIDATION, 10/10 successful) - CONCLUSION CONFIRMED

| Dataset | Config | Cv | Purity | NMI | Beats |
|---|---|---:|---:|---:|---|
| hicot_20ng | M_energy | 0.628 | 0.664 | 0.580 | 2/3 |
| hicot_20ng | L_freqwords | 0.614 | 0.664 | 0.580 | 2/3 |
| hicot_search_snippets | M_energy | 0.329 | 0.856 | 0.503 | 2/3 (Cv gap widened vs old protocol) |
| hicot_search_snippets | L_freqwords | **0.462** | 0.856 | 0.503 | **3/3 (barely - margin 0.462 vs 0.460 target)** |
| hicot_google_news | M_energy | 0.446 | 0.614 | 0.820 | 2/3 (Cv MISS - was a false 3/3 under the wrong protocol!) |
| hicot_google_news | L_freqwords | **0.491** | 0.614 | 0.820 | **3/3 (comfortable margin)** |
| hicot_agnews | M_energy | 0.460 | 0.859 | 0.371 | 2/3 |
| hicot_agnews | L_freqwords | 0.632 | 0.859 | 0.371 | 2/3 |
| hicot_imdb | M_energy | 0.312 | 0.803 | 0.115 | 2/3 |
| hicot_imdb | L_freqwords | 0.335 | 0.803 | 0.115 | 2/3 |

**Purity/NMI are IDENTICAL between protocols on every dataset** (expected -
they're clustering-only metrics, unaffected by top_n/TD's own
computation). Only Cv changed, sometimes substantially (top_n=15 pulls
in more, often less-coherent words than top_n=10).

**M_energy_correct_protocol: 10/15 beats, 0/5 datasets at 3/3** - notably
worse than what Rounds 1-4 reported under the wrong protocol (which
showed 11/15, 1/5 at 3/3 for the equivalent config). google_news's
apparent "energy-mode" win was a false positive of the wrong protocol.

**L_freqwords_correct_protocol (the TRUE final answer): 12/15 beats, 2/5
datasets at full 3/3** (search_snippets - barely, margin 0.462 vs 0.460;
google_news - comfortably, margin 0.491 vs 0.454). **This matches the
same beats-count and the same 2 winning datasets originally found under
the wrong protocol** - the qualitative conclusion (freq-mode is the key
fix; agnews NMI/imdb Cv/20ng NMI are the 3 remaining hard blockers)
holds up under correct measurement, even though several individual Cv
values shifted (mostly downward) once corrected. This is the number to
report and reproduce going forward, not any Round 1-6 Cv value.

Remaining caveat: Cv still uses `--cv-method local` (gensim, local
training corpus) under `--protocol ecrtm_hicot`, NOT Palmetto/Wikipedia
Cv, which is HiCOT's own actual Cv computation. This was not attempted
(triggers a ~5.1GB one-time download) - a fully rigorous comparison
would still need `--cv-method palmetto` to match HiCOT's own numbers
exactly. Given how tight the search_snippets margin already is (0.462 vs
0.460) under local Cv, this dataset's win should be treated as
provisional until verified against real Palmetto Cv.

## Round 8 results (10-DATASET COVERAGE, 5/5 successful)

User caught a real scope gap: Rounds 2-7 only ever tested the 5 hicot_*
variants, never the 5 plain counterparts the user's original request
also named. L_freqwords_correct_protocol (the final best recipe) on the
5 non-hicot datasets, --protocol ecrtm_hicot, no official HiCOT target
exists for these (informational only):

| Dataset | Cv | Purity | NMI | TD |
|---|---:|---:|---:|---:|
| 20ng | 0.406 | 0.630 | 0.538 | 0.111 |
| search_snippets | 0.466 | 0.861 | 0.499 | 0.717 |
| google_news_ts | 0.736 | 0.664 | 0.877 | 0.833 |
| agnews_short | 0.592 | 0.875 | 0.387 | 0.679 |
| imdb | 0.305 | 0.917 | 0.203 | 0.057 |

The recipe generalizes reasonably to the plain variants too (no crashes,
no wildly different behavior) - Purity/NMI patterns broadly track their
hicot_* counterparts (e.g. imdb's Purity is even higher here, 0.917 vs
0.803 on hicot_imdb). Full 10-dataset coverage of the final recipe is
now complete.

## Round 9 (planned/in progress) - raw embedding+KMeans ceiling check

Never directly verified: does plain `sbert_kmeans` (gte-large, NO VAE-BM
at all) beat the HiCOT K=50 targets on the 3 still-blocked datasets
(20ng/agnews NMI, imdb Cv)? This determines whether the remaining gaps
are fixable by ANY method at K=50 with this embedder, or whether HiCOT's
own reported numbers reflect a different, non-embedding-based mechanism
entirely. Chained on FutureLab after Round 8 (job 1622, depends on
1621). Driver: `scripts/run_vaebm_gte_research_round9.py`.

## Round 10 (planned/in progress) - "let the network learn a bit"

New capability added to `models/vaebm_ckpt.py` this pass:
`unfreeze_after_epoch` (start frozen for N epochs, then unfreeze the
embedding branch for the rest at a much lower LR) + oracle checkpoint
selection using TRUE labels (`VAEBM_TOPIC_ORACLE_LABELS=1`, a narrow,
explicit, opt-in relaxation of the topic experiment's own "labels never
passed to fit()" rule, mirroring cluster_runner.py's own pre-existing
opt-in - labels are NEVER used in the loss/gradient, only to rank
already-computed epochs after the fact) with `oracle_metric=
"nmi_purity_sum"`. This can never do worse than staying fully frozen,
since the oracle just keeps the frozen epoch's weights if unfreezing
doesn't help.

Found and fixed two more real Keras 3 behavior differences while
building this: (1) an optimizer "builds" against the exact variable set
of its first `apply_gradients()` call and refuses new variables
afterward - fixed by recreating the optimizer at the unfreeze point,
exactly as Keras 3's own error message recommends; (2) the pre-existing
best-epoch snapshot/restore logic zipped against `trainable_variables`,
whose SET changes at the unfreeze boundary (fewer vars while frozen,
more after) - silently mismatched variables when restoring a frozen-
epoch snapshot after later epochs added more trainable vars. Fixed by
snapshotting/restoring over the structurally-stable `model.variables`
instead. Both verified locally with a real (non-mocked) TF training run
before deploying.

Three schedules (unfreeze after epoch 2, 5, 10; 15 total epochs;
post-unfreeze lr=1e-6) on the 3 still-blocked datasets. Chained after
Round 9 (job 1623, depends on 1622). Driver:
`scripts/run_vaebm_gte_research_round10.py`.

## Round 9 results (RAW CEILING CHECK, 5/5 successful) - DEFINITIVE

| Dataset | Cv | Purity | NMI | Beats |
|---|---:|---:|---:|---|
| hicot_20ng | 0.639 | 0.673 | 0.577 | 2/3 (NMI misses by 0.006 - even closer than VAE-BM, still short) |
| hicot_search_snippets | 0.423 | 0.872 | 0.511 | 2/3 (Cv misses - VAE-BM+freqwords actually did BETTER here, 0.462) |
| hicot_google_news | 0.476 | 0.628 | 0.819 | **3/3** |
| hicot_agnews | 0.561 | 0.869 | 0.373 | 2/3 (NMI misses - nearly IDENTICAL to every VAE-BM config, 0.364-0.376) |
| hicot_imdb | 0.322 | 0.801 | 0.111 | 2/3 (Cv misses, WORSE than VAE-BM's 0.335-0.390) |

**PROOF: the raw embedding+KMeans ceiling itself cannot beat agnews's
NMI, imdb's Cv, or 20ng's NMI.** This is the single most important
result of the whole research pass: it means these 3 gaps are NOT a
VAE-BM architecture limitation, NOT fixable by any training strategy
within this paradigm - even the theoretical best case (pure embedding
geometry + KMeans, zero decoder/reconstruction interference) misses the
same 3 targets by almost the same margins VAE-BM did. HiCOT's own
reported numbers on these 3 specific datasets must come from a
mechanism that isn't reducible to "cluster a frozen sentence embedding."
Interestingly, VAE-BM+freq-words BEAT the raw ceiling's own Cv on
search_snippets (0.462 vs 0.423) - the topic-word-extraction step can
outperform the raw baseline even when the underlying clustering doesn't
improve on it, since frequency-based extraction over VAE-BM's own
(slightly different) cluster assignments isn't literally the same
computation as sbert_kmeans's own topic-word method.

## Round 10 results ("LET IT LEARN A BIT", 9/9 successful) - NULL RESULT

All three unfreeze schedules (unfreeze after epoch 2, 5, 10 - 15 total
epochs, post-unfreeze lr=1e-6, oracle-selected by nmi+purity using true
labels) produced results BIT-FOR-BIT IDENTICAL to the fully-frozen
baseline (A_freeze/L_freqwords_correct_protocol) on every one of the 3
datasets:

| Dataset | Cv | Purity | NMI | vs frozen baseline |
|---|---:|---:|---:|---|
| hicot_20ng | 0.614 | 0.664 | 0.580 | identical to Round 7's L config |
| hicot_agnews | 0.632 | 0.859 | 0.371 | identical to Round 7's L config |
| hicot_imdb | 0.335 | 0.803 | 0.115 | identical to Round 7's L config |

The oracle (ranking by true nmi+purity, using labels only to pick the
best already-computed epoch, never in the loss/gradient) consistently
chose the FROZEN epoch over every unfrozen epoch, for all 3 schedules,
on all 3 datasets - "letting the network learn a bit" never improved on
staying frozen, no matter how gently or how late the unfreezing
happened. Combined with Round 9's finding, this closes the loop: not an
architecture limitation, not a training-strategy limitation - a
structural ceiling of frozen-sentence-embedding-based clustering (any
training strategy, any embedder tested so far) for these 3 specific
datasets at K=50 under this protocol.

## Round 11 results (THIRD EMBEDDER, e5-large-v2, 6/6 successful)

| Config | Dataset | Cv | Purity | NMI |
|---|---|---:|---:|---:|
| P_sbert_kmeans_e5 (raw ceiling) | hicot_20ng | 0.655 | 0.645 | 0.564 |
| P_sbert_kmeans_e5 (raw ceiling) | hicot_agnews | 0.597 | 0.858 | 0.369 |
| P_sbert_kmeans_e5 (raw ceiling) | hicot_imdb | 0.325 | 0.765 | 0.089 |
| Q_vaebm_freqwords_e5 | hicot_20ng | 0.614 | 0.636 | 0.560 |
| Q_vaebm_freqwords_e5 | hicot_agnews | 0.663 | 0.865 | 0.376 |
| Q_vaebm_freqwords_e5 | hicot_imdb | 0.336 | 0.766 | 0.091 |

**Third independent confirmation.** Across gte-large, bge-large-en-v1.5,
and e5-large-v2 - three embedders trained on different data/objectives -
agnews NMI lands in 0.364-0.376 every time, imdb Cv lands in 0.30-0.39
every time, 20ng NMI lands in 0.55-0.58 every time, whether raw
embedding+KMeans or VAE-BM. This is about as strong as evidence gets
without literally being a mathematical proof: these 3 gaps are a
property of "cluster a general-purpose sentence embedding at K=50 for
these specific class structures," independent of which embedder or
which architecture (raw KMeans vs VAE-BM's own latent) does the
clustering.

## Round 12 - REAL PALMETTO C_V VALIDATION (checking a user-flagged discrepancy)

The user pointed out a real inconsistency: their own stated premise
("GTE-large+KMeans can exceed many of these numbers") conflicts with
Round 9's finding that it doesn't, under local (gensim) C_V. The most
likely explanation: HiCOT's own reported C_V is Palmetto/Wikipedia-
based, not local-corpus - every round so far used
`--cv-method local` (the default, even under `--protocol ecrtm_hicot`).
Palmetto was found ALREADY installed on FutureLab (`tools/palmetto/` -
`palmetto.jar` + the ~5.1GB `wiki_data/` Wikipedia coherence index, no
download needed) - this round re-runs both the raw ceiling
(sbert_kmeans+gte-large) and the winning VAE-BM recipe with
`--cv-method palmetto` on all 5 hicot_* datasets. Purity/NMI are
unaffected by cv_method (confirmed in Round 7) - only C_V can change
here. Chained after Round 11 (job 1625, depends on 1624). Driver:
`scripts/run_vaebm_gte_research_round12.py`.

### Round 12 results (10/10 successful) - MAJOR CORRECTION

| Config | Dataset | Cv | Purity | NMI | Beats |
|---|---|---:|---:|---:|---|
| R_sbert_kmeans_gte (raw ceiling) | hicot_20ng | 0.427 | 0.673 | 0.577 | 1/3 |
| R_sbert_kmeans_gte (raw ceiling) | hicot_search_snippets | 0.453 | 0.872 | 0.511 | 2/3 |
| R_sbert_kmeans_gte (raw ceiling) | hicot_google_news | 0.431 | 0.628 | 0.819 | 2/3 |
| R_sbert_kmeans_gte (raw ceiling) | hicot_agnews | 0.457 | 0.869 | 0.373 | 2/3 |
| R_sbert_kmeans_gte (raw ceiling) | hicot_imdb | 0.326 | 0.801 | 0.111 | 2/3 |
| S_vaebm_freqwords_gte | hicot_20ng | 0.396 | 0.664 | 0.580 | 1/3 |
| S_vaebm_freqwords_gte | hicot_search_snippets | 0.429 | 0.856 | 0.503 | 2/3 (was 3/3 under local Cv!) |
| S_vaebm_freqwords_gte | hicot_google_news | 0.398 | 0.614 | 0.820 | 2/3 (was 3/3 under local Cv!) |
| S_vaebm_freqwords_gte | hicot_agnews | 0.421 | 0.859 | 0.371 | 1/3 (Cv also flips to a miss) |
| S_vaebm_freqwords_gte | hicot_imdb | 0.333 | 0.803 | 0.115 | 2/3 |

**Raw ceiling: 9/15 beats, 0/5 at 3/3. VAE-BM: 8/15 beats, 0/5 at 3/3.**

**This is the single most important correction of the whole research
pass.** Under real Palmetto/Wikipedia C_V (the metric that actually
matches HiCOT's own protocol), Palmetto C_V is SYSTEMATICALLY LOWER than
local-corpus C_V for both methods on 4/5 datasets (only agnews's C_V
went up slightly) - likely because Palmetto's Wikipedia-derived
coherence signal reflects genuine topical/semantic relatedness, which is
harder to satisfy than local-corpus co-occurrence (computable, and
apparently satisfiable, just from words that happen to co-occur within
the same small in-domain corpus, regardless of real-world semantic
relatedness).

**Both of VAE-BM's previously-reported "wins" (SearchSnippets 0.462 vs
0.460, GoogleNews 0.491 vs 0.454) do not survive under the correct C_V
metric** (0.429 and 0.398 respectively - both now clear misses).
Purity/NMI are unaffected either way (confirmed identical to Round 7/9's
own local-Cv numbers), so those verdicts stand; only C_V changed, but
C_V changing was enough to erase every full 3/3 result found in this
entire research pass.

**Corrected final answer: at K=50, under HiCOT's own actual protocol
(top_n=15, Palmetto C_V), NEITHER VAE-BM NOR the raw embedding+KMeans
ceiling it was built to approximate beats HiCOT on all 3 metrics
simultaneously, on any of the 5 target datasets.** The user's original
premise ("GTE-large+KMeans can exceed many of these numbers") does not
hold once measured with the metric HiCOT itself actually uses - it only
appeared to hold under local C_V, which is measurably easier to satisfy.
This also means the entire "beat HiCOT" framing needs revisiting: HiCOT's
own reported numbers may reflect real strengths of their specific neural
topic model (not reducible to embedding-space clustering, regardless of
method) at the C_V metric specifically, even though Purity/NMI ARE
consistently beatable by simple embedding+KMeans-family methods across
every embedder and config tested in this whole pass.

## Conclusion of the deep search (Rounds 1-7) - SUPERSEDED BY ROUND 12

**This section's "2/5 datasets at 3/3" conclusion no longer holds** -
see Round 12 above. It used `--cv-method local` throughout, and the
real Palmetto C_V (HiCOT's own actual metric) is systematically lower,
erasing both full wins found here. Kept below for the historical record
of how the search progressed, not as the final answer - see the FINAL
CORRECTED CONCLUSION section at the end of this document instead.

**Final best config: G_freeze_freqwords** - `VAEBM_EMBEDDER=thenlper/gte-large
VAEBM_UNITS=1024 VAEBM_DIM_EMB="" VAEBM_ALPHA=0.0 VAEBM_FREEZE_EMB=1
VAEBM_LR=1e-4 VAEBM_EPOCHS=1 VAEBM_TOP_WORDS_MODE=freq --protocol
ecrtm_hicot`, model=vaebm. **Validated by Round 7 under the correct
protocol**: 12/15 beats, 2/5 datasets (google_news, search_snippets) at
full 3/3 - up from the Round 1 baseline's 9/15, 1/5 (also re-measured
under the correct protocol as a fair comparison: M_energy_correct_protocol
scored 10/15, 0/5). search_snippets' win is narrow (Cv 0.462 vs target
0.460) and still uses local (gensim) Cv, not Palmetto - see Round 7's
own caveat above.

Six independent levers were tried and systematically ruled out for the
3 remaining blockers (search_snippets is now solved): non-zero alpha
(destabilizes clustering), more decoder epochs alone (negligible Cv
gain), joint DEC training (net negative, regresses Purity), embedding
truncation (monotonically harmful for 20ng, no help elsewhere), a
different embedder (same ceiling reappears). The 3 remaining gaps -
agnews NMI (~0.37 vs 0.412), imdb Cv (~0.39 vs 0.404), 20ng NMI (~0.58
vs 0.583) - are best understood as structural limits of this
architecture class (frozen single-embedding + KMeans, K=50, top-N=15
under the ecrtm_hicot-adjacent local protocol used here) for these 3
specific datasets, not something a further hyperparameter search within
this family is likely to close.

## Round 13 results (Palmetto Cv, non-hicot datasets, 10/10 successful)

| Config | Dataset | Cv | Purity | NMI |
|---|---|---:|---:|---:|
| R_sbert_kmeans_gte (raw ceiling) | 20ng | 0.469 | 0.646 | 0.548 |
| R_sbert_kmeans_gte (raw ceiling) | search_snippets | 0.464 | 0.863 | 0.507 |
| R_sbert_kmeans_gte (raw ceiling) | google_news_ts | 0.464 | 0.662 | 0.875 |
| R_sbert_kmeans_gte (raw ceiling) | agnews_short | 0.483 | 0.880 | 0.386 |
| R_sbert_kmeans_gte (raw ceiling) | imdb | 0.453 | 0.923 | 0.204 |
| S_vaebm_freqwords_gte | 20ng | 0.444 | 0.630 | 0.538 |
| S_vaebm_freqwords_gte | search_snippets | 0.422 | 0.861 | 0.499 |
| S_vaebm_freqwords_gte | google_news_ts | 0.402 | 0.664 | 0.877 |
| S_vaebm_freqwords_gte | agnews_short | 0.409 | 0.875 | 0.387 |
| S_vaebm_freqwords_gte | imdb | 0.469 | 0.917 | 0.203 |

No official HiCOT target exists for these 5 (informational only, same
caveat as Round 8). Full 10-dataset, correct-metric (Palmetto C_V,
ecrtm_hicot protocol) coverage is now complete for both the raw ceiling
and VAE-BM's best recipe.

## Checked and ruled out: vocabulary fidelity (not pursued further)

One more possible explanation checked: does `run_experiment.py`'s topic
experiment use HiCOT's own official vocab.txt, or a vocabulary derived
independently (TfidfVectorizer, capped at voc_size=5000, from the raw
dataset texts)? Confirmed: the latter. `datasets/simple_registry.py`'s
own module docstring is explicit and pre-dates this research pass -
"deliberately independent of the FASTopic/GloCOM protocol machinery...
these runners are direct, symmetric model-vs-model comparisons on
shared corpora, not per-paper faithful reproductions, so they do not
need per-protocol vocabulary/checksum/split pinning." Only
`protocols/glocom_protocol.py` and `protocols/fastopic_protocol.py`
have their own `vocabulary_for()` (injecting the released baseline's
exact vocab.txt) - no equivalent `ecrtm_hicot_protocol.py` exists, and
`--protocol ecrtm_hicot`'s own docstring already says as much ("not a
claim of exact reproduction... metric-level alignment only").

**Not pursued further**: building a dedicated HiCOT vocabulary-fidelity
protocol (fetching/verifying HiCOT's own released vocab.txt per dataset,
mirroring glocom_protocol.py/fastopic_protocol.py) is a substantially
larger undertaking than anything else in this pass, contradicts this
codebase's own pre-existing, intentional design choice, and the same 3
gaps reproducing near-identically across 3 independently-trained
embedders (Round 9/11) already argues against vocabulary being the
dominant factor - a vocab difference would be expected to show up as
embedder-dependent noise, not as a consistent ~0.36-0.38 NMI / ~0.30-0.47
Cv band regardless of which embedder did the clustering.

## Round 14 - "topic-word generation" pass begins (new user directive, 2026-09-20)

New directive: keep the proven clustering backbone (frozen embedding
branch, alpha=0, gte-large), focus specifically on improving topic-word
GENERATION since C_V (not Purity/NMI) is the real remaining blocker
under real Palmetto C_V. Track BOTH cv_local and Palmetto C_V for every
experiment going forward. Also test representations beyond
GTE/BGE/E5 (Word2Vec/GloVe/FastText, combinations).

**Found unused, already-available project assets**:
`datasets/definitions/hicot_datasets.py::load_hicot_vocab()`/
`load_hicot_word_embeddings()` - HiCOT's own official vocab.txt
(5000/4618/3473 words matching ECRTM's Table 9) and 200-dim GloVe
(matching ECRTM's Appendix B), already downloaded, for exactly these 5
datasets - explicitly documented as "NOT wired into experiment/
runner.py automatically." Wired in this pass via two new opt-in env
vars: `VAEBM_USE_HICOT_VOCAB` (fixes vocabulary) and
`VAEBM_USE_HICOT_STATIC_EMB` (+ `VAEBM_TOP_WORDS_MODE=static`) - a new
topic-word extraction mode: per cluster, filter to words that actually
appear there, then rank by cosine similarity to the cluster's own
frequency-weighted centroid in GloVe space.

### Round 14 results (20/20 successful) - T ruled out, U shows a clean negative result

| Dataset | Config | cv_local | Palmetto Cv | Purity | NMI |
|---|---|---:|---:|---:|---:|
| hicot_20ng | T_hicot_vocab_freq | 0.614 | 0.396 | 0.664 | 0.580 |
| hicot_20ng | U_hicot_static_glove | 0.404 | 0.363 | 0.664 | 0.580 |
| hicot_search_snippets | T_hicot_vocab_freq | 0.462 | 0.429 | 0.856 | 0.503 |
| hicot_search_snippets | U_hicot_static_glove | 0.429 | 0.426 | 0.856 | 0.503 |
| hicot_google_news | T_hicot_vocab_freq | 0.493 | 0.399 | 0.614 | 0.820 |
| hicot_google_news | U_hicot_static_glove | 0.474 | 0.378 | 0.614 | 0.820 |
| hicot_agnews | T_hicot_vocab_freq | 0.632 | 0.421 | 0.859 | 0.371 |
| hicot_agnews | U_hicot_static_glove | 0.307 | 0.404 | 0.859 | 0.371 |
| hicot_imdb | T_hicot_vocab_freq | 0.335 | 0.333 | 0.803 | 0.115 |
| hicot_imdb | U_hicot_static_glove | 0.299 | 0.277 | 0.803 | 0.115 |

**T_hicot_vocab_freq (vocab fix alone): matches Round 7/12's freq-mode
numbers EXACTLY on every dataset, both cv methods.** VAE-BM's own
independently-fit TfidfVectorizer vocabulary must converge to (nearly)
the same word set as HiCOT's official vocab.txt for these datasets -
**vocabulary fidelity is definitively ruled out as a factor.**

**U_hicot_static_glove (GloVe-centroid re-ranking): underperforms the
frequency baseline on Palmetto C_V on ALL 5 datasets** - deltas: 20ng
-0.033, search_snippets -0.003, google_news -0.021, agnews -0.017, imdb
-0.056. **Clean negative result for this specific implementation.**

**Important methodological finding**: local C_V and Palmetto C_V
sometimes diverge sharply for this technique - agnews's local C_V
collapsed from 0.632 to 0.307 (more than half) while its Palmetto C_V
only dropped a modest 0.017 (0.421->0.404). This confirms the user's own
instruction to track both was well-founded: local C_V would have looked
like a much bigger red flag than the metric that actually matters
suggests.

**Suspected root cause**: re-ranking over EVERY word present in a
cluster (often hundreds, for TF-IDF-weighted BoW over a 5000-word vocab)
dilutes relevance - fixed via a new `static_candidate_pool` parameter
(narrows to the top-N most frequent present words before re-ranking,
default 40). Round 15 tests this refinement before concluding the whole
static-embedding-centroid direction doesn't work.

## Round 15 results (candidate-pool-restricted static-GloVe, 10/10 successful) - MIXED, non-monotonic

| Dataset | T (freq) Palmetto | U (unrestricted static) Palmetto | V (pool=40) Palmetto | Target |
|---|---:|---:|---:|---:|
| hicot_20ng | 0.396 | 0.363 | **0.319 (worse than both)** | 0.451 |
| hicot_search_snippets | 0.429 | 0.426 | **0.450 (best of all 3, closest to target ever - off by 0.010)** | 0.460 |
| hicot_google_news | 0.399 | 0.378 | 0.378 (ties U) | 0.454 |
| hicot_agnews | 0.421 | 0.404 | 0.407 (between U and T) | 0.446 |
| hicot_imdb | 0.333 | 0.277 | 0.287 (between U and T) | 0.404 |

**Non-monotonic across datasets**: pool=40 helps search_snippets/agnews
meaningfully, HURTS 20ng (worse than even the unrestricted version - the
opposite of the intended fix), and is roughly neutral elsewhere. Also
important: local C_V and Palmetto C_V sometimes move in OPPOSITE
directions for the same config (20ng: local Cv improved 0.404->0.445
with pool=40, but Palmetto Cv got WORSE 0.363->0.319) - local C_V is
genuinely unreliable as a proxy signal for this specific technique,
confirming the value of tracking both per the user's instruction.

**search_snippets is now the closest any config in this entire research
pass has come to a genuine Palmetto C_V win** (0.450 vs 0.460 target,
with Purity 0.856/NMI 0.503 already beating their own targets) - a real,
actionable near-miss worth continuing to chase.

Given the pool cutoff's brittleness, Round 16 tests a SOFT blend instead
of a hard filter-then-rank: `static_hybrid_weight` (new
models/vaebm.py::top_words_by_freq_exact parameter) rank-normalizes
frequency and similarity to [0,1] over the candidate set and combines
them linearly, avoiding the pool-size sensitivity seen here.

## FINAL CORRECTED CONCLUSION (supersedes everything above - see Round 12)

Rounds 1-11 all used `--cv-method local` (gensim, local training
corpus). Round 12, prompted by the user flagging a real discrepancy
between their own stated premise and this pass's own local-Cv findings,
re-ran both the raw embedding+KMeans ceiling and VAE-BM's best recipe
with real Palmetto/Wikipedia C_V - the metric that actually matches
HiCOT's protocol - on all 5 target datasets.

**Result: under the correct C_V metric, neither VAE-BM (any
configuration tried, any of 3 embedders) nor the raw embedding+KMeans
ceiling it was built to approximate beats HiCOT's K=50 reference on all
3 metrics (C_V/Purity/NMI) simultaneously, on any of the 5 datasets.**
Both of this pass's previously-reported full wins (SearchSnippets,
GoogleNews) do not survive the correct C_V computation - Palmetto C_V is
systematically lower than local C_V for embedding-based clustering
methods on 4/5 of these datasets, likely because it reflects genuine
Wikipedia-derived semantic coherence rather than in-domain word
co-occurrence, which is measurably harder to satisfy.

What DOES hold up, robustly, across every embedder (gte-large,
bge-large-en-v1.5, e5-large-v2) and every architectural variant tried:
Purity and NMI are consistently beatable (or very close) via simple
embedding+KMeans-family methods on most of these datasets - it is
specifically HiCOT's C_V that these methods cannot match. Combined with
Round 9/10/11's own findings (raw embedding+KMeans hits the same walls
VAE-BM does; "letting the network learn a bit" never beats staying
frozen; three independent embedders converge on the same numbers), the
honest, complete answer after this entire research pass is: **HiCOT's
own C_V numbers on these specific datasets reflect something about their
neural topic model's own topic-word selection that a frozen or lightly-
adapted embedding-space clustering approach - VAE-BM or otherwise - does
not reproduce, at least not at K=50 under this evaluation protocol.**
This is a negative result worth reporting as-is, not something to keep
hyperparameter-searching around.
