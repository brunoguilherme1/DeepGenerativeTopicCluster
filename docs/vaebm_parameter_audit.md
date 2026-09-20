# VAE-BM Parameter & Architecture Audit

2026-09-20, user-authorized systematic search (see
`docs/vaebm_leaderboard.md` and `docs/vaebm_mimic_gte_research_log.md`).
Full audit of every parameter/architectural choice across `vaebm.py`,
`vaebm_poe.py`, `vaebm_dec.py`, `vaebm_ckpt.py`, adapters, and
`runner.py`/`cluster_runner.py`/`scientific_models.py`, before launching
Stage B/D experiments per the user's explicit instruction not to assume
the previously-tested parameters are the complete search space.

## Critical structural finding

Three separate builder layers exist and do **not** expose the same env
vars:

| Builder layer | File | Env vars honored |
|---|---|---|
| Topic experiment | `experiment/runner.py` (`_VAEBM_DEFAULTS`) | UNITS, EPOCHS, LR, DIM_EMB, ALPHA, EMBEDDER, KL_WEIGHT, FREEZE_EMB, TOP_WORDS_MODE + CKPT_* + USE_HICOT_* + STATIC_* + TOPIC_ORACLE_LABELS |
| Cluster experiment | `experiment/cluster_runner.py` | UNITS, EPOCHS, LR, DIM_EMB, ALPHA, EMBEDDER (**vaebm only**); CKPT_ALPHA/CKPT_LR |
| Classification experiment | `experiment/scientific_models.py` | same as cluster_runner |

All of this research pass's work is in the topic experiment, so this
mostly doesn't block us - noted for completeness. `vaebm_poe`/
`vaebm_dec`/`vaebm_ckpt` get `units=50, epochs=50, batch_size=128,
lr=1e-3, dim=(1500,1000,500), dim_emb=(368,)` hardcoded in
cluster_runner/scientific_models (not the topic experiment, which
already reads these from `_VAEBM_DEFAULTS` for all 4 models).

## Real bugs found (fix before/alongside the search - they corrupt results otherwise)

1. **`best_metric = -1.0` should be `float("-inf")`** - `vaebm_poe.py`,
   `vaebm_dec.py`, `vaebm_ckpt.py`. In the UNSUPERVISED fallback path
   (no oracle labels - i.e. every topic-experiment run of these 3
   models so far), `metric = -mean_loss` where `mean_loss` is a large
   positive number (negative ELBO), so `metric > -1.0` is essentially
   never true early on and best_weights/best_kmeans frequently never
   get set from a genuinely-better epoch - these variants have been
   effectively selecting close to the FINAL epoch, not the best-loss
   epoch, in every non-oracle topic-experiment run to date (including
   the vaebm_dec "net negative" finding in Rounds 3-4 of the earlier
   research pass - that conclusion should be treated as provisional
   until re-tested with this fixed).
2. **`if i == 10` should almost certainly be `i == 0`** - `vaebm.py`,
   `vaebm_poe.py`. With `dim=(1500,1000,500)`, index 10 never exists,
   so the BoW branch's hidden layers are ALWAYS fully glorot-initialized
   (never identity), while the embedding branch's first layer IS
   identity-initialized when `dim_emb` is non-empty. Asymmetric by
   accident. Does not affect any alpha=0 result (BoW branch is
   gradient-decoupled there regardless of its init), but matters for
   any alpha>0 or vaebm_poe/vaebm_dec experiment.
3. **DEC drops `kl_weight`** - `vaebm_dec.py`'s inline training loop
   recomputes `recon - kl` directly instead of calling `VAEBM.call()`,
   so `VAEBMDECAdapter(kl_weight=...)` is silently ignored for DEC.
4. **KMeans `random_state=22` is a literal**, not tied to the run's own
   seed, at every model's final-clustering call site - multi-seed
   sweeps vary VAE weight init but never KMeans init.
5. **Embedding encode `batch_size=32` at fit vs `128` at inference** -
   asymmetric, likely harmless numerically, noted for completeness.

## Tier 1 - highest-value new env vars (implemented this pass, see commits)

1. `VAEBM_BATCH_SIZE` - batch size, was a `128` literal at every builder.
2. `VAEBM_KMEANS_SEED` / `VAEBM_KMEANS_N_INIT` - final-clustering KMeans
   init, was `random_state=22, n_init="auto"` literal everywhere.
3. `VAEBM_NORMALIZE_MU` - L2-normalize `mu` before KMeans (the cheapest
   way to get cosine/spherical-KMeans-like clustering behavior, since no
   spherical KMeans implementation exists anywhere in this codebase or
   its dependencies - confirmed via audit).
4. `VAEBM_NORMALIZE_EMB` - L2-normalize the raw SentenceTransformer
   output before it enters the model (some embedders already have this
   baked in via their own `Normalize` module - gte-large/bge-large/
   e5-large all confirmed to; this matters for embedders that don't).
5. `VAEBM_DEC_LAMBDA_C` - DEC loss weight, was a `0.1` literal at every
   call site. Re-testing vaebm_dec at various weights is now meaningful
   given the checkpoint-selection bug fix above.
6. Fixed the `best_metric`/`i == 10` bugs above directly (not env-gated
   - these are correctness fixes, not research knobs).

## Tier 2 - noted, not yet implemented (lower priority / bigger lift)

- `VAEBM_DIM` (BoW hidden widths) - large existing knob, but currently
  MOOT for every alpha=0 config tested so far (BoW branch gradient-
  decoupled regardless of its architecture) - only relevant if alpha>0
  or vaebm_poe/vaebm_dec (both branches contribute) is revisited.
- `recon_weight` (symmetric to `kl_weight`) - does not exist at all.
- Optimizer type/weight decay/gradient clipping - Adam only, no
  regularization anywhere.
- Encoder-internal `tf.random.set_seed(1234)` hardcoded, overrides the
  run's own seed at model construction - blocks genuine multi-seed
  sensitivity analysis.
- Activation function (`tanh`, hardcoded), EarlyStopping `patience=1`,
  `top_m` internal candidate count, Identity-init gain, decoder R init
  scale, PoE prior precision, DEC target-update period/ν, vectorizer
  min_df/max_df/ngram_range.

## Confirmed absent (do not go looking for these)

Any cosine/spherical KMeans or custom clustering class; any
`sklearn.preprocessing.normalize` usage in the vaebm path; any
alternative fusion mechanism (concat/attention/gating/learnable alpha -
the code comment at `vaebm.py`'s alpha blend says "could be made
learnable" but this was never implemented); `recon_weight`; temperature/
logit scaling; dropout; batchnorm; skip connections; gradient clipping;
LR schedules; KL annealing/free-bits.
