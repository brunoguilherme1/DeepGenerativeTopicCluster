# RQ1 final protocol (locked, for main.tex)

This document is the single source of truth for how RQ1's headline VAE-BM
numbers are produced, superseding the ~35 exploratory "research round" runs
in `docs/vaebm_leaderboard.md`. It was written after an independent
two-agent audit (propose + critique) of the existing RQ1 artifacts found:
9/10 main-table cells traceable and correct, 1/10 a copy-paste transcription
error (SearchSnippets K50), and two methodological concerns worth fixing
before the numbers are final: (a) `lambda_relevance` had been hand-tuned per
dataset/K using the same Palmetto C_V being reported, and (b) most of the
reported Purity/NMI gains are attributable to the GTE-large embedder itself,
not to VAE-BM's own mechanism.

## Locked configuration

- Embedder: `thenlper/gte-large` for 20NG/AGNews/SearchSnippets/GoogleNews;
  `BAAI/bge-large-en-v1.5` for IMDB (matches the existing appendix config
  table; not a new choice).
- Encoder embedding branch: frozen (`VAEBM_FREEZE_EMB=1`).
- `alpha=0.0` (embedding branch contributes only through the frozen
  embedding vector fed into the shared latent space, not a residual mixing
  coefficient).
- `units=1024`, `dim_emb` left at the embedder's native size.
- `epochs=1` — disclosed here plainly and in the paper text. This is not a
  new deviation; every prior "research round" also used epochs=1.
- Top-words mode: `relevance`.
- **lambda_relevance = 0.1, GLOBAL across every dataset and K.** This is the
  one deliberate change from the prior per-dataset-tuned values. It costs a
  small number of cells (e.g. GoogleNews K100 becomes a loss vs. HiCOT) but
  removes the same-metric selection circularity. The full per-dataset
  lambda-sensitivity sweep from Rounds 26-35 is kept, unchanged, as an
  appendix table — explicitly labeled "lambda tuned on test C_V (upper
  bound)," never used to justify the main table's numbers.
- Coherence: Palmetto C_V only (`--cv-method palmetto`) — this is the final
  reported number, not an exploratory local-corpus estimate.

## Seeds

- 5 seeds (1-5) for 20NG, AGNews, SearchSnippets, GoogleNews.
- 3 seeds (1-3) for IMDB (longer per-seed runtime; run on a dedicated GPU).
- K in {50, 100} for every (dataset, seed) pair.

## Zero-cost ablation (same run, no extra training)

A GTE-large + KMeans row (same top-words method, same lambda=0.1 where
applicable) is reported alongside VAE-BM in the MAIN table, not the
appendix, per the critique finding that this ablation is necessary to
separate VAE-BM's own contribution from the embedder's.

## Decision rule (hard freeze at 21:30 local time)

- For any (dataset, K) cell with >=3 finished seeds by the freeze time:
  report mean +/- std, computed only over the seeds that finished.
- For any cell with 1-2 finished seeds: report the single/available value(s)
  and label the cell "single seed" (or "n=2") in the table footnote — never
  silently presented as a multi-seed mean.
- A VAE-BM vs. HiCOT comparison is only called a "win" in the text if
  VAE-BM's mean exceeds HiCOT's own reported mean by more than the sum of
  the two methods' stds (i.e., outside HiCOT's own reported uncertainty
  band). Anything smaller is reported as "comparable," not a win.
- No label information, and no selection on the classification/clustering
  downstream task, is used anywhere in this process — only unsupervised
  Palmetto C_V, and only for the (already-fixed, non-swept) lambda=0.1
  config, so there is nothing left to select on at this stage.

## Execution

`scripts/run_vaebm_rq1_final_multiseed.py`, adapted from the Round 27-35
harness (`scripts/run_vaebm_gte_research_round34_k100_lambda.py`): one
subprocess per (dataset, K, seed) combo, calling the same
`scripts/run_experiment.py --experiment topic --models vaebm --protocol
ecrtm_hicot --cv-method palmetto` CLI used throughout this project, with a
JSON checkpoint for crash-safe resume and per-combo timeout/retry.
