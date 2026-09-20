# VAE-BM K=50 Leaderboard (live)

2026-09-20, user-authorized systematic search. Goal: beat HiCOT K=50 on
Cv/Purity/NMI for as many of the 5 target datasets as possible, treating
each `hicot_*`/plain pair as a joint diagnostic unit. See
`docs/vaebm_mimic_gte_research_log.md` for the full history this builds
on (Rounds 1-16) and `docs/vaebm_parameter_audit.md` for the full
parameter/architecture audit.

**Rule**: Palmetto Cv is the official metric vs. HiCOT's target. cv_local
is tracked alongside for every experiment (never used alone to pick a
winner - the two have been observed to move in opposite directions).
Any result using labels for anything beyond argmax topic assignment is
tagged `[ORACLE]` and excluded from "beats HiCOT" claims.

## HiCOT official targets (K=50)

| Dataset | Cv | Purity | NMI |
|---|---:|---:|---:|
| SearchSnippets | 0.460 | 0.818 | 0.478 |
| GoogleNews | 0.454 | 0.465 | 0.657 |
| 20NG | 0.451 | 0.626 | 0.583 |
| AGNews | 0.446 | 0.857 | 0.412 |
| IMDB | 0.404 | 0.737 | 0.082 |

## Current best (Stage A baseline, from Rounds 1-16 - to be re-verified)

| Dataset | Configuration | Embedder | cv_local | Palmetto Cv | Purity | NMI | Δ Cv | Δ Purity | Δ NMI | 3/3? |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| hicot_20ng | frozen, alpha=0, freq words | gte-large | 0.614 | 0.396 | 0.664 | 0.580 | -0.055 | +0.038 | -0.003 | no |
| 20ng | frozen, alpha=0, freq words | gte-large | 0.444 | ? (no target) | 0.630 | 0.538 | - | - | - | n/a |
| hicot_search_snippets | frozen, alpha=0, GloVe pool=40 | gte-large + HiCOT GloVe | 0.490 | 0.450 | 0.856 | 0.503 | -0.010 | +0.038 | +0.025 | no |
| search_snippets | frozen, alpha=0, freq words | gte-large | 0.422 | ? (no target) | 0.861 | 0.499 | - | - | - | n/a |
| hicot_google_news | frozen, alpha=0, freq words | gte-large | 0.493 | 0.399 | 0.614 | 0.820 | -0.055 | +0.149 | +0.163 | no |
| google_news_ts | frozen, alpha=0, freq words | gte-large | 0.402 | ? (no target) | 0.664 | 0.877 | - | - | - | n/a |
| hicot_agnews | frozen, alpha=0, freq words | gte-large | 0.632 | 0.421 | 0.859 | 0.371 | -0.025 | +0.002 | -0.041 | no |
| agnews_short | frozen, alpha=0, freq words | gte-large | 0.409 | ? (no target) | 0.875 | 0.387 | - | - | - | n/a |
| hicot_imdb | frozen, alpha=0, freq words | gte-large | 0.335 | 0.333 | 0.803 | 0.115 | -0.071 | +0.066 | +0.033 | no |
| imdb | frozen, alpha=0, freq words | gte-large | 0.469 (best VAE-BM Cv result in whole pass) | ? (no target) | 0.917 | 0.203 | - | - | - | n/a |

Raw embedding+KMeans ceiling (Palmetto, gte-large, for reference):
hicot_20ng 0.427/0.673/0.577, hicot_search_snippets 0.453/0.872/0.511,
hicot_google_news 0.431/0.628/0.819, hicot_agnews 0.457/0.869/0.373,
hicot_imdb 0.326/0.801/0.111. Plain (Palmetto): 20ng 0.469/0.646/0.548,
search_snippets 0.464/0.863/0.507, google_news_ts 0.464/0.662/0.875,
agnews_short 0.483/0.880/0.386, imdb 0.453/0.923/0.204.

## Per-dataset blocker summary (from Rounds 1-16)

- **20NG**: NMI off by only 0.003 (0.580 vs 0.583) - closest miss. Cv already beats target comfortably.
- **SearchSnippets**: Purity/NMI already beat target. Cv off by 0.010 (0.450 vs 0.460) - closest Cv miss of any dataset.
- **GoogleNews**: Purity/NMI beat target by wide margins. Cv is the only gap (0.399 vs 0.454, -0.055).
- **AGNews**: Purity near target (+0.002). Both Cv (-0.025) and NMI (-0.041) need work.
- **IMDB**: Purity/NMI beat target comfortably. Cv is the big gap (-0.071) - but the PLAIN imdb dataset already exceeds the HiCOT target on Cv (0.469 > 0.404) under the identical recipe. This is the single most promising diagnostic lead - see Stage C below.

## Stage C: hicot_* vs plain diagnostic investigation

Two diagnostic agents traced exact dataset provenance for all 5 pairs.
Headline finding: the apparent "plain beats HiCOT" results above are
**not** evidence of a transferable modeling advantage in 4/5 cases -
they are measurement/provenance artifacts. Only one (IMDB) pointed at a
real, fixable measurement bug; the rest are just "these are different
documents than what HiCOT scored."

- **IMDB (the standout case, investigated first)**: plain `imdb` is raw
  HuggingFace `stanfordnlp/imdb` text - mixed case, full punctuation,
  stopwords present, and an **unstripped `<br />` HTML tag** (a
  formatting artifact from the original scrape, present in nearly every
  review). `hicot_imdb` is HiCOT's own cleaned artifact: lowercased,
  punctuation/digits stripped, stopwords removed, vocab hard-capped at
  5,000 content words. Because Palmetto Cv (NPMI-style, scored against a
  Wikipedia reference corpus) rewards words that co-occur with almost
  everything, plain imdb's topic words got dominated by function words
  ("the", "and", "is") and `br` itself - both score artificially high on
  Cv without reflecting genuine topic quality. **This is a measurement
  confound, not a real win.** Fix implemented this pass:
  `VAEBM_EXCLUDE_STOPWORDS=1` (sklearn's English stopword list, applied
  via a single `counts_k` zeroing operation in
  `vaebm.py::top_words_by_freq_exact` that transparently covers all
  three topic-word modes). Round 17 (running) re-tests plain imdb under
  this fix - the hypothesis is its Cv drops from 0.469 toward the
  0.33-0.40 band, while hicot_imdb (already stopword-free upstream)
  stays roughly flat.
- **20NG**: `hicot_20ng` is HiCOT's own artifact, which does **not**
  strip email headers/footers/quoted-reply text. `20ng` here uses
  `fetch_20newsgroups(subset="all", remove=("headers","footers","quotes"))`
  - a materially cleaner document. Not a fair comparison; header junk
  words (`subject`, `organization`, `nntp`, `posting`, `host`, `lines`,
  `writes`, `article`, `edu`) likely inflate hicot_20ng's own topic-word
  quality in the opposite direction. Follow-up: try
  `VAEBM_EXTRA_EXCLUDE_WORDS` with that header-junk list specifically on
  hicot_20ng.
- **GoogleNews**: `hicot_google_news` is **title-only** (avg 5.75
  tokens/doc). The plain dataset used through Round 16,
  `google_news_ts`, is title+snippet (avg 27.95 tokens/doc) - a
  different, much richer document with **zero exact-text overlap** with
  hicot's corpus. `google_news_t` (title-only, 5,842/11,019 docs
  overlapping with hicot_google_news) is the correct plain counterpart
  and replaces `google_news_ts` starting Round 17.
- **AGNews**: `agnews_short` (8,000 docs, STC2-style cut) and
  `hicot_agnews` (12,500 docs, ECRTM cut) are genuinely different
  subsamples of the source corpus - only 1 document in common.
  `agnews_short` additionally has **unescaped HTML entity artifacts**
  (`lt`, `gt`, `quot`, `href`, `aspx` literally appearing as tokens) from
  an un-decoded scrape, a second, independent source of Cv inflation on
  top of the subsampling mismatch. Follow-up: `VAEBM_EXTRA_EXCLUDE_WORDS`
  with that HTML-entity list on agnews_short.
- **SearchSnippets**: the smallest, most benign difference of the four
  non-IMDB pairs - both datasets come from the same underlying STC2
  corpus; HiCOT applies additional vocab pruning and hyphen-splitting.
  No artifact found here; this pair's cv_local/Palmetto gap (if any)
  should be treated as a genuine signal, not provenance noise.

**Implication for the search strategy**: only IMDB's plain-vs-hicot gap
was a real, fixable measurement bug (now patched, pending Round 17
confirmation). The 20NG/GoogleNews/AGNews plain variants are not valid
"this property should transfer to hicot_*" evidence on their own -
they're different documents, not a cleaner view of the same documents.
Stage C's actual transferable lesson so far is narrower than originally
hoped: **strip stopwords/junk-tokens from topic-word extraction**, not
"whatever makes the plain dataset easier."

## Round 17 (running, job 1630): Stage A re-baseline with stopword fix

Re-tests all 10 datasets (vaebm + sbert_kmeans, cv_local + Palmetto) with
`VAEBM_EXCLUDE_STOPWORDS=1` and the corrected `google_news_t` pairing.
Results pending - table above still reflects pre-fix Rounds 1-16 numbers
and should be treated as provisional, especially plain `imdb`'s 0.469.
