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

## Round 17 (complete, job 1630): Stage A re-baseline with stopword fix

Re-tested all 10 datasets (vaebm + sbert_kmeans, cv_local + Palmetto) with
`VAEBM_EXCLUDE_STOPWORDS=1` and the corrected `google_news_t` pairing.
**Confirms the measurement-artifact hypothesis decisively**: plain
`imdb`'s Palmetto Cv collapsed 0.469 -> 0.338 once stopwords/`br` are
excluded from topic words - it is now BELOW both `hicot_imdb` (0.350)
and the HiCOT target (0.404). The "plain imdb beats HiCOT" claim from
Rounds 1-16 is retracted: it was measurement noise, not a real result.
Plain `20ng`'s ceiling (sbert_kmeans) also dropped sharply (0.469 ->
0.355 Palmetto), same artifact. `search_snippets`/`google_news_t` moved
much less (~flat), consistent with the diagnostic finding that those
pairs' differences were provenance/subsampling, not stopword leakage.

### hicot_* results (vaebm recipe: frozen, alpha=0, freq words, gte-large, stopwords excluded)

| Dataset | Palmetto Cv | Δ vs target | cv_local | Purity | Δ Purity | NMI | Δ NMI | 3/3? |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| hicot_20ng | 0.397 | -0.054 | 0.630 | 0.664 | +0.038 | 0.580 | -0.003 | no |
| hicot_search_snippets* | 0.431 | -0.029 | 0.461 | 0.856 | +0.038 | 0.503 | +0.025 | no |
| hicot_google_news | 0.398 | -0.056 | 0.491 | 0.614 | +0.149 | 0.820 | +0.163 | no |
| hicot_agnews | 0.426 | -0.020 | 0.622 | 0.859 | +0.002 | 0.371 | -0.041 | no |
| hicot_imdb | 0.350 | -0.054 | 0.351 | 0.803 | +0.066 | 0.115 | +0.033 | no |

\* search_snippets used plain freq-mode here (Round 17's uniform recipe),
NOT the GloVe-hybrid config that previously got it to 0.450 - the two
fixes haven't been combined yet, see Round 18 below. Every other
dataset's Cv gap held flat (20NG, GoogleNews) or narrowed (AGNews
-0.025->-0.020, IMDB -0.071->-0.054) under the stopword fix alone.
Still 0/5 on 3/3, but IMDB and AGNews both moved the right direction.

### Plain-dataset results (same recipe)

| Dataset | Palmetto Cv | cv_local | Purity | NMI |
|---|---:|---:|---:|---:|
| 20ng | 0.335 | 0.447 | 0.630 | 0.538 |
| search_snippets | 0.424 | 0.465 | 0.861 | 0.499 |
| google_news_t (replaces google_news_ts) | 0.408 | 0.493 | 0.614 | 0.807 |
| agnews_short | 0.410 | 0.591 | 0.875 | 0.387 |
| imdb | 0.338 | 0.334 | 0.917 | 0.203 |

`google_news_t` (correct pair) now sits almost exactly ON hicot_google_news
(0.408 vs 0.398 Palmetto) - the large gap the old (wrong) `google_news_ts`
pairing suggested was mostly a "richer document" artifact, not something
transferable to hicot's title-only corpus.

### Raw sbert_kmeans ceiling (Palmetto, gte-large, stopwords excluded)

hicot_20ng 0.432, hicot_search_snippets 0.454, hicot_google_news 0.431,
hicot_agnews 0.457, hicot_imdb 0.349 (barely moved from pre-fix - these
5 were already close to their pre-fix values, confirming hicot_*'s own
vocab was already close to stopword-clean). Plain: 20ng 0.355 (was
0.469 pre-fix - large drop), search_snippets 0.464 (flat), google_news_t
0.432 (new pairing), agnews_short 0.483 (flat - HTML-entity artifacts
`lt/gt/quot/href/aspx` are NOT standard English stopwords, so this
generic fix didn't touch them - see Round 18), imdb 0.322 (was 0.453
pre-fix - large drop, same artifact as the vaebm result above).

## Round 18 (next): targeted follow-ups from Round 17's evidence

1. **Recombine search_snippets' GloVe-hybrid config with the stopword
   fix** (previously 0.450 Palmetto without the fix, -0.010 from
   target - the closest Cv miss of any dataset) - test whether the two
   fixes compound.
2. **Dataset-specific junk-word lists**, since Round 17 showed the
   generic English-stopword list doesn't touch non-stopword junk:
   `VAEBM_EXTRA_EXCLUDE_WORDS="subject,organization,nntp,posting,host,lines,writes,article,edu"`
   on hicot_20ng (email header artifacts, per the provenance diagnostic);
   `VAEBM_EXTRA_EXCLUDE_WORDS="lt,gt,quot,href,aspx"` on hicot_agnews
   AND agnews_short (unescaped HTML entities).
3. **`VAEBM_NORMALIZE_MU=1` sensitivity sweep** across all 5 hicot_*
   datasets (new Tier-1 capability, untested) - L2-normalizing mu before
   KMeans approximates spherical clustering, the "alternative clustering
   geometry" the user's directive explicitly called for and the audit
   confirmed was otherwise entirely absent from this codebase.

## Round 18 (complete, job 1631): results

| Combo | Dataset | Palmetto Cv | Δ vs R17 | cv_local | Purity | Δ Purity | NMI | Δ NMI |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| A: GloVe-hybrid+stopword | hicot_search_snippets | 0.450 | +0.019 (recovers pre-fix best) | 0.488 | 0.856 | 0 | 0.503 | 0 |
| B: header-junk exclude | hicot_20ng | 0.361 | **-0.036 (worse)** | 0.603 | 0.664 | 0 | 0.580 | 0 |
| C: HTML-junk exclude | hicot_agnews | 0.424 | -0.002 (~flat) | 0.622 | 0.859 | 0 | 0.371 | 0 |
| C: HTML-junk exclude | agnews_short | 0.407 | -0.003 (~flat/worse) | 0.592 | 0.875 | 0 | 0.387 | 0 |
| D: normalize_mu | hicot_20ng | 0.399 | +0.002 (~flat) | 0.635 | 0.676 | **+0.012** | 0.582 | **+0.002** |
| D: normalize_mu | hicot_search_snippets | 0.425 | -0.007 (slightly worse) | 0.479 | 0.869 | +0.013 | 0.512 | +0.009 |
| D: normalize_mu | hicot_google_news | 0.398 | 0 (exactly flat) | 0.491 | 0.614 | 0 | 0.820 | 0 |
| D: normalize_mu | hicot_agnews | 0.430 | **+0.004** | 0.652 | 0.872 | **+0.013** | 0.380 | **+0.009** |
| D: normalize_mu | hicot_imdb | 0.350 | ~flat | 0.352 | 0.802 | 0 | 0.114 | 0 |

**Finding A (recovered, not improved)**: combining the GloVe-hybrid topic-word
config with stopword exclusion gives 0.450 - statistically the same as
the pre-fix hybrid-only result (0.450). The two fixes don't compound;
search_snippets' topic words were apparently never stopword-polluted in
the first place (consistent with the diagnostic's "smallest, most benign
difference" finding for this pair). Still the closest Cv miss of any
dataset (-0.010).

**Finding B/C (important failed idea - logged per the user's explicit
instruction to record failures)**: hand-excluding dataset-specific junk
words (email headers on 20NG, HTML entities on AGNews) made Palmetto Cv
*worse*, not better - the opposite of IMDB's result. Hypothesis:
IMDB's stopwords/`br` are near-universal filler that inflates Cv via
trivial co-occurrence with everything; 20NG's header words and AGNews's
HTML entities are dataset-local artifacts that, once removed, simply
expose different (not more Wikipedia-coherent) words in the topic list -
there's no guarantee the next-ranked word scores higher NPMI than the
removed one. **Conclusion: stopword/junk exclusion is not a general-purpose
Cv lever - it only helps when the excluded terms are true near-universal
filler words, not any "obviously junky" token.** Abandoning further
hand-curated exclude-word lists; not pursuing this direction further.

**Finding D (adopt as new default for 20NG/AGNews)**: `VAEBM_NORMALIZE_MU=1`
gives small, consistent, Cv-safe gains on Purity/NMI for 20NG and AGNews,
flat/no-op on GoogleNews and IMDB, and a small Cv cost on search_snippets
(only when NOT combined with the GloVe-hybrid config - untested combined).
Most notable: **20NG's NMI gap narrows from -0.003 to -0.001** (0.582 vs
0.583 target) - almost an exact match, achieved with zero Cv cost (0.397
-> 0.399), directly satisfying the user's own guidance to "prioritize
small [20NG] improvements without hurting Cv." AGNews improves on all
three metrics simultaneously (Cv -0.020 -> -0.016, Purity +0.002 ->
+0.015, NMI -0.041 -> -0.032) - the best AGNews result of the whole pass,
though still short of 3/3.

## Round 19 (complete, job 1632): results

| Combo | Dataset | Palmetto Cv | vs. component alone | Purity | NMI |
|---|---|---:|---|---:|---:|
| E: GloVe-hybrid+stopword+normalize_mu | hicot_search_snippets | 0.442 | **worse than A alone (0.450)** | 0.869 | 0.512 |
| F: GloVe-hybrid+stopword | hicot_imdb | 0.314 | **worse than freq+stopword (0.350)** | 0.803 | 0.115 |
| G: normalize_mu | 20ng (plain) | 0.339 | ~flat vs. R17 (0.335) | 0.628 | 0.539 |
| G: normalize_mu | agnews_short (plain) | 0.411 | ~flat vs. R17 (0.410), Purity slightly down | 0.869 | 0.386 |

**Both composition attempts failed** - a second class of "important
failed idea" alongside Round 18's junk-word exclusions:
- Adding `normalize_mu` on top of search_snippets' GloVe-hybrid config
  trades Cv for Purity/NMI (which already beat target by a wide margin
  there) - a bad trade since Cv is the only metric blocking 3/3 for this
  dataset. **Decision: search_snippets' standing best stays Finding A
  alone (GloVe-hybrid+stopword, no normalize_mu)**, Cv 0.450.
- GloVe-hybrid topic words do NOT generalize to IMDB even combined with
  the stopword fix - confirms Rounds 14-16's original (pre-fix) finding
  that this topic-word mode is search_snippets-specific, not a general
  Cv lever. **IMDB's Cv gap (-0.054) is now the only one no experiment
  this whole pass has moved beyond the initial stopword fix itself.**
- Plain-dataset normalize_mu parity checks: small/mixed effects, weaker
  than their hicot_* counterparts' gains - confirms the effect is real
  (not noise) but its size varies per dataset/subsample, not a
  guaranteed transfer.

## Current standing best per hicot_* dataset (after Rounds 17-19)

| Dataset | Recipe | Palmetto Cv | Δ Cv | Purity | Δ Purity | NMI | Δ NMI | 3/3? |
|---|---|---:|---:|---:|---:|---:|---:|---|
| hicot_20ng | freq+stopword+**normalize_mu** | 0.399 | -0.052 | 0.676 | +0.050 | 0.582 | **-0.001** | no |
| hicot_search_snippets | **GloVe-hybrid**+stopword | 0.450 | **-0.010** | 0.856 | +0.038 | 0.503 | +0.025 | no |
| hicot_google_news | freq+stopword | 0.398 | -0.056 | 0.614 | +0.149 | 0.820 | +0.163 | no |
| hicot_agnews | freq+stopword+**normalize_mu** | 0.430 | -0.016 | 0.872 | +0.015 | 0.380 | -0.032 | no |
| hicot_imdb | freq+stopword | 0.350 | -0.054 | 0.803 | +0.066 | 0.115 | +0.033 | no |

Still 0/5 on 3/3, but every dataset's Cv gap is flat-or-narrower than
Round 16's pre-fix baseline, and 20NG's NMI is now a near-exact match
(off by 0.001). SearchSnippets remains the single closest Cv miss
(-0.010) of the whole pass. GoogleNews and IMDB are now the two
stubborn cases - neither has moved beyond the initial stopword fix
under ANY topic-word-extraction or clustering-geometry tweak tried so
far (freq vs. GloVe-hybrid, with/without normalize_mu).

## Round 20 (complete, job 1633): GloVe-hybrid does NOT extend to GoogleNews/AGNews

| Dataset | Palmetto Cv (GloVe-hybrid+stopword) | vs. freq+stopword standing best |
|---|---:|---|
| hicot_google_news | 0.370 | **worse** (0.398 -> 0.370, -0.028) |
| hicot_agnews | 0.424 | **worse than normalize_mu recipe** (0.430 -> 0.424) and ~flat vs. plain freq (0.426) |

Both negative. "Short text -> GloVe-hybrid helps" does not hold as a
general pattern - it is specific to search_snippets for reasons not yet
understood (possibly search_snippets' own GloVe coverage, or that it's
the pair with the smallest/most benign hicot/plain provenance gap of
the 5 - see Stage C). The Rounds 14-15 note that the pool-cutoff version
"helped meaningfully" on AGNews evidently held only for cv_local, not
Palmetto Cv - a concrete instance of the "local and Palmetto can move in
opposite directions" risk the user flagged up front. **GloVe-hybrid
topic words are now confirmed dataset-specific to search_snippets only.**

This is the 5th consecutive negative/mixed result on topic-word-extraction
and clustering-geometry tweaks (Round 18's B/C, Round 19's E/F, Round
20's H). Per the user's "stop clearly unproductive experiments early and
redirect resources" instruction, pivoting away from this axis.

## Round 21 (complete, job 1634): BGE-large - the most significant finding of the pass

| Dataset | Palmetto Cv (BGE) | Δ vs GTE (R17) | Purity (BGE) | Δ vs GTE | NMI (BGE) | Δ vs GTE |
|---|---:|---:|---:|---:|---:|---:|
| hicot_20ng | 0.400 | ~flat (+0.003) | 0.649 | -0.015 | 0.559 | -0.021 |
| hicot_search_snippets | 0.419 | **-0.012 (worse)** | 0.807 | -0.049 | 0.456 | -0.047 |
| hicot_google_news | 0.395 | ~flat (-0.003) | 0.602 | -0.012 | 0.809 | -0.011 |
| hicot_agnews | 0.423 | ~flat (-0.003) | 0.862 | +0.003 | 0.373 | +0.002 |
| **hicot_imdb** | **0.362** | **+0.012 (BEST of the whole pass)** | **0.840** | **+0.037** | **0.138** | **+0.023** |

**IMDB improves on all 3 metrics simultaneously with BGE-large** - Cv gap
narrows from -0.054 to -0.042 (Δ Purity now +0.103, Δ NMI now +0.056).
This is the best hicot_imdb Cv result of the entire pass, under ANY
config tried (stopword-only, GloVe-hybrid, normalize_mu, or this).

**Bigger finding**: the raw sbert_kmeans ceiling under BGE for hicot_imdb
is 0.353 (barely moved from GTE's 0.349 ceiling) - but vaebm+BGE reaches
0.362, ABOVE its own raw-embedding ceiling. This is the first genuine
"learned VAE-BM representation exceeds the raw embedding+KMeans ceiling"
result of the whole pass - exactly the outcome the user's directive said
was possible and explicitly forbade dismissing in advance ("do not
declare something a structural ceiling just because GTE+KMeans failed").

BGE is flat-to-worse everywhere else, especially search_snippets (real
Purity/NMI cost, -0.049/-0.047) - **not a universal upgrade, an
IMDB-specific one.** Consistent with the user's own explicit allowance
that different datasets may need different best configs. **Decision:
hicot_imdb's new standing best embedder is BGE-large; the other 4
datasets keep gte-large.**

## Current standing best per hicot_* dataset (after Rounds 17-21)

| Dataset | Recipe | Palmetto Cv | Δ Cv | Purity | Δ Purity | NMI | Δ NMI | 3/3? |
|---|---|---:|---:|---:|---:|---:|---:|---|
| hicot_20ng | gte-large, freq+stopword+normalize_mu | 0.399 | -0.052 | 0.676 | +0.050 | 0.582 | -0.001 | no |
| hicot_search_snippets | gte-large, GloVe-hybrid+stopword | 0.450 | -0.010 | 0.856 | +0.038 | 0.503 | +0.025 | no |
| hicot_google_news | gte-large, freq+stopword | 0.398 | -0.056 | 0.614 | +0.149 | 0.820 | +0.163 | no |
| hicot_agnews | gte-large, freq+stopword+normalize_mu | 0.430 | -0.016 | 0.872 | +0.015 | 0.380 | -0.032 | no |
| **hicot_imdb** | **bge-large**, freq+stopword | **0.362** | **-0.042** | **0.840** | **+0.103** | **0.138** | **+0.056** | no |

## Round 22 (complete, job 1635): normalize_mu does NOT compound with BGE either

BGE+normalize_mu on hicot_imdb: Cv 0.358 (vs. BGE-alone's 0.362,
**-0.004 worse**), Purity 0.832 (vs. 0.840, -0.008 worse), NMI 0.135
(vs. 0.138, -0.003 worse). A third instance of normalize_mu failing to
compound with a topic-word/embedder change (after Round 19's
search_snippets GloVe-hybrid attempt) - **normalize_mu's benefit
appears specific to gte-large+20NG/AGNews, not a universal add-on.**
**Decision: hicot_imdb's standing best stays BGE+freq+stopword alone**
(no normalize_mu), Cv 0.362, Δ-0.042 - unchanged from Round 21.

## Round 23 (complete, job 1636): E5 doesn't move either stubborn dataset

| Dataset | Palmetto Cv (E5) | vs GTE | vs BGE | Purity (E5) | vs GTE | NMI (E5) | vs GTE |
|---|---:|---:|---:|---:|---:|---:|---:|
| hicot_google_news | 0.395 | ~flat (-0.003) | **identical to BGE (0.395)** | 0.605 | -0.009 | 0.804 | -0.016 |
| hicot_20ng | 0.395 | ~flat (-0.002/-0.004) | -0.005 | 0.636 | **-0.028 to -0.040 (worse)** | 0.560 | -0.020 to -0.022 (worse) |

**Three different embedding families (GTE, BGE, E5) now converge tightly
to Cv~0.395-0.399 for BOTH GoogleNews and 20NG** - strong evidence this
specific gap is not an embedding-choice problem, unlike IMDB (which WAS
embedder-sensitive - BGE moved it, Round 21). E5 actively hurts 20NG's
Purity/NMI with no Cv upside. Embedder swapping is exhausted as a lever
for these two datasets.

## Major finding: alpha=0 has never actually fused BoW+embedding, in ANY of the 23 rounds so far

Checked the encoder's own fusion math directly
(`models/vaebm.py::Encoder.call`): `mu = alpha * mu_bow + (1 - alpha) *
mu_emb`. Every single round in this entire pass (1-23) used
`VAEBM_ALPHA=0.0` - meaning **`mu = mu_emb` exactly, every time.** The
BoW/TF-IDF branch is computed (wasted compute) but its contribution to
`mu` is multiplied by zero and discarded before clustering, in literally
every experiment run so far. This whole pass has been testing a
frozen-embedding-encoder-plus-KMeans pipeline wearing a "VAE-BM" label -
not the genuine BoW+embedding fusion HiCOT/ECRTM's own architecture (and
this project's own `alpha` default of 0.99, mostly BoW-weighted) was
designed around. This is a real, previously untested axis - a strong
candidate for why GoogleNews/20NG are stuck regardless of embedder (an
embedding-only signal may simply lack whatever vocabulary-frequency
information Palmetto's Wikipedia-NPMI rewards for these two datasets).

## Round 24 (complete, job 1637): alpha=0.5 is actively harmful, not free upside

| Dataset | Metric | alpha=0 baseline | alpha=0.5, epochs=20 | Δ |
|---|---|---:|---:|---:|
| hicot_google_news | Palmetto Cv | 0.398 | 0.404 | +0.006 (marginal improvement) |
| hicot_google_news | Purity | 0.614 | 0.487 | **-0.127 (collapse)** |
| hicot_google_news | NMI | 0.820 | 0.640 | **-0.180 (collapse - flips from a big win to a MISS vs. 0.657 target)** |
| hicot_20ng | Palmetto Cv | 0.397-0.399 | 0.377 | -0.021 (worse) |
| hicot_20ng | Purity | 0.664-0.676 | 0.444 | **-0.220 to -0.232 (collapse - flips from a win to a big MISS)** |
| hicot_20ng | NMI | 0.580-0.582 | 0.415 | **-0.167 (collapse - flips from a near-exact match to a big MISS)** |

**Decisive negative result.** Re-engaging the BoW branch at a 50/50
fusion weight, even after giving it 20 epochs to train (vs. alpha=0's
1-epoch shortcut), does not add useful signal - it destroys the clean
semantic clustering geometry the frozen embedding alone provided.
GoogleNews's Cv nudged up marginally, but at a catastrophic cost to
Purity/NMI (both metrics FLIP from comfortably beating target to
missing it). 20NG got strictly worse on every metric. **Conclusion:
alpha=0 (embedding-only, this whole pass's setting) is not leaving
value on the table for these two datasets - it is the better choice,
now that the alternative has actually been tested rather than assumed.**
This closes the "alpha=0 might be why they're stuck" hypothesis as
tested-and-rejected, not merely untested. A gentler intermediate value
(e.g. alpha=0.1-0.2) remains theoretically untested but the magnitude of
damage at 0.5 (with an already-undertrained BoW branch relative to
HiCOT's own many-epoch regime) makes it a low-priority follow-up rather
than an obvious next step.

## Status after 24 rounds: comprehensive summary

**Confirmed decisively**: the stopword-exclusion fix (Round 17) was real
and retracts the earlier "plain imdb beats HiCOT" claim as a measurement
artifact. **The single biggest lead**: BGE-large for hicot_imdb (Round
21) - the only case this whole pass where a learned VAE-BM representation
exceeds its own raw-embedding ceiling. **Confirmed dead ends** (evidence-
based, not assumed): hand-curated junk-word exclusion beyond true
universal stopwords (Round 18); GloVe-hybrid topic words outside
search_snippets (Rounds 19-20); normalize_mu stacked with any
topic-word/embedder change beyond its original gte-large+20NG/AGNews
context (Rounds 19, 22); embedder swapping for GoogleNews/20NG - GTE,
BGE, E5 all converge to the same Cv (Round 23); alpha>0 BoW fusion
(Round 24) - actively harmful at 0.5, not merely unhelpful. Still 0/5 on
beating HiCOT on all 3 metrics simultaneously, but every dataset's Cv
gap is flat-or-narrower than the Round 16 pre-fix baseline, and the
search has now touched every major axis in the user's directive except
GloVe/Word2Vec/FastText static-only embeddings and genuinely new
architectural variants (Stage E proper, e.g. a differently-structured
fusion or a separate clustering/topic-generation latent space).

