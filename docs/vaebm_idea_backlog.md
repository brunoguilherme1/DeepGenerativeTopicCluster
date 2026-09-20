# VAE-BM Idea Backlog (2026-09-20)

Exhaustive brainstorm requested by the user after Round 25: "no
limit/border - vocabulary, embedding, TF-IDF params, new ways to get
top words, and so on." Organized by category, each idea marked with why
it's plausible and whether it's genuinely new or a variant of something
already tried. Cross-reference `docs/vaebm_leaderboard.md` for what's
already confirmed/ruled out before re-deriving it here.

Every idea below stays inside this project's standing rules: no labels
in training/loss (oracle-only if used, clearly tagged), no silent change
to the as-supplied Encoder/Decoder/ELBO math, cv_local + Palmetto Cv
both tracked, opt-in env vars only.

## Category 1: Vocabulary / vectorizer parameters

1. **`voc_size` sweep** - fixed at 5000 every round so far. Try 2000,
   3000, 8000, 10000, 15000. Bigger vocab = more candidate topic words
   (may surface more specific/coherent terms); smaller vocab forces more
   common (possibly more Wikipedia-legible) words to the top.
2. **`min_df`/`max_df` filtering** on the vectorizer (currently unset -
   audit Tier 2 item, never implemented). Drop ultra-rare terms (noise)
   and/or ultra-common terms (candidate junk) BEFORE they ever compete
   for top-word slots, rather than excluding by hand-curated list after
   the fact (Round 18's hand lists backfired - filtering by frequency
   statistics is a different, more principled mechanism).
3. **N-gram vocabulary** (`ngram_range=(1,2)` or `(1,3)`) - bigrams/
   trigrams as candidate topic "words" (e.g. "stock market", "climate
   change"). Phrases are less ambiguous than single generic words and
   may score differently on Wikipedia NPMI.
4. **`sublinear_tf=True`** on the TfidfVectorizer (log-scaled term
   frequency instead of raw count) - changes which words dominate
   within-cluster ranking.
5. **Lemmatization/stemming** before vectorizing - merges word-form
   variants ("running"/"runs"/"ran"), reducing sparsity and possibly
   surfacing cleaner base-form topic words.
6. **POS-tag filtering** - restrict candidate topic words to nouns/
   adjectives only (classic topic-modeling trick; verbs and function
   words rarely make good topic labels even after stopword removal).
7. **Automatic ubiquitous-word suppression (data-driven, not hand-
   curated)** - Round 18 showed hand lists for 20NG/AGNews backfired.
   Alternative: compute each word's document-frequency ACROSS clusters
   (fraction of the K=50 clusters where it appears in the top candidate
   list) and down-weight/exclude words above a threshold, adaptively per
   dataset instead of a fixed list. Should in principle generalize where
   hand-curation didn't.

## Category 2: Embeddings (clustering representation)

8. **More contextual embedder families untried**: `all-mpnet-base-v2`,
   `multi-qa-mpnet-base-dot-v1`, `nomic-ai/nomic-embed-text-v1`,
   `jinaai/jina-embeddings-v2/v3`, `hkunlp/instructor-xl` (instruction-
   conditioned - could be prompted differently per dataset domain).
9. **Domain-matched embedders per dataset**: a sentiment/review-tuned
   model for IMDB; a news-tuned model for AGNews/GoogleNews.
10. **Pooling strategy** - GTE/BGE/E5 all use their own default pooling;
    try mean-pooling vs CLS-token pooling vs max-pooling from the SAME
    base model's token embeddings (SentenceTransformer exposes this).
11. **Truncation length** - check `embedder.encode()`'s default
    `max_seq_length` (often 256-384 tokens). IMDB reviews are long;
    confirm whether reviews are silently truncated, losing the back half
    of every document - if so, chunk long documents and average
    chunk-level embeddings instead of one truncated encode() call.
12. **Concatenated multi-embedding fusion**: GTE + BGE concatenated
    (then optionally PCA'd back down) as a single richer document
    vector - an ensemble-of-embedders approach, untried this whole pass
    (every round used exactly one embedder).
13. **Weighted two-embedder fusion**: `w * E_gte + (1-w) * E_bge`
    (requires both to be pre-normalized/same-dim, or projected to a
    shared dim first) - cheaper than concatenation, sweep `w`.

## Category 3: BoW branch / alpha fusion (re-examine post-Round-24)

14. **Gentle alpha values**: Round 24 found alpha=0.5 catastrophic, but
    only tested one point far from 0. Sweep alpha in {0.02, 0.05, 0.1,
    0.15, 0.2} - a much gentler blend might add useful signal without
    destroying the clean embedding-only geometry.
15. **`vectorizer_type="count"` instead of `"tfidf"`** for the BoW
    branch, combined with a small alpha - different input scale/
    distribution to the BoW MLP.
16. **L2-normalize the TF-IDF vectors** before the BoW branch (currently
    `norm=None` explicitly) - could stabilize the BoW branch's own
    contribution once alpha>0.

## Category 4: New ways to generate topic words (the big "propose new methods" ask)

17. **Same-space KeyBERT-style ranking**: rank candidate words by cosine
    similarity to the cluster centroid IN THE SAME EMBEDDING SPACE used
    for clustering (e.g. encode each vocab word with the SAME gte-large/
    bge-large model, not a separate 200-dim GloVe space like the
    existing GloVe-hybrid mode). More representationally consistent than
    mixing a contextual clustering space with a static GloVe ranking
    space - untried.
18. **Wikipedia-NPMI-optimized candidate selection** - the most direct
    lever: since Palmetto's own Wikipedia coherence index is already
    self-hosted (`scripts/setup_palmetto.py`), score CANDIDATE words by
    their own Wikipedia NPMI (via Palmetto's index or a cached
    co-occurrence table) and greedily/exactly pick the top-10 that
    maximize the SAME confirmation measure Cv itself computes, instead
    of ranking by frequency or embedding similarity first and hoping it
    correlates. This is legitimate (no labels, no test-set leakage - the
    Wikipedia index is a fixed external resource, exactly what Cv itself
    already uses at evaluation time) and directly targets the metric
    rather than a proxy for it.
19. **LDAvis-style relevance scoring**: `relevance = lambda * log(p(w|topic))
    + (1-lambda) * log(p(w|topic)/p(w))`, sweep lambda in [0, 1] - a
    well-established, principled alternative to raw frequency that
    explicitly balances "frequent in topic" against "exclusive to topic."
20. **Maximal Marginal Relevance (MMR) reranking** - balance relevance
    (frequency/similarity) against diversity (penalize a candidate
    that's too similar to already-selected words), avoiding topics
    padded with near-synonyms.
21. **c-TF-IDF ported into vaebm** (currently only `sbert_kmeans`'s
    ceiling uses this formula) - a new `top_words_mode="ctfidf"` option
    for VAE-BM's own clusters, not just the raw-embedding baseline.
22. **Coherence-guided greedy beam search**: build each topic's word
    list incrementally, at each step adding whichever remaining
    candidate most increases the LOCAL corpus's own NPMI-based
    coherence score so far - directly optimizes cv_local as a proxy
    (with the caveat, already known this pass, that cv_local and
    Palmetto can diverge - validate the result on Palmetto before
    trusting it).
23. **Ensemble/voting across existing modes**: take words appearing in
    the top-10 of at least 2 of {freq, energy, GloVe-hybrid} as a more
    robust consensus list.
24. **Sweep `top_m`** (the internal candidate-pool size before final
    ranking, currently a hardcoded default of 20, only exposed for
    static mode via `VAEBM_STATIC_CANDIDATE_POOL`) - expose for freq/
    energy modes too and sweep 10/20/30/50.

## Category 5: Clustering algorithm alternatives (beyond plain KMeans)

25. **True spherical KMeans** (cosine-distance-native, not just
    L2-normalize + Euclidean KMeans as `normalize_mu` approximates) -
    e.g. via a manual cosine-KMeans implementation.
26. **Gaussian Mixture Model** on `mu` instead of hard KMeans - soft
    assignments could change which documents contribute to each
    cluster's word-frequency counts.
27. **PCA/UMAP dimensionality reduction on `mu`** before KMeans - reduce
    noise dimensions before clustering, a well-established practice for
    high-dim embedding clustering.
28. **Best-of-N unsupervised model selection**: run KMeans with several
    seeds/`n_init` values, pick the run with the best Silhouette/
    Davies-Bouldin score (label-free) rather than a single fixed seed -
    legitimate, no labels used, could reduce variance and pick a
    genuinely better partition.
29. **Consensus/ensemble clustering**: run KMeans multiple times
    (different seeds and/or different embedders), build a co-association
    matrix, cluster that - more robust than any single run.

## Category 6: Architecture (Stage E proper)

30. **Decoupled clustering vs. topic-generation latent spaces** - the
    user's own Section 4 suggestion: one encoder/representation
    optimized for clustering quality, a separate one (or the BoW branch
    specifically) optimized for topic-word decoding, rather than forcing
    one `mu` to serve both jobs.
31. **`units` (latent dim) sweep** - fixed at 1024 every round. Try 50
    (original VAE-BM default), 128, 256, 512 - a more compressed latent
    space might cluster more cleanly than a 1024-dim near-copy of the
    raw embedding.
32. **KL annealing / free-bits** (confirmed absent by the audit) - could
    allow a more expressive/useful `mu` without posterior collapse.
33. **A real hidden layer in the embedding branch** (`dim_emb` is
    currently empty - direct `Dense(units)` from the raw embedding, no
    nonlinearity) - try `dim_emb=(512,)` or `(768, 256)` so the branch
    can learn a genuinely transformed representation, not just a linear
    projection.
34. **Residual/skip connection** from the raw embedding to `mu` (absent
    per audit) - lets the branch learn a residual correction instead of
    the whole mapping from scratch.
35. **Re-test `vaebm_dec`/`vaebm_poe` from scratch** now that the
    `best_metric` checkpoint-selection bug is fixed (Round 17's fix) -
    the earlier "vaebm_dec is net negative" conclusion predates the fix
    and was never revisited.

## Category 7: Multi-seed / statistical robustness

36. **Multi-seed runs (5-10 seeds) of the current best config per
    dataset** - every result this pass is a single seed (42); report
    mean/std, and optionally pick the best seed by an unsupervised
    criterion (Silhouette score, NOT labels) rather than trusting one
    draw.

## Category 8: Dataset-specific engineering

37. **IMDB**: chunk long reviews (e.g. 3x256-token windows) and average
    chunk embeddings instead of one truncated encode() call - directly
    tests the truncation-loss hypothesis from #11.
38. **GoogleNews**: try a short-text-specialized embedder (many
    STS-tuned small models are specifically benchmarked on titles/
    queries) given its docs are only ~5.75 tokens - GTE/BGE/E5 are all
    general-purpose, none specialized for this length regime.
39. **AGNews**: de-duplicate near-identical headlines before fitting -
    could reduce noise in cluster formation.
40. **20NG**: a milder header-cleaning regex (strip only boilerplate
    lines like "Subject:"/"Lines:" prefixes, not a stopword-style full
    exclusion) - Round 18's blanket exclusion made things worse; a more
    surgical removal (matching what `20ng`'s own `remove=` argument does)
    might behave differently.

## Priority ranking (highest expected value first)

Given everything already ruled out this pass, the ideas most likely to
move Cv specifically (the binding constraint on every dataset) are, in
order:

1. **#18 - Wikipedia-NPMI-optimized candidate selection** (most direct;
   attacks the metric's own formula instead of a proxy for it)
2. **#2 - min_df/max_df vocabulary filtering** (principled, data-driven,
   unlike the hand-curated lists that backfired)
3. **#19 - LDAvis-style relevance scoring** (well-established, cheap to
   implement, a genuinely different ranking principle than anything
   tried)
4. **#17 - same-space KeyBERT-style ranking** (fixes the representation-
   space mismatch in the existing GloVe-hybrid approach)
5. **#14 - gentle alpha sweep** (0.02-0.2) (Round 24 only tested one
   extreme point)
6. **#7 - automatic ubiquitous-word suppression** (a smarter, adaptive
   version of the hand-curated exclusion that failed)
7. **#31 - units/latent-dim sweep** (architecture-level, cheap to test)
