# Methodological notes

This file exists because the task brief for this repository is explicit:
*"do not silently change the mathematical formulation of VAE-BM... if you
find a methodological/model issue, document it separately rather than
silently changing the model."* Everything below is a documented issue or
deliberate protocol choice, not a code fix.

## 1. `mu` is not `p(z|d)`

`VaeBmKMeansFit`/`Encoder` (`src/vaebm_benchmark/models/vaebm.py`) produce
`mu`, the mean of a diagonal Gaussian `q(z|x)`. It is:

- a point in a continuous latent space, unconstrained in sign or scale;
- **not** normalized to sum to 1 across any axis;
- **not** a distribution over discrete topics the way FASTopic's
  `transform()` or GloCOM's `get_theta()` are (both produce a
  `softmax`-normalized document-topic distribution, shape `[N, K]`,
  interpretable as `p(topic | doc)`).

Consequences for this repo's own comparisons:

- **Clustering metrics (NMI/ARI/AMI/Purity/ACC)** are unaffected - they
  only need a hard cluster id per document, which
  `KMeans(n_clusters=K).fit(mu)` provides exactly like any other
  embedding-space clustering. No apples-to-oranges issue here.
- **Topic-word metrics (NPMI/C_V/Diversity/IRBO)** are also unaffected -
  they operate on the top-`n` word lists per cluster
  (`get_topics_energy`/`get_topics_freq`), not on `mu` directly.
- **Anything that would compare VAE-BM's `mu` against FASTopic's/GloCOM's
  soft document-topic distribution index-for-index** (e.g. a topic-level
  correlation, or treating cluster `k` as "the same topic" across models)
  is **not attempted anywhere in this repo**, precisely because there is
  no natural bijection between VAE-BM's latent dimensions/cluster ids and
  a baseline's topic ids. If a future addition wants that kind of
  comparison, it needs its own alignment step (e.g. Hungarian matching on
  top-word overlap) - which does not currently exist here and should not
  be assumed to be free.

## 2. Two topic-word views, not one

VAE-BM's supplied `top_words_by_freq_exact()` returns two different
rankings per cluster: `"energy"` (decoder logit `R @ mu + b`, masked to
words observed in that cluster's documents) and `"freq"` (raw term
frequency within the cluster). These measure different things - "energy"
reflects what the decoder's learned word vectors predict as likely given
the cluster's mean latent position; "freq" reflects what's literally most
common in the cluster's documents, independent of the model. This repo
reports **both** (`get_topics_both_views()`) rather than picking one, and
`configs/models/vaebm.yaml`'s default `get_topics()` view is `"energy"`
(the model's own learned signal) for any code path that needs a single
list. Comparisons against baseline topic-quality metrics use whichever
view is specified in the relevant `results/<baseline>/vaebm_results.csv`
column - check the column name before reading a single NPMI number as "the"
VAE-BM score.

## 3. GloCOM's official protocol has no held-out test split

Inspecting `github.com/qducnguyen/GloCOM`'s own `dataloader.py` shows:

```python
self.test_bow = scipy.sparse.load_npz(f'{path}/bow.npz')
```

i.e. it reloads the *training* BOW file as "test" too. The paper's own
reference results are computed transductively - fit and evaluate on the
same full corpus. `protocols/glocom_protocol.py`'s `split_strategy` is
`"full_corpus"` for exactly this reason: reproducing the paper's own
un-split evaluation is more faithful than inventing a split it never
used, even though a genuinely held-out test would be more rigorous in
the abstract. Whether FASTopic's paper does the same is recorded in
`protocols/fastopic_protocol.py` and `results/fastopic/protocol.json`
once verified against the paper/repo - see that file, not an assumption
copied from GloCOM, since the two papers are not required to share an
evaluation convention.

## 4. `num_global_clusters` is an adapter choice, not a GloCOM hyperparameter

GloCOM's paper describes building a "global document" per semantic
cluster (SBERT embeddings, K-Means, pool member BOW vectors) but the
official repo ships only the *precomputed* result for its own bundled
dataset(s) - no code that builds it from raw text for a new corpus. This
repo's `glocom_adapter.py` implements that construction directly from the
paper's description (not a guess: embed → K-Means → pool), and
`num_global_clusters` is this adapter's own preprocessing knob (clamped to
never exceed corpus size), tuned per dataset to match the paper's own
reported choice where stated. See `protocols/glocom_protocol.py` for the
per-dataset values and their provenance.

## 5. Neither paper's headline coherence metric is exactly reproducible here

Both FASTopic and GloCOM report their main topic-coherence number (CV /
TC) against an external reference corpus neither official repo ships or
names precisely:

- **FASTopic** (paper §4.1, verbatim): "We use a widely-used large
  Wikipedia article collection as the external reference corpus to
  compute CV." Neither `github.com/bobxwu/FASTopic` nor its companion
  `github.com/BobXWu/TopMost` ships or names that corpus - the shipped
  demos/tests use the training corpus itself as a stand-in, purely to
  illustrate the metric API.
- **GloCOM**: computes TC via the **Palmetto** Java library against a
  bundled Wikipedia reference index (`evaluations/palmetto.jar` +
  `wiki_data/wikipedia_bd/`, both external downloads from
  `hobbitdata.informatik.uni-leipzig.de`, requiring a Java runtime). Its
  own repo separately ships an unused-by-default gensim-based
  alternative (`compute_topic_coherence()`) taking any reference corpus.

This project uses each protocol's own in-repo fallback (train corpus as
reference, via gensim's `CoherenceModel`) for both baselines, and marks
the resulting `cv`/`npmi` comparison as a **documented deviation** in
`protocols/fastopic_protocol.py`/`protocols/glocom_protocol.py`'s
`verify()` output and in `results/<baseline>/comparison.csv`'s
`published_source` column - never silently presented as equivalent to
the papers' own Wikipedia/Palmetto-based numbers. Wiring up Palmetto+a
real Wikipedia dump for GloCOM, or sourcing the actual reference corpus
FASTopic used, is future work, not something this project claims to have
solved.

## 6. Datasets each paper reports but this project does not currently run

- **FASTopic**: WoS (Web of Science) is used in the paper's Tables
  1/2/4/5/7/9 but **is not distributed by either `bobxwu/FASTopic` or
  `BobXWu/TopMost`** (confirmed via the full git history of TopMost's
  `data/` folder - it has never existed there). Reproducing it would
  require independently sourcing and filtering the raw WoS-11967/HDLTex
  corpus (Kowsari et al. 2017) to match the paper's reported 10,000-doc/
  7-label/10,000-vocab subset, with no documented filtering procedure to
  check against.
- **GloCOM**: only **SearchSnippets** ships precomputed artifacts in the
  official repo. GoogleNews/StackOverflow/Biomedical (also used in the
  paper) require external STTM (`github.com/qiang2100/STTM`) preprocessing
  followed by TopMost filtering (drop words with corpus frequency < 3,
  drop documents with < 2 remaining tokens) that GloCOM's own repo
  documents only in prose, not in a runnable script - see
  `configs/datasets/stack_overflow.yaml`/`biomedical.yaml` for the
  separate (different-artifact, different-doc-count) mirror this project
  already has provenance for, kept for future use via
  `glocom_adapter.py`'s from-raw-text `fit()` fallback, not the
  official-artifact `fit_precomputed()` path used for SearchSnippets.

## 7. Neither official codebase sets a random seed

Confirmed by grepping both cloned repos directly (`grep -rni seed` across
every `.py` file in each): neither `bobxwu/FASTopic`/`BobXWu/TopMost` nor
`qducnguyen/GloCOM` sets a training-time seed anywhere. Both papers
instead report results averaged over multiple runs (FASTopic: undisclosed
count, with significance markers whose test method is never stated;
GloCOM: explicitly 3 runs, mean±std). This project's own `seeds` list per
protocol is therefore its OWN choice for reproducible re-runs of ITS OWN
experiments, not a value recovered from either paper - exact numeric
reproduction of a paper's specific published mean/std is not something
either official codebase's own conventions would guarantee even if you
had their exact seed.

## 8. Published vs. reproduced vs. VAE-BM are never collapsed into one number

`results/<baseline>/comparison.csv` always keeps three columns:
`published` (verbatim from the paper's own table, cited by
`PublishedResult.source`), `reproduced_baseline` (this repo's own run of
the official implementation), and `vaebm`. A gap between `published` and
`reproduced_baseline` is expected and informative (different hardware,
package versions, exact preprocessing edge cases, or a "smoke test"
document subsample rather than the paper's full corpus - see each
protocol's `notes`) - it is not evidence of a bug in this repo, and it
must be visible before any claim that "VAE-BM outperforms X."
