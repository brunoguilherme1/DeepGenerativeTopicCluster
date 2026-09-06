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

## 8. VAE-BM's supplied default learning rate (1e-2) diverges at these vocab scales

Confirmed by direct diagnostic (fit VAE-BM under the GloCOM protocol,
same seed/data/architecture, `lr` in `{1e-2, 1e-3, 1e-4}`, all else
identical):

| lr | outcome (20 epochs, SearchSnippets, vocab=4618) |
|---|---|
| **1e-2** (the supplied notebook's own default) | loss explodes past batch ~95 of epoch 2 (`loss: 670891244127709560832.0000` -> `NaN`); Keras's `TerminateOnNaN` callback stops training; `EarlyStopping` restores epoch-1 weights (themselves already at loss≈2×10^11, nowhere near converged). Resulting KMeans clusters are near-random: **Purity=0.224, NMI=0.014, glocom_td=0.157** (versus GloCOM's own reproduced 0.821/0.530/0.976). |
| 1e-3 | trains stably for the full 20 epochs (loss 187 -> 137, monotonically decreasing, `EarlyStopping` never triggers). Non-degenerate results: **Purity=0.742, NMI=0.393, cv=0.499, glocom_td=0.496** - competitive with, if somewhat below, GloCOM's own reproduced numbers. |
| 1e-4 | still improving slowly after 5 epochs (loss 188.4 -> 186.9) - undertrained within a 20-epoch smoke-test budget, not a fair comparison point here. |

This is a genuine numerical-stability issue in the model AS SUPPLIED,
at this specific combination of vocab size, `alpha=0.99` (i.e. the
decoder's BoW-energy branch dominates gradients almost entirely - see
`models/vaebm.py`'s `Encoder.call`), and Adam's default lack of gradient
clipping - not something introduced by this project's adapter code, and
not something this project silently patched inside `models/vaebm.py`
(that file's math is untouched). Instead,
`protocols/glocom_protocol.py`/`protocols/fastopic_protocol.py`'s
`build_vaebm()` explicitly override `lr=1e-3` for the actual comparison
runs (a hyperparameter substitution, not a change to VAE-BM's
architecture/loss), with this section as the citable record of why and
what the supplied default actually does at this scale. Anyone re-running
`configs/models/vaebm.yaml`'s literal default (`lr: 0.01`, kept as
documentation of the AS-SUPPLIED value, not silently edited to 1e-3
there) should expect the same divergence this section describes.

This is exactly the kind of finding a smoke test is supposed to catch
before scaling up - see this project's own instructions ("test only
small experiments... only expand after the smoke test succeeds").

## 9. Published vs. reproduced vs. VAE-BM are never collapsed into one number

`results/<baseline>/comparison.csv` always keeps three columns:
`published` (verbatim from the paper's own table, cited by
`PublishedResult.source`), `reproduced_baseline` (this repo's own run of
the official implementation), and `vaebm`. A gap between `published` and
`reproduced_baseline` is expected and informative (different hardware,
package versions, exact preprocessing edge cases, or a "smoke test"
document subsample rather than the paper's full corpus - see each
protocol's `notes`) - it is not evidence of a bug in this repo, and it
must be visible before any claim that "VAE-BM outperforms X."

## 10. `--protocol ecrtm_hicot`: metric-level alignment with ECRTM/HiCOT, not a reproduction

`scripts/run_experiment.py --experiment topic --protocol ecrtm_hicot`
(default `--protocol generic`, unchanged from before this option existed)
changes how `experiment/runner.py`'s topic-quality metrics are computed,
to match ECRTM (Wu et al., ICML 2023, Table 2/3) and HiCOT (2025) as
closely as possible **without touching datasets or preprocessing** -
those were explicitly left alone, so this is metric-level alignment
only, never claimed as an exact reproduction of either paper's numbers.
See the prior "fair comparison" analysis of this same runner
(conversation record, not a file in this repo) for the fuller gap list
this section deliberately does NOT close: Yahoo Answer is still absent,
`agnews_short`/`agnews_full` are still not the papers' own 12,500-doc AG
News subsample, and neither dataset's preprocessing follows Card et al.
(2018)'s 5-step pipeline (lowercase, strip punctuation, drop number-
containing tokens, drop tokens <3 chars, remove stopwords).

What `ecrtm_hicot` DOES change, all in `experiment/runner.py::run_single()`:

- **top_n = 15**, not 10 - both papers explicitly select the top-15
  words of each discovered topic for CV/TD (ECRTM §4.1: "We select the
  top 15 words of discovered topics for the above topic quality
  evaluation").
- **C_V, by default, still via the local-training-corpus gensim
  fallback** (`metrics/topic_quality.py::coherence()`) - `ecrtm_hicot`
  does NOT, by itself, switch to Palmetto/Wikipedia. ECRTM §4.1,
  verbatim: "We use the public Wikipedia article collection as the
  external reference corpus. This removes the bias of using relatively
  small datasets (e.g., training sets) as the reference corpus." - so
  the paper-faithful C_V source IS Palmetto, but it is a ~5.1GB one-time
  download (`scripts/setup_palmetto.py`) this project deliberately never
  triggers just because `--protocol ecrtm_hicot` was passed (a real cost
  a caller may not want on every run, e.g. a quick smoke test). Pass
  `--cv-method palmetto` EXPLICITLY (works under either `--protocol`) to
  opt in - only then is `metrics/palmetto.py::palmetto_cv` used, and
  only then does `scripts/run_experiment.py` auto-install Palmetto/the
  Wikipedia index first if not already present
  (`scripts/setup_palmetto.py::ensure_palmetto_ready` - the jar, ~5.9MB,
  checked into `HoangTran223/HiCOT`'s own repo directly; the
  Wikipedia_bd coherence index, ~5.1GB compressed, official DICE/AKSW
  host, verified reachable before that script was written - both landing
  at exactly the paths `metrics/palmetto.py`'s own `DEFAULT_JAR`/
  `DEFAULT_WIKI_INDEX` expect; idempotent, prints a step-by-step debug
  trail throughout since it is meant to run unattended on a Colab/SLURM
  session). If Palmetto remains unavailable even after that attempt
  (e.g. the install failed), `cv` is recorded as `None` (`N/A` in the
  printed table) - it is never silently replaced with the local-corpus
  number under the same `cv` column, matching how `glocom_protocol.py`
  already treats this same distinction (see #5 above).
- **TD via the fixed-K*15-denominator Dieng definition**
  (`metrics/topic_quality.py::topic_diversity_dieng_fixed_k`), not the
  pre-existing `topic_diversity()`. The two diverge whenever a model
  returns fewer than K non-empty topic-word lists (a degenerate/collapsed
  cluster) - `topic_diversity()` shrinks its own denominator to match
  whatever was actually returned, silently understating how much topic
  collapsing should hurt the score; the fixed `K*15` denominator (the
  papers' own definition) does not have this problem.
- Result metadata now records `evaluation_protocol`, `top_n`,
  `cv_source` (`"palmetto_wikipedia"` vs `"cv_local_corpus"`), and
  `td_definition` (`"dieng_unique_words_top15"` vs
  `"unique_over_returned_slots"`) - persisted in
  `results/experiment_results.csv`/`.json`, so a row is always
  self-describing about which metric variant produced it.

**Document-assignment semantics (Purity/NMI)** - independent of
`--protocol`, recorded for every run regardless:

ECRTM's own official implementation exposes a genuine document-topic
distribution via `get_theta()`, and computes Purity/NMI from
`argmax(theta)` against ground truth. This repo's `run_single()` now
does the same **whenever a model actually has one** - but the check is
NOT "does `get_document_topics()` return non-`None`": VAEBMAdapter's own
`get_document_topics()` always returns `mu` (never `None` - it is reused
there as the storage slot `get_mu()`/`get_document_embeddings()` also
read from elsewhere in this codebase), even though `mu` is explicitly
**not** a topic distribution (#1 above). Naively checking "non-`None`"
would silently treat `mu` as `theta` and `argmax` over it - precisely
the "pretend mu is theta" mistake this must avoid. Instead,
`_assignment_source_for_model()` gates on the **model name**: VAE-BM/
BERTopic/sbert_kmeans (and their registered variants) are explicitly
known to never use `argmax_theta`, regardless of what their own
`get_document_topics()` returns; only a model this repo has no explicit
knowledge of yet (e.g. a future real ECRTM baseline) can take that
branch, and only if its `get_document_topics()` actually returns
something.

- **VAE-BM**: `mu` (the encoder's latent Gaussian mean) is explicitly
  documented (#1 above) as **not** a topic distribution - it is never
  passed through a softmax to manufacture a fake `theta`, and never
  reaches the `argmax_theta` branch regardless of what
  `get_document_topics()` returns. VAE-BM keeps its existing
  `KMeans(n_clusters=K).fit(mu)` assignment,
  `assignment_source="kmeans_on_latent_mu"`. VAE-BM is therefore **not
  structurally identical to ECRTM** here - a real difference, not a bug,
  and not something this change silently papers over.
- **SBERT+KMeans**: has no doc-topic distribution at all by
  construction (a plain embed-then-cluster pipeline) -
  `assignment_source="kmeans_on_embeddings"`.
- **BERTopic**: same - its `hdbscan_model` is swapped for `KMeans` over
  UMAP-reduced SBERT embeddings (see `bertopic_adapter.py`'s own module
  docstring) - `assignment_source="kmeans_on_embeddings"`.

Every result also records `topic_source`: `"native"` when a model's
`get_topics()` is its own learned/native output (VAE-BM's decoder
energy/freq view, BERTopic's own c-TF-IDF), or `"cluster-derived"` for
SBERT+KMeans, whose topic words are computed here, after the fact, from
cluster membership via class-based TF-IDF (`sbert_kmeans_adapter.py`'s
own module docstring) - never a native output of that model family.

Usage: `python scripts/run_experiment.py --experiment topic --protocol
ecrtm_hicot --models vaebm bertopic sbert_kmeans --datasets 20ng imdb
--k 50 100 --seed 42`. Everything else about the CLI (`--models`,
`--datasets`, `--k`, `--vaebm-configs`, `--sbert-configs`, ...) is
unchanged; `--protocol` only ever changes how CV/TD are computed and
what metadata a result carries.

## 11. `hicot_*` dataset ids: HiCOT's own preprocessed artifacts, used verbatim

`hicot_20ng`/`hicot_imdb`/`hicot_agnews`/`hicot_search_snippets`/
`hicot_google_news` (`datasets/definitions/hicot_datasets.py`) download
the exact `train_texts.txt`/`train_labels.txt`/`test_texts.txt`/
`test_labels.txt` (plus `vocab.txt`/`word_embeddings.npz`/
`{train,test}_bow.npz`, kept on disk but not auto-wired into any model -
see below) files from `github.com/HoangTran223/HiCOT`'s own
`datasets/<Folder>/` - no re-tokenizing, re-splitting, or resampling.
These are a SEPARATE artifact from this project's pre-existing `20ng`/
`imdb`/`agnews_short`/`agnews_full`/`search_snippets`/`google_news_*`
datasets (different preprocessing, different vocab) - never silently
merged with them under the same id.

**Why `hicot_agnews` matters specifically**: it is the first AG News
artifact in this project that actually matches ECRTM's own Table 9 -
10,000 train + 2,500 test = 12,500 documents, exactly the paper's own
subsample size. `agnews_short` (8,000 docs, a different STC-family
subsample) and `agnews_full` (127,600 docs, the full HF `ag_news`) are
both different cuts of the same underlying corpus, neither matching the
paper - see the "fair comparison" gap list this project's own prior
analysis identified (not itself a file in this repo, but the reasoning
behind adding `hicot_agnews` rather than trying to resample the existing
two).

**Train/test combination rule** - verified directly against the actual
files, not assumed:

- **20NG/IMDB/AGNews**: train and test are genuinely disjoint (line
  counts verified: 11,314+7,532=18,846 / 25,000+25,000=50,000 /
  10,000+2,500=12,500 - all three matching ECRTM Table 9 exactly).
  `_load_raw()` concatenates them (train first, then test, in file
  order) into one corpus, since this project's topic experiment - like
  ECRTM's own Table 2/3 - evaluates topic quality transductively over
  the whole corpus, not via a held-out split (the paper's own train/test
  split is for its downstream text-classification experiment, Sec 4.4,
  not for topic-quality evaluation).
- **SearchSnippets/GoogleNews**: HiCOT's own `train_texts.txt` and
  `test_texts.txt` are **byte-identical** (verified with a direct diff -
  same for the label files) - there is no genuine held-out split for
  these two, only one corpus duplicated across both filenames.
  `_load_raw()` detects this (exact equality of both texts and labels)
  and uses train alone, rather than silently doubling every document
  (which would corrupt corpus size and every topic-quality metric
  computed over it).

**Labels**: already 0-indexed contiguous integers in the official files
(verified: AGNews has exactly `{0,1,2,3}`, IMDB exactly `{0,1}`, 20NG
exactly 20 distinct values) - used as-is, `num_classes = max(labels)+1`.

**Vocab/BoW/word-embeddings are downloaded but not auto-wired into any
model.** `load_hicot_vocab()`/`load_hicot_bow()`/
`load_hicot_word_embeddings()` (same module) expose HiCOT's own 200-dim
GloVe embeddings (matching ECRTM Appendix B's spec) and BoW matrices
(scipy-sparse `.npz`) for a future caller - e.g. passing
`load_hicot_vocab(...)` as VAEBMAdapter's own `vocabulary=` parameter
(`models/vaebm.py::fit_predict` already supports a fixed external
vocabulary, for exactly this kind of protocol-fidelity need). This was
deliberately NOT wired into `experiment/runner.py` automatically: forcing
VAE-BM (or any model) onto HiCOT's own fixed vocabulary for these
datasets is a model-behavior decision with real consequences for
`--voc-size`, left for an explicit future change rather than decided
silently here.

**Still not changed by adding these dataset ids** (see #10 above for the
metric-side alignment, which is independent of this): the actual
preprocessing pipeline these texts went through is HiCOT's own, not
independently re-verified against Card et al. (2018)'s exact 5 steps
here - it is used because it is the paper's own artifact, not because
this project re-derived or checked it.

## 12. `classification`/`cluster`/`all` experiments: model set, representation, and every judgment call

Two new `scripts/run_experiment.py` experiment types, plus `--experiment
all` to run topic+classification+cluster in sequence (three separate
outputs, never merged) - model set `{vaebm, fastopic, lda, hicot}`
throughout (`experiment/scientific_models.py`), with `bertopic`/`glocom`
also usable in `cluster` (both pre-existing there already).

**`representation_source`/`assignment_source` are looked up by MODEL
NAME, never by probing what `get_document_topics()` returns.**
`VAEBMAdapter.get_document_topics()` always returns `mu` (never `None` -
it is reused there as the storage slot `get_mu()`/
`get_document_embeddings()` also read from), even though `mu` is
explicitly not a topic distribution (#1 above). A naive "non-`None` ->
must be theta" check would silently treat `mu` as `theta` for VAE-BM -
exactly the mistake this project has now caught and fixed twice (first
in `experiment/runner.py`'s `ecrtm_hicot` protocol work, #10 above; the
same fix is applied here via `scientific_models.py`'s
`representation_source_for_model()`/`assignment_source_for_model()`,
gated on model name, reused by both new experiments). Values:
`"theta"`/`"argmax_theta"` for fastopic/glocom/lda/hicot (all four
expose a genuine document-topic probability simplex via
`get_document_topics()`/`transform()`); `"mu"`/`"kmeans_on_latent_mu"`
for vaebm (its EXISTING, unchanged behavior - `mu` is never softmaxed
into a fake `theta`); `"embeddings"`/`"kmeans_on_embeddings"` for
bertopic (a plain SBERT-embedding KMeans swap, no theta or mu at all).

**LDA** (`models/lda_adapter.py`): scikit-learn's own
`LatentDirichletAllocation` (`learning_method="online"`, its own
default) - no reimplementation, straightforward `transform()` ->
genuine `theta`.

**HiCOT** (`models/_hicot_source.py` + `models/hicot_adapter.py`): the
actual model vendored from `github.com/HoangTran223/HiCOT`
(`HiCOT/{HiCOT,ECR,DT,TP,_model_utils}.py`), mirroring this project's
existing GloCOM-vendoring precedent (`models/_glocom_source.py`) rather
than reimplementing from the paper alone - see `_hicot_source.py`'s own
module docstring for exactly what was dropped as unused
(`torch_kmeans`, `sklearn.cluster.KMeans`, `scipy.spatial.distance.
squareform`, `sentence_transformers`, `utils.static_utils` - all
imported upstream but never referenced in `HiCOT.py`'s own body;
`hdbscan` made a lazy import, only reached by the non-default
`method_CL="HDBSCAN"` path). Hyperparameter defaults are taken from
upstream's own `utils/config.py` argparse defaults (verified against
`main.py`'s actual `HiCOT(...)` construction call), NOT `HiCOT`'s own
class-signature defaults, which differ in three places: `weight_loss_ECR`
(argparse 40.0 vs. class-signature 250.0), `max_clusters` (argparse 9 vs.
class-signature 50), `threshold_cluster` (argparse 10 vs.
class-signature 30). `use_pretrainWE` defaults to `False`, matching
upstream's own argparse default, even when a real
`load_hicot_word_embeddings()` artifact is available - this adapter does
not silently opt every run into pretrained embeddings.

**Document embeddings computed live, not downloaded.** Upstream's own
`datasethandler/basic_dataset_handler.py` loads a SEPARATE, precomputed
`doc2vec/doc_embeddings_384_.npz` artifact this project's own
`hicot_datasets.py` does NOT fetch (only the 8 files listed in #11
above). `HiCOTAdapter.fit()` instead encodes documents live via
`sentence_transformers` (default `all-MiniLM-L6-v2`, 384-dim, matching
`HiCOT`'s own `doc2vec_size=384` default) - mathematically the same
encoding process, just not reusing the cached file. This is what lets
`HiCOTAdapter` fit on ANY document list (the `cluster` experiment's own
generic-corpus philosophy - see `cluster_runner.py`'s own module
docstring), not only the pinned `hicot_*` datasets.

**`cluster` uses HiCOT's own official vocab/word-embeddings only for
`classification`, not for itself.** `scientific_models.py::build_hicot`
takes an optional `dataset_id` - when it is a `hicot_*` id,
`load_hicot_vocab()`/`load_hicot_word_embeddings()` (#11 above) are
injected; `cluster_runner.py`'s own `_build_hicot` leaves `dataset_id`
unset (generic self-fit vocab/random embedding init), matching how
`_build_fastopic`/`_build_glocom` already treat that experiment's own
philosophy ("compares models on a SHARED generic corpus, not a specific
paper's own pinned dataset artifact" - the module's own pre-existing
docstring). `classification_runner.py` always passes `dataset_id`,
since its own experiment is explicitly about HiCOT's official protocol.

**Why `classification` requires a real train/test split and `cluster`
does not.** Classification accuracy is only a meaningful generalization
measurement against a genuinely held-out test set; `cluster`/`topic`
evaluate transductively over the full corpus (matching ECRTM's own Table
2/3 methodology and this project's other datasets/protocols, #3/#10
above). `datasets/definitions/hicot_datasets.py::load_hicot_split()`
raises for `hicot_search_snippets`/`hicot_google_news` specifically,
since HiCOT ships the identical corpus under both `train_texts.txt` and
`test_texts.txt` for those two (verified directly, #11 above) - silently
handing back that "split" would report a data-leaked accuracy, not a
real one.

**The SVM protocol itself.** ECRTM Sec 4.4 / HiCOT's own `--tune_SVM`
reference: "we use the doc-topic distributions learned by topic models
as document features and train SVMs to predict the class of each
document." Neither paper's own text/code (HiCOT's `evaluations/` package
has no SVM module) specifies an exact kernel/C - this experiment uses
scikit-learn's `SVC(kernel="linear", C=1.0)` (`--svm-kernel`/`--svm-c`
overridable), a documented default, not a claim of reproducing either
paper's own SVM tuning.

**Multi-seed semantics differ between the two new experiments' own
established conventions, deliberately.** `cluster` (pre-existing,
unchanged) and `classification` (new) both refit the WHOLE model per
seed (not just re-seed a downstream classifier/KMeans step) - the same
"seed reseeds the whole pipeline" convention `experiment/runner.py`'s
topic experiment already uses, capturing a model's own training
variance rather than only a downstream step's. `classification`'s own
aggregation (`utils/stats.py::summarize`, mirroring - not importing,
per this project's independence from DTEA - the same Student's-t
single-sample CI approach `--protocol`-adjacent work already documented,
#10 above) additionally reports `n_runs`/mean/std/CI, since accuracy/F1
`std` and a 95% CI are only meaningful with 2+ seeds - a single seed
still runs (n_runs=1, ci_lower==ci_upper==the one value, std=0.0), never
a fabricated interval.

**The label-based vs. label-free metric distinction** (`cluster`'s
ACC/NMI/ARI/AMI/Homogeneity/Completeness/V-measure/Purity vs.
Silhouette/Davies-Bouldin/Calinski-Harabasz,
`metrics/clustering_quality.py`) mirrors the reference architecture at
`github.com/brunoguilherme1/document-topic-evaluatio-arena`'s own
`metrics/definitions/{clustering_quality,geometry}.py` split (same
scikit-learn functions, same citations) - reimplemented independently
here, not imported, per this project's own stated independence from
that repo (`models/base.py`'s own docstring).

**`sbert_kmeans` added as a fifth `classification`/`cluster` model** -
plain SBERT embeddings (`all-MiniLM-L6-v2` default) + scikit-learn
KMeans, no trained topic model of its own. Mirrors DTEA's own registry
(`document-topic-evaluatio-arena`'s README lists `sbert_kmeans` as a
first-class baseline, "embedding + clustering") - included here as a
cheap, always-available baseline to contrast against the four genuine
topic models, and useful as a fast sanity check (or a fallback) when
`fastopic`/`tensorflow` aren't installed in a given environment.
`representation_source="embeddings"`, `assignment_source=
"kmeans_on_embeddings"`, `topic_source="cluster-derived"` (its topic
words come from class-based TF-IDF over cluster membership, computed
after the fact - see `models/sbert_kmeans_adapter.py`'s own module
docstring - never a native model output). `classification_runner.py`'s
`_representation()` gained an `"embeddings"` branch to use it as SVM
input directly - a standard "SBERT embeddings -> SVM" baseline in its
own right, not merely a fallback for models with no theta/mu.

**A real quirk inherited from HiCOT's own vendored code, observed on a
real run, not fixed here** (per this project's own rule: document a
methodological/model issue, don't silently change the model): scipy
raises `ClusterWarning: The symmetric non-negative hollow observation
matrix looks suspiciously like an uncondensed distance matrix` from
`_hicot_source.py::HiCOT.create_group_topic()`'s call to
`scipy.cluster.hierarchy.linkage(distances, ...)`, where `distances` is
already a full pairwise (num_topics x num_topics) distance matrix, not
the condensed 1-D form (or raw observation vectors) `linkage()` expects.
scipy still runs - it silently reinterprets the square matrix as
`num_topics` raw observations in `num_topics`-dimensional space and
computes its OWN new pairwise distances from that, which is related to
but not identical to clustering directly on the intended distance
matrix. This is upstream `HiCOT.py` source, vendored verbatim (see
`_hicot_source.py`'s own module docstring) - not something this project
introduced or has silently corrected. It may contribute to occasional
degenerate topic-grouping behavior (`create_group_topic()`'s HAC-based
grouping), on top of the more likely dominant cause: this project's own
`build_hicot()` reduces `epochs` to 50 from upstream's own 500 (a
smoke-run default, not a paper reproduction, see
`scientific_models.py`'s own module docstring) - a real fit at a much
higher `K` (e.g. GoogleNews's 152 classes) with only 50 epochs across
four jointly-trained loss terms is a plausible, sufficient explanation
for a collapsed/degenerate clustering on its own, without needing the
scipy quirk to explain it.
