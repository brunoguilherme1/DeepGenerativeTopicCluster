# VAE-BM Baselines Comparison

Faithful, per-baseline reproduction of published topic-model protocols -
**FASTopic** and **GloCOM** - each compared against **VAE-BM** under the
*exact* experimental conditions the original paper used, not a
standardized benchmark.

## Relationship to DTEA

This repository is a sibling of, and takes architectural inspiration
from, [`document-topic-evaluatio-arena`](../document-topic-evaluatio-arena)
(DTEA) - but answers a different question, on purpose:

| | DTEA | This repo |
|---|---|---|
| Question | *Under one common benchmark, how do topic models and clustering methods compare?* | *Under the exact experimental conditions used by established published methods, how competitive is VAE-BM?* |
| Priority | **Standardization** - same datasets, splits, metrics, protocol, for every model | **Faithful reproduction** - each baseline keeps its own paper's dataset artifact, preprocessing, split strategy, K, and metrics |
| Splits | Generates its own reproducible splits (`RandomSplit` etc.) for every dataset | Uses a split ONLY if the original paper/repo used one; most topic-model papers (both baselines here) evaluate transductively on the full corpus, and this repo reproduces that rather than inventing a split |
| Adding a baseline | One model adapter, registered once | One model adapter + **one `BaselineProtocol`** + one config |

DTEA is not modified by anything in this repository. Two adapters here
(`fastopic_adapter.py`, `glocom_adapter.py`) independently reuse the same
official upstream sources DTEA's own adapters wrap (the `fastopic` PyPI
package; the vendored `qducnguyen/GloCOM` source) - both repos point at
the same authors' code, not at each other.

## The core rule

> **Never claim a fair comparison unless VAE-BM and the baseline were
> evaluated using the same dataset artifact, preprocessing, K, metrics,
> and experimental protocol for that specific paper.**

Concretely: if FASTopic's paper uses `K=20` on 20 Newsgroups, VAE-BM gets
`KMeans(n_clusters=20)` on the exact same preprocessed 20 Newsgroups - K
is never independently tuned for VAE-BM. See
`src/vaebm_benchmark/protocols/base.py`'s `BaselineProtocol` docstring.

## Repository structure

```text
vaebm-baselines-comparision/
├── README.md
├── pyproject.toml
├── docs/
│   └── methodological_notes.md   # documented issues/decisions, per the "don't silently change VAE-BM" rule
├── configs/
│   ├── models/                   # provenance: citation, official repo, implementation notes
│   ├── datasets/                 # provenance: source URL, citation, license, expected doc/class counts
│   └── experiments/               # smoke-test experiment configs
├── src/vaebm_benchmark/
│   ├── models/
│   │   ├── base.py               # ProtocolModelAdapter - the common fit()/get_topics()/get_document_clusters() interface
│   │   ├── vaebm.py              # VAE-BM AS SUPPLIED - unmodified math (Encoder/Decoder/VAEBM/VaeBmKMeansFit)
│   │   ├── vaebm_adapter.py      # thin adapter wrapping vaebm.py to the common interface
│   │   ├── fastopic_adapter.py   # wraps the official `fastopic` PyPI package
│   │   ├── glocom_adapter.py     # this repo's data pipeline around the vendored GloCOM model
│   │   ├── _glocom_source.py     # vendored verbatim from github.com/qducnguyen/GloCOM
│   │   └── _topmost_bases.py     # shared topmost.Preprocess/BasicTrainer plumbing
│   ├── protocols/
│   │   ├── base.py               # BaselineProtocol - the abstraction everything else is organized around
│   │   ├── fastopic_protocol.py
│   │   └── glocom_protocol.py
│   ├── datasets/                 # provenance-tracked, checksummed dataset fetchers
│   ├── metrics/                  # topic_quality.py (NPMI/C_V/Diversity/IRBO), clustering_quality.py (NMI/ARI/AMI/Purity/ACC)
│   ├── evaluation/                # runner.py (fit+evaluate+persist), registry.py (protocol name -> class)
│   └── utils/                     # paths, provenance/checksums, seeding
├── scripts/
│   ├── verify_protocol.py        # prints + persists a protocol's own provenance/verdict report
│   ├── run_baseline.py
│   ├── run_vaebm.py
│   └── compare.py                # published vs. reproduced vs. vaebm, side by side
├── notebooks/
│   └── VAEBM_Baseline_Comparison_Colab.ipynb
├── tests/
├── data/{raw,processed}/          # gitignored; checksummed manifests recorded alongside raw data
└── results/
    ├── fastopic/{protocol.json, baseline_results.csv, vaebm_results.csv, comparison.csv}
    └── glocom/{protocol.json, baseline_results.csv, vaebm_results.csv, comparison.csv}
```

## Quickstart (local or Colab)

```bash
git clone <this-repo-url>
cd vaebm-baselines-comparision
pip install -e ".[fastopic,glocom,vaebm,metrics,viz,dev]"

python scripts/verify_protocol.py --baseline fastopic
python scripts/run_baseline.py    --baseline fastopic --dataset nyt
python scripts/run_vaebm.py       --protocol fastopic --dataset nyt
python scripts/compare.py         --baseline fastopic --dataset nyt

python scripts/verify_protocol.py --baseline glocom
python scripts/run_baseline.py    --baseline glocom --dataset search_snippets
python scripts/run_vaebm.py       --protocol glocom --dataset search_snippets
python scripts/compare.py         --baseline glocom --dataset search_snippets
```

Add `--full` to any of the above to use paper-scale settings (e.g. 200
epochs) instead of the smoke-test default (20 epochs) - see
`protocols/fastopic_protocol.py`/`protocols/glocom_protocol.py`.

**Datasets**: FASTopic's smoke test uses **NYT** (9,172 docs, via
`topmost`'s official mirror); GloCOM's uses **SearchSnippets** (12,294
docs, via the official repo's own precomputed artifact) - see
`docs/methodological_notes.md` for why these were chosen over the other
datasets each paper reports.

See `notebooks/VAEBM_Baseline_Comparison_Colab.ipynb` for a runnable,
step-by-step Colab version of the same flow (clone → install → GPU check
→ verify protocol → run baseline → run VAE-BM → compare → final table).

## Adding a future baseline

1. One model adapter under `src/vaebm_benchmark/models/` implementing
   `ProtocolModelAdapter` (`base.py`).
2. One `BaselineProtocol` subclass under
   `src/vaebm_benchmark/protocols/` pinning dataset artifact,
   preprocessing, K, split strategy, metrics, and seeds from that paper.
3. One config under `configs/models/<baseline>.yaml` recording provenance.
4. One entry in `src/vaebm_benchmark/evaluation/registry.py`.

No other file should need to change - this is deliberately smaller-scope
than DTEA's full registry/catalog machinery, since this repo's own size
guideline (`configs`/`protocols`, not a general model zoo) doesn't need it.

## Current scope

Only **FASTopic** and **GloCOM** are implemented, each against **one**
small official dataset, **one** baseline seed, **one** VAE-BM seed - a
smoke test, per this project's own instructions, not a full experiment
sweep. Expand datasets/seeds only after confirming the smoke test
reproduces sane, non-degenerate results for both systems.

## Additional experiment runner: `scripts/run_experiment.py`

A separate, simpler runner - independent of the paper-fidelity
`protocols/*.py` track above - for direct model-vs-model comparisons on
shared corpora (not a specific paper's own pinned artifact/split). Three
modes:

- `--experiment topic` (default): VAE-BM vs. BERTopic topic-quality
  comparison (C_V, Purity, NMI, TD) at requested K values.
- `--experiment cluster`: pure clustering-quality comparison (ACC via
  Hungarian-matched accuracy, NMI) across any registered model
  (`vaebm`, `bertopic`, `fastopic`, `glocom`) that exposes hard document
  clusters - K is always the dataset's own number of ground-truth
  classes, never independently tuned.
- `--experiment llm_cluster_refinement`: an LLM-based post-clustering
  refinement stage (LLMEdgeRefine-style) applied on top of the same hard
  clusters `cluster` mode produces - before/after ACC/NMI/ARI/AMI/Purity
  plus Silhouette/Davies-Bouldin/Calinski-Harabasz. Requires a GPU
  runtime for real use (4-bit Mistral-7B via `pip install -e ".[llm]"`);
  see `notebooks/LLM_Cluster_Refinement_Colab.ipynb` and
  `src/vaebm_benchmark/llm/*.py`'s module docstrings for the full design
  (edge-point detection, candidate-cluster selection, cluster-context
  construction, deterministic prompting/parsing, conservative
  reassignment, persistent decision cache, `--resume` support).

Run `python scripts/run_experiment.py --help` for the full flag list.
