# Repository comparison report: Repo A vs. Repo B

**Repo A** = `nois-vaebm-baselines-comparision` (independently built, studied fresh for this report)
**Repo B** = `vaebm-baselines-comparision` (this repository, "our own")

This report was produced by reading Repo A's actual source, configs, results, vendored upstream
code, and test suite — not its README — and by attempting to run it. Every claim below cites a
specific file/line/command. Where a repo's own documentation and its code disagreed, that is
flagged explicitly. This report does not assume Repo B is better; several findings below favor
Repo A, and are adopted into Repo B in the improvements section.

## A. Executive verdict

Neither repository is simply "better." Their strengths are complementary and mostly orthogonal:

- **Repo A is stronger on**: dataset-checksum enforcement (fails loud on mismatch *before* use,
  not just after-the-fact recording), a formal `MatchStatus` enum + aggregate `fair_comparison`
  boolean, using the *exact* official GloCOM `run.py` via subprocess for training, and refusing to
  silently substitute a coherence metric when Palmetto is unavailable.
- **Repo B is stronger on**: actually having *run* both protocols end-to-end and produced real,
  committed published/reproduced/VAE-BM numbers; preserving VAE-BM's exact supplied architecture
  (initializers, checkpointing/EarlyStopping/NaN-termination, TF-IDF+log1p preprocessing) rather
  than a from-scratch reimplementation; correctly distinguishing GloCOM's own TD formula from the
  generic one; and a substantially deeper, more targeted test suite (22 tests vs. 7, several of
  which assert numeric edge cases, not just structural contracts).
- **Repo A has a critical, verified defect**: it discards the *official* GloCOM `run.py`'s own
  correctly-computed metrics (which it already ran via subprocess) and recomputes them itself in
  Python — using, for Topic Diversity, a formula that is **provably different** from the official
  one it vendors verbatim at `data/upstream/glocom/evaluations/topic_diversity.py` (see §C.1).
  Combined with `subprocess.run(..., check=True)` around a script that calls Palmetto internally
  with no error handling, Repo A's GloCOM pipeline **cannot produce any result at all** without a
  full Java + Palmetto + multi-GB Wikipedia index installation — confirmed by direct attempt on
  this machine (§F).
- **Best final design** = Repo B's execution discipline and VAE-BM fidelity + Repo A's checksum
  enforcement, `MatchStatus`/`fair_comparison` formalization, and (fixed) official-artifact reuse
  patterns. The improvements section below adopts exactly this combination into Repo B.

## B. Detailed comparison table

| Area | Repo A | Repo B | Better | Reason |
|---|---|---|---|---|
| Architecture | Single generic `BaselineProtocol` dataclass driven entirely by one YAML per baseline (`configs/experiments/<id>.yaml`); adapters resolved by dotted import path string | One `BaselineProtocol` *class* per baseline in Python + a YAML for provenance only | **A** | Repo A's design needs *zero* new Python classes to add a baseline whose logic fits the generic contract — genuinely closer to the task's "1 adapter + 1 protocol + 1 config" ideal (protocol *is* the config). Repo B still requires a hand-written `FASTopicProtocol`/`GloCOMProtocol` subclass, which is more code per baseline (mitigated in the improvements below). |
| Protocol fidelity — FASTopic | Injects the *exact* TopMost-released `train_bow.npz`/`vocab.txt` into FASTopic via a custom `ReleasedArtifactPreprocess` (`models/fastopic.py:18-22`), bypassing re-tokenization entirely | Re-runs `topmost.Preprocess(vocab_size=10000, stopwords="English")` on `train_texts` (re-derives vocab/BoW rather than reusing the shipped `.npz`/`vocab.txt`) | **A** | Reusing the byte-identical shipped artifact is strictly more faithful than re-deriving an equivalent one, even if the recipe is documented (see improvements: adopted). |
| Protocol fidelity — GloCOM (baseline) | Subprocess-executes the *actual* official `run.py` (`models/glocom.py:28-35`) — zero reimplementation risk in training | Vendors only the 2 `nn.Module` classes; reimplements training via `topmost.BasicTrainer` + a duck-typed dataset | **A** (training) / **B** (robustness — see next row) | Repo A's approach is more faithful *when it works*. But see the critical defect below: it is not robust, and its own recomputed metrics disagree with the official code it subprocesses. |
| GloCOM metric fidelity | Recomputes TD/Purity/NMI itself in Python from raw artifacts *after* discarding `run.py`'s own printed, correct values; its `topic_diversity()` (`metrics/paper_metrics.py:12-16`) computes `unique_words/(K·top_n)`, **not** the official `TD=(TF==1).sum()/(K·T)` from `evaluations/topic_diversity.py` it vendors verbatim | Implements both: `topic_diversity()` (generic/FASTopic) and `topic_diversity_glocom()` (official TF==1 formula), selected per-protocol via `MetricSpec.name`, unit-tested to prove they diverge on repeated words | **B** | Verified directly (§C.1): the two formulas give different values whenever any top-word repeats across topics — SearchSnippets topics do repeat words, so Repo A's GloCOM TD number, if it were ever produced, would not match the paper's own metric. |
| GloCOM baseline robustness | `subprocess.run(..., check=True)`; official `run.py` calls Palmetto internally with **no** try/except (`data/upstream/glocom/run.py`, final block) | Own training loop; Palmetto/Wikipedia not wired in, but coherence computed via a documented gensim fallback that never blocks TD/Purity/NMI/topics | **B** | Verified by direct attempt on this machine (§F): no Java runtime installed, no `tools/palmetto/`; the official script would raise past the coherence line, and Repo A's `check=True` means the *entire* subprocess result (including already-written `top_words_15.txt`/`test_theta.npy`) is discarded on that non-zero exit. Repo A's GloCOM baseline literally cannot report anything without a full Palmetto+Wikipedia install. |
| Protocol fidelity — FASTopic dataset choice | 20 Newsgroups (20,846 docs), TopMost's official `20NG.zip`, K=50 | NYT (9,172 docs), TopMost's official `NYT.zip`, K=50 | **Tie** | Both are real datasets the FASTopic paper reports at K=50 with an official pre-split TopMost artifact (verified independently in this project's own prior research, not just Repo A's claim). 20NG is the more commonly-cited "flagship" dataset in topic-model literature; NYT is ~2x smaller/faster, better matching this project's own "small experiments only" instruction. Neither choice is a fidelity error. |
| Split-semantics honesty (FASTopic) | Marks `split_verification: unknown` — refuses to assert the released 11,314/7,532 partition is what the paper's tables used | Marks `split_strategy` as MATCH, backed by evidence that TopMost's own reference `FASTopicTrainer` fits on `train_texts` and evaluates held-out `test_texts` for clustering (independently verified against `topmost/trainers/basic/FASTopic_trainer.py` in this project's own research) | **Tie** (see note) | Repo A is more conservative; Repo B's MATCH is evidence-backed, not asserted. Both are legitimate positions on a genuinely ambiguous point (the paper itself never states the mapping) — presented as a disagreement, not a bug on either side. |
| VAE-BM integration | **Rewritten from scratch**: no custom Identity-gain kernel initializers from the supplied notebook, no `EarlyStopping`/`ModelCheckpoint`/`TerminateOnNaN` callbacks, no `init_R_from_vocab` option, feeds the *raw* released `bow.npz` counts directly (no TF-IDF weighting, no `log1p`) | `models/vaebm.py` is the supplied notebook code ported with unchanged architecture, initializers, and callbacks; `TfidfVectorizer(...); log1p` preprocessing preserved; optional `vocabulary=` param added (additive only) to pin the baseline's exact vocab | **B** | The supplied code's own preprocessing step is "documents → BoW/TF-IDF → ... → VAE-BM" (task's own words) — Repo A's adapter skips the TF-IDF/log1p step entirely, which is a real, unflagged deviation from "the supplied formulation" it claims to preserve in its own docs (`docs/METHODOLOGICAL_ISSUES.md`). Repo A also has no safety net for the exact NaN-divergence failure mode this project discovered and documented (§C.2) — if run, it would likely produce silently-garbage `mu` with no diagnostic. |
| VAE-BM K/vocab matching | `build_vaebm()` is one method shared by *both* protocols in `protocols/base.py:73-83`, hardcoding `epochs=20, lr=0.01, alpha=0.99` identically for both baselines; does correctly pass the protocol's own `topic_count` as K | Each protocol's own `build_vaebm()` sets K, vocabulary (exact list, not just size), and an embedder matched to that baseline's own `doc_embed_model`/`embedding_model` | **B** | Repo A never diverges K from the baseline (good), but its one-size-fits-all VAE-BM config ignores per-protocol embedder choice and vocabulary-list matching (it happens to inherit the right vocab only because it reuses the released `bow.npz` directly — see the row above for why that specific reuse has its own preprocessing-fidelity cost). |
| Fair-comparison guardrails | Formal `MatchStatus{MATCH,DIFFERENCE,UNKNOWN}` enum + `ProtocolCheck` dataclass + a single derived `fair_comparison = all(status is MATCH)` boolean, persisted into every result record | Free-text `"verdict": "MATCH - ..."` strings inside a dict returned by `verify()`; no enum, no aggregate boolean | **A** (structure) | Repo A's structure is exactly what the task's §7 asks for and is genuinely better engineering. However (§C.3), most of Repo A's individual checks compare a config value to *itself* (tautological MATCH), so the aggregate boolean's rigor is currently shallower than its shape suggests — both repos' *content* is comparably self-asserted; Repo A's *shape* is better. Adopted into Repo B (§D). |
| Reproducibility metadata | Per-*run* `results/<id>/runs/<dataset>-<model>-seed<N>/metadata.json` (immutable, one per run) with environment/package versions, seed, metric errors | One shared `protocol.json` (overwritten per `verify_protocol.py` call) + one `results/<id>/<system>_results.csv` (append-only rows) + one topics JSON overwritten per dataset | **A** | Per-run immutable metadata is strictly better: Repo B's shared `protocol.json`/topics-JSON files get silently overwritten by the next run, losing history a per-run directory would keep. Adopted into Repo B (§D). |
| Dataset checksum enforcement | Hardcoded *expected* SHA256 in the YAML, compared at download time; **raises and aborts** on mismatch (`datasets/download.py:41-42,59-60`) | Computes and records SHA256 into a manifest *after* download for later self-consistency checks; no independently-recorded "expected" hash to catch a changed upstream artifact | **A** | Repo A can detect "the upstream file silently changed since we pinned it"; Repo B can only detect "the local copy was tampered with after our own download," a strictly weaker guarantee. Adopted into Repo B (§D). |
| Colab notebook | Single parameterized notebook; user sets `BASELINE`/`DATASET` and reruns the whole notebook once per protocol; computes `DEVICE` correctly before invoking scripts; explicitly documents the Palmetto external prerequisite | One notebook with separate, already-filled-in cells for *both* protocols in one pass; documents `--full` flag | **B** (turnkey) / **A** (prerequisite honesty) | Repo B's notebook needs no re-parameterization to cover both protocols. Repo A's notebook is more explicit that Palmetto is an unmet external prerequisite by default (both should state this — Repo B's did not call this out as sharply before this report; fixed in §D). |
| Tests | 7 tests / 60 lines total, entirely structural ("protocol registered," "method is callable," "field equals string") | 22 tests / passing (incl. one requiring live network), several assert actual numeric behavior: `glocom_td` vs. `topic_diversity` disagree on repeats, purity/NMI/ARI/IRBO edge cases, checksum-tamper detection, K-never-diverges-from-baseline enforced by *building* real adapters | **B** | Verified by running both suites (§F): Repo A's tests would pass even if `topic_diversity()`'s formula were wrong for GloCOM (and it is — no test catches this). Repo B's test suite would have caught that specific class of bug. |
| Extensibility | Adding a baseline = one adapter + one YAML (protocol *is* the YAML) — no new Python class required for the common case | Adding a baseline = one adapter + one YAML (provenance) + one `BaselineProtocol` subclass in Python | **A** | Genuinely less code per new baseline for the common case. Not fully adopted (Repo B's per-protocol Python `verify()`/`build_vaebm()` customization has real value — see improvements — but the config-as-protocol idea is worth partially adopting). |
| Documentation | `README.md` + `docs/METHODOLOGICAL_ISSUES.md`, concise, accurate to what's implemented (does not overclaim results it hasn't produced) | `README.md` + `docs/methodological_notes.md`, longer, includes real numeric findings (lr divergence table, checkpoint-collision bug) because real runs were done | **Tie** | Different content because of different execution state, not different quality. Both are honest about their own limitations. |
| Result design (published vs. reproduced vs. VAE-BM kept separate) | Yes, structurally (`published_results.csv`, `baseline_results.csv`, `vaebm_results.csv`, `comparison.csv`) — but the latter two have never been populated | Yes, same separation, and populated with real numbers from actual runs | **B** | A separation that is never exercised with real data is unverified in practice; Repo B's separation is proven to work end-to-end (checked by re-running after fixing two real bugs found along the way). |
| Dependency isolation | `pyproject.toml` extras split cleanly (`fastopic`, `glocom`, `vaebm`); GloCOM's own official pinned versions documented in README (Python 3.10.14, torch 2.4.1, sentence-transformers 2.7.0, numpy 1.26.3, scipy 1.10.1, sklearn 1.5.1, gensim 4.3.3) but **not enforced** — `pyproject.toml`'s own `glocom` extra allows `torch>=2.4` (unpinned) and `gensim==4.3.3` (pinned), an inconsistent mix | Extras split similarly; `truststore` added as a core dependency to work around a real corporate-proxy TLS failure encountered while actually installing/running | **Tie** | Repo A documents GloCOM's exact upstream pins more thoroughly (a genuine plus); Repo B discovered and fixed a real installation-blocking issue Repo A never hit because it never finished installing the ML extras (venv confirmed to have zero ML packages installed, §F). |

## C. Verified findings (not opinions)

### C.1 — Repo A's GloCOM `topic_diversity()` does not match the official formula it vendors

Repo A's own vendored copy, `data/upstream/glocom/evaluations/topic_diversity.py`:

```python
def compute_TD(texts):
    K = len(texts); T = len(texts[0].split())
    ...
    TF = counter.sum(axis=0)
    TD = (TF == 1).sum() / (K * T)          # words in >=2 topics score ZERO, not one
```

Repo A's own `src/vaebm_benchmark/metrics/paper_metrics.py:12-16`:

```python
def topic_diversity(topics, top_n=15):
    ...
    return len({word for topic in selected for word in topic}) / (len(selected) * top_n)
    # every unique word counts once, regardless of how many topics repeat it
```

These are different functions whenever any top-word repeats across ≥2 topics (common — SearchSnippets' topic lists do repeat words). `evaluate_output()` (`evaluation/runner.py:28`) calls the generic version for *both* protocols. This project's own `metrics/topic_quality.py` implements both formulas separately (`topic_diversity` and `topic_diversity_glocom`) and has a unit test (`tests/test_metrics.py::test_topic_diversity_glocom_differs_from_standard_on_repeats`) proving they diverge — precisely to prevent this class of bug.

### C.2 — Repo A's VAE-BM has no protection against the exact divergence this project found and fixed

This project's own diagnostic (documented in `docs/methodological_notes.md` §8) proved the supplied VAE-BM's own default `lr=1e-2` diverges to `inf`/`NaN` within ~2 epochs at vocab scales of 4,618-10,000 words, producing near-random KMeans clusters (Purity 0.224, NMI 0.014) unless caught. Repo A's `VAEBMAdapter.fit_evaluate()` (`models/vaebm.py:110-118`) calls `model.fit(...)` with **no callbacks at all** — no `TerminateOnNaN`, no `EarlyStopping`, no checkpoint/restore — and its `build_vaebm()` (`protocols/base.py:79`) defaults to the **same** `learning_rate: 0.01`. If/when Repo A actually runs VAE-BM (it has not — §F), it is very likely to hit the identical failure mode this project already characterized, with no mechanism to detect or recover from it.

### C.3 — Repo A's `MatchStatus` checks are structurally strong but mostly tautological

`protocols/base.py:93-116` — most `ProtocolCheck` entries compare a config value to *the same config value* (e.g. `ProtocolCheck("paper", self.paper["citation"], self.paper["citation"], MatchStatus.MATCH)`), which trivially always passes and verifies nothing against an independent source. Only `checksum` (compared against a runtime-computed manifest), `split_verification` (hand-set per dataset), and `seeds` (hand-set to `None`/known) carry real epistemic content. The *shape* (enum + aggregate boolean) is worth adopting; the current *content* is not meaningfully more "verified against upstream" than Repo B's narrative verdict strings, which are also self-asserted but backed by cited research (this project's own multi-source investigation of both papers/repos, done before either repo was built).

## D. Features to adopt from Repo A (genuinely better ideas)

1. **Checksum enforcement at download time, not just after-the-fact recording.** Record an
   *expected* SHA256 per file in each dataset config; fail loudly if the downloaded artifact
   doesn't match. Adopted in `utils/provenance.py`/dataset fetchers.
2. **A formal `MatchStatus` enum + `ProtocolCheck` dataclass + aggregate `fair_comparison`
   boolean**, computed and persisted with every run, not just described in prose. Adopted in
   `protocols/base.py`.
3. **Per-run immutable metadata directories** (`results/<protocol>/runs/<dataset>-<system>-seed<N>/`)
   instead of one shared, overwritable `protocol.json`/topics-JSON per protocol. Adopted in
   `evaluation/runner.py`.
4. **Reusing a baseline's released BoW/vocab artifact directly for VAE-BM's input** where a
   protocol has one (e.g. GloCOM's official `bow.npz`), rather than re-deriving an equivalent BoW
   via a fresh vectorizer — but *without* dropping VAE-BM's own TF-IDF+log1p preprocessing step
   (Repo A's mistake, per row above). Implemented as an optional `raw_bow=` path.
5. **A real, optional Palmetto integration** (`palmetto_cv()` calling the actual jar), used when
   present, falling back to the existing documented gensim approximation when absent — rather than
   only ever having the approximation. Adopted in `metrics/topic_quality.py`.

## E. Features from Repo A that should NOT be adopted

1. **Discarding official code's own computed metrics after subprocessing it, and recomputing them
   independently in Python** — this is exactly what produced the TD-formula bug (§C.1). If a
   baseline's official script prints/saves its own metric values, use those values, don't
   re-derive them from intermediate artifacts with a second implementation that can silently
   diverge.
2. **`subprocess.run(..., check=True)` around a script with an unguarded external dependency
   (Palmetto) that runs *after* the artifacts you actually need are already written.** This turns
   a soft, cosmetic metric failure into a hard failure of the entire baseline run. If official
   code is subprocessed, its exit code should not be trusted as a proxy for "did the outputs I
   need get written" — check for the files directly.
3. **A from-scratch VAE-BM reimplementation that drops the supplied initializers, training
   safety-nets, and the TF-IDF/log1p preprocessing step**, while documentation claims "the
   supplied formulation" is preserved. Preserve the supplied code close to verbatim instead (as
   Repo B already does).
4. **A default `--device cuda` on a CLI script** (`scripts/run_baseline.py`) — footguns on any
   CPU-only or Mac environment unless the caller remembers to override it (the notebook does;
   direct CLI use does not default sensibly).
5. **Tautological self-consistency checks presented as protocol "verification."** A check that
   always reads the same field twice and calls it MATCH adds structure without adding evidence;
   worth fixing rather than replicating as-is (Repo B's adoption of the `MatchStatus` shape backs
   its checks with the same cited evidence its narrative verdicts already had, not with new
   tautologies).

## F. Test/run results (this session, both repos)

**Repo A** (`.venv`, Python 3.12.13, only base deps installed — numpy/scipy/sklearn/pyyaml/requests/pytest;
no fastopic/topmost/gensim/sentence-transformers/torch/tensorflow installed):
- `pytest tests/` → **7 passed** (all structural; see §B).
- `data/raw/{fastopic,glocom}/...` present (datasets *were* downloaded, checksums matched).
- `results/{fastopic,glocom}/{comparison.csv,protocol.json,published_results.csv}` present, but
  `reproduced_baseline` and `vaebm` columns are **empty in both** — no `baseline_results.csv`,
  no `vaebm_results.csv`, no `results/*/runs/` directory exists anywhere in the repo.
- Confirmed blockers to completing a real run on this machine: no ML extras installed in the
  venv; no Java runtime (`java -version` → "Unable to locate a Java Runtime"); no
  `tools/palmetto/` directory. GloCOM's official `run.py` calls Palmetto unconditionally at its
  final step with no exception handling — this would abort the whole subprocess (and, given
  `check=True`, abort Repo A's adapter) even though training/TD/Purity/NMI complete first.

**Repo B** (this repo): 22 tests passing (`pytest tests/`, full ML extras installed); both
protocols actually executed end-to-end with real committed results in `results/{fastopic,glocom}/`
(see `docs/methodological_notes.md` for the full published/reproduced/VAE-BM tables and the two
real bugs found and fixed along the way — a checkpoint-path collision and a VAE-BM learning-rate
divergence).

## G. Critical scientific issues found, ranked

- **CRITICAL** — Repo A: GloCOM's `topic_diversity()` formula does not match the official code it
  vendors and subprocesses (§C.1). Any number it eventually produces for GloCOM's TD would be
  silently wrong relative to the paper's own metric.
- **CRITICAL** — Repo A: the GloCOM baseline pipeline cannot produce *any* result without a full
  Java + Palmetto + Wikipedia_bd install, due to `check=True` around a script that fails late and
  unconditionally on a metric that is not needed to obtain TD/Purity/NMI/topics (§F).
- **HIGH** — Repo A: VAE-BM reimplementation drops the supplied TF-IDF+log1p preprocessing step
  and the initializer scheme, and has no divergence safety-net for a default learning rate this
  project proved unstable (§C.2). If run as-is, results would likely be silently degenerate.
- **HIGH** — Repo B (fixed during this project, see `docs/methodological_notes.md` §8): the
  supplied VAE-BM's own default `lr=1e-2` diverges to NaN at these vocab scales; both protocols
  now override to `lr=1e-3`, documented rather than silently changed.
- **MEDIUM** — Repo A: `MatchStatus` checks are structurally sound but mostly tautological (§C.3)
  — currently adds an appearance of rigor beyond what is actually verified.
- **MEDIUM** — Repo B (pre-existing, addressed below): no formal `fair_comparison` boolean; a
  reader has to read prose verdicts to determine whether a comparison is defensible.
- **LOW** — Repo A: `--device cuda` CLI default is a footgun off the documented notebook path.
- **LOW** — Both repos: neither repo's "verify" step re-fetches or independently re-derives values
  from a live copy of the official source at run time; both encode what their authors already
  researched, which can go stale if the upstream repo changes.

## H. Features both repositories are missing

- **A live protocol-diff mechanism**: neither repo re-fetches the official repo/paper at verify
  time to detect drift (e.g., an upstream commit moving, a changed default hyperparameter). Both
  encode a point-in-time research finding as a static config value.
- **Paper-vs-code discrepancy tracking as structured data** (both have it as prose in markdown
  docs, not as a queryable field alongside each protocol check).
- **Container/lockfile-level reproduction** (a Dockerfile or `pip freeze`-style lockfile pinned to
  exact resolved versions) — both rely on loose `pyproject.toml` ranges plus prose about what was
  "actually used."
- **An automatic environment snapshot taken at the moment of each individual metric computation**
  (both capture environment once per run, not per metric — irrelevant while metrics run in the
  same process, but would matter if a future baseline shells out to a different environment, as
  GloCOM's subprocess path already does in Repo A).
- **Better Colab isolation**: neither creates a fresh virtualenv/conda env inside the Colab
  notebook itself; both `pip install -e .` directly into the Colab base environment, risking
  conflicts with Colab's preinstalled package set (notably: Colab ships its own numpy/torch that
  the `numpy<2.0` pin and `tensorflow`/`torch` co-installation could fight with).
