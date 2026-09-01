# SLURM/Jupyter GPU setup

This branch (`slurm-gpu-experiments`) exists to make this project reliable
on a SLURM cluster with GPUs, without changing anything about how it runs
on Colab or a laptop. Nothing in `pyproject.toml`'s dependency ranges was
tightened to do this - see the comments next to `glocom`/`llm`/
`sbert_experiment` there for why an unbounded `torch>=2.0` stays unbounded
(same reasoning already applied to `numpy` in this file: a cap risks a
multi-minute pip resolver backtrack on an environment that's already
working). Everything CUDA-version-specific lives here instead, as an
install-order recipe you run once per environment.

## The problem this solves

`pip install torch` (with no index pinned) resolves to whatever CUDA build
is newest for your platform - not the newest build your driver actually
supports. On an environment with:

```text
NVIDIA H200
Driver Version: 570.195.03
Driver-supported CUDA: 12.8
```

`pip` can still resolve `torch==2.13.0+cu130` (built for CUDA 13.0), and
`torch.cuda.is_available()` silently returns `False` - no exception, no
error, just a build too new for the driver to run. `nvidia-smi` still sees
the GPUs fine, which is what makes this confusing: the driver and hardware
are working, only the specific torch wheel is wrong.

Run `python scripts/check_gpu.py` any time to check for exactly this - see
"Diagnosing the CUDA mismatch" below.

## Recommended install workflow

```bash
git clone https://github.com/brunoguilherme1/DeepGenerativeTopicCluster.git
cd DeepGenerativeTopicCluster
git checkout slurm-gpu-experiments
```

**Install a CUDA-matched torch build FIRST**, before installing this
project's own extras. Once torch is already installed, pip's resolver sees
`sentence-transformers`'/`bertopic`'s own unbounded `torch` requirement as
already satisfied and leaves it alone - it will NOT silently upgrade or
replace it (pip does not touch an already-installed package unless a
version conflict forces it to, or you pass `--upgrade`). This install-order
trick is what keeps the CUDA pin "isolated" from `pyproject.toml` itself.

```bash
# Match the wheel index to your DRIVER's supported CUDA version (12.8 in
# the example above -> the cu126 index; check the exact available
# version/tag against https://pytorch.org/get-started/locally/ or
# https://pytorch.org/get-started/previous-versions/ at install time -
# PyTorch ships new releases regularly and the note below (torch 2.10.x)
# is an example, not a guarantee it's still the latest cu126 build.
pip install torch --index-url https://download.pytorch.org/whl/cu126
# e.g. explicitly: pip install torch==2.10.0+cu126 --index-url https://download.pytorch.org/whl/cu126

python scripts/check_gpu.py   # confirm torch.cuda.is_available() == True BEFORE installing anything else
```

Then install this project's own dependencies, choosing the extra that
matches what you're actually running:

```bash
# SBERT-vs-BERTopic topic experiments (sbert_kmeans, sbert_mpnet,
# sbert_t5large, bertopic, ...) - no VAE-BM, no tensorflow:
pip install -e ".[sbert_experiment]"

# VAE-BM comparisons too (model "vaebm" or any --vaebm-configs variant) -
# pulls in tensorflow, needed by src/vaebm_benchmark/models/vaebm.py:
pip install -e ".[experiment]"

# LLM cluster refinement (--experiment llm_cluster_refinement) - its own,
# separate, much heavier extra (transformers/accelerate/bitsandbytes):
pip install -e ".[llm]"
```

Avoid `pip install -e ".[all,llm]"` for a plain SBERT run - it forces
tensorflow, FASTopic/GloCOM's own dependency chains, and the entire LLM
stack (bitsandbytes, accelerate) onto an install that needs none of them,
each one more surface area for a CUDA/toolkit version conflict on a
cluster where you're already fighting one.

Re-run `python scripts/check_gpu.py` after each of these installs -
`sentence-transformers` and `bertopic` both declare their own (unbounded)
`torch` requirement, so it's worth confirming a later `pip install` step
didn't quietly pull in a second, incompatible torch build.

## Running the experiment (CLI unchanged - `--sbert-configs` required)

`sbert_mpnet`/`sbert_t5large` are not built-in model names - they are
named `sbert_kmeans` variants (a chosen embedder, registered under a
name of your choice), defined via `--sbert-configs`. Running just
`--models sbert_mpnet sbert_t5large` with no `--sbert-configs` at all
fails with `KeyError: Unknown model 'sbert_mpnet'`. The CLI itself is
unchanged (`--sbert-configs` already existed before this branch) - this
is only about actually supplying it:

```bash
python scripts/run_experiment.py \
  --experiment topic \
  --models sbert_mpnet sbert_t5large \
  --datasets search_snippets biomedical stack_overflow google_news_ts 20ng \
  --k 20 80 \
  --seed 42 \
  --sbert-configs '{"sbert_mpnet": {"embedder": "all-mpnet-base-v2"}, "sbert_t5large": {"embedder": "t5-large"}}'
```

See `--sbert-configs`'s own `--help` text, or `experiment/runner.py`'s
`register_sbert_kmeans_variants()`, for the general mechanism (also
usable with any number of other embedders, not just these two).

## In Jupyter specifically: `!pip` vs. the kernel's own Python

A Jupyter cell's `!pip install ...` runs whatever `pip` is first on
`PATH` for the shell the cell spawns - which is not guaranteed to be the
same Python environment the notebook's kernel is actually running under
(common on SLURM/JupyterHub setups with multiple conda envs/kernels
registered). A `!pip install torch` that silently installs into the wrong
environment produces the exact same symptom this doc is about
(`torch.cuda.is_available() == False`) but for a completely different
reason - the notebook is importing a torch that was never touched by the
install at all.

Always install using the *kernel's own* interpreter explicitly:

```python
import sys
!{sys.executable} -m pip install torch --index-url https://download.pytorch.org/whl/cu126
!{sys.executable} -m pip install -e ".[sbert_experiment]"
```

`{sys.executable}` is guaranteed to be the interpreter the notebook is
actually running - `!pip`/`!python` are not.

## Diagnosing the CUDA mismatch

```bash
python scripts/check_gpu.py
```

Prints the Python executable, torch version + CUDA build
(`torch.version.cuda`), `torch.cuda.is_available()`, GPU count/names,
`CUDA_VISIBLE_DEVICES`, and `nvidia-smi`'s own output if it's on `PATH`.
When CUDA is unavailable, it parses the driver's supported CUDA version
out of `nvidia-smi` and, if torch's build is newer, prints e.g.:

```text
Driver supports CUDA 12.8, but PyTorch was compiled for CUDA 13.0 - this
torch build is too new for this driver, so torch.cuda.is_available() is
silently False (no exception). Install a torch build matching the
driver's CUDA version instead - see docs/slurm_gpu_setup.md.
```

which is exactly the "install the wrong wheel first" install-order fix
above.

## The `pypi.nvidia.com` SSL issue

If installing an `nvidia-*` package (a transitive dependency of some torch
CUDA wheels) produces something like:

```text
CERTIFICATE_VERIFY_FAILED
self-signed certificate in certificate chain
```

it means this environment's outbound network sits behind a TLS-inspecting
proxy (common on managed institutional networks) that re-signs HTTPS
traffic with the institution's own certificate authority - one Python's
default trust store (`certifi`'s bundled CA list) doesn't know about,
even though the OS itself trusts it. This project already hits the same
class of issue for `huggingface_hub`/`sentence-transformers` downloads -
see `truststore` in `pyproject.toml`'s core `dependencies` and
`src/vaebm_benchmark/__init__.py`, which makes Python's `ssl` module trust
the OS certificate store instead of only `certifi`'s bundle. That fix
already covers this project's own code; `pip install` itself (before any
of this project's code has even run) is a separate process pip's own SSL
handling doesn't automatically get.

**Preferred fix**, in order:

1. **Use the institution's own CA certificate, if you can get it from
   your SLURM cluster's IT/security team.** Point pip (and any other
   Python tool) at it explicitly:
   ```bash
   pip install --cert /path/to/institution-ca.pem torch --index-url https://download.pytorch.org/whl/cu126
   # or, for every pip invocation in the session:
   export PIP_CERT=/path/to/institution-ca.pem
   ```
   This is the only option that actually verifies the proxy's certificate
   rather than skipping verification - safest by construction, use it
   whenever the CA bundle is obtainable.

2. **`--trusted-host` as a fallback only**, when you cannot obtain the
   institution's CA certificate (e.g. a managed environment where you have
   no path to it) and you have independently confirmed the network is a
   trusted, managed/institutional one (not an open or unknown network):
   ```bash
   pip install --trusted-host pypi.nvidia.com --trusted-host download.pytorch.org \
       torch --index-url https://download.pytorch.org/whl/cu126
   ```
   This disables certificate verification for exactly the named hosts -
   it is not a general "ignore SSL errors" flag, but it does mean a
   network-level attacker positioned as that host could serve a
   malicious package undetected. Do not reach for this on an untrusted or
   unknown network, and do not use a blanket
   `PIP_TRUSTED_HOST` env var covering hosts beyond the ones actually
   failing.

## Known remaining limitations

- This branch does not pin an exact known-good `torch==X.Y.Z+cu12W`
  version in any file, since the correct pin is a function of the
  cluster's specific driver version (find yours with `nvidia-smi`) and
  changes as PyTorch ships new releases - `scripts/check_gpu.py` is the
  mechanism for confirming whatever you installed actually works, not a
  substitute for checking PyTorch's own install matrix.
- `tensorflow>=2.15` (needed only for `vaebm`/`vaebm_*` models, via the
  `experiment` extra - not `sbert_experiment`) brings its own
  `nvidia-cudnn-cu12`/`nvidia-cublas-cu12`-style dependencies, which can
  still conflict with a hand-picked torch CUDA build if you install both
  extras into the same environment. If you need both VAE-BM and SBERT
  models in one run, install torch first (as above), then `.[experiment]`,
  then re-run `scripts/check_gpu.py` to confirm torch wasn't silently
  changed.
- This doc does not cover SLURM job-script/`sbatch` GPU allocation itself
  (`--gres=gpu:1`, etc.) - it assumes a job/session that already has a GPU
  allocated and visible to `nvidia-smi`.
