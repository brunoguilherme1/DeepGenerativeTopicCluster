"""Best-effort GPU/accelerator memory release between successive model
fits within the same process. `experiment/{runner,cluster_runner,
classification_runner}.py`'s `run_single()` all build-fit-evaluate-
discard a fresh model per (model, dataset, k[, seed]) combination -
PyTorch/TensorFlow do not release GPU memory back to the OS between
model loads in the same process by default, so a heavy model (e.g.
BGE-M3) can fail with CUDA OOM purely because an EARLIER model in the
same sweep left memory allocated/fragmented, not because the current
model+dataset combination itself doesn't fit on the GPU. Each runner
calls `release_accelerator_memory()` from a `finally` block, so it runs
after BOTH a successful fit and a caught exception - a failed/OOM'd
model is exactly the case that most needs its partially-allocated
tensors released before the NEXT combination tries to fit.

Best-effort and silent about what isn't installed: a project without
torch/tensorflow available (e.g. this project's own macOS dev
environment usually has neither at once) just runs `gc.collect()` and
returns - never raises for a missing framework.
"""

from __future__ import annotations

import gc


def release_accelerator_memory() -> None:
    """Call with no live reference to the just-discarded model (e.g.
    `del model` right before this, or simply let it fall out of scope
    first) - Python's own garbage collector needs the refcount to drop
    to zero before torch/tensorflow can actually reclaim the underlying
    GPU memory."""
    gc.collect()

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except ImportError:
        pass

    try:
        import tensorflow as tf

        tf.keras.backend.clear_session()
    except ImportError:
        pass
