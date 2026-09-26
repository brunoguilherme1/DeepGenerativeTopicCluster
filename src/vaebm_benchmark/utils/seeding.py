"""Single place every script sets every relevant random seed. Called once,
at the top of a run, before any data loading, model construction, or
training - never scattered ad hoc through the codebase."""

from __future__ import annotations

import os
import random
import sys


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

    # Only touch tensorflow if THIS process already imported it - never
    # import it here just to seed it. Importing tensorflow after torch has
    # touched CUDA at all (even just torch.cuda.is_available()/
    # manual_seed_all, no real model) can SIGSEGV natively on this system
    # (verified 2026-09-25 on labuai - see utils/gpu_memory.py's own
    # comment for the fuller writeup of this class of bug).
    if "tensorflow" in sys.modules:
        import tensorflow as tf

        tf.random.set_seed(seed)
