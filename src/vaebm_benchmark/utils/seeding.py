"""Single place every script sets every relevant random seed. Called once,
at the top of a run, before any data loading, model construction, or
training - never scattered ad hoc through the codebase."""

from __future__ import annotations

import os
import random


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

    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except Exception:
        pass
