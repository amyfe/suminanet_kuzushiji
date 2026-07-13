"""Shared GPU cleanup helper for Optuna objective() functions."""
from __future__ import annotations

import gc
import torch


def release_cuda_memory() -> None:
    """Reclaim CUDA memory after an Optuna trial ends.

    Call from a `finally` block, after `del`-ing the trial's own local
    model/optimizer/scheduler/scaler/dataloader references in the caller's
    frame — this function can't do that `del` on the caller's behalf since
    passing objects as arguments only adds a reference in this frame.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
