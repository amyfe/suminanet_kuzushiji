"""Shared utilities for all visualization modules."""

from pathlib import Path

import matplotlib.pyplot as plt

SAVEFIG_KWARGS = dict(dpi=150, bbox_inches="tight")

GRID_ALPHA   = 0.3
BAR_ALPHA    = 0.7
MARKER_SIZE  = 4


def savefig(fig: plt.Figure, path: "str | Path") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), **SAVEFIG_KWARGS)
    plt.close(fig)
    return path
