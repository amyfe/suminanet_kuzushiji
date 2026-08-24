"""Shared utilities for all visualization modules."""

from pathlib import Path

import matplotlib.pyplot as plt

SAVEFIG_KWARGS = dict(dpi=150, bbox_inches="tight")

GRID_ALPHA   = 0.3
BAR_ALPHA    = 0.7
MARKER_SIZE  = 4

# Priority order for CJK tick-label fonts: HanaMin has the fullest coverage
# of rare/kuzushiji-specific kanji (including CJK Extension B, which Droid
# Sans Fallback lacks), then Noto CJK, with Droid as a last resort. Each
# font is checked both as a per-user install (~/.local/share/fonts — no
# root needed, e.g. via `apt download <pkg> && dpkg -x <pkg>.deb <dir>`)
# and as a system-wide install, since not every environment allows one.
_USER_FONT_DIR = Path.home() / ".local/share/fonts"
_CJK_FONT_CANDIDATES = [
    _USER_FONT_DIR / "HanaMinA.ttf",
    "/usr/share/fonts/truetype/hanazono/HanaMinA.ttf",
    _USER_FONT_DIR / "HanaMinB.ttf",
    "/usr/share/fonts/truetype/hanazono/HanaMinB.ttf",
    _USER_FONT_DIR / "NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    _USER_FONT_DIR / "NotoSerifCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
]


def cjk_font_prop(size: float = 10):
    """Return a FontProperties for the fullest-coverage installed CJK font,
    or None if none of the candidates are present."""
    import matplotlib.font_manager as fm

    for path in _CJK_FONT_CANDIDATES:
        if Path(path).exists():
            try:
                return fm.FontProperties(fname=path, size=size)
            except Exception:
                continue
    return None


def pastel_color(hex_color: str, factor: float = 0.55) -> tuple[float, float, float]:
    """Blend a hex color toward white by `factor` (0 = unchanged, 1 = white)."""
    import matplotlib.colors as mcolors

    r, g, b = mcolors.to_rgb(hex_color)
    return (r + (1 - r) * factor, g + (1 - g) * factor, b + (1 - b) * factor)


def savefig(fig: plt.Figure, path: "str | Path") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Explicit facecolor so figures with a non-default fig.patch color (e.g.
    # a black-background gallery) save correctly instead of matplotlib's
    # savefig defaulting to a white background — a no-op for every existing
    # white-background plot, since fig.get_facecolor() is already white then.
    fig.savefig(str(path), facecolor=fig.get_facecolor(), **SAVEFIG_KWARGS)
    plt.close(fig)
    return path
