"""Figure export convenience wrapper.

Ported from QuantToolbox/tools/save_graphic.m (+ save_graphic2.m).
QuantToolbox/export/*.m, the third-party MATLAB ``export_fig`` package,
is NOT ported -- matplotlib's native ``savefig`` already handles
resolution/format/tight-bbox export directly, with no equivalent gap to
fill.

Translation notes:

- The original saves up to 7 resolution/format variants (pdf at 3
  resolutions, png at 3 resolutions, plus one via the third-party
  export_fig) into a fixed ``Figures/<variant>/`` directory tree. This is
  simplified to a single function that saves whichever
  formats/resolutions are requested into one output directory --
  matplotlib's ``savefig(dpi=...)`` already parameterizes resolution
  directly, so there's no need for MATLAB's separate low/med/high-res
  code paths.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.figure


def save_graphic(
    fig: matplotlib.figure.Figure,
    name: str,
    output_dir: str | Path = "Figures",
    formats: tuple[str, ...] = ("pdf", "png"),
    resolutions: tuple[int, ...] = (600, 300, 150),
) -> list[Path]:
    """Save a matplotlib figure in the given formats, each at the given
    DPI resolutions, into output_dir/<format>-<resolution>dpi/<name>.<format>.

    Original: tools/save_graphic.m (+ save_graphic2.m)
    """
    output_dir = Path(output_dir)
    saved: list[Path] = []

    for fmt in formats:
        for dpi in resolutions:
            subdir = output_dir / f"{fmt}-{dpi}dpi"
            subdir.mkdir(parents=True, exist_ok=True)
            path = subdir / f"{name}.{fmt}"
            fig.savefig(path, dpi=dpi, bbox_inches="tight", format=fmt)
            saved.append(path)

    return saved
