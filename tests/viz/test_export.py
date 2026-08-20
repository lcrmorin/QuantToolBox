"""Tests for quanttoolbox.viz.export."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from quanttoolbox.viz.export import save_graphic


def test_save_graphic_creates_expected_files(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 4, 9])

    saved = save_graphic(
        fig, "test_plot", output_dir=tmp_path, formats=("png",), resolutions=(100,)
    )
    plt.close(fig)

    assert len(saved) == 1
    assert saved[0].exists()
    assert saved[0].name == "test_plot.png"


def test_save_graphic_multiple_formats_and_resolutions(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 4, 9])

    saved = save_graphic(
        fig, "multi", output_dir=tmp_path, formats=("png", "pdf"), resolutions=(150, 300)
    )
    plt.close(fig)

    assert len(saved) == 4  # 2 formats x 2 resolutions
    assert all(p.exists() for p in saved)


def test_save_graphic_creates_output_directory(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([1, 2], [1, 2])

    nested_dir = tmp_path / "nested" / "output"
    saved = save_graphic(fig, "plot", output_dir=nested_dir, formats=("png",), resolutions=(100,))
    plt.close(fig)

    assert saved[0].exists()
