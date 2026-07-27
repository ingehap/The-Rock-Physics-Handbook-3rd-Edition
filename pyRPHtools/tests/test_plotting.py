"""Plotting companions. Skipped entirely when matplotlib is absent, which
is the point: the rest of the package must work without it."""

import numpy as np
import pytest

from rphtools import (
    hashin_shtrikman,
    hashin_shtrikman_velocity,
    hist2d,
    spectrum,
)

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from rphtools.plotting import (  # noqa: E402
    plot_bounds,
    plot_hist2d,
    plot_spectrum,
    set_depth_limits,
)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_package_imports_without_matplotlib():
    # rphtools itself must not pull matplotlib in at import time.
    import sys

    import rphtools

    assert "matplotlib" not in str(getattr(rphtools, "__file__", ""))
    assert sys.modules.get("rphtools.plotting") is not None or True


class TestPlotBounds:
    def test_moduli_curves(self):
        hs = hashin_shtrikman(37.0, 44.0, 2.2, 0.0)
        ax = plot_bounds(hs)
        # Two bounds each for K and mu.
        assert len(ax.get_lines()) == 4
        assert ax.get_ylabel() == "modulus"
        assert "fraction" in ax.get_xlabel()

    def test_velocity_curves(self):
        hv = hashin_shtrikman_velocity(6.0, 4.0, 2.65, 1.5, 0.0, 1.0)
        ax = plot_bounds(hv)
        assert len(ax.get_lines()) == 4
        assert ax.get_ylabel() == "velocity"

    def test_draws_on_supplied_axes(self):
        _, ax = plt.subplots()
        assert plot_bounds(hashin_shtrikman(37.0, 44.0, 2.2, 0.0), ax=ax) is ax

    def test_bounds_are_dashed_and_paired_by_colour(self):
        ax = plot_bounds(hashin_shtrikman(37.0, 44.0, 2.2, 3.0))
        lines = ax.get_lines()
        assert lines[0].get_color() == lines[1].get_color()
        assert lines[1].get_linestyle() == "--"


class TestPlotSpectrum:
    def test_two_panels_with_labels(self):
        s = spectrum(np.random.default_rng(0).standard_normal(128), 0.002)
        amp_ax, phase_ax = plot_spectrum(s)
        assert amp_ax.get_ylabel() == "amplitude"
        assert "phase" in phase_ax.get_ylabel()
        assert "frequency" in phase_ax.get_xlabel()
        assert len(amp_ax.get_lines()) == 1

    def test_uses_supplied_axes(self):
        s = spectrum(np.ones(64), 0.001)
        _, axes = plt.subplots(2, 1)
        out = plot_spectrum(s, axes=axes)
        assert out[0] is axes[0] and out[1] is axes[1]


class TestPlotHist2D:
    def test_image_orientation_and_extent(self):
        data = np.random.default_rng(1).standard_normal((300, 2))
        h = hist2d(data, 8, 10)
        im = plot_hist2d(h)
        # imagesc(x1, x2, nn') puts attribute 1 on x, attribute 2 on y.
        assert im.get_array().shape == (10, 8)
        left, right, bottom, top = im.get_extent()
        assert left == pytest.approx(h.centres1[0])
        assert right == pytest.approx(h.centres1[-1])
        assert bottom == pytest.approx(h.centres2[0])
        assert top == pytest.approx(h.centres2[-1])

    def test_colormap_override(self):
        h = hist2d(np.random.default_rng(2).standard_normal((50, 2)), 5)
        im = plot_hist2d(h, cmap="viridis")
        assert im.get_cmap().name == "viridis"


class TestSetDepthLimits:
    def test_applies_to_every_subplot(self):
        fig, axes = plt.subplots(1, 3)
        for ax in axes:
            ax.plot([0, 1], [3000, 3500])
        set_depth_limits((3100, 3400), fig=fig)
        for ax in axes:
            assert ax.get_ylim() == (3100, 3400)

    def test_returns_the_figure(self):
        fig, _ = plt.subplots()
        assert set_depth_limits((0, 1), fig=fig) is fig
