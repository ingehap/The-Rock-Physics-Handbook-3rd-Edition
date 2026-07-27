"""Optional matplotlib companions to the computational functions.

The rest of `rphtools` never plots: functions return arrays and you draw
them however you like. This module collects the plots the MATLAB drew as a
side effect, so nothing is lost — but it is the only module that touches
matplotlib, and it imports it lazily, so the package works without it.

============================  ====================  =======================
MATLAB                        Python                Notes
============================  ====================  =======================
``logax.m``                   `set_depth_limits`
``hash.m`` / ``hashv.m``      `plot_bounds`         The bound curves those
                                                    drew with no outputs.
``fftplot.m``                 `plot_spectrum`       Its two-panel figure.
``hist2d.m``                  `plot_hist2d`         Its ``imagesc`` view.
============================  ====================  =======================

Install with ``pip install "./pyRPHtools[plot]"`` to get matplotlib.

Every function takes an optional ``ax`` (or ``axes``) and returns it, so
plots compose into larger figures.
"""

from __future__ import annotations

__all__ = [
    "plot_bounds",
    "plot_hist2d",
    "plot_spectrum",
    "set_depth_limits",
]


def _pyplot():
    """Import matplotlib.pyplot on demand, with a helpful error."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            'plotting requires matplotlib; install it with pip install "./pyRPHtools[plot]"'
        ) from exc
    return plt


def set_depth_limits(limits, fig=None):
    """Set the depth (y) axis limits on every subplot of a figure.

    Parameters
    ----------
    limits : sequence of float
        ``(ymin, ymax)``.
    fig : matplotlib.figure.Figure, optional
        Figure to adjust. Defaults to the current figure.

    Returns
    -------
    matplotlib.figure.Figure
        The figure that was adjusted.

    Notes
    -----
    Port of ``logax.m``, which did this for log-track subplots.
    """
    plt = _pyplot()
    fig = plt.gcf() if fig is None else fig
    for ax in fig.get_axes():
        ax.set_ylim(limits)
    return fig


def plot_bounds(curves, ax=None, **kwargs):
    """Plot upper and lower bound curves against composition.

    Parameters
    ----------
    curves : HSBoundCurves or HSVelocityCurves
        Result of `rphtools.hashin_shtrikman` or
        `rphtools.hashin_shtrikman_velocity`.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. Defaults to the current axes.
    **kwargs
        Passed to each ``plot`` call.

    Returns
    -------
    matplotlib.axes.Axes

    Notes
    -----
    Reproduces the figure ``hash.m`` and ``hashv.m`` drew when called with
    no output arguments.
    """
    plt = _pyplot()
    ax = plt.gca() if ax is None else ax

    fields = curves._fields
    if "k_upper" in fields:
        pairs = [("k_upper", "k_lower", "bulk modulus"), ("mu_upper", "mu_lower", "shear modulus")]
        ax.set_ylabel("modulus")
    else:
        pairs = [("vp_upper", "vp_lower", "Vp"), ("vs_upper", "vs_lower", "Vs")]
        ax.set_ylabel("velocity")

    for upper, lower, label in pairs:
        line = ax.plot(curves.f2, getattr(curves, upper), label=f"{label} upper", **kwargs)[0]
        ax.plot(
            curves.f2,
            getattr(curves, lower),
            linestyle="--",
            color=line.get_color(),
            label=f"{label} lower",
            **kwargs,
        )
    ax.set_xlabel("fraction of material 2")
    ax.legend()
    return ax


def plot_spectrum(spec, axes=None, **kwargs):
    """Plot amplitude and phase spectra in two stacked panels.

    Parameters
    ----------
    spec : Spectrum
        Result of `rphtools.spectrum`.
    axes : sequence of matplotlib.axes.Axes, optional
        Two axes to draw on. A new figure is created if omitted.
    **kwargs
        Passed to each ``plot`` call.

    Returns
    -------
    tuple of matplotlib.axes.Axes
        The amplitude and phase axes.

    Notes
    -----
    Reproduces ``fftplot.m``'s two-panel figure.
    """
    plt = _pyplot()
    if axes is None:
        _, axes = plt.subplots(2, 1, sharex=True)
    amp_ax, phase_ax = axes

    amp_ax.plot(spec.freq, spec.amplitude, **kwargs)
    amp_ax.set_ylabel("amplitude")
    phase_ax.plot(spec.freq, spec.phase, **kwargs)
    phase_ax.set_ylabel("phase (radians)")
    phase_ax.set_xlabel("frequency (Hz)")
    return amp_ax, phase_ax


def plot_hist2d(hist, ax=None, **kwargs):
    """Plot a bivariate histogram as an image.

    Parameters
    ----------
    hist : Histogram2D
        Result of `rphtools.hist2d`.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. Defaults to the current axes.
    **kwargs
        Passed to ``imshow`` (e.g. ``cmap``).

    Returns
    -------
    matplotlib.image.AxesImage

    Notes
    -----
    Reproduces ``hist2d.m``'s ``imagesc(x1, x2, nn')`` view: the first
    attribute runs along x, the second along y, origin at the lower left.
    """
    plt = _pyplot()
    ax = plt.gca() if ax is None else ax
    kwargs.setdefault("cmap", "gray_r")
    kwargs.setdefault("aspect", "auto")
    extent = (
        hist.centres1[0],
        hist.centres1[-1],
        hist.centres2[0],
        hist.centres2[-1],
    )
    return ax.imshow(hist.counts.T, origin="lower", extent=extent, **kwargs)
