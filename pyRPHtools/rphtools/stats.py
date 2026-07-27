"""Histograms, Bayes classification, and Monte-Carlo simulation.

Ports of the following RPHtools MATLAB functions:

=================  ========================  =================================
MATLAB             Python                    Notes
=================  ========================  =================================
``hist2d.m``       `hist2d`                  Rebuilt on ``np.histogramdd``.
``hist3d.m``       `hist3d`                  Handles 1-3 columns.
``bayesclass.m``   `bayes_classify`          The ROOT version, not the
                                             different ``private/`` one.
``monte.m``        `monte_carlo_cdf`
``monteccdf.m``    `monte_carlo_ccdf`        Its three subfunctions become
                                             private helpers.
``pdfbayes.m``     `pdf_bayes`               RECONSTRUCTION, not a port:
                                             its engines are missing.
=================  ========================  =================================

All three histogram-based routines share one binning convention, factored
into `_bin_index` here. Given bin *centres* ``x``, the MATLAB built edges as
``[x0 - w0/2, x + w/2]`` with ``w = [diff(x), 0]`` — note the trailing zero,
which makes the final edge the last centre itself rather than
``centre + half-width``. Samples below the first edge or above the last are
clamped into the first and last bins. That is preserved exactly, quirk
included.

Behavior notes (deliberate changes from MATLAB, see PORTING_PLAN.md):

- ``hist3d.m``'s weighted path was broken: it called ``hist2d`` with four
  arguments where ``hist2d.m`` accepts three, and delegated 1-column input
  to a ``hist1d`` that does not exist in RPHtools. Rebuilding on
  ``np.histogramdd`` fixes both.
- `monte_carlo_cdf` and `monte_carlo_ccdf` take an optional
  ``numpy.random.Generator`` so runs are reproducible.
- Plotting side effects are removed; see `rphtools.plotting`.

References
----------
The Rock Physics Handbook, statistics chapter.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

__all__ = [
    "BayesClassification",
    "Histogram2D",
    "HistogramND",
    "bayes_classify",
    "hist2d",
    "hist3d",
    "monte_carlo_ccdf",
    "monte_carlo_cdf",
    "pdf_bayes",
    "PdfBayesResult",
]


def _edges_from_centres(centres):
    """MATLAB's centre-to-edge rule, trailing-zero quirk and all.

    ``binwidth = [diff(x) 0]``; ``edges = [x(1) - binwidth(1)/2, x + binwidth/2]``,
    so the last edge coincides with the last centre.
    """
    x = np.ravel(np.asarray(centres, float))
    width = np.append(np.diff(x), 0.0)
    return np.concatenate([[x[0] - width[0] / 2.0], x + width / 2.0])


def _bin_index(values, spec):
    """Bin index (0-based) of each value, plus the bin centres.

    `spec` is either a bin count (equal-width bins spanning the data) or a
    vector of bin centres. Out-of-range values are clamped into the end
    bins, as the MATLAB did.
    """
    v = np.ravel(np.asarray(values, float))
    spec_arr = np.atleast_1d(np.asarray(spec, float))

    if spec_arr.size == 1:
        nbin = int(spec_arr[0])
        lo, hi = float(np.min(v)), float(np.max(v))
        if lo == hi:  # degenerate: centre the bins on the single value
            lo = lo - np.floor(nbin / 2) - 0.5
            hi = hi + np.ceil(nbin / 2) - 0.5
        width = (hi - lo) / nbin
        centres = lo + width * (np.arange(nbin) + 0.5)
        idx = np.ceil((v - lo) / width).astype(int)
        idx = np.clip(idx, 1, nbin) - 1
    else:
        centres = spec_arr
        nbin = centres.size
        edges = _edges_from_centres(centres)
        idx = np.sum(v[:, None] >= edges[None, :], axis=1)
        idx = np.clip(idx, 1, nbin) - 1
    return idx, centres, nbin


class Histogram2D(NamedTuple):
    """Bivariate histogram."""

    counts: np.ndarray
    """Counts, shape ``(nbin1, nbin2)``."""
    centres1: np.ndarray
    """Bin centres along the first column."""
    centres2: np.ndarray
    """Bin centres along the second column."""


def hist2d(data, bins1=15, bins2=None, weights=None):
    """Bivariate histogram of two-column data.

    Parameters
    ----------
    data : array_like
        ``(n, 2)`` array; one row per sample.
    bins1 : int or array_like, optional
        Number of bins, or the bin centres, for the first column.
        Defaults to 15 equally spaced bins.
    bins2 : int or array_like, optional
        Same for the second column. Defaults to `bins1`.
    weights : array_like, optional
        Per-sample weights. ``None`` (default) counts each sample once.

    Returns
    -------
    Histogram2D
        Named tuple ``(counts, centres1, centres2)``.

    Notes
    -----
    Port of ``hist2d.m``, rebuilt on NumPy. Samples outside the bin range
    are clamped into the end bins, matching the original.
    """
    data = np.atleast_2d(np.asarray(data, float))
    if data.shape[1] != 2:
        raise ValueError("data must have exactly two columns")
    if bins2 is None:
        bins2 = bins1

    i1, c1, n1 = _bin_index(data[:, 0], bins1)
    i2, c2, n2 = _bin_index(data[:, 1], bins2)
    w = None if weights is None else np.ravel(np.asarray(weights, float))
    counts = np.zeros((n1, n2))
    np.add.at(counts, (i1, i2), 1.0 if w is None else w)
    return Histogram2D(counts=counts, centres1=c1, centres2=c2)


class HistogramND(NamedTuple):
    """Histogram of one, two, or three attributes."""

    counts: np.ndarray
    """Counts; one axis per input column."""
    centres: tuple
    """Bin centres, one array per input column."""


def hist3d(data, bins=15, weights=None):
    """Histogram of one-, two-, or three-column data.

    Parameters
    ----------
    data : array_like
        ``(n, k)`` array with ``k`` of 1, 2, or 3.
    bins : int, array_like, or sequence, optional
        A bin count applied to every column, a vector of bin centres
        applied to every column, or a sequence of one specification per
        column.
    weights : array_like, optional
        Per-sample weights.

    Returns
    -------
    HistogramND
        Named tuple ``(counts, centres)``.

    Notes
    -----
    Port of ``hist3d.m``. The original's weighted path was broken (it
    called ``hist2d`` with four arguments where that function takes three)
    and its 1-column path called a ``hist1d`` missing from RPHtools; the
    NumPy rebuild handles both.
    """
    data = np.atleast_2d(np.asarray(data, float))
    ncol = data.shape[1]
    if not 1 <= ncol <= 3:
        raise ValueError("data must have one, two, or three columns")

    # A list/tuple of the right length is one spec per column; anything
    # else (a scalar count, or one vector of centres) applies to them all.
    if isinstance(bins, (list, tuple)) and len(bins) == ncol:
        specs = list(bins)
    else:
        specs = [bins] * ncol

    idx, centres, shape = [], [], []
    for k in range(ncol):
        i, c, n = _bin_index(data[:, k], specs[k])
        idx.append(i)
        centres.append(c)
        shape.append(n)

    w = None if weights is None else np.ravel(np.asarray(weights, float))
    counts = np.zeros(shape)
    np.add.at(counts, tuple(idx), 1.0 if w is None else w)
    return HistogramND(counts=counts, centres=tuple(centres))


class BayesClassification(NamedTuple):
    """Facies assignment from a non-parametric PDF."""

    code: np.ndarray
    """Index of the most probable facies for each sample."""
    probs: np.ndarray
    """Conditional probability of each facies, ``(n, nfacies)``."""
    max_prob: np.ndarray
    """The winning probability for each sample."""


def bayes_classify(data, pdf, axes):
    """Bayes classification of samples against a non-parametric PDF.

    Each sample is binned onto the PDF's axes and assigned the facies with
    the highest conditional probability in that cell.

    Parameters
    ----------
    data : array_like
        ``(n, k)`` array of attributes, ``k`` of 1, 2, or 3.
    pdf : array_like
        Multivariate PDF with one axis per attribute plus a trailing
        facies axis, e.g. ``(nx, ny, nz, nfacies + 1)``. The **last**
        slice along the facies axis is excluded from the comparison — it
        holds the marginal, per the ``pdfbayes`` convention.
    axes : sequence of array_like
        Bin centres defining each attribute axis of `pdf`.

    Returns
    -------
    BayesClassification
        Named tuple ``(code, probs, max_prob)``. `code` is a 0-based
        facies index (the MATLAB's was 1-based).

    Notes
    -----
    Port of the root ``bayesclass.m``. RPHtools also ships a *different*
    implementation at ``private/bayesclass.m`` with cruder binning; this
    is not that one.
    """
    data = np.atleast_2d(np.asarray(data, float))
    pdf = np.asarray(pdf, float)
    ncol = data.shape[1]
    if ncol > len(axes):
        raise ValueError("need one axis per attribute column")

    idx = [_bin_index(data[:, k], axes[k])[0] for k in range(ncol)]
    # Attributes the PDF has but the data does not are singleton axes.
    while len(idx) < pdf.ndim - 1:
        idx.append(np.zeros(data.shape[0], int))

    probs = pdf[tuple(idx)][..., :-1]
    code = np.argmax(probs, axis=-1)
    return BayesClassification(code=code, probs=probs, max_prob=np.max(probs, axis=-1))


def _empirical_cdf(x, shrink=0.01):
    """Non-parametric CDF over the distinct values of `x`.

    Flat spots are dropped and a zero-probability point is prepended just
    below the smallest value so the CDF spans [0, 1].
    """
    x = np.ravel(np.asarray(x, float))
    x = x[np.isfinite(x)]
    values, counts = np.unique(x, return_counts=True)
    cdf = np.cumsum(counts) / counts.sum()

    keep = np.append(np.diff(cdf) != 0, True)
    cdf, values = cdf[keep], values[keep]

    if cdf[0] != 0:
        cdf = np.concatenate([[0.0], cdf])
        first = values[0]
        values = np.concatenate([[first - shrink * abs(first)], values])
    return cdf, values


def monte_carlo_cdf(params, n, rng=None):
    """Monte-Carlo draws from a non-parametric CDF, with linear correlation.

    The first column is drawn by inverting its empirical CDF. Every other
    column is regressed linearly on the first and simulated as the fit
    plus Gaussian noise with the residual standard deviation.

    Parameters
    ----------
    params : array_like
        ``(ndata, nvar)`` array of observations; the primary variable is
        the first column. A 1-D input is treated as a single column.
    n : int
        Number of draws.
    rng : numpy.random.Generator, optional
        Random source, for reproducibility.

    Returns
    -------
    ndarray
        ``(n, nvar)`` array of simulated values.

    See Also
    --------
    monte_carlo_ccdf : conditional-CDF version, which preserves
        non-linear and heteroscedastic relationships.

    Notes
    -----
    Port of ``monte.m``.
    """
    p = np.asarray(params, float)
    if p.ndim == 1:
        p = p[:, None]
    n = int(n)
    rng = np.random.default_rng() if rng is None else rng

    cdf, values = _empirical_cdf(p[:, 0], shrink=0.01)
    out = np.zeros((n, p.shape[1]))
    out[:, 0] = np.interp(rng.random(n), cdf, values)

    for k in range(1, p.shape[1]):
        slope, intercept = np.polyfit(p[:, 0], p[:, k], 1)
        residual = p[:, k] - (slope * p[:, 0] + intercept)
        out[:, k] = (
            slope * out[:, 0] + intercept + np.std(residual, ddof=1) * rng.standard_normal(n)
        )
    return out


def _conditional_cdfs(x, y):
    """Empirical CDFs of `y` within percentile bins of `x`."""
    edges = np.percentile(x, np.arange(5, 101, 5))
    cdfs = []
    masks = [x < edges[0]]
    for k in range(1, edges.size):
        masks.append((x < edges[k]) & (x >= edges[k - 1]))
    for mask in masks:
        subset = y[mask]
        cdfs.append(_empirical_cdf(subset) if subset.size else None)
    return cdfs, edges


def _draw_conditional(cdfs, edges, conditioning, rng):
    """Draw from the conditional CDF selected by each conditioning value."""
    conditioning = np.ravel(conditioning)
    out = np.zeros(conditioning.size)
    r = rng.random(conditioning.size)

    def fill(mask, entry):
        if entry is None or not np.any(mask):
            return
        cdf, values = entry
        out[mask] = np.interp(r[mask], cdf, values)

    fill(conditioning < edges[0], cdfs[0])
    for k in range(1, edges.size):
        fill((conditioning < edges[k]) & (conditioning >= edges[k - 1]), cdfs[k])
    fill(conditioning >= edges[-1], cdfs[-1])
    return out


def monte_carlo_ccdf(params, n, rng=None):
    """Monte-Carlo draws using conditional non-parametric CDFs.

    The first column is drawn by inverting its empirical CDF; every other
    column is drawn from the empirical CDF of that variable *conditioned*
    on which percentile bin of the primary variable the draw fell into.
    Unlike `monte_carlo_cdf` this needs no linearity or constant-variance
    assumption.

    Parameters
    ----------
    params : array_like
        ``(ndata, nvar)`` array of observations; primary variable first.
    n : int
        Number of draws.
    rng : numpy.random.Generator, optional
        Random source, for reproducibility.

    Returns
    -------
    ndarray
        ``(n, nvar)`` array of simulated values.

    Notes
    -----
    Port of ``monteccdf.m``, whose ``makecdf``/``makeccdf``/``drawccdf``
    subfunctions become private helpers here. Conditioning uses 20
    five-percentile bins of the primary variable, as in the original.
    """
    p = np.asarray(params, float)
    if p.ndim == 1:
        p = p[:, None]
    n = int(n)
    rng = np.random.default_rng() if rng is None else rng

    cdf, values = _empirical_cdf(p[:, 0])
    out = np.zeros((n, p.shape[1]))
    out[:, 0] = np.interp(rng.random(n), cdf, values)

    for k in range(1, p.shape[1]):
        cdfs, edges = _conditional_cdfs(p[:, 0], p[:, k])
        out[:, k] = _draw_conditional(cdfs, edges, out[:, 0], rng)
    return out


class PdfBayesResult(NamedTuple):
    """Non-parametric PDF estimate and its Bayes classification statistics."""

    pdf: np.ndarray
    """Smoothed class-conditional PDFs, shape ``(*nbins, nfacies + 1)``.
    The trailing slice is the prior-weighted marginal, matching the layout
    `bayes_classify` expects."""
    axes: tuple
    """Bin centres, one array per attribute."""
    bandwidth: np.ndarray
    """Standard deviation of the smoothing kernel per attribute, in data
    units — interpretable as the measurement error of each attribute."""
    joint: np.ndarray
    """Joint probability ``p(predicted=i, true=j)``, shape
    ``(nfacies, nfacies)``."""
    conditional: np.ndarray
    """``p(true=j | predicted=i)``, the rows of `joint` normalized."""
    success_rate: float
    """Probability of correct classification (the trace of `joint`)."""
    error: float
    """Total Bayes classification error, ``1 - success_rate``."""
    entropy: float
    """``H(facies)``, the unconditional entropy of the facies, in bits."""
    cond_entropy: float
    """``H(facies | attributes)``, in bits."""
    cond_info: float
    """``I(facies; attributes)``, the mutual information, in bits."""
    cond_entropy_norm: float
    """``H(facies | attributes) / H(facies)``; 0 means the attributes
    determine the facies completely, 1 means they say nothing. Being a
    ratio, it does not depend on the logarithm base."""


def _entropy(p):
    """Shannon entropy in bits of a probability array (zeros ignored)."""
    p = np.asarray(p, float)
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    p = p / p.sum()
    return float(-np.sum(p * np.log2(p)))


def pdf_bayes(data, code, weights=None, bins=10, smoothing=0.1, priors=None):
    """Non-parametric PDF estimation with Bayes error and information.

    Builds a smoothed class-conditional probability density for each
    facies over 1-3 attributes, then reports how well those attributes
    separate the facies: the Bayes classification error, the joint and
    conditional probability tables, and the entropy and mutual
    information of the facies given the attributes.

    Parameters
    ----------
    data : array_like
        ``(n, k)`` array of attributes, ``k`` of 1, 2, or 3.
    code : array_like
        Facies code for each sample. Any hashable labels; they are sorted
        and mapped to the rows/columns of `joint` in that order.
    weights : array_like, optional
        Per-sample weight (the MATLAB's optional second column of
        ``CODE``). Defaults to 1 for every sample.
    bins : int or sequence, optional
        Number of bins per attribute, or explicit bin centres per
        attribute. Defaults to 10.
    smoothing : float or sequence, optional
        Kernel width as a fraction of each attribute's standard
        deviation, the MATLAB's ``OBS``. Defaults to 0.1.
    priors : array_like, optional
        Prior probability of each facies. Defaults to the observed
        (weighted) frequencies.

    Returns
    -------
    PdfBayesResult

    See Also
    --------
    bayes_classify : apply the resulting `pdf` to new samples.

    Notes
    -----
    **This is a reconstruction, not a port.** ``pdfbayes.m``'s two
    computational engines (``pdfgendraw`` and ``pdfstat``) and the
    primitives its ``private/`` helpers call (``histnd``, ``entropdf``)
    are all missing from RPHtools, so the function cannot be translated.
    The pipeline here follows what the surviving pieces specify:
    ``private/cpdf.m`` for weighted per-facies histograms normalized to
    unit sum with a prior-weighted marginal appended, ``pdfbayes.m``'s own
    documentation for the Gaussian kernel width
    (``obs * std(attribute)``), ``private/bayes.m`` for the joint
    probability table, and ``private/centropy.m`` for the entropy
    combinations.

    Two details could not be recovered and are chosen here: smoothing uses
    ``scipy.ndimage.gaussian_filter`` (the MATLAB smoothed the histogram
    with a Gaussian kernel of that width, so this matches the described
    algorithm), and entropies are in **bits** — ``entropdf`` is missing, so
    its logarithm base is unknown. `cond_entropy_norm` is a ratio and is
    unaffected by that choice.

    Unlike ``private/bayes.m``, which broke ties between equally probable
    facies by adding random noise, ties here go to the lowest facies
    index, so results are reproducible.
    """
    from scipy.ndimage import gaussian_filter

    data = np.atleast_2d(np.asarray(data, float))
    ncol = data.shape[1]
    if not 1 <= ncol <= 3:
        raise ValueError("data must have one, two, or three attribute columns")
    code = np.ravel(np.asarray(code))
    if code.size != data.shape[0]:
        raise ValueError("code must have one entry per sample")
    w = np.ones(code.size) if weights is None else np.ravel(np.asarray(weights, float))

    facies = np.unique(code)
    nfac = facies.size

    specs = list(bins) if isinstance(bins, (list, tuple)) else [bins] * ncol
    idx, axes, shape = [], [], []
    for k in range(ncol):
        i, c, n = _bin_index(data[:, k], specs[k])
        idx.append(i)
        axes.append(c)
        shape.append(n)

    scale = np.atleast_1d(np.asarray(smoothing, float))
    if scale.size == 1:
        scale = np.repeat(scale, ncol)
    bandwidth = scale * np.std(data, axis=0)
    # Kernel width in bins: the physical width over the bin spacing.
    sigma = [
        bandwidth[k] / (axes[k][1] - axes[k][0]) if len(axes[k]) > 1 else 0.0 for k in range(ncol)
    ]

    pdf = np.zeros(tuple(shape) + (nfac + 1,))
    observed = np.zeros(nfac)
    for f, label in enumerate(facies):
        mask = code == label
        hist = np.zeros(shape)
        np.add.at(hist, tuple(i[mask] for i in idx), w[mask])
        observed[f] = w[mask].sum()
        hist = gaussian_filter(hist, sigma=sigma, mode="nearest")
        total = hist.sum()
        pdf[..., f] = hist / total if total else hist

    prior = observed / observed.sum() if priors is None else np.asarray(priors, float)
    prior = prior / prior.sum()
    pdf[..., nfac] = np.sum(pdf[..., :nfac] * prior, axis=-1)

    # --- Bayes statistics on the prior-weighted PDFs ---------------------
    weighted = pdf[..., :nfac] * prior
    predicted = np.argmax(weighted, axis=-1)
    joint = np.zeros((nfac, nfac))
    for i in range(nfac):
        cell = predicted == i
        for j in range(nfac):
            joint[i, j] = weighted[..., j][cell].sum()

    row_sums = joint.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        conditional = np.where(row_sums > 0, joint / row_sums, 0.0)
    success = float(np.trace(joint))

    # --- entropies (private/centropy.m) ----------------------------------
    h_given_facies = float(np.sum([prior[f] * _entropy(pdf[..., f]) for f in range(nfac)]))
    h_attributes = _entropy(pdf[..., nfac])
    info = h_attributes - h_given_facies
    h_facies = _entropy(prior)
    h_facies_given = h_facies - info

    return PdfBayesResult(
        pdf=pdf,
        axes=tuple(axes),
        bandwidth=bandwidth,
        joint=joint,
        conditional=conditional,
        success_rate=success,
        error=1.0 - success,
        entropy=h_facies,
        cond_entropy=h_facies_given,
        cond_info=info,
        cond_entropy_norm=h_facies_given / h_facies if h_facies else 0.0,
    )
