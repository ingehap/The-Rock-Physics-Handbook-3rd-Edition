"""Spectra, instantaneous attributes, and log/trace utilities.

Ports of the following RPHtools MATLAB functions:

===============  ===========================  ===============================
MATLAB           Python                       Notes
===============  ===========================  ===============================
``fftplot.m``    `spectrum`                   Compute only; no plotting.
``iatrib.m``     `instantaneous_attributes`
``blockav.m``    `block_average`
``ft1axis.m``    `fft_axis`                   ``axis=0``.
``ft2axis.m``    `fft_axis`                   ``axis=1``.
===============  ===========================  ===============================

Behavior notes (deliberate changes from MATLAB, see PORTING_PLAN.md):

- ``ft1axis.m`` and ``ft2axis.m`` were the same routine along two different
  axes; they are one function with an ``axis`` argument.
- `spectrum` returns the frequency axis, amplitude, and phase instead of
  drawing a two-panel figure.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy.signal import hilbert

__all__ = [
    "InstantaneousAttributes",
    "Spectrum",
    "block_average",
    "fft_axis",
    "instantaneous_attributes",
    "spectrum",
]


class Spectrum(NamedTuple):
    """One-sided amplitude and phase spectrum."""

    freq: np.ndarray
    """Frequencies from 0 to Nyquist (Hz)."""
    amplitude: np.ndarray
    """Magnitude of the complex spectrum."""
    phase: np.ndarray
    """Phase angle in radians (principal value)."""


def spectrum(data, dt):
    """One-sided amplitude and phase spectrum of a real time series.

    Parameters
    ----------
    data : array_like
        Real-valued time series.
    dt : float
        Sampling interval in seconds.

    Returns
    -------
    Spectrum
        Named tuple ``(freq, amplitude, phase)`` covering 0 to Nyquist.
        The spectra of a real series are symmetric, so only the positive
        side is returned.

    Notes
    -----
    Port of ``fftplot.m`` without its plotting. The frequency axis is
    ``0`` to ``0.5/dt`` in steps of ``1/(n*dt)``, matching the original.
    """
    data = np.ravel(np.asarray(data, float))
    n = data.size
    nyquist = 0.5 / dt
    step = 2.0 * nyquist / n
    freq = np.arange(0.0, nyquist + step / 2.0, step)
    spec = np.fft.fft(data)[: freq.size]
    return Spectrum(freq=freq, amplitude=np.abs(spec), phase=np.angle(spec))


class InstantaneousAttributes(NamedTuple):
    """Complex-trace (Hilbert) attributes of a trace or section."""

    amplitude: np.ndarray
    """Instantaneous amplitude (envelope)."""
    phase: np.ndarray
    """Instantaneous phase in radians (wrapped, as in the original)."""
    frequency: np.ndarray
    """Instantaneous frequency: the sample-to-sample phase difference, so
    one sample shorter than the input along `axis`."""


def instantaneous_attributes(x, axis=0):
    """Instantaneous amplitude, phase, and frequency of a trace or section.

    Parameters
    ----------
    x : array_like
        Seismic trace, or a section with time along `axis`.
    axis : int, optional
        Time axis. Defaults to 0, matching MATLAB's column-wise
        ``hilbert`` on an ``[nt, nx]`` section.

    Returns
    -------
    InstantaneousAttributes
        Named tuple ``(amplitude, phase, frequency)``.

    Notes
    -----
    Port of ``iatrib.m``. As in the original, the phase is wrapped to
    ``(-pi, pi]`` and the "frequency" is its raw first difference, so it
    shows 2*pi jumps at wrap points — unwrap the phase first
    (``np.unwrap``) if you need a continuous frequency estimate. The
    result is also a difference per *sample*, not per second: divide by
    ``2*pi*dt`` for Hz.
    """
    x = np.asarray(x, float)
    analytic = hilbert(x, axis=axis)
    phase = np.angle(analytic)
    return InstantaneousAttributes(
        amplitude=np.abs(analytic),
        phase=phase,
        frequency=np.diff(phase, axis=axis),
    )


def block_average(log, nb):
    """Block average of a log or signal over `nb` samples.

    Each output sample is the mean of the block it belongs to, so the
    output has the same length as the input (a blocked, staircase version
    of the log rather than a decimated one).

    Parameters
    ----------
    log : array_like
        Log as a 1-D vector or a 2-D array of columns. NaN marks missing
        values and is excluded from each block's mean.
    nb : int
        Block length in samples. The log is padded with copies of its last
        row so its length is a multiple of `nb`; the padding is discarded
        from the output.

    Returns
    -------
    ndarray
        Blocked log, same shape as the input.

    Notes
    -----
    Port of ``blockav.m``. A block that is entirely NaN yields NaN and
    raises no error (NumPy warns; the original's ``nanmean`` was silent).
    """
    arr = np.asarray(log, float)
    one_d = arr.ndim == 1
    if one_d:
        arr = arr[:, None]
    nr, nc = arr.shape
    nb = int(nb)
    if nb < 1:
        raise ValueError("nb must be a positive integer")

    npad = nb - (nr % nb)
    padded = np.vstack([arr, np.repeat(arr[-1:], npad, axis=0)])
    blocks = padded.reshape(-1, nb, nc)
    with np.errstate(invalid="ignore"):
        means = np.nanmean(blocks, axis=1)
    out = np.repeat(means, nb, axis=0)[:nr]
    return out[:, 0] if one_d else out


def fft_axis(cx, adjoint=False, sign=1, axis=0):
    """Centred Fourier transform along one axis (Claerbout's convention).

    Alternate samples are negated before or after the transform so that
    zero frequency lands in the middle of the output rather than at the
    first sample.

    Parameters
    ----------
    cx : array_like
        Input array.
    adjoint : bool, optional
        ``False`` (default) transforms forward — negate, then transform;
        ``True`` applies the adjoint — transform, then negate. (The
        MATLAB's ``adj`` flag, where 0 is the forward direction.)
    sign : {1, -1}, optional
        Sign convention of the exponent. With ``sign=1`` the forward
        direction uses an inverse FFT and the adjoint a forward FFT, as in
        the original.
    axis : int, optional
        Axis to transform. 0 reproduces ``ft1axis.m``, 1 ``ft2axis.m``.

    Returns
    -------
    ndarray
        Transformed array, same shape as `cx`.

    Notes
    -----
    Merges ``ft1axis.m`` and ``ft2axis.m``, which differed only in the
    axis they acted on.

    References
    ----------
    Claerbout, J. F., Basic Earth Imaging.
    """
    if sign not in (1, -1):
        raise ValueError("sign must be 1 or -1")
    cx = np.array(cx, dtype=complex, copy=True)

    flip = [slice(None)] * cx.ndim
    flip[axis] = slice(1, None, 2)
    flip = tuple(flip)

    if not adjoint:
        cx[flip] = -cx[flip]
        cx = np.fft.ifft(cx, axis=axis) if sign == 1 else np.fft.fft(cx, axis=axis)
    else:
        cx = np.fft.fft(cx, axis=axis) if sign == 1 else np.fft.ifft(cx, axis=axis)
        cx[flip] = -cx[flip]
    return cx
