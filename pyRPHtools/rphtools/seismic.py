"""Normal-incidence wave propagation and synthetic seismograms in 1-D media.

Ports of the following RPHtools MATLAB functions:

==================  ===============================  =======================
MATLAB              Python                           Notes
==================  ===============================  =======================
``kennet.m``        `kennett`                        Invariant imbedding.
``kennett_aux.m``   (merged into `kennett`)          Its taper fix is
                                                     adopted; its
                                                     ``save omega1D.mat``
                                                     side effect is not.
``pgator.m``        `propagator_seis`                Propagator matrix.
``kenfdisp.m``      `kennett_frazer_dispersion`
``kenfrtt.m``       `kennett_frazer_traveltimes`
``ezseis.m``        `quick_seismic_section`          Dialog/plot stripped.
(missing)           `ricker`                         Stands in for the
                                                     absent ``sourcewvlt``.
==================  ===============================  =======================

Layer stacks are ``(n, 3)`` arrays of ``[velocity, density, thickness]``,
one row per layer, exactly as in the MATLAB.

Behavior notes (deliberate changes from MATLAB, see PORTING_PLAN.md):

- ``sourcewvlt`` — the default wavelet of ``kennet.m`` and ``pgator.m`` — is
  missing from RPHtools, so `wavelet=None` uses `ricker` instead. The
  original wavelet is unknowable; pass one explicitly for reproducibility.
- The half-band taper uses ``np.hanning(m + 2)[1:-1]``, which is MATLAB's
  ``hanning(m)`` (MATLAB's window excludes the zero endpoints; NumPy's
  includes them). Getting this wrong shifts the taper by one sample and
  silently changes every seismogram.
- `kennett` adopts the rounded, length-clipped taper of ``kennett_aux.m``,
  which is what makes odd-length wavelets work.
- `quick_seismic_section` requires its parameters as keyword arguments
  instead of ``inputdlg`` prompts, and fixes a latent crash: ``ezseis.m``
  uses the filter cutoff ``fc`` without defining it whenever no decimation
  is needed (``R <= 1``).

References
----------
Kennett, B. L. N., 1974, 1983. Frazer, L. N., 1994.
The Rock Physics Handbook, wave-propagation chapter.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy.signal import decimate, filtfilt, firwin

__all__ = [
    "KennettFrazerTraveltimes",
    "KennettResult",
    "PropagatorResult",
    "SeismicSection",
    "kennett",
    "kennett_frazer_dispersion",
    "kennett_frazer_traveltimes",
    "propagator_seis",
    "quick_seismic_section",
    "ricker",
]


def _matlab_hanning(m):
    """MATLAB's ``hanning(m)``: the symmetric Hann window without its zero
    endpoints, which is ``np.hanning(m + 2)[1:-1]`` (NOT ``np.hanning(m)``)."""
    return np.hanning(m + 2)[1:-1]


def ricker(freq=25.0, dt=0.001, n=None):
    """Zero-phase Ricker wavelet.

    Parameters
    ----------
    freq : float, optional
        Peak (dominant) frequency in Hz.
    dt : float, optional
        Sampling interval in seconds.
    n : int, optional
        Number of samples. Defaults to an odd length covering +/- 2 / freq
        seconds, which contains the wavelet's significant energy.

    Returns
    -------
    w : ndarray
        Wavelet samples, peaking at the centre.
    t : ndarray
        Time axis, centred on zero.

    Notes
    -----
    Stands in for ``sourcewvlt``, the default wavelet of ``kennet.m`` and
    ``pgator.m``, which is absent from RPHtools. Since the original is
    unknowable, this is new behavior rather than a port; pass an explicit
    wavelet when reproducibility against MATLAB matters.
    """
    if n is None:
        n = int(2 * round(2.0 / (freq * dt))) + 1
    t = (np.arange(n) - (n - 1) / 2.0) * dt
    a = (np.pi * freq * t) ** 2
    return (1.0 - 2.0 * a) * np.exp(-a), t


def _band_taper(n_wavelet, n_half):
    """Flat-then-cosine taper applied to the transfer function.

    Unity over the lower half of the retained band, then the decaying half
    of a Hann window. Follows ``kennett_aux.m``'s rounded, length-clipped
    form, which is what lets odd-length wavelets through.
    """
    m = int(np.floor(n_wavelet / 2 + 0.5))  # MATLAB round()
    flat = int(np.floor(n_wavelet / 4 + 0.5))
    w = _matlab_hanning(m)
    taper = np.concatenate([np.ones(flat), w[flat:m]])
    return taper[:n_half]


def _hermitian_synthesis(spec, zero_dc):
    """Rebuild a real time series from a half-spectrum, as the MATLAB did.

    ``[x0, x1..x_{L-1}, 0, conj(x_{L-1}..x_1)]`` then a forward FFT
    (MATLAB's analysis/synthesis pair is swapped relative to the usual
    convention; NumPy's 1/n placement matches, so it ports directly).
    """
    head = np.array([0.0 + 0.0j]) if zero_dc else spec[:1]
    full = np.concatenate([head, spec[1:], [0.0], np.conj(spec[:0:-1])])
    return np.real(np.fft.fft(full))


def _frequency_axis(n, dt):
    """MATLAB's ``om = (2*pi/(n*dt)) * [0 : n/2 - 1]``."""
    k = np.arange(int(np.floor(n / 2 - 1)) + 1)
    return (2.0 * np.pi / (n * dt)) * k


def _convolve_same(row, kernel):
    """MATLAB's ``conv(row, kernel, 'same')``: always the length of `row`.

    ``np.convolve(..., mode='same')`` returns ``max(len(row), len(kernel))``
    instead, which silently widens the section when the Fresnel-zone box is
    wider than the number of traces.
    """
    full = np.convolve(row, kernel)
    start = (kernel.size - 1) // 2
    return full[start : start + row.size]


def _layer_columns(layers):
    lyr = np.atleast_2d(np.asarray(layers, float))
    if lyr.shape[1] != 3:
        raise ValueError("layers must be an (n, 3) array of [velocity, density, thickness]")
    return lyr[:, 0], lyr[:, 1], lyr[:, 2]


class KennettResult(NamedTuple):
    """Kennett synthetic seismograms and the medium's transfer function."""

    wz: np.ndarray
    """Seismogram at the top of the stack (reflected)."""
    pz: np.ndarray
    """Seismogram at the bottom of the stack (transmitted)."""
    freq: np.ndarray
    """Frequencies of `reflectivity` and `transmissivity` (Hz)."""
    reflectivity: np.ndarray
    """Complex reflection transfer function."""
    transmissivity: np.ndarray
    """Complex transmission transfer function."""


def kennett(layers, wavelet=None, dt=0.001, multiples="all", free_surface=False):
    """Synthetic seismograms by Kennett's invariant-imbedding method.

    Plane-wave, normal-incidence propagation through a 1-D layered medium,
    accumulated recursively from the bottom of the stack upward.

    Parameters
    ----------
    layers : array_like
        ``(n, 3)`` array of ``[velocity, density, thickness]`` per layer.
    wavelet : array_like, optional
        Source wavelet. Defaults to `ricker` at 25 Hz (see the module note:
        the MATLAB's own default, ``sourcewvlt``, is missing from RPHtools).
    dt : float, optional
        Time sampling interval of the wavelet, in seconds.
    multiples : {'all', 'primaries', 'first-order'}, optional
        Which reverberations to include. ``'primaries'`` is the MATLAB's
        ``mopt=0``, ``'first-order'`` its ``mopt=1``, and ``'all'``
        (default) anything else.
    free_surface : bool, optional
        Include free-surface multiples (the MATLAB's ``fs=1``).

    Returns
    -------
    KennettResult
        Named tuple ``(wz, pz, freq, reflectivity, transmissivity)``.

    See Also
    --------
    propagator_seis : the propagator-matrix equivalent.

    Notes
    -----
    Port of ``kennet.m``, adopting the taper fix from its ``kennett_aux.m``
    variant. The ``'first-order'`` option clamps the real and imaginary
    parts of the transfer functions at 1, as the MATLAB did, to keep the
    truncated series from diverging.
    """
    modes = {"primaries": 0, "first-order": 1, "all": 2}
    if multiples not in modes:
        raise ValueError(f"multiples must be one of {tuple(modes)}")
    mopt = modes[multiples]

    v, rho, d = _layer_columns(layers)
    nlr = v.size
    if wavelet is None:
        wavelet = ricker(dt=dt)[0]
    wvlt = np.ravel(np.asarray(wavelet, float))
    n = wvlt.size

    om = _frequency_axis(n, dt)
    freq = om / (2.0 * np.pi)
    p0 = np.fft.ifft(wvlt)

    imp = rho * v
    deno = imp[1:] + imp[:-1]
    rd = np.concatenate([[-1.0 if free_surface else 0.0], (imp[:-1] - imp[1:]) / deno])
    td = np.concatenate([[1.0], 2.0 * np.sqrt(imp[1:] * imp[:-1]) / deno])
    ru, tu = -rd, td

    rdhat = np.zeros(om.shape, complex)
    tdhat = np.ones(om.shape, complex)
    for j in range(nlr - 1, -1, -1):
        ed = np.exp(1j * (d[j] / v[j]) * om)
        if mopt == 0:
            reverb = np.ones(om.shape, complex)
        elif mopt == 1:
            reverb = 1.0 + ru[j] * ed * rdhat * ed
        else:
            reverb = 1.0 / (1.0 - ru[j] * ed * rdhat * ed)

        rdhat = rd[j] + tu[j] * ed * rdhat * ed * reverb * td[j]
        tdhat = tdhat * ed * reverb * td[j]

        if mopt == 1:
            rdhat = np.clip(rdhat.real, None, 1.0) + 1j * np.clip(rdhat.imag, None, 1.0)
            tdhat = np.clip(tdhat.real, None, 1.0) + 1j * np.clip(tdhat.imag, None, 1.0)

    taper = _band_taper(n, om.size)
    pz_spec = tdhat * p0[: om.size] * taper
    wz_spec = rdhat * p0[: om.size] * taper

    return KennettResult(
        wz=_hermitian_synthesis(wz_spec, zero_dc=True),
        pz=_hermitian_synthesis(pz_spec, zero_dc=False),
        freq=freq,
        reflectivity=rdhat,
        transmissivity=tdhat,
    )


class PropagatorResult(NamedTuple):
    """Propagator-matrix synthetic seismograms."""

    pz: np.ndarray
    """Seismogram at the bottom of the stack (transmitted pressure)."""
    wz: np.ndarray
    """Seismogram at the top (reflected particle velocity)."""


def propagator_seis(layers, wavelet=None, dt=0.001, alpha=0.0):
    """Synthetic seismograms by the propagator-matrix method.

    Plane-wave, normal-incidence propagation through a 1-D layered medium,
    accumulated as a product of layer propagator matrices.

    Parameters
    ----------
    layers : array_like
        ``(n, 3)`` array of ``[velocity, density, thickness]`` per layer.
    wavelet : array_like, optional
        Source wavelet (pressure). Defaults to `ricker` at 25 Hz.
    dt : float, optional
        Time sampling interval of the wavelet, in seconds.
    alpha : float, optional
        Attenuation coefficient; 0 (default) is lossless.

    Returns
    -------
    PropagatorResult
        Named tuple ``(pz, wz)``.

    See Also
    --------
    kennett : the invariant-imbedding equivalent.

    Notes
    -----
    Port of ``pgator.m``. Shares the wavelet handling, frequency axis, band
    taper, and Hermitian synthesis with `kennett`.
    """
    v, rho, d = _layer_columns(layers)
    if wavelet is None:
        wavelet = ricker(dt=dt)[0]
    wvlt = np.ravel(np.asarray(wavelet, float))
    n = wvlt.size

    om = _frequency_axis(n, dt)
    p0 = np.fft.ifft(wvlt)

    a11 = np.ones(om.shape, complex)
    a12 = np.zeros(om.shape, complex)
    a21 = np.zeros(om.shape, complex)
    a22 = np.ones(om.shape, complex)
    for j in range(v.size):
        k = om / v[j]
        ck = k + 1j * alpha * k
        wdv = d[j] * ck
        imp = rho[j] * v[j]
        c11 = np.cos(wdv)
        c12 = 1j * imp * np.sin(wdv)
        c21 = (1j / imp) * np.sin(wdv)
        a11, a12, a21, a22 = (
            c11 * a11 + c12 * a21,
            c11 * a12 + c12 * a22,
            c21 * a11 + c11 * a21,
            c21 * a12 + c11 * a22,
        )

    zn = rho[-1] * v[-1]
    denom = a12 - zn * a22
    pz_spec = (zn * (a12 * a21 - a11 * a22) / denom) * p0[: om.size]
    wz_spec = ((zn * a21 - a11) / denom) * p0[: om.size]

    taper = _band_taper(n, om.size)
    pz_spec = pz_spec * taper
    wz_spec = wz_spec * taper

    return PropagatorResult(
        pz=_hermitian_synthesis(pz_spec, zero_dc=False),
        wz=_hermitian_synthesis(wz_spec, zero_dc=True),
    )


def _kennett_frazer_phase(v, rho, d, omega):
    """Accumulated ``log(t1)`` phase term of the Kennett-Frazer recursion.

    Returns the cumulative excess traveltime contribution at each layer,
    evaluated for a (possibly vector) angular frequency `omega`.
    """
    n = v.size
    ve = np.append(v, v[-1])
    rhoe = np.append(rho, rho[-1])
    de = np.append(d, d[-1])
    imp = rhoe * ve

    rru = np.zeros_like(np.asarray(omega, complex))
    accum = np.zeros_like(np.asarray(omega, complex))
    excess = []
    total_thickness = 0.0
    for k in range(n):
        den = imp[k] + imp[k + 1]
        rd = (imp[k + 1] - imp[k]) / den
        ru = -rd
        td = 2.0 * np.sqrt(imp[k] * imp[k + 1]) / den

        theta = np.exp(1j * (de[k] / ve[k]) * omega)
        t1 = td / (1.0 - rd * rru * theta**2)
        rru = (ru + rru * theta**2) / (1.0 - rd * rru * theta**2)

        accum = accum + np.log(t1) / (1j * omega)
        total_thickness += de[k]
        excess.append(np.real(accum))
    return np.array(excess), total_thickness


def kennett_frazer_dispersion(layers, freq):
    """Scattering (stratigraphic) velocity dispersion in 1-D layered media.

    The apparent velocity of a normally incident plane wave, from the
    Kennett-Frazer invariant-imbedding recursion. This is scattering
    dispersion, not intrinsic viscoelastic dispersion.

    Parameters
    ----------
    layers : array_like
        ``(n, 3)`` array of ``[velocity, density, thickness]`` per layer.
    freq : array_like
        Frequencies in Hz, e.g. from ``np.logspace``.

    Returns
    -------
    freq : ndarray
        The input frequencies.
    velocity : ndarray
        Apparent velocity at each frequency.

    Notes
    -----
    Port of ``kenfdisp.m``.

    The velocity crosses three regimes as frequency rises. Where the
    wavelength is much longer than a layer *and* the stack is many
    wavelengths thick, it sits at the effective-medium (Backus) velocity;
    where the wavelength approaches twice the layer spacing, a periodic
    stack shows strong resonant excursions; above that it tends to the
    ray-theory (time-average) velocity. Two edge effects are worth
    knowing: as the frequency drops far enough that the whole stack is a
    small fraction of a wavelength there is no accumulated scattering left
    to measure, so the curve returns toward the ray-theory value; and the
    recursion accumulates ``log`` on its principal branch, so results
    degrade once the per-layer phase becomes large.
    """
    v, rho, d = _layer_columns(layers)
    freq = np.atleast_1d(np.asarray(freq, float))
    omega = 2.0 * np.pi * freq

    ray_time = np.cumsum(d / v)
    excess, total_thickness = _kennett_frazer_phase(v, rho, d, omega)
    travel_time = ray_time[-1] + excess[-1]
    return freq, total_thickness / travel_time


class KennettFrazerTraveltimes(NamedTuple):
    """Traveltimes accumulated layer by layer down the stack."""

    tt: np.ndarray
    """Exact traveltimes, including multiples and thin-layer effects."""
    rt: np.ndarray
    """Ray-theory (time-average) traveltimes."""
    emtt: np.ndarray
    """Effective-medium traveltimes."""


def kennett_frazer_traveltimes(layers, freq):
    """Exact, ray-theory, and effective-medium traveltimes in layered media.

    Parameters
    ----------
    layers : array_like
        ``(n, 3)`` array of ``[velocity, density, thickness]`` per layer.
    freq : float
        Frequency in Hz.

    Returns
    -------
    KennettFrazerTraveltimes
        Named tuple ``(tt, rt, emtt)``, each of length ``n``: the
        cumulative traveltime to the base of each layer.

    Notes
    -----
    Port of ``kenfrtt.m``. Its effective-medium velocity uses the
    *unweighted* harmonic mean of the P-wave moduli over the layers
    traversed so far and the unweighted mean density — unlike
    ``kenfdisp.m``, which weights both by fractional thickness. Both are
    kept as the originals had them.
    """
    v, rho, d = _layer_columns(layers)
    m = rho * v**2
    omega = 2.0 * np.pi * float(freq)

    rt = np.cumsum(d / v)
    excess, _ = _kennett_frazer_phase(v, rho, d, omega)
    tt = rt + np.ravel(excess)

    emtt = np.empty(v.size)
    length = 0.0
    for k in range(v.size):
        length += d[k]
        m_harm = (k + 1) / np.sum(1.0 / m[: k + 1])
        v_em = np.sqrt(m_harm / np.mean(rho[: k + 1]))
        emtt[k] = length / v_em
    return KennettFrazerTraveltimes(tt=tt, rt=rt, emtt=emtt)


class SeismicSection(NamedTuple):
    """Quick normal-incidence synthetic section."""

    section: np.ndarray
    """Synthetic seismogram(s), ``(n_time, n_trace)``."""
    time: np.ndarray
    """Two-way time axis (s)."""
    filter_taps: np.ndarray
    """Coefficients of the low-pass FIR filter applied."""


def quick_seismic_section(vel, dens, dx=1.0, dz=1.0, top=0.0, freq=25.0, noise_ratio=0.0, rng=None):
    """Quick normal-incidence synthetic seismic section from velocity and density.

    Computes reflectivity from impedance, converts depth to two-way time,
    low-pass filters to the requested seismic frequency, and averages
    horizontally over a Fresnel zone.

    Parameters
    ----------
    vel, dens : array_like
        Velocity and density, as 1-D logs or 2-D images of the same shape.
    dx, dz : float, optional
        Horizontal and vertical grid spacing.
    top : float, optional
        Depth to the top of the model.
    freq : float, optional
        Seismic frequency in Hz.
    noise_ratio : float, optional
        Noise-to-signal *energy* ratio (note: not signal-to-noise), usually
        below 1. 0 (default) adds no noise.
    rng : numpy.random.Generator, optional
        Source of the random noise, for reproducibility.

    Returns
    -------
    SeismicSection
        Named tuple ``(section, time, filter_taps)``. A 1-D input log
        produces a section of 25 identical traces, as in the MATLAB.

    Notes
    -----
    Port of ``ezseis.m`` with its ``inputdlg`` prompts replaced by keyword
    arguments and its plotting removed. Fixes a latent crash: the MATLAB
    used the filter cutoff ``fc`` without defining it whenever no
    decimation was needed (``R <= 1``).
    """
    vel = np.asarray(vel, float)
    dens = np.asarray(dens, float)
    if vel.ndim == 1:
        vel = vel[:, None]
    if dens.ndim == 1:
        dens = dens[:, None]
    if vel.shape != dens.shape:
        raise ValueError("vel and dens must have the same shape")

    imped = vel * dens
    t_top = 2.0 * top / np.mean(vel[0, :])
    tt = np.cumsum(2.0 * dz / vel, axis=0) + t_top

    dtt = 0.5 * np.min(np.diff(tt, axis=0))
    t_uniform = np.arange(np.min(tt[0, :]), np.max(tt[-1, :]) + dtt, dtt)

    # Pad one sample above and below so every query lies inside the range.
    tt_pad = np.vstack(
        [
            np.full((1, tt.shape[1]), t_uniform[0] - dtt),
            tt,
            np.full((1, tt.shape[1]), t_uniform[-1] + dtt),
        ]
    )
    imped_pad = np.vstack([imped[:1], imped, imped[-1:]])

    imped_t = np.empty((t_uniform.size, imped.shape[1]))
    for k in range(imped.shape[1]):
        imped_t[:, k] = np.interp(t_uniform, tt_pad[:, k], imped_pad[:, k])
    reflectivity = 0.5 * np.diff(np.log(imped_t), axis=0)

    nyquist = 0.5 / dtt
    target_nyquist = 5.0 * freq
    ratio = int(np.floor(nyquist / target_nyquist))
    if ratio > 1:
        traces = [decimate(reflectivity[:, k], ratio, ftype="fir") for k in range(imped.shape[1])]
        reflectivity = np.column_stack(traces)
        cutoff = min(ratio * freq / nyquist, 0.99)
    else:
        ratio = 1
        # ezseis.m left fc undefined on this branch; the equivalent cutoff
        # without decimation is freq / nyquist.
        cutoff = min(freq / nyquist, 0.99)

    taps = firwin(10, cutoff)
    section = filtfilt(taps, 1.0, reflectivity, axis=0)

    depth = top + dz * 0.5 * vel.shape[0]
    wavelength = np.mean(vel) / freq
    fresnel = np.sqrt(depth * wavelength)
    box_n = max(2, int(np.floor(fresnel / dx)))
    box = np.ones(box_n) / box_n
    if section.shape[1] > 1:
        section = np.apply_along_axis(_convolve_same, 1, section, box)
    else:
        section = np.repeat(section, 25, axis=1)

    if noise_ratio != 0:
        rng = np.random.default_rng() if rng is None else rng
        section = section + np.sqrt(noise_ratio) * np.std(section) * rng.standard_normal(
            section.shape
        )

    dt_out = ratio * dtt
    time = t_uniform[0] + dt_out * np.arange(section.shape[0])
    return SeismicSection(section=section, time=time, filter_taps=taps)
