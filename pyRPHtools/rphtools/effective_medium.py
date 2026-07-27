"""Inclusion-based effective-medium models: Berryman self-consistent and DEM.

Ports of the following RPHtools MATLAB functions:

===============  =======================  =====================================
MATLAB           Python                   Notes
===============  =======================  =====================================
``berrysc.m``    `berryman_sc`            Fraction sweep over the
                                          `berryman_scm` core.
``berryscm.m``   `berryman_scm`           N-phase self-consistent solver.
``berryscp.m``   `berryman_sc_pressure`   Pressure loop with crack closing.
``dem.m``        `dem`                    ODE via ``scipy.integrate.solve_ivp``.
``dem1.m``       `dem_at_fraction`        Single-porosity DEM.
``demyprime.m``  (private ``_dem_rhs``)   Explicit arguments instead of the
                                          ``global DEMINPT`` side channel.
``ode45m.m``     (not ported)             Superseded by SciPy's solver.
===============  =======================  =====================================

The spheroid geometry factors (theta, f) and the Berryman polarization
factors P, Q — copied three times in the MATLAB — are implemented once
(`_spheroid_theta_fn`, `_berryman_pq`) and shared by every model here.

Behavior notes (deliberate changes from MATLAB, see PORTING_PLAN.md):

- As in MATLAB, an aspect ratio of exactly 1 is remapped to 0.99 so the
  general spheroid expressions apply; pass e.g. 0.999 for near-spheres.
- `berryman_sc_pressure` picks the mineral phase with ``argmax(k)`` — the
  MATLAB ``find(k==max(k))`` crashed on ties.
- `dem_at_fraction` integrates at the same tight tolerance as `dem`
  (``dem1.m`` used a looser 1e-5).
- Plotting side effects are removed.

References
----------
Berryman, J. G., 1980, Long-wavelength propagation in composite elastic
media: J. Acoust. Soc. Am., 68, 1809-1831.
Berryman, J. G., 1992, Single-scattering approximations for coefficients in
Biot's equations of poroelasticity: J. Acoust. Soc. Am., 91, 551-571.
The Rock Physics Handbook, effective-medium chapter.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy.integrate import solve_ivp

__all__ = [
    "BerrymanSCCurves",
    "DEMResult",
    "berryman_sc",
    "berryman_sc_pressure",
    "berryman_scm",
    "dem",
    "dem_at_fraction",
]


def _spheroid_theta_fn(aspect):
    """Spheroid geometry factors theta and f for oblate/prolate inclusions.

    Aspect ratios of exactly 1 are remapped to 0.99, as in the MATLAB.
    """
    asp = np.atleast_1d(np.asarray(aspect, float)).copy()
    asp[asp == 1.0] = 0.99
    theta = np.empty_like(asp)
    fn = np.empty_like(asp)

    ob = asp < 1.0
    a = asp[ob]
    theta[ob] = (a / (1.0 - a**2) ** 1.5) * (np.arccos(a) - a * np.sqrt(1.0 - a**2))
    fn[ob] = (a**2 / (1.0 - a**2)) * (3.0 * theta[ob] - 2.0)

    pr = asp > 1.0
    a = asp[pr]
    theta[pr] = (a / (a**2 - 1.0) ** 1.5) * (a * np.sqrt(a**2 - 1.0) - np.arccosh(a))
    fn[pr] = (a**2 / (a**2 - 1.0)) * (2.0 - 3.0 * theta[pr])

    return theta, fn


def _berryman_pq(k, mu, k_bg, mu_bg, theta, fn):
    """Berryman polarization factors P, Q for spheroidal inclusions
    (moduli `k`, `mu`) embedded in a background (`k_bg`, `mu_bg`)."""
    nu = (3.0 * k_bg - 2.0 * mu_bg) / (2.0 * (3.0 * k_bg + mu_bg))
    r = (1.0 - 2.0 * nu) / (2.0 * (1.0 - nu))
    a = mu / mu_bg - 1.0
    b = (k / k_bg - mu / mu_bg) / 3.0

    f1 = 1.0 + a * (1.5 * (fn + theta) - r * (1.5 * fn + 2.5 * theta - 4.0 / 3.0))
    f2 = (
        1.0
        + a * (1.0 + 1.5 * (fn + theta) - (r / 2.0) * (3.0 * fn + 5.0 * theta))
        + b * (3.0 - 4.0 * r)
        + (a / 2.0)
        * (a + 3.0 * b)
        * (3.0 - 4.0 * r)
        * (fn + theta - r * (fn - theta + 2.0 * theta**2))
    )
    f3 = 1.0 + a * (1.0 - (fn + 1.5 * theta) + r * (fn + theta))
    f4 = 1.0 + (a / 4.0) * (fn + 3.0 * theta - r * (fn - theta))
    f5 = a * (-fn + r * (fn + theta - 4.0 / 3.0)) + b * theta * (3.0 - 4.0 * r)
    f6 = 1.0 + a * (1.0 + fn - r * (fn + theta)) + b * (1.0 - theta) * (3.0 - 4.0 * r)
    f7 = (
        2.0
        + (a / 4.0) * (3.0 * fn + 9.0 * theta - r * (3.0 * fn + 5.0 * theta))
        + b * theta * (3.0 - 4.0 * r)
    )
    f8 = a * (1.0 - 2.0 * r + (fn / 2.0) * (r - 1.0) + (theta / 2.0) * (5.0 * r - 3.0)) + b * (
        1.0 - theta
    ) * (3.0 - 4.0 * r)
    f9 = a * ((r - 1.0) * fn - r * theta) + b * theta * (3.0 - 4.0 * r)

    p = f1 / f2
    q = (2.0 / f3 + 1.0 / f4 + (f4 * f5 + f6 * f7 - f8 * f9) / (f2 * f4)) / 5.0
    return p, q


def berryman_scm(k, mu, aspect, fraction, tol=None, max_iter=3000):
    """Berryman self-consistent moduli of an N-phase composite.

    Solves the coupled fixed-point equations
    ``sum x_i (k_i - k_sc) P_i = 0``, ``sum x_i (mu_i - mu_sc) Q_i = 0``
    by direct iteration from the Voigt average, as in the MATLAB.

    Parameters
    ----------
    k, mu : array_like
        Bulk and shear moduli of the N phases.
    aspect : array_like
        Aspect ratio of each phase's inclusions (< 1 oblate, > 1 prolate;
        exactly 1 is remapped to 0.99).
    fraction : array_like
        Volume fraction of each phase; should sum to 1.
    tol : float, optional
        Absolute convergence tolerance on the bulk modulus. Defaults to
        ``1e-6 * k[0]`` (the MATLAB default).
    max_iter : int, optional
        Iteration cap (MATLAB: 3000).

    Returns
    -------
    k_sc, mu_sc : float
        Effective bulk and shear moduli.

    See Also
    --------
    berryman_sc : two-phase version swept over a fraction range.
    berryman_sc_pressure : pressure-dependent version with crack closing.

    Notes
    -----
    Port of ``berryscm.m``.
    """
    k, mu, x = (np.atleast_1d(np.asarray(a, float)) for a in (k, mu, fraction))
    theta, fn = _spheroid_theta_fn(aspect)
    if not (k.shape == mu.shape == theta.shape == x.shape):
        raise ValueError("k, mu, aspect, and fraction must have the same length")
    if tol is None:
        tol = 1e-6 * k[0]

    k_sc = float(np.sum(k * x))
    mu_sc = float(np.sum(mu * x))
    delta = np.inf
    n = 0
    while delta > abs(tol) and n < max_iter:
        p, q = _berryman_pq(k, mu, k_sc, mu_sc, theta, fn)
        k_new = float(np.sum(x * k * p) / np.sum(x * p))
        mu_new = float(np.sum(x * mu * q) / np.sum(x * q))
        delta = abs(k_sc - k_new)
        k_sc, mu_sc = k_new, mu_new
        n += 1
    return k_sc, mu_sc


class BerrymanSCCurves(NamedTuple):
    """Self-consistent moduli versus fraction of phase 2."""

    k: np.ndarray
    """Effective bulk modulus at each fraction."""
    mu: np.ndarray
    """Effective shear modulus at each fraction."""
    f2: np.ndarray
    """Fraction of phase 2 actually used (endpoints clamped by 1e-7)."""


def berryman_sc(k1, mu1, k2, mu2, asp1, asp2, f2=None):
    """Berryman self-consistent moduli of a two-phase composite vs fraction.

    Parameters
    ----------
    k1, mu1 : float
        Bulk and shear moduli of phase 1.
    k2, mu2 : float
        Bulk and shear moduli of phase 2.
    asp1, asp2 : float
        Aspect ratios of the two phases' inclusions (< 1 oblate, > 1
        prolate; exactly 1 is remapped to 0.99).
    f2 : array_like, optional
        Fractions of phase 2 at which to evaluate. Defaults to the MATLAB
        sweep ``0:0.01:1``; endpoint fractions are clamped away from pure
        phases by 1e-7, as in the original.

    Returns
    -------
    BerrymanSCCurves
        Named tuple ``(k, mu, f2)``.

    Notes
    -----
    Port of ``berrysc.m``, implemented as a sweep over the `berryman_scm`
    core (the MATLAB duplicated the entire solver inline).
    """
    f2 = np.linspace(0.0, 1.0, 101) if f2 is None else np.asarray(f2, float)
    eps = 1e-7
    f2 = np.clip(f2, eps, 1.0 - eps)

    k_out = np.empty_like(f2)
    mu_out = np.empty_like(f2)
    for i, x2 in enumerate(f2):
        k_out[i], mu_out[i] = berryman_scm(
            [k1, k2], [mu1, mu2], [asp1, asp2], [1.0 - x2, x2], tol=1e-6 * k1
        )
    return BerrymanSCCurves(k=k_out, mu=mu_out, f2=f2)


def berryman_sc_pressure(k, mu, aspect, fraction, pressures):
    """Berryman self-consistent moduli versus effective pressure.

    Pressure dependence comes from thinning and closing penny-shaped
    fluid-filled components (shear modulus 0 and aspect ratio < 0.2);
    stiffer pores and solid phases are unaffected by stress.

    Parameters
    ----------
    k, mu : array_like
        Bulk and shear moduli of the N phases.
    aspect : array_like
        Aspect ratio of each phase's inclusions.
    fraction : array_like
        Volume fraction of each phase; should sum to 1.
    pressures : array_like
        Effective pressures, in the same units as the moduli.

    Returns
    -------
    k_eff, mu_eff : ndarray
        Effective moduli at each pressure.

    Notes
    -----
    Port of ``berryscp.m``. The mineral phase (used for the crack-closing
    rate) is the one with the largest bulk modulus, chosen with ``argmax``
    — the MATLAB ``find(k==max(k))`` failed when two phases tied.
    """
    k, mu, asp, x = (np.atleast_1d(np.asarray(a, float)) for a in (k, mu, aspect, fraction))
    asp = asp.copy()
    asp[asp == 1.0] = 0.99
    pressures = np.atleast_1d(np.asarray(pressures, float))

    imin = int(np.argmax(k))
    k_min_phase, mu_min_phase = k[imin], mu[imin]
    pr_min = (3.0 * k_min_phase - 2.0 * mu_min_phase) / (6.0 * k_min_phase + 2.0 * mu_min_phase)

    k_out = np.empty_like(pressures)
    mu_out = np.empty_like(pressures)
    for i, p in enumerate(pressures):
        dasp = p * 2.0 * (1.0 - pr_min) / (np.pi * mu_min_phase)
        dasp = np.where((asp < 0.2) & (mu == 0.0), dasp, 0.0)
        dasp = np.minimum(dasp, asp)

        asp_p = asp - dasp
        x_p = x * (1.0 - dasp / asp)
        x_p = x_p / x_p.sum()

        keep = asp_p != 0.0
        k_out[i], mu_out[i] = berryman_scm(k[keep], mu[keep], asp_p[keep], x_p[keep])
    return k_out, mu_out


# ---------------------------------------------------------------------------
# Differential effective medium
# ---------------------------------------------------------------------------


def _dem_rhs(t, y, k2, mu2, theta, fn):
    """DEM right-hand side (port of ``demyprime.m`` with explicit args)."""
    k, mu = y
    p, q = _berryman_pq(k2, mu2, k, mu, theta, fn)
    return [
        float((k2 - k) * p) / (1.0 - t),
        float((mu2 - mu) * q) / (1.0 - t),
    ]


class DEMResult(NamedTuple):
    """DEM effective moduli along the porosity path."""

    k: np.ndarray
    """Effective bulk modulus at each porosity."""
    mu: np.ndarray
    """Effective shear modulus at each porosity."""
    phi: np.ndarray
    """Porosity (fraction of phase 2)."""


def _dem_solve(k1, mu1, k2, mu2, aspect, phi_c, t_eval, rtol):
    theta, fn = _spheroid_theta_fn(aspect)
    theta, fn = float(theta[0]), float(fn[0])
    sol = solve_ivp(
        _dem_rhs,
        (0.0, float(t_eval[-1])),
        [float(k1), float(mu1)],
        t_eval=t_eval,
        args=(float(k2), float(mu2), theta, fn),
        rtol=rtol,
        atol=1e-12 * max(abs(k1), abs(mu1), 1.0),
        method="RK45",
    )
    if not sol.success:
        raise RuntimeError(f"DEM integration failed: {sol.message}")
    return sol.y[0], sol.y[1], phi_c * sol.t


def dem(k1, mu1, k2, mu2, aspect, phi_c=1.0, phi=None, rtol=1e-10):
    """Differential-effective-medium moduli along a porosity path.

    Phase 2 is incrementally added to the phase-1 matrix; the coupled ODEs
    ``dK/dt = (K2 - K) P / (1 - t)`` (and likewise for mu with Q) are
    integrated with ``scipy.integrate.solve_ivp``.

    Parameters
    ----------
    k1, mu1 : float
        Bulk and shear moduli of the background matrix.
    k2, mu2 : float
        Bulk and shear moduli of the inclusions.
    aspect : float
        Aspect ratio of the inclusions (< 1 oblate, > 1 prolate; exactly 1
        is remapped to 0.99).
    phi_c : float, optional
        Percolation porosity for the modified DEM model; 1 (default) gives
        the usual DEM. Phase 2 then has moduli ``(k2, mu2)`` at
        concentration ``phi_c``.
    phi : array_like, optional
        Porosities at which to report the moduli. Defaults to 100 points
        from 0 to ``0.99999 * phi_c`` (the MATLAB integration range).
    rtol : float, optional
        Relative tolerance of the integrator (MATLAB used 1e-10).

    Returns
    -------
    DEMResult
        Named tuple ``(k, mu, phi)``.

    See Also
    --------
    dem_at_fraction : single-porosity variant.

    Notes
    -----
    Port of ``dem.m``/``demyprime.m``; the MATLAB ``global DEMINPT`` +
    ``feval`` + bundled ``ode45m`` machinery is replaced by a closure and
    SciPy's RK45.
    """
    if phi is None:
        phi = np.linspace(0.0, 0.99999 * phi_c, 100)
    else:
        phi = np.atleast_1d(np.asarray(phi, float))
    t_eval = phi / phi_c
    if np.any(t_eval >= 1.0) or np.any(t_eval < 0.0) or np.any(np.diff(t_eval) <= 0):
        raise ValueError("phi must be increasing and within [0, phi_c)")
    k, mu, phi_out = _dem_solve(k1, mu1, k2, mu2, aspect, phi_c, t_eval, rtol)
    return DEMResult(k=k, mu=mu, phi=phi_out)


def dem_at_fraction(k1, mu1, k2, mu2, aspect, phi, phi_c=1.0, rtol=1e-10):
    """Differential-effective-medium moduli at a single porosity.

    Parameters
    ----------
    k1, mu1 : float
        Bulk and shear moduli of the background matrix.
    k2, mu2 : float
        Bulk and shear moduli of the inclusions.
    aspect : float
        Aspect ratio of the inclusions.
    phi : float
        Porosity (fraction of phase 2) at which to evaluate.
    phi_c : float, optional
        Percolation porosity for the modified DEM model (1 = usual DEM).
    rtol : float, optional
        Relative tolerance of the integrator. ``dem1.m`` used a loose 1e-5;
        the default here matches `dem` (1e-10).

    Returns
    -------
    k, mu : float
        Effective bulk and shear moduli at `phi`.

    Notes
    -----
    Port of ``dem1.m``.
    """
    if phi == 0:
        return float(k1), float(mu1)
    t_final = float(phi) / phi_c
    if not 0.0 < t_final < 1.0:
        raise ValueError("phi must lie in (0, phi_c)")
    k, mu, _ = _dem_solve(k1, mu1, k2, mu2, aspect, phi_c, np.array([t_final]), rtol)
    return float(k[-1]), float(mu[-1])
