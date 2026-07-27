"""Backus averaging of thin isotropic layers.

Ports of the following RPHtools MATLAB functions:

=============  ====================  ===========================================
MATLAB         Python                Notes
=============  ====================  ===========================================
``bkus.m``     `backus_average`      Velocities + TI constants.
``bkusc.m``    `backus_average_c`    Full 6x6 stiffness matrix.
``bkuslog.m``  `backus_average_log`  Backus average of an entire well log.
=============  ====================  ===========================================

``bkus.m`` and ``bkusc.m`` implemented identical mathematics with different
argument orders and output formats; here both are thin views over one shared
core, with a single argument order ``(f, vp, vs, rho)``.

Behavior notes (deliberate changes from MATLAB, see PORTING_PLAN.md):

- Fractions are always normalized by their sum (as ``bkusc.m`` did), so raw
  layer thicknesses are accepted too; non-positive or non-finite fractions
  raise ``ValueError`` (``bkus.m`` instead printed a warning and blocked on
  ``pause``).
- ``bkus.m``'s runtime check that ``c66 == (c11 - c12)/2`` is dropped: the
  equality is a mathematical identity of the Backus average (both reduce to
  ``sum(f * mu)``) and is asserted in the test suite instead.
- `backus_average_log` raises on non-monotonic depth, which the original
  silently mishandled.

References
----------
Backus, G. E., 1962, Long-wave elastic anisotropy produced by horizontal
layering: J. Geophys. Res., 67, 4427-4440.
The Rock Physics Handbook, Backus-average section.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from .tensors import ti_velocities, ti_voigt_matrix

__all__ = [
    "BackusLogResult",
    "BackusResult",
    "backus_average",
    "backus_average_c",
    "backus_average_log",
]


def _backus_core(f, vp, vs, rho):
    """Shared Backus core: five TI constants (plus c12) and average density."""
    f, vp, vs, rho = (np.atleast_1d(np.asarray(a, float)) for a in (f, vp, vs, rho))
    if not (f.shape == vp.shape == vs.shape == rho.shape):
        raise ValueError("f, vp, vs, and rho must have the same shape")
    if not np.all(np.isfinite(f)) or np.any(f < 0) or f.sum() <= 0:
        raise ValueError("fractions must be finite, non-negative, and sum to > 0")
    f = f / f.sum()

    mu = rho * vs**2
    lam = rho * vp**2 - 2.0 * mu

    x = np.sum(f * mu * (lam + mu) / (lam + 2.0 * mu))
    y = np.sum(f * mu * lam / (lam + 2.0 * mu))
    z = np.sum(f * lam / (lam + 2.0 * mu))
    u = np.sum(f / (lam + 2.0 * mu))
    v = np.sum(f / mu)
    w = np.sum(f * mu)

    c11 = 4.0 * x + z * z / u
    c12 = 2.0 * y + z * z / u
    c13 = z / u
    c33 = 1.0 / u
    c44 = 1.0 / v
    c66 = w
    rho_avg = float(np.sum(f * rho))
    return c11, c12, c13, c33, c44, c66, rho_avg


class BackusResult(NamedTuple):
    """Backus-average velocities, TI constants, and density."""

    vp0: float
    """P velocity along the symmetry axis (MATLAB ``vp33``)."""
    vp45: float
    """P velocity at 45 degrees from the symmetry axis (MATLAB ``vp13``)."""
    vp90: float
    """P velocity in the layering plane (MATLAB ``vp11``)."""
    vs0: float
    """S velocity along the symmetry axis (MATLAB ``vs33``)."""
    vsh90: float
    """SH velocity in the layering plane (MATLAB ``vsh11``)."""
    c11: float
    c12: float
    c13: float
    c33: float
    c44: float
    c66: float
    rho: float
    """Volume-average density."""


def backus_average(f, vp, vs, rho):
    """Backus average of thin isotropic layers: velocities and TI constants.

    Parameters
    ----------
    f : array_like
        Layer volume fractions (or raw thicknesses; normalized by their sum).
    vp, vs : array_like
        Layer P- and S-wave velocities.
    rho : array_like
        Layer densities. All four inputs must have the same shape.

    Returns
    -------
    BackusResult
        Named tuple with the characteristic velocities ``(vp0, vp45, vp90,
        vs0, vsh90)``, the TI constants ``(c11, c12, c13, c33, c44, c66)``,
        and the volume-average density ``rho``.

    See Also
    --------
    backus_average_c : same average returned as a full 6x6 matrix.
    backus_average_log : Backus average of an entire well log.

    Notes
    -----
    Port of ``bkus.m`` (argument order there was ``(f, rho, vp, vs)``).
    The effective medium is VTI with the symmetry axis perpendicular to the
    layers; ``vp45`` is evaluated with `rphtools.tensors.ti_velocities`.
    """
    c11, c12, c13, c33, c44, c66, rho_avg = _backus_core(f, vp, vs, rho)
    vp45 = float(ti_velocities(c11, c33, c44, c66, c13, rho_avg, 45.0).vp)
    return BackusResult(
        vp0=float(np.sqrt(c33 / rho_avg)),
        vp45=vp45,
        vp90=float(np.sqrt(c11 / rho_avg)),
        vs0=float(np.sqrt(c44 / rho_avg)),
        vsh90=float(np.sqrt(c66 / rho_avg)),
        c11=float(c11),
        c12=float(c12),
        c13=float(c13),
        c33=float(c33),
        c44=float(c44),
        c66=float(c66),
        rho=rho_avg,
    )


def backus_average_c(f, vp, vs, rho):
    """Backus average of thin isotropic layers as a 6x6 stiffness matrix.

    Parameters
    ----------
    f : array_like
        Layer volume fractions (or raw thicknesses; normalized by their sum).
    vp, vs : array_like
        Layer P- and S-wave velocities.
    rho : array_like
        Layer densities. All four inputs must have the same shape.

    Returns
    -------
    c : ndarray
        The ``(6, 6)`` VTI stiffness matrix of the effective medium.
    rho_avg : float
        Volume-average density.

    See Also
    --------
    backus_average : same average returned as velocities and constants.

    Notes
    -----
    Port of ``bkusc.m`` (argument order there was ``(f, vp, vs, den)``).
    """
    c11, c12, c13, c33, c44, c66, rho_avg = _backus_core(f, vp, vs, rho)
    return ti_voigt_matrix(c11, c12, c13, c33, c44, c66), rho_avg


class BackusLogResult(NamedTuple):
    """Backus average of a well log."""

    c: np.ndarray
    """The ``(6, 6)`` VTI stiffness matrix of the whole-log average."""
    rho: float
    """Thickness-weighted average density."""
    phi: float | None
    """Thickness-weighted average porosity (``None`` if not provided)."""


def backus_average_log(depth, vp, vs, rho, phi=None):
    """Backus average of an entire well log.

    Each depth sample is treated as the top of a thin isotropic layer; the
    bottom layer is extrapolated with the thickness of the last interval.
    The average runs over the whole input log — window it beforehand for a
    running average.

    Parameters
    ----------
    depth : array_like
        Depths (any units), strictly monotonic (increasing or decreasing).
    vp, vs : array_like
        Log P- and S-wave velocities.
    rho : array_like
        Log densities.
    phi : array_like, optional
        Log porosities; when given, the thickness-weighted average porosity
        is returned as well.

    Returns
    -------
    BackusLogResult
        Named tuple ``(c, rho, phi)`` — the 6x6 VTI stiffness matrix,
        average density, and average porosity (``None`` if `phi` was not
        provided).

    Raises
    ------
    ValueError
        If `depth` is not strictly monotonic (the original silently produced
        wrong results for non-monotonic depth).

    Notes
    -----
    Port of ``bkuslog.m``.
    """
    depth = np.asarray(depth, float)
    d = np.diff(depth)
    if not (np.all(d > 0) or np.all(d < 0)):
        raise ValueError("depth must be strictly monotonic")
    thick = np.abs(np.append(d, d[-1]))

    c, rho_avg = backus_average_c(thick, vp, vs, rho)
    phi_avg = None
    if phi is not None:
        phi = np.asarray(phi, float)
        phi_avg = float(np.sum(thick / thick.sum() * phi))
    return BackusLogResult(c=c, rho=rho_avg, phi=phi_avg)
