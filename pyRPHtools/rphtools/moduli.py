"""Isotropic elastic moduli and velocity conversions.

Ports of the following RPHtools MATLAB functions:

=============  ======================  =========================================
MATLAB         Python                  Notes
=============  ======================  =========================================
``ku2v.m``     `moduli_to_velocity`
``lm2v.m``     `lame_to_velocity`
(missing)      `velocity_to_moduli`    Reconstruction of ``v2ku``, listed in
                                       ``Contents.m`` but absent from RPHtools.
(missing)      `velocity_to_lame`      Reconstruction of ``v2lm`` (same).
``critpor.m``  `critical_porosity`
=============  ======================  =========================================

All functions accept scalars or array_likes and broadcast with NumPy rules.
Any *consistent* unit system works; the Handbook's examples typically use GPa
for moduli, g/cm^3 for density, and km/s for velocities, which are mutually
consistent.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

__all__ = [
    "CriticalPorosity",
    "critical_porosity",
    "lame_to_velocity",
    "moduli_to_velocity",
    "velocity_to_lame",
    "velocity_to_moduli",
]


def moduli_to_velocity(k, mu, rho):
    """P- and S-wave velocities from bulk modulus, shear modulus, and density.

    .. math:: V_p = \\sqrt{(K + \\tfrac{4}{3}\\mu)/\\rho}, \\qquad
              V_s = \\sqrt{\\mu/\\rho}

    Parameters
    ----------
    k : array_like
        Bulk modulus (e.g. GPa).
    mu : array_like
        Shear modulus (same units as `k`).
    rho : array_like
        Density (e.g. g/cm^3; GPa and g/cm^3 yield km/s).

    Returns
    -------
    vp, vs : ndarray
        P- and S-wave velocities.

    See Also
    --------
    velocity_to_moduli : the inverse conversion.

    Notes
    -----
    Port of ``ku2v.m``.
    """
    k, mu, rho = np.asarray(k, float), np.asarray(mu, float), np.asarray(rho, float)
    vp = np.sqrt((k + 4.0 / 3.0 * mu) / rho)
    vs = np.sqrt(mu / rho)
    return vp, vs


def velocity_to_moduli(vp, vs, rho):
    """Bulk and shear moduli from P- and S-wave velocities and density.

    .. math:: K = \\rho\\,(V_p^2 - \\tfrac{4}{3}V_s^2), \\qquad
              \\mu = \\rho\\,V_s^2

    Parameters
    ----------
    vp, vs : array_like
        P- and S-wave velocities (e.g. km/s).
    rho : array_like
        Density (e.g. g/cm^3; km/s and g/cm^3 yield GPa).

    Returns
    -------
    k, mu : ndarray
        Bulk and shear moduli.

    See Also
    --------
    moduli_to_velocity : the inverse conversion.

    Notes
    -----
    Reconstruction of ``v2ku``, which is listed in the RPHtools ``Contents.m``
    index (and called by ``hertzmindv.m``) but missing from the distribution.
    """
    vp, vs, rho = np.asarray(vp, float), np.asarray(vs, float), np.asarray(rho, float)
    mu = rho * vs**2
    k = rho * (vp**2 - 4.0 / 3.0 * vs**2)
    return k, mu


def lame_to_velocity(lam, mu, rho):
    """P- and S-wave velocities from Lame parameters and density.

    .. math:: V_p = \\sqrt{(\\lambda + 2\\mu)/\\rho}, \\qquad
              V_s = \\sqrt{\\mu/\\rho}

    Parameters
    ----------
    lam : array_like
        Lame's first parameter, lambda (e.g. GPa).
    mu : array_like
        Shear modulus (Lame's second parameter, same units as `lam`).
    rho : array_like
        Density (e.g. g/cm^3; GPa and g/cm^3 yield km/s).

    Returns
    -------
    vp, vs : ndarray
        P- and S-wave velocities.

    See Also
    --------
    velocity_to_lame : the inverse conversion.

    Notes
    -----
    Port of ``lm2v.m``.
    """
    lam, mu, rho = np.asarray(lam, float), np.asarray(mu, float), np.asarray(rho, float)
    vp = np.sqrt((lam + 2.0 * mu) / rho)
    vs = np.sqrt(mu / rho)
    return vp, vs


def velocity_to_lame(vp, vs, rho):
    """Lame parameters from P- and S-wave velocities and density.

    .. math:: \\lambda = \\rho\\,(V_p^2 - 2 V_s^2), \\qquad
              \\mu = \\rho\\,V_s^2

    Parameters
    ----------
    vp, vs : array_like
        P- and S-wave velocities (e.g. km/s).
    rho : array_like
        Density (e.g. g/cm^3; km/s and g/cm^3 yield GPa).

    Returns
    -------
    lam, mu : ndarray
        Lame's first parameter and the shear modulus.

    See Also
    --------
    lame_to_velocity : the inverse conversion.

    Notes
    -----
    Reconstruction of ``v2lm``, which is listed in the RPHtools ``Contents.m``
    index but missing from the distribution.
    """
    vp, vs, rho = np.asarray(vp, float), np.asarray(vs, float), np.asarray(rho, float)
    mu = rho * vs**2
    lam = rho * (vp**2 - 2.0 * vs**2)
    return lam, mu


class CriticalPorosity(NamedTuple):
    """Rock properties at critical porosity (field order follows ``critpor.m``)."""

    vp: np.ndarray
    """P-wave velocity at critical porosity."""
    vs: np.ndarray
    """S-wave velocity at critical porosity."""
    rho: np.ndarray
    """Density at critical porosity (arithmetic average)."""
    m: np.ndarray
    """P-wave modulus at critical porosity (Reuss average)."""
    k: np.ndarray
    """Bulk modulus at critical porosity (Reuss average)."""
    mu: np.ndarray
    """Shear modulus at critical porosity (Reuss average)."""


def critical_porosity(vp1, vs1, rho1, vp2, vs2, rho2, phi_c):
    """Velocities, density, and moduli at critical porosity.

    In Nur's critical-porosity model the load-bearing rock frame exists only
    below the critical porosity ``phi_c``; at ``phi_c`` the moduli are the
    Reuss (isostress) averages of the two end members, and the density is the
    arithmetic (volume) average.

    Parameters
    ----------
    vp1, vs1, rho1 : array_like
        Velocities and density of material 1 (the mineral end member).
    vp2, vs2, rho2 : array_like
        Velocities and density of material 2 (the pore-fill end member).
    phi_c : array_like
        Critical porosity, the volume fraction of material 2 (0 to 1).

    Returns
    -------
    CriticalPorosity
        Named tuple ``(vp, vs, rho, m, k, mu)`` evaluated at ``phi_c``.

    Notes
    -----
    Port of ``critpor.m``. As in the original, the P-wave modulus `m`, bulk
    modulus `k`, and shear modulus `mu` are Reuss-averaged *independently*
    (so ``m != k + 4/3 mu`` in general); `vp` is computed from `k` and `mu`,
    not from `m`.

    References
    ----------
    The Rock Physics Handbook, section on the critical-porosity model
    (Nur et al., 1998).
    """
    vp1, vs1, rho1 = (np.asarray(a, float) for a in (vp1, vs1, rho1))
    vp2, vs2, rho2 = (np.asarray(a, float) for a in (vp2, vs2, rho2))
    phi_c = np.asarray(phi_c, float)

    m1, m2 = rho1 * vp1**2, rho2 * vp2**2
    mu1, mu2 = rho1 * vs1**2, rho2 * vs2**2
    k1, k2 = m1 - 4.0 / 3.0 * mu1, m2 - 4.0 / 3.0 * mu2

    m = (m1 * m2) / ((1 - phi_c) * m2 + phi_c * m1)
    mu = (mu1 * mu2) / ((1 - phi_c) * mu2 + phi_c * mu1)
    k = (k1 * k2) / ((1 - phi_c) * k2 + phi_c * k1)
    rho = (1 - phi_c) * rho1 + phi_c * rho2

    vs = np.sqrt(mu / rho)
    vp = np.sqrt((k + 4.0 / 3.0 * mu) / rho)
    return CriticalPorosity(vp=vp, vs=vs, rho=rho, m=m, k=k, mu=mu)
