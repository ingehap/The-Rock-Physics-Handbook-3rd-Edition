"""Upper and lower bounds on the elastic moduli of aggregates.

Ports of the following RPHtools MATLAB functions:

===========  ==============================  ================================
MATLAB       Python                          Notes
===========  ==============================  ================================
``bound.m``  `bounds`                        ``ib`` flag becomes ``method``.
``hash.m``   `hashin_shtrikman`              Compute-only; no plotting.
``hashv.m``  `hashin_shtrikman_velocity`     Compute-only; no plotting.
===========  ==============================  ================================

Behavior notes (deliberate changes from MATLAB, see PORTING_PLAN.md):

- ``bound.m`` required ``sum(f) == 1`` *exactly*, which fails for perfectly
  valid inputs like ten fractions of 0.1 (floating-point sum 0.9999...).
  `bounds` uses a small tolerance instead.
- Plotting side effects are removed; use ``rphtools.plotting`` (future) or
  matplotlib directly on the returned curves.

References
----------
Hashin, Z., and Shtrikman, S., 1963, A variational approach to the theory of
the elastic behaviour of multiphase materials: J. Mech. Phys. Solids, 11,
127-140.
Berryman, J. G., 1993, Mixture theories for rock properties.
The Rock Physics Handbook, bounds chapter.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

__all__ = [
    "ElasticBounds",
    "HSBoundCurves",
    "HSVelocityCurves",
    "bounds",
    "hashin_shtrikman",
    "hashin_shtrikman_velocity",
]

_SUM_TOL = 1e-6


class ElasticBounds(NamedTuple):
    """Upper/lower bounds and their average for an aggregate."""

    k_upper: float
    k_lower: float
    mu_upper: float
    mu_lower: float
    k_avg: float
    """Arithmetic average of the bulk-modulus bounds (Hill average for
    Voigt-Reuss)."""
    mu_avg: float
    """Arithmetic average of the shear-modulus bounds."""


def bounds(f, k, mu, method="hashin-shtrikman"):
    """Upper and lower elastic bounds of an isotropic multiphase aggregate.

    Parameters
    ----------
    f : array_like
        Volume fractions of the phases; must sum to 1.
    k : array_like
        Bulk moduli of the phases.
    mu : array_like
        Shear moduli of the phases. All three must have the same length.
    method : {'hashin-shtrikman', 'hs', 'voigt-reuss', 'vr'}, optional
        ``'hashin-shtrikman'`` (default) gives the narrowest possible bounds
        for an isotropic aggregate; ``'voigt-reuss'`` gives the simple
        arithmetic/harmonic bounds.

    Returns
    -------
    ElasticBounds
        Named tuple ``(k_upper, k_lower, mu_upper, mu_lower, k_avg, mu_avg)``.
        The averages are the arithmetic means of the bounds — the
        Voigt-Reuss-Hill average when ``method='voigt-reuss'``.

    Notes
    -----
    Port of ``bound.m`` (``ib=0`` -> ``'voigt-reuss'``, ``ib=1`` ->
    ``'hashin-shtrikman'``). A phase with ``mu = 0`` (fluid) drives the
    lower shear bound (and the Reuss bound) to 0, as physics requires.
    """
    f, k, mu = (np.atleast_1d(np.asarray(a, float)) for a in (f, k, mu))
    if not (f.shape == k.shape == mu.shape):
        raise ValueError("f, k, and mu must have the same length")
    if abs(f.sum() - 1.0) > _SUM_TOL:
        raise ValueError(f"fractions must sum to 1 (got {f.sum()})")

    method = method.lower()
    if method in ("voigt-reuss", "vr"):
        with np.errstate(divide="ignore"):
            k_u = float(np.sum(f * k))
            k_l = float(1.0 / np.sum(f / k))
            mu_u = float(np.sum(f * mu))
            mu_l = float(1.0 / np.sum(f / mu))
    elif method in ("hashin-shtrikman", "hs"):
        c = 4.0 / 3.0
        kmx, kmn = float(k.max()), float(k.min())
        umx, umn = float(mu.max()), float(mu.min())

        with np.errstate(divide="ignore"):
            k_u = float(1.0 / np.sum(f / (k + c * umx)) - c * umx)
            k_l = float(1.0 / np.sum(f / (k + c * umn)) - c * umn)

            eta_mx = umx * (9.0 * kmx + 8.0 * umx) / (kmx + 2.0 * umx) / 6.0
            eta_mn = umn * (9.0 * kmn + 8.0 * umn) / (kmn + 2.0 * umn) / 6.0
            mu_u = float(1.0 / np.sum(f / (mu + eta_mx)) - eta_mx)
            mu_l = float(1.0 / np.sum(f / (mu + eta_mn)) - eta_mn)
    else:
        raise ValueError(f"unknown method {method!r}")

    return ElasticBounds(
        k_upper=k_u,
        k_lower=k_l,
        mu_upper=mu_u,
        mu_lower=mu_l,
        k_avg=(k_u + k_l) / 2.0,
        mu_avg=(mu_u + mu_l) / 2.0,
    )


class HSBoundCurves(NamedTuple):
    """Hashin-Shtrikman bound curves versus fraction of material 2."""

    k_upper: np.ndarray
    k_lower: np.ndarray
    mu_upper: np.ndarray
    mu_lower: np.ndarray
    f2: np.ndarray
    """Volume fraction of material 2 at which the bounds are evaluated."""


def _default_f2():
    """The MATLAB evaluation grid: 0:0.01:1 with the first point at 1e-7.

    The 1e-7 avoids a 0/0 when material 2 is vacuum (``k2 = mu2 = 0``).
    """
    f2 = np.linspace(0.0, 1.0, 101)
    f2[0] = 1e-7
    return f2


def hashin_shtrikman(k1, mu1, k2, mu2, f2=None):
    """Hashin-Shtrikman bound curves for a two-phase mixture.

    Parameters
    ----------
    k1, mu1 : float
        Bulk and shear moduli of material 1 (the stiffer phase for the
        upper/lower labeling to hold).
    k2, mu2 : float
        Bulk and shear moduli of material 2.
    f2 : array_like, optional
        Volume fractions of material 2 at which to evaluate the bounds.
        Defaults to the MATLAB grid ``0:0.01:1`` with the first point moved
        to 1e-7 (which avoids a 0/0 when material 2 is vacuum).

    Returns
    -------
    HSBoundCurves
        Named tuple ``(k_upper, k_lower, mu_upper, mu_lower, f2)``.

    See Also
    --------
    bounds : bounds of an N-phase aggregate at a single composition.
    hashin_shtrikman_velocity : the same curves converted to velocities.

    Notes
    -----
    Port of ``hash.m``. If material 1 is the softer phase, the "upper" and
    "lower" outputs swap roles, as in the original. For a vacuum phase 2
    (``k2 = mu2 = 0``) the lower shear-bound curve is NaN (0/0 in its
    zeta term), exactly as in MATLAB; the k bounds remain finite thanks to
    the 1e-7 first grid point.
    """
    f2 = _default_f2() if f2 is None else np.asarray(f2, float)
    k1, mu1, k2, mu2 = (np.float64(a) for a in (k1, mu1, k2, mu2))

    with np.errstate(divide="ignore", invalid="ignore"):
        k_u = k2 + (1.0 - f2) * (k1 - k2) * (k2 + 4.0 * mu1 / 3.0) / (
            k2 + 4.0 * mu1 / 3.0 + f2 * (k1 - k2)
        )
        k_l = k2 + (1.0 - f2) * (k1 - k2) * (k2 + 4.0 * mu2 / 3.0) / (
            k2 + 4.0 * mu2 / 3.0 + f2 * (k1 - k2)
        )
        zeta1 = mu1 * (9.0 * k1 + 8.0 * mu1) / (6.0 * (k1 + 2.0 * mu1))
        zeta2 = mu2 * (9.0 * k2 + 8.0 * mu2) / (6.0 * (k2 + 2.0 * mu2))
        mu_u = mu2 + (mu1 - mu2) * (1.0 - f2) * (mu2 + zeta1) / (mu2 + zeta1 + f2 * (mu1 - mu2))
        mu_l = mu2 + (mu1 - mu2) * (1.0 - f2) * (mu2 + zeta2) / (mu2 + zeta2 + f2 * (mu1 - mu2))

    return HSBoundCurves(k_upper=k_u, k_lower=k_l, mu_upper=mu_u, mu_lower=mu_l, f2=f2)


class HSVelocityCurves(NamedTuple):
    """Hashin-Shtrikman velocity bound curves versus fraction of material 2."""

    vp_upper: np.ndarray
    vp_lower: np.ndarray
    vs_upper: np.ndarray
    vs_lower: np.ndarray
    f2: np.ndarray
    """Volume fraction of material 2 at which the bounds are evaluated."""


def hashin_shtrikman_velocity(vp1, vs1, rho1, vp2, vs2, rho2, f2=None):
    """Hashin-Shtrikman bound curves for velocities of a two-phase mixture.

    Parameters
    ----------
    vp1, vs1, rho1 : float
        P velocity, S velocity, and density of material 1 (assumed the
        faster phase for the upper/lower labeling to hold).
    vp2, vs2, rho2 : float
        P velocity, S velocity, and density of material 2.
    f2 : array_like, optional
        Volume fractions of material 2; same default grid as
        `hashin_shtrikman`.

    Returns
    -------
    HSVelocityCurves
        Named tuple ``(vp_upper, vp_lower, vs_upper, vs_lower, f2)``.

    Notes
    -----
    Port of ``hashv.m``: moduli bounds from `hashin_shtrikman` combined with
    the arithmetic-average density.
    """
    rho1, rho2 = float(rho1), float(rho2)
    mu1 = rho1 * float(vs1) ** 2
    mu2 = rho2 * float(vs2) ** 2
    k1 = rho1 * float(vp1) ** 2 - 4.0 / 3.0 * mu1
    k2 = rho2 * float(vp2) ** 2 - 4.0 / 3.0 * mu2

    hs = hashin_shtrikman(k1, mu1, k2, mu2, f2=f2)
    rho = (1.0 - hs.f2) * rho1 + hs.f2 * rho2

    return HSVelocityCurves(
        vp_upper=np.sqrt((hs.k_upper + 4.0 / 3.0 * hs.mu_upper) / rho),
        vp_lower=np.sqrt((hs.k_lower + 4.0 / 3.0 * hs.mu_lower) / rho),
        vs_upper=np.sqrt(hs.mu_upper / rho),
        vs_lower=np.sqrt(hs.mu_lower / rho),
        f2=hs.f2,
    )
