"""Fluid substitution, poroelasticity, and velocity dispersion.

Ports of the following RPHtools MATLAB functions:

==============  ==============================  ================================
MATLAB          Python                          Notes
==============  ==============================  ================================
``gassmnk.m``   `gassmann_k`
``gassmnv.m``   `gassmann_vel`                  Delegates to `gassmann_k`.
``patchw.m``    `white_patchy`                  Plotting stripped.
``biot.m``      `biot_dispersion`               ``bessel`` -> ``scipy.special``.
``biothf.m``    `biot_hf`
``biothfb.m``   `biot_hf_geertsma_smit`
``biothfgs.m``  (not ported)                    Computes sqrt(vp1^2 + vp2^2),
                                                not vp1 — an approximation
                                                neglecting the slow wave;
                                                used as a test oracle only.
``BKs2d.m``     `brown_korringa_sat_to_dry`
``BKd2s.m``     `brown_korringa_dry_to_sat`
``BKs2s.m``     `brown_korringa_s`
``BKc2c.m``     `brown_korringa_c`
``bkti.m``      `brown_korringa_ti`             Takes fluid bulk modulus, not
                                                compressibility.
``mmti.m``      `squirt_ti`
==============  ==============================  ================================

Behavior notes (deliberate changes from MATLAB, see PORTING_PLAN.md):

- `gassmann_vel` delegates to `gassmann_k` and therefore inherits its
  ``phi == 0`` pass-through guard, which the MATLAB inline copy lacked.
- The Biot-family functions and `white_patchy` drop the mineral shear
  modulus argument that the MATLAB accepted but never used.
- `biot_dispersion` takes an explicit frequency array instead of
  ``(d1, d2)`` log-range bounds.

Units: moduli/velocity/density functions work in any consistent unit system.
The frequency-dependent models (`biot_dispersion`, `white_patchy`) mix
moduli, viscosity, permeability, and frequency, so use one coherent system
throughout — SI (Pa, kg/m^3, Pa*s, m^2, m, Hz, m/s) is the safe choice.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy.special import jv

from .tensors import isotropic_cs

__all__ = [
    "BiotDispersion",
    "WhitePatchyResult",
    "biot_dispersion",
    "biot_hf",
    "biot_hf_geertsma_smit",
    "brown_korringa_c",
    "brown_korringa_dry_to_sat",
    "brown_korringa_s",
    "brown_korringa_sat_to_dry",
    "brown_korringa_ti",
    "gassmann_k",
    "gassmann_vel",
    "squirt_ti",
    "white_patchy",
]


# ---------------------------------------------------------------------------
# Gassmann
# ---------------------------------------------------------------------------


def gassmann_k(k_sat1, k_fl1, k_fl2, k_min, phi):
    """Gassmann fluid substitution on the bulk modulus.

    Parameters
    ----------
    k_sat1 : array_like
        Rock bulk modulus saturated with fluid 1 (use the dry-rock modulus
        with ``k_fl1 = 0`` for dry-to-saturated substitution).
    k_fl1 : array_like
        Bulk modulus of the initial fluid.
    k_fl2 : array_like
        Bulk modulus of the new fluid.
    k_min : array_like
        Mineral bulk modulus.
    phi : array_like
        Porosity (0 to 1). Where ``phi == 0``, `k_sat1` is returned
        unchanged.

    Returns
    -------
    ndarray
        Rock bulk modulus saturated with fluid 2.

    Notes
    -----
    Port of ``gassmnk.m``. Passing P-wave moduli in place of bulk moduli
    performs the approximate Gassmann calculation, as in the original.

    References
    ----------
    Gassmann, F., 1951; The Rock Physics Handbook, fluid-substitution
    section.
    """
    k1, kf1, kf2, k0, phi = (np.asarray(a, float) for a in (k_sat1, k_fl1, k_fl2, k_min, phi))
    with np.errstate(divide="ignore", invalid="ignore"):
        a = k1 / (k0 - k1) - kf1 / (phi * (k0 - kf1)) + kf2 / (phi * (k0 - kf2))
        k2 = k0 * a / (1.0 + a)
    return np.where(phi == 0, k1, k2)


def gassmann_vel(vp1, vs1, rho1, rho_fl1, k_fl1, rho_fl2, k_fl2, k_min, phi):
    """Gassmann fluid substitution with velocities as input and output.

    Parameters
    ----------
    vp1, vs1, rho1 : array_like
        Rock P velocity, S velocity, and bulk density with fluid 1.
    rho_fl1, k_fl1 : array_like
        Density and bulk modulus of the initial fluid.
    rho_fl2, k_fl2 : array_like
        Density and bulk modulus of the new fluid.
    k_min : array_like
        Mineral bulk modulus.
    phi : array_like
        Porosity (0 to 1).

    Returns
    -------
    vp2, vs2, rho2, k2 : ndarray
        P velocity, S velocity, bulk density, and bulk modulus of the rock
        with fluid 2.

    Notes
    -----
    Port of ``gassmnv.m``. Giving ``vs1 = 0`` and the mineral P-wave modulus
    in place of `k_min` performs the approximate Gassmann calculation.
    Unlike the MATLAB (which duplicated the formula inline), this delegates
    to `gassmann_k` and therefore inherits its ``phi == 0`` guard.
    """
    vp1, vs1, rho1 = (np.asarray(a, float) for a in (vp1, vs1, rho1))
    rho_fl1, rho_fl2, phi = (np.asarray(a, float) for a in (rho_fl1, rho_fl2, phi))

    rho2 = rho1 - phi * rho_fl1 + phi * rho_fl2
    mu = rho1 * vs1**2
    k1 = rho1 * vp1**2 - 4.0 / 3.0 * mu
    k2 = gassmann_k(k1, k_fl1, k_fl2, k_min, phi)
    vp2 = np.sqrt((k2 + 4.0 / 3.0 * mu) / rho2)
    vs2 = np.sqrt(mu / rho2)
    return vp2, vs2, rho2, k2


# ---------------------------------------------------------------------------
# Brown-Korringa (arbitrary anisotropy, 6x6 compliance/stiffness domain)
# ---------------------------------------------------------------------------


def _bk_svect_beta(s):
    """Row-sum vector and compressibility used by the Brown-Korringa forms."""
    svect = s[..., :3, :].sum(axis=-2)
    beta = s[..., :3, :3].sum(axis=(-2, -1))
    return svect, beta


def brown_korringa_sat_to_dry(s_sat, k_min, mu_min, k_fl, phi):
    """Brown-Korringa saturated-to-dry substitution on the compliance matrix.

    Parameters
    ----------
    s_sat : array_like
        6x6 compliance matrix of the saturated rock.
    k_min, mu_min : float
        Isotropic mineral bulk and shear moduli.
    k_fl : float
        Bulk modulus of the (initial) saturating fluid.
    phi : float
        Porosity (0 to 1).

    Returns
    -------
    ndarray
        6x6 compliance matrix of the dry rock.

    Notes
    -----
    Port of ``BKs2d.m``; exact inverse of `brown_korringa_dry_to_sat`.

    References
    ----------
    Brown, R., and Korringa, J., 1975, On the dependence of the elastic
    properties of a porous rock on the compressibility of the pore fluid:
    Geophysics, 40, 608-616.
    """
    s_sat = np.asarray(s_sat, float)
    beta0 = 1.0 / k_min
    svect_min, _ = _bk_svect_beta(isotropic_cs(k_min, mu_min).s)
    svect_sat, beta_sat = _bk_svect_beta(s_sat)

    factor = k_fl / (-k_fl * (beta_sat - beta0) + phi * (1.0 - k_fl * beta0))
    svect = svect_sat - svect_min
    return s_sat + factor * np.einsum("...i,...j->...ij", svect, svect)


def brown_korringa_dry_to_sat(s_dry, k_min, mu_min, k_fl, phi):
    """Brown-Korringa dry-to-saturated substitution on the compliance matrix.

    Parameters
    ----------
    s_dry : array_like
        6x6 compliance matrix of the dry rock.
    k_min, mu_min : float
        Isotropic mineral bulk and shear moduli.
    k_fl : float
        Bulk modulus of the new saturating fluid.
    phi : float
        Porosity (0 to 1).

    Returns
    -------
    ndarray
        6x6 compliance matrix of the saturated rock.

    Notes
    -----
    Port of ``BKd2s.m``.
    """
    s_dry = np.asarray(s_dry, float)
    beta0 = 1.0 / k_min
    svect_min, _ = _bk_svect_beta(isotropic_cs(k_min, mu_min).s)
    svect_dry, beta_dry = _bk_svect_beta(s_dry)

    factor = k_fl / (k_fl * (beta_dry - beta0) + phi * (1.0 - k_fl * beta0))
    svect = svect_dry - svect_min
    return s_dry - factor * np.einsum("...i,...j->...ij", svect, svect)


def brown_korringa_s(s_sat1, k_min, mu_min, k_fl1, k_fl2, phi):
    """Brown-Korringa fluid-to-fluid substitution in the compliance domain.

    Parameters
    ----------
    s_sat1 : array_like
        6x6 compliance matrix of the rock saturated with fluid 1.
    k_min, mu_min : float
        Isotropic mineral bulk and shear moduli.
    k_fl1, k_fl2 : float
        Bulk moduli of the initial and new fluids.
    phi : float
        Porosity (0 to 1).

    Returns
    -------
    ndarray
        6x6 compliance matrix of the rock saturated with fluid 2.

    Notes
    -----
    Port of ``BKs2s.m``: saturated -> dry -> saturated.
    """
    s_dry = brown_korringa_sat_to_dry(s_sat1, k_min, mu_min, k_fl1, phi)
    return brown_korringa_dry_to_sat(s_dry, k_min, mu_min, k_fl2, phi)


def brown_korringa_c(c_sat1, k_min, mu_min, k_fl1, k_fl2, phi):
    """Brown-Korringa fluid-to-fluid substitution in the stiffness domain.

    Parameters
    ----------
    c_sat1 : array_like
        6x6 stiffness matrix of the rock saturated with fluid 1.
    k_min, mu_min : float
        Isotropic mineral bulk and shear moduli.
    k_fl1, k_fl2 : float
        Bulk moduli of the initial and new fluids.
    phi : float
        Porosity (0 to 1).

    Returns
    -------
    ndarray
        6x6 stiffness matrix of the rock saturated with fluid 2.

    Notes
    -----
    Port of ``BKc2c.m``: invert, substitute in the compliance domain,
    invert back.
    """
    s_sat1 = np.linalg.inv(np.asarray(c_sat1, float))
    s_sat2 = brown_korringa_s(s_sat1, k_min, mu_min, k_fl1, k_fl2, phi)
    return np.linalg.inv(s_sat2)


def brown_korringa_ti(s_dry, s_min, k_fl, phi):
    """Brown-Korringa substitution specialized to TI five-constant packing.

    Parameters
    ----------
    s_dry : array_like
        Dry-rock TI compliances, shape ``(..., 5)`` in the order
        ``(s11, s12, s13, s33, s44)``.
    s_min : array_like
        Mineral TI compliances, same packing (broadcast against `s_dry`).
    k_fl : array_like
        Bulk modulus of the saturating fluid (the MATLAB took the fluid
        *compressibility* ``1/k_fl`` instead).
    phi : array_like
        Porosity (0 to 1).

    Returns
    -------
    ndarray
        Saturated-rock TI compliances, shape ``(..., 5)``, same packing.
        ``s44`` is unchanged by fluid substitution.

    Notes
    -----
    Port of ``bkti.m``. For TI symmetry this is algebraically identical to
    `brown_korringa_dry_to_sat` on the corresponding 6x6 matrices (tested).
    """
    s_dry = np.asarray(s_dry, float)
    s_min = np.broadcast_to(np.asarray(s_min, float), s_dry.shape)
    beta_fl = 1.0 / np.asarray(k_fl, float)
    phi = np.asarray(phi, float)

    s11o, s12o, s13o, s33o, _ = np.moveaxis(s_min, -1, 0)
    s11d, s12d, s13d, s33d, s44d = np.moveaxis(s_dry, -1, 0)

    s1a_min = s11o + s12o + s13o
    s3a_min = 2.0 * s13o + s33o
    beta_min = 2.0 * (s11o + s12o + 2.0 * s13o) + s33o

    s1a_dry = s11d + s12d + s13d
    s3a_dry = 2.0 * s13d + s33d
    beta_dry = 2.0 * (s11d + s12d + 2.0 * s13d) + s33d

    denom = beta_dry - beta_min + (beta_fl - beta_min) * phi
    d1 = s1a_dry - s1a_min
    d3 = s3a_dry - s3a_min

    return np.stack(
        [
            s11d - d1 * d1 / denom,
            s12d - d1 * d1 / denom,
            s13d - d1 * d3 / denom,
            s33d - d3 * d3 / denom,
            s44d,
        ],
        axis=-1,
    )


# ---------------------------------------------------------------------------
# Squirt (TI)
# ---------------------------------------------------------------------------


def squirt_ti(s_dry, s_dry_hp):
    """Unrelaxed wet-frame TI compliances from dry compliances (squirt model).

    Parameters
    ----------
    s_dry : array_like
        Dry-rock TI compliances at the pressure(s) of interest, shape
        ``(..., 5)`` in the order ``(s11, s12, s13, s33, s44)``.
    s_dry_hp : array_like
        Dry-rock TI compliances at high pressure (crack-free reference),
        shape ``(5,)``.

    Returns
    -------
    ndarray
        Unrelaxed wet-frame compliances, shape ``(..., 5)``, same packing.

    Notes
    -----
    Port of ``mmti.m``. Check applicability before use (remove a linear
    trend if necessary, as the original advises). If ``s_dry == s_dry_hp``
    (no crack-related compliance change) the result is 0/0 = NaN, as in
    MATLAB.

    References
    ----------
    Mukerji, T., and Mavko, G., 1994, Pore fluid effects on seismic velocity
    in anisotropic rocks: Geophysics, 59, 233-244.
    """
    s_dry = np.asarray(s_dry, float)
    s_dry_hp = np.asarray(s_dry_hp, float)
    ds = s_dry - s_dry_hp
    ds1, ds2, ds3, ds4, ds5 = np.moveaxis(ds, -1, 0)

    ds_aabb = 2.0 * (ds1 + ds2 + 2.0 * ds3) + ds4
    ds_abab = 2.0 * ds1 + ds4 + 4.0 * ds5 + 4.0 * (ds1 - ds2)
    a = (ds_abab / ds_aabb - 1.0) / 4.0

    t1, t2, t3, t4, t5 = (d / ds_aabb for d in (ds1, ds2, ds3, ds4, ds5))
    b = 1.0 - 4.0 * a
    g1 = t1 - (4.0 * a / b) * (t2 + t3)
    g2 = t2 / b
    g3 = t3 / b
    g4 = t4 - (8.0 * a / b) * t3
    g5 = t5 / b - (t1 + t4) / b / 4.0 + (g1 + g4) / 4.0

    g = np.stack([g1, g2, g3, g4, g5], axis=-1)
    return s_dry - ds_aabb[..., None] * g


# ---------------------------------------------------------------------------
# Biot theory
# ---------------------------------------------------------------------------


def _biot_dry_frame(vp_dry, vs_dry, rho_min, rho_fl, phi, tortuosity):
    """Shared Biot-family preamble: dry moduli and the three densities."""
    rho_dry = (1.0 - phi) * rho_min
    rho = (1.0 - phi) * rho_min + phi * rho_fl
    mu_dry = rho_dry * vs_dry**2
    k_dry = rho_dry * vp_dry**2 - 4.0 / 3.0 * mu_dry
    rho_biot = rho_min * (1.0 - phi) + phi * rho_fl * (1.0 - 1.0 / tortuosity)
    return k_dry, mu_dry, rho_dry, rho, rho_biot


class BiotDispersion(NamedTuple):
    """Biot dispersion and attenuation curves (all arrays over frequency)."""

    freq: np.ndarray
    """Frequencies (Hz)."""
    vp1: np.ndarray
    """Fast P-wave velocity."""
    vp2: np.ndarray
    """Slow P-wave velocity."""
    vs: np.ndarray
    """S-wave velocity."""
    q1_inv: np.ndarray
    """Fast P-wave attenuation, 1/Q."""
    q2_inv: np.ndarray
    """Slow P-wave attenuation, 1/Q."""
    qs_inv: np.ndarray
    """S-wave attenuation, 1/Q."""


def biot_dispersion(
    vp_dry,
    vs_dry,
    k_min,
    rho_min,
    rho_fl,
    k_fl,
    eta,
    phi,
    perm,
    pore_size,
    tortuosity,
    freq,
):
    """Full Biot velocity dispersion and attenuation at all frequencies.

    Parameters
    ----------
    vp_dry, vs_dry : float
        P and S velocities of the dry porous rock (m/s).
    k_min : float
        Mineral bulk modulus (Pa).
    rho_min : float
        Mineral density (kg/m^3).
    rho_fl, k_fl : float
        Pore-fluid density (kg/m^3) and bulk modulus (Pa).
    eta : float
        Pore-fluid viscosity (Pa*s).
    phi : float
        Porosity (0 to 1).
    perm : float
        Absolute permeability (m^2).
    pore_size : float
        Pore-size parameter (m); usually ~1/6 to 1/7 of the grain diameter.
    tortuosity : float
        Tortuosity parameter (always > 1, usually 1 to 3).
    freq : array_like
        Frequencies (Hz) at which to evaluate, e.g.
        ``np.logspace(0, 6, 100)`` (the MATLAB took log-range bounds
        ``d1, d2`` instead).

    Returns
    -------
    BiotDispersion
        Named tuple ``(freq, vp1, vp2, vs, q1_inv, q2_inv, qs_inv)``.

    Notes
    -----
    Port of ``biot.m``. The MATLAB called the legacy ``bessel`` function;
    here the viscous-flow correction uses ``scipy.special.jv`` with the same
    magnitude cutoffs (Bessel evaluation for ``zeta <= 1e3``, asymptote
    above, unity below ``zeta <= 0.1``). The mineral shear modulus the
    MATLAB accepted was never used and is not a parameter here.

    References
    ----------
    Biot, M. A., 1956; The Rock Physics Handbook, Biot-theory section.
    """
    freq = np.atleast_1d(np.asarray(freq, float))
    k_dry, mu_dry, _, rho, _ = _biot_dry_frame(vp_dry, vs_dry, rho_min, rho_fl, phi, tortuosity)

    d = k_min * (1.0 + phi * (k_min / k_fl - 1.0))
    h = k_dry + 4.0 / 3.0 * mu_dry + (k_min - k_dry) ** 2 / (d - k_dry)
    c = k_min * (k_min - k_dry) / (d - k_dry)
    m = k_min**2 / (d - k_dry)

    om = 2.0 * np.pi * freq
    zeta = np.sqrt((rho_fl * pore_size**2 / eta) * om)

    t = np.full(zeta.shape, (1.0 + 1j) / np.sqrt(2.0), dtype=complex)
    small = zeta <= 1e3
    zs = zeta[small]
    arg = np.exp(-1j * np.pi / 4.0) * zs
    t[small] = np.exp(1j * 3.0 * np.pi / 4.0) * jv(1, arg) / jv(0, arg)

    f = 0.25 * zeta * t / (1.0 + 2j * t / zeta)
    f[zeta <= 1e-1] = 1.0

    qf = tortuosity * rho_fl / phi - (1j * eta / perm) * f / om
    b0 = rho_fl**2 - rho * qf
    b1 = h * qf + m * rho - 2.0 * c * rho_fl
    b2 = c**2 - m * h

    disc = np.sqrt(b1**2 - 4.0 * b2 * b0)
    sl1 = (-b1 + disc) / (2.0 * b2)
    sl2 = (-b1 - disc) / (2.0 * b2)
    sls = (rho * qf - rho_fl**2) / (mu_dry * qf)

    with np.errstate(divide="ignore", invalid="ignore"):
        vp1 = 1.0 / np.real(np.sqrt(sl1))
        vp2 = 1.0 / np.real(np.sqrt(sl2))
        vs = 1.0 / np.real(np.sqrt(sls))
        q1_inv = np.imag(1.0 / sl1) / np.real(1.0 / sl1)
        q2_inv = np.imag(1.0 / sl2) / np.real(1.0 / sl2)
        qs_inv = np.imag(1.0 / sls) / np.real(1.0 / sls)

    return BiotDispersion(
        freq=freq, vp1=vp1, vp2=vp2, vs=vs, q1_inv=q1_inv, q2_inv=q2_inv, qs_inv=qs_inv
    )


def biot_hf(vp_dry, vs_dry, k_min, rho_min, rho_fl, k_fl, phi, tortuosity):
    """High-frequency limiting velocities of Biot theory (Johnson-Plona).

    Parameters
    ----------
    vp_dry, vs_dry : array_like
        P and S velocities of the dry rock.
    k_min : array_like
        Mineral bulk modulus.
    rho_min : array_like
        Mineral density.
    rho_fl, k_fl : array_like
        Pore-fluid density and bulk modulus.
    phi : array_like
        Porosity (0 to 1).
    tortuosity : array_like
        Tortuosity parameter (always > 1, usually 1 to 3).

    Returns
    -------
    vp1, vp2, vs : ndarray
        High-frequency limits of the fast P, slow P, and S velocities.

    Notes
    -----
    Port of ``biothf.m`` (its scalar-expansion loops are replaced by NumPy
    broadcasting; the unused mineral shear modulus is dropped).

    References
    ----------
    Johnson, D. L., and Plona, T. J., 1982: J. Acoust. Soc. Am., 72,
    556-565.
    """
    vp_dry, vs_dry, k_min, rho_min, rho_fl, k_fl, phi, tortuosity = (
        np.asarray(a, float)
        for a in (vp_dry, vs_dry, k_min, rho_min, rho_fl, k_fl, phi, tortuosity)
    )
    k_dry, mu_dry, _, _, rho_biot = _biot_dry_frame(
        vp_dry, vs_dry, rho_min, rho_fl, phi, tortuosity
    )
    b = k_dry / k_min

    den = 1.0 - phi - b + phi * k_min / k_fl
    p = ((1.0 - phi) * (1.0 - phi - b) * k_min + phi * (k_min / k_fl) * k_dry) / den
    p = p + 4.0 / 3.0 * mu_dry
    q = (1.0 - phi - b) * phi * k_min / den
    r = phi**2 * k_min / den

    rho12 = (1.0 - tortuosity) * phi * rho_fl
    rho11 = (1.0 - phi) * rho_min - rho12
    rho22 = phi * rho_fl * tortuosity

    t1 = p * rho22 + r * rho11 - 2.0 * q * rho12
    t2 = p * r - q**2
    t3 = rho11 * rho22 - rho12**2
    disc = np.sqrt(t1**2 - 4.0 * t2 * t3)

    vp1 = np.sqrt((t1 + disc) / (2.0 * t3))
    vp2 = np.sqrt((t1 - disc) / (2.0 * t3))
    vs = np.sqrt(mu_dry / rho_biot)
    return vp1, vp2, vs


def biot_hf_geertsma_smit(vp_dry, vs_dry, k_min, rho_min, rho_fl, k_fl, phi, tortuosity):
    """Approximate Biot high-frequency limit (Geertsma-Smit).

    Slightly overestimates the exact high-frequency limiting velocity of
    `biot_hf`.

    Parameters
    ----------
    vp_dry, vs_dry : array_like
        P and S velocities of the dry rock.
    k_min : array_like
        Mineral bulk modulus.
    rho_min : array_like
        Mineral density.
    rho_fl, k_fl : array_like
        Pore-fluid density and bulk modulus.
    phi : array_like
        Porosity (0 to 1).
    tortuosity : array_like
        Tortuosity parameter (always > 1, usually 1 to 3).

    Returns
    -------
    vp1, vs : ndarray
        Approximate high-frequency limits of the fast P and S velocities.

    Notes
    -----
    Port of ``biothfb.m`` (Geertsma-Smit as recast by Bourbie et al.).
    """
    vp_dry, vs_dry, k_min, rho_min, rho_fl, k_fl, phi, tortuosity = (
        np.asarray(a, float)
        for a in (vp_dry, vs_dry, k_min, rho_min, rho_fl, k_fl, phi, tortuosity)
    )
    k_dry, mu_dry, _, rho, rho_biot = _biot_dry_frame(
        vp_dry, vs_dry, rho_min, rho_fl, phi, tortuosity
    )
    b = k_dry / k_min

    t1 = phi * rho / (rho_fl * tortuosity) + (1.0 - b) * (1.0 - b - 2.0 * phi / tortuosity)
    t2 = (1.0 - b - phi) / k_min + phi / k_fl
    vp1 = np.sqrt((k_dry + 4.0 / 3.0 * mu_dry + t1 / t2) / rho_biot)
    vs = np.sqrt(mu_dry / rho_biot)
    return vp1, vs


# ---------------------------------------------------------------------------
# White's patchy-saturation model
# ---------------------------------------------------------------------------


class WhitePatchyResult(NamedTuple):
    """White patchy-saturation dispersion results (arrays over frequency)."""

    vp: np.ndarray
    """P-wave velocity at each frequency."""
    k: np.ndarray
    """Complex bulk modulus at each frequency."""
    attenuation: np.ndarray
    """Attenuation coefficient at each frequency (as in the MATLAB:
    ``om * tan(theta/2) / vp``)."""
    k_inf: float
    """High-frequency limiting bulk modulus."""
    k_lf: float
    """Low-frequency limiting bulk modulus."""


def white_patchy(k_dry, mu_dry, k_min, rho_min, phi, perm, fluid1, fluid2, sg1, radius, freq):
    """White's patchy-saturation model with the Dutta-Ode correction.

    Fluid 1 occupies a central sphere of radius `radius`, surrounded by a
    shell saturated with fluid 2; `sg1` is the overall saturation of
    fluid 1 (so the outer shell radius is ``radius / sg1**(1/3)``).

    Parameters
    ----------
    k_dry, mu_dry : float
        Dry-rock bulk and shear moduli (Pa).
    k_min : float
        Mineral bulk modulus (Pa).
    rho_min : float
        Mineral density (kg/m^3).
    phi : float
        Porosity (0 to 1).
    perm : float
        Absolute permeability (m^2).
    fluid1, fluid2 : tuple of float
        ``(k_fl, rho_fl, eta)`` — bulk modulus (Pa), density (kg/m^3), and
        viscosity (Pa*s) of each fluid (the MATLAB packed these as a 3x2
        matrix ``FL``).
    sg1 : float
        Overall saturation of fluid 1 (0 to 1).
    radius : float
        Radius of the fluid-1 patch (m).
    freq : array_like
        Frequencies (Hz) at which to evaluate.

    Returns
    -------
    WhitePatchyResult
        Named tuple ``(vp, k, attenuation, k_inf, k_lf)``.

    Notes
    -----
    Port of ``patchw.m`` with plotting and the display-only intermediate
    outputs stripped; the unused mineral shear modulus is dropped.

    References
    ----------
    White, J. E., 1975; Dutta, N. C., and Ode, H., 1979; The Rock Physics
    Handbook, patchy-saturation section.
    """
    kf1, rof1, nu1 = (float(a) for a in fluid1)
    kf2, rof2, nu2 = (float(a) for a in fluid2)
    freq = np.atleast_1d(np.asarray(freq, float))
    om = 2.0 * np.pi * freq
    a = float(radius)
    b = a / sg1 ** (1.0 / 3.0)

    k1 = gassmann_k(k_dry, 0.0, kf1, k_min, phi)
    k2 = gassmann_k(k_dry, 0.0, kf2, k_min, phi)
    mu1 = mu2 = mu_dry

    r1 = (k1 - k_dry) * (3.0 * k2 + 4.0 * mu2)
    r1 /= (1.0 - k_dry / k_min) * (k2 * (3.0 * k1 + 4.0 * mu2) + 4.0 * mu2 * (k1 - k2) * sg1)
    r2 = (k2 - k_dry) * (3.0 * k1 + 4.0 * mu1)
    r2 /= (1.0 - k_dry / k_min) * (k2 * (3.0 * k1 + 4.0 * mu2) + 4.0 * mu2 * (k1 - k2) * sg1)

    ka1 = 1.0 / (phi / kf1 + (1.0 - phi) / k_min - k_dry / k_min**2)
    ka2 = 1.0 / (phi / kf2 + (1.0 - phi) / k_min - k_dry / k_min**2)
    q1 = (1.0 - k_dry / k_min) * ka1 / k1
    q2 = (1.0 - k_dry / k_min) * ka2 / k2
    ke1 = kf1 * (1.0 - k1 / k_min) * (1.0 - k_dry / k_min) / (phi * k1 * (1.0 - kf1 / k_min))
    ke1 = (1.0 - ke1) * ka1
    ke2 = kf2 * (1.0 - k2 / k_min) * (1.0 - k_dry / k_min) / (phi * k2 * (1.0 - kf2 / k_min))
    ke2 = (1.0 - ke2) * ka2

    alpha1 = np.sqrt(1j * om * nu1 / (perm * ke1))
    alpha2 = np.sqrt(1j * om * nu2 / (perm * ke2))

    z1 = (1.0 - np.exp(-2.0 * a * alpha1)) * nu1 / perm
    z1 /= (alpha1 * a - 1.0) + (alpha1 * a + 1.0) * np.exp(-2.0 * a * alpha1)

    z2 = ((alpha2 * b + 1.0) + (alpha2 * b - 1.0) * np.exp(2.0 * (b - a) * alpha2)) * nu2 / perm
    z2 /= (alpha2 * b + 1.0) * (alpha2 * a - 1.0) - (alpha2 * b - 1.0) * (
        alpha2 * a + 1.0
    ) * np.exp(2.0 * alpha2 * (b - a))
    z2 = -z2

    w = 3.0 * a * (r1 - r2) * (q2 - q1) / (1j * b**3 * om * (z1 + z2))

    k_inf = k2 * (3.0 * k1 + 4.0 * mu2) + 4.0 * mu2 * (k1 - k2) * sg1
    k_inf /= (3.0 * k1 + 4.0 * mu2) - 3.0 * (k1 - k2) * sg1
    k_lf = (k2 * (k1 - k_dry) + sg1 * k_dry * (k2 - k1)) / ((k1 - k_dry) + sg1 * (k2 - k1))

    k = k_inf / (1.0 - k_inf * w)

    rho = (1.0 - phi) * rho_min + phi * sg1 * rof1 + phi * (1.0 - sg1) * rof2
    m = k + 4.0 * mu_dry / 3.0
    theta = np.arctan(np.imag(m) / np.real(m))
    vp = np.sqrt(np.abs(m) / rho) / np.cos(theta / 2.0)
    attenuation = om * np.tan(theta / 2.0) / vp

    return WhitePatchyResult(
        vp=vp, k=k, attenuation=attenuation, k_inf=float(k_inf), k_lf=float(k_lf)
    )
