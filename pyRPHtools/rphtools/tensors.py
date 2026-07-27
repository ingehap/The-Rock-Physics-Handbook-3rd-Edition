"""Stiffness/compliance tensor utilities for isotropic and TI media.

Ports of the following RPHtools MATLAB functions:

============  =====================  ==========================================
MATLAB        Python                 Notes
============  =====================  ==========================================
``CSiso.m``   `isotropic_cs`         Closed-form compliance instead of ``inv``.
``c2anis.m``  `thomsen_params`
``c2sti.m``   `ti_c_to_s`            Involutive: converts both directions.
``c2vti.m``   `ti_velocities`
``cti2v.m``   `cti_to_velocities`
``ezbond.m``  `bond_rotation`        Plus the `bond_matrix` helper.
(new)         `ti_voigt_matrix`      Builds the 6x6 VTI matrix from the five
                                     independent constants.
(missing)     `ti_from_velocities`   Reconstruction of ``v2cti``: the exact
                                     inverse of `thomsen_params`.
============  =====================  ==========================================

Stiffness/compliance matrices use the 6x6 Voigt notation of the Handbook,
stored as ``(..., 6, 6)`` ndarrays (leading axes broadcast). MATLAB's two
packed 5-vector conventions (``[c11 c33 c44 c66 c13]`` and
``(a11, a12, a13, a33, a44)``) are replaced by explicit named parameters.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

__all__ = [
    "CTIVelocities",
    "IsotropicCS",
    "ThomsenParams",
    "TICompliance5",
    "TIVelocities",
    "bond_matrix",
    "bond_rotation",
    "cti_to_velocities",
    "isotropic_cs",
    "thomsen_params",
    "ti_c_to_s",
    "ti_velocities",
    "ti_from_velocities",
    "ti_voigt_matrix",
]


def ti_voigt_matrix(c11, c12, c13, c33, c44, c66=None):
    """Build the 6x6 Voigt matrix of a VTI medium from its constants.

    Parameters
    ----------
    c11, c12, c13, c33, c44 : array_like
        The five independent TI stiffnesses (symmetry axis along x3).
    c66 : array_like, optional
        Defaults to ``(c11 - c12) / 2``, which TI symmetry requires.

    Returns
    -------
    ndarray
        Shape ``(..., 6, 6)``; leading axes follow NumPy broadcasting of the
        inputs.
    """
    if c66 is None:
        c66 = (np.asarray(c11, float) - np.asarray(c12, float)) / 2.0
    c11, c12, c13, c33, c44, c66 = np.broadcast_arrays(
        *(np.asarray(a, float) for a in (c11, c12, c13, c33, c44, c66))
    )
    c = np.zeros(c11.shape + (6, 6))
    c[..., 0, 0] = c[..., 1, 1] = c11
    c[..., 0, 1] = c[..., 1, 0] = c12
    c[..., 0, 2] = c[..., 2, 0] = c13
    c[..., 1, 2] = c[..., 2, 1] = c13
    c[..., 2, 2] = c33
    c[..., 3, 3] = c[..., 4, 4] = c44
    c[..., 5, 5] = c66
    return c


class IsotropicCS(NamedTuple):
    """6x6 stiffness and compliance of an isotropic material."""

    c: np.ndarray
    """Stiffness matrix, shape ``(..., 6, 6)``."""
    s: np.ndarray
    """Compliance matrix, shape ``(..., 6, 6)``. Contains ``inf`` if mu = 0."""


def isotropic_cs(k, mu):
    """6x6 stiffness and compliance matrices of an isotropic material.

    Parameters
    ----------
    k : array_like
        Bulk modulus.
    mu : array_like
        Shear modulus.

    Returns
    -------
    IsotropicCS
        Named tuple ``(c, s)`` of stiffness and compliance, each of shape
        ``(..., 6, 6)`` following NumPy broadcasting of `k` and `mu`.

    Notes
    -----
    Port of ``CSiso.m``, which returned ``[S, C]`` (compliance first) and
    computed ``S`` by numerical inversion. Here the compliance is closed-form
    (``s11 = 1/E``, ``s12 = -nu/E``, ``s44 = 1/mu`` with
    ``E = 9 K mu / (3K + mu)``), which is exact and well defined even where
    ``inv`` would struggle. For a fluid (``mu = 0``) the stiffness is valid
    but the compliance entries involving shear are ``inf``.
    """
    k, mu = np.broadcast_arrays(np.asarray(k, float), np.asarray(mu, float))
    lam = k - 2.0 * mu / 3.0
    c = np.zeros(k.shape + (6, 6))
    c[..., 0, 0] = c[..., 1, 1] = c[..., 2, 2] = lam + 2.0 * mu
    for i, j in ((0, 1), (0, 2), (1, 2)):
        c[..., i, j] = c[..., j, i] = lam
    c[..., 3, 3] = c[..., 4, 4] = c[..., 5, 5] = mu

    with np.errstate(divide="ignore", invalid="ignore"):
        e = 9.0 * k * mu / (3.0 * k + mu)
        nu = (3.0 * k - 2.0 * mu) / (2.0 * (3.0 * k + mu))
        s11 = 1.0 / e
        s12 = -nu / e
        s44 = 1.0 / mu
    s = np.zeros(k.shape + (6, 6))
    s[..., 0, 0] = s[..., 1, 1] = s[..., 2, 2] = s11
    for i, j in ((0, 1), (0, 2), (1, 2)):
        s[..., i, j] = s[..., j, i] = s12
    s[..., 3, 3] = s[..., 4, 4] = s[..., 5, 5] = s44
    return IsotropicCS(c=c, s=s)


class ThomsenParams(NamedTuple):
    """Thomsen (1986) anisotropy parameters of a TI medium."""

    epsilon: np.ndarray
    """P-wave anisotropy, ``(c11 - c33) / (2 c33)``."""
    gamma: np.ndarray
    """SH-wave anisotropy, ``(c66 - c44) / (2 c44)``."""
    delta: np.ndarray
    """Exact delta, controlling near-vertical P-wave moveout."""
    delta_sv: np.ndarray
    """SV-wave counterpart of delta."""


def thomsen_params(c11, c33, c44, c66, c13):
    """Thomsen anisotropy parameters from TI stiffnesses.

    Parameters
    ----------
    c11, c33, c44, c66, c13 : array_like
        TI stiffnesses (symmetry axis along x3); broadcast together.

    Returns
    -------
    ThomsenParams
        Named tuple ``(epsilon, gamma, delta, delta_sv)``.

    Notes
    -----
    Port of ``c2anis.m`` (input there was packed as ``[c11 c33 c44 c66 c13]``).

    References
    ----------
    Thomsen, L., 1986, Weak elastic anisotropy: Geophysics, 51, 1954-1966.
    """
    c11, c33, c44, c66, c13 = (np.asarray(a, float) for a in (c11, c33, c44, c66, c13))
    epsilon = (c11 - c33) / c33 / 2.0
    gamma = (c66 - c44) / c44 / 2.0
    a = c13 + c44
    b = c33 - c44
    d = c11 - c44
    delta = (a * a - b * b) / c33 / b / 2.0
    delta_sv = (d * b - a * a) / c44 / b / 2.0
    return ThomsenParams(epsilon=epsilon, gamma=gamma, delta=delta, delta_sv=delta_sv)


class TICompliance5(NamedTuple):
    """Five TI constants in the ``(m11, m12, m13, m33, m44)`` packing."""

    m11: np.ndarray
    m12: np.ndarray
    m13: np.ndarray
    m33: np.ndarray
    m44: np.ndarray


def ti_c_to_s(m11, m12, m13, m33, m44):
    """Convert between TI stiffness and compliance constants.

    The transformation is involutive: passing stiffnesses returns
    compliances, and passing compliances returns stiffnesses. (Note that the
    sixth constant transforms as ``s66 = 1/c66`` and is not handled here,
    consistent with the original.)

    Parameters
    ----------
    m11, m12, m13, m33, m44 : array_like
        The five TI constants, all in the same domain (all stiffnesses or all
        compliances); broadcast together.

    Returns
    -------
    TICompliance5
        Named tuple ``(m11, m12, m13, m33, m44)`` in the other domain.

    Notes
    -----
    Port of ``c2sti.m`` (input there was packed as ``(a11,a12,a13,a33,a44)``
    — note the packing differs from the ``[c11 c33 c44 c66 c13]`` convention
    used by `thomsen_params` and `ti_velocities`).
    """
    a11, a12, a13, a33, a44 = (np.asarray(a, float) for a in (m11, m12, m13, m33, m44))
    det = a33 * (a11 + a12) - 2.0 * a13 * a13
    b11 = (a33 / det + 1.0 / (a11 - a12)) / 2.0
    b12 = (a33 / det - 1.0 / (a11 - a12)) / 2.0
    b13 = -a13 / det
    b33 = (a11 + a12) / det
    b44 = 1.0 / a44
    return TICompliance5(m11=b11, m12=b12, m13=b13, m33=b33, m44=b44)


class TIVelocities(NamedTuple):
    """Phase velocities in a TI medium (field order follows ``c2vti.m``)."""

    vp: np.ndarray
    """Quasi-P phase velocity."""
    vsh: np.ndarray
    """SH phase velocity."""
    vsv: np.ndarray
    """Quasi-SV phase velocity."""


def ti_velocities(c11, c33, c44, c66, c13, rho, angle_deg):
    """Phase velocities at an angle from the symmetry axis of a TI medium.

    Parameters
    ----------
    c11, c33, c44, c66, c13 : array_like
        TI stiffnesses (symmetry axis along x3).
    rho : array_like
        Density (units consistent with the stiffnesses).
    angle_deg : array_like
        Angle(s) from the symmetry axis, in degrees. Broadcasts against the
        stiffnesses (e.g. scalar constants with an array of angles, or
        arrays of constants with a scalar angle).

    Returns
    -------
    TIVelocities
        Named tuple ``(vp, vsh, vsv)`` of phase velocities at `angle_deg`.

    Notes
    -----
    Port of ``c2vti.m``. Exact TI phase-velocity expressions (Handbook,
    elastic anisotropy chapter), not the weak-anisotropy approximation.
    """
    c11, c33, c44, c66, c13, rho = (np.asarray(a, float) for a in (c11, c33, c44, c66, c13, rho))
    zeta = np.deg2rad(np.asarray(angle_deg, float))
    s2 = np.sin(zeta) ** 2
    c2 = np.cos(zeta) ** 2
    s22 = np.sin(2.0 * zeta) ** 2

    mm = np.sqrt(((c11 - c44) * s2 - (c33 - c44) * c2) ** 2 + (c13 + c44) ** 2 * s22)
    mp2 = (c11 * s2 + c33 * c2 + c44 + mm) / 2.0
    msv2 = (c11 * s2 + c33 * c2 + c44 - mm) / 2.0
    msh2 = c66 * s2 + c44 * c2

    return TIVelocities(vp=np.sqrt(mp2 / rho), vsh=np.sqrt(msh2 / rho), vsv=np.sqrt(msv2 / rho))


class CTIVelocities(NamedTuple):
    """Fast/slow velocities and Thomsen parameters from a full TI matrix."""

    vp_slow: np.ndarray
    """P velocity along the symmetry (slow) axis."""
    vs_slow: np.ndarray
    """S velocity polarized across the symmetry axis (slow)."""
    vp_fast: np.ndarray
    """P velocity orthogonal to the symmetry axis (fast)."""
    vs_fast: np.ndarray
    """S velocity orthogonal to the symmetry axis (fast)."""
    epsilon: np.ndarray
    gamma: np.ndarray
    delta: np.ndarray


def cti_to_velocities(c, rho):
    """Fast/slow velocities and Thomsen parameters from 6x6 TI matrices.

    Works for both VTI and HTI orientations: ``c11``/``c33`` and
    ``c44``/``c66`` are sorted so that "slow" is along the symmetry axis
    (e.g. the crack normal) and "fast" orthogonal to it.

    Parameters
    ----------
    c : array_like
        TI stiffness matrices, shape ``(..., 6, 6)`` (MATLAB used
        ``6 x 6 x n``).
    rho : array_like
        Density, broadcastable against the leading axes of `c`.

    Returns
    -------
    CTIVelocities
        Named tuple ``(vp_slow, vs_slow, vp_fast, vs_fast, epsilon, gamma,
        delta)``.

    Notes
    -----
    Port of ``cti2v.m``. As in the original, `delta` uses the *unsorted*
    ``c[0, 2]`` together with the sorted ``c33``/``c44``.
    """
    c = np.asarray(c, float)
    rho = np.asarray(rho, float)
    if c.shape[-2:] != (6, 6):
        raise ValueError(f"c must have shape (..., 6, 6), got {c.shape}")

    c11_raw, c33_raw = c[..., 0, 0], c[..., 2, 2]
    c44_raw, c66_raw = c[..., 3, 3], c[..., 5, 5]
    c13 = c[..., 0, 2]

    c11 = np.maximum(c11_raw, c33_raw)
    c33 = np.minimum(c11_raw, c33_raw)
    c66 = np.maximum(c44_raw, c66_raw)
    c44 = np.minimum(c44_raw, c66_raw)

    epsilon = (c11 - c33) / (2.0 * c33)
    gamma = (c66 - c44) / (2.0 * c44)
    delta = ((c13 + c44) ** 2 - (c33 - c44) ** 2) / (2.0 * c33 * (c33 - c44))

    return CTIVelocities(
        vp_slow=np.sqrt(c33 / rho),
        vs_slow=np.sqrt(c44 / rho),
        vp_fast=np.sqrt(c11 / rho),
        vs_fast=np.sqrt(c66 / rho),
        epsilon=epsilon,
        gamma=gamma,
        delta=delta,
    )


def bond_matrix(theta_deg):
    """6x6 Bond transformation matrix for a rotation about the vertical axis.

    Parameters
    ----------
    theta_deg : float
        Rotation angle of the x1 axis about x3, in degrees
        (counter-clockwise positive).

    Returns
    -------
    ndarray
        The ``(6, 6)`` Bond matrix ``M`` such that a stiffness matrix
        transforms as ``M @ c @ M.T``.

    References
    ----------
    The Rock Physics Handbook, "Coordinate transformations" section.
    """
    theta = np.deg2rad(float(theta_deg))
    ct, st = np.cos(theta), np.sin(theta)
    # Direction cosines b_ij between new (rows) and old (columns) axes.
    b = np.array([[ct, st, 0.0], [-st, ct, 0.0], [0.0, 0.0, 1.0]])

    m = np.zeros((6, 6))
    m[:3, :3] = b**2
    for i in range(3):
        m[i, 3] = 2.0 * b[i, 1] * b[i, 2]
        m[i, 4] = 2.0 * b[i, 2] * b[i, 0]
        m[i, 5] = 2.0 * b[i, 0] * b[i, 1]
    pairs = ((1, 2), (2, 0), (0, 1))
    for r, (i, j) in enumerate(pairs):
        for col in range(3):
            m[3 + r, col] = b[i, col] * b[j, col]
        for c_idx, (k, w) in enumerate(pairs):
            m[3 + r, 3 + c_idx] = b[i, k] * b[j, w] + b[i, w] * b[j, k]
    return m


def bond_rotation(c, theta_deg):
    """Rotate a 6x6 stiffness matrix about the vertical (x3) axis.

    The x3 axis is unchanged; x1 rotates counter-clockwise by `theta_deg`.
    Useful for HTI media, e.g. rotating vertically fractured rock so the
    fracture normal makes an angle with a seismic line.

    Parameters
    ----------
    c : array_like
        Stiffness matrix, shape ``(..., 6, 6)``.
    theta_deg : float
        Rotation angle in degrees (counter-clockwise positive).

    Returns
    -------
    ndarray
        The rotated stiffness matrix, ``M @ c @ M.T``, same shape as `c`.

    See Also
    --------
    bond_matrix : the transformation matrix itself.

    Notes
    -----
    Port of ``ezbond.m``. Rotating by ``theta`` and then ``-theta`` recovers
    the original matrix.
    """
    m = bond_matrix(theta_deg)
    return m @ np.asarray(c, float) @ m.T


def ti_from_velocities(vp0, vs0, rho, epsilon, gamma, delta):
    """TI stiffnesses from vertical velocities and Thomsen parameters.

    The exact inverse of `thomsen_params` combined with
    ``c33 = rho vp0^2``, ``c44 = rho vs0^2``.

    Parameters
    ----------
    vp0, vs0 : array_like
        P and S velocities along the symmetry axis.
    rho : array_like
        Density (units consistent with the velocities).
    epsilon, gamma, delta : array_like
        Thomsen (1986) anisotropy parameters.

    Returns
    -------
    c11, c12, c13, c33, c44, c66 : ndarray
        The TI stiffnesses. Pass them to `ti_voigt_matrix` for the 6x6
        form.

    Notes
    -----
    Reconstruction of ``v2cti``, which the RPHtools ``Contents.m`` lists
    but which is absent from the distribution. Nothing calls it, so no
    existing code path depended on it. Unlike the empirical models still
    missing from the toolbox, this one is pure algebra: inverting the
    Thomsen definitions is exact, and the round trip against
    `thomsen_params` is asserted in the test suite.

    ``delta`` fixes ``c13`` through
    ``(c13 + c44)^2 = 2 c33 (c33 - c44) delta + (c33 - c44)^2``; the
    positive root is taken, which is the physical branch for
    ``c13 + c44 > 0``.
    """
    vp0, vs0, rho = (np.asarray(a, float) for a in (vp0, vs0, rho))
    epsilon, gamma, delta = (np.asarray(a, float) for a in (epsilon, gamma, delta))

    c33 = rho * vp0**2
    c44 = rho * vs0**2
    c11 = c33 * (1.0 + 2.0 * epsilon)
    c66 = c44 * (1.0 + 2.0 * gamma)
    c12 = c11 - 2.0 * c66
    c13 = np.sqrt(2.0 * c33 * (c33 - c44) * delta + (c33 - c44) ** 2) - c44
    return c11, c12, c13, c33, c44, c66
