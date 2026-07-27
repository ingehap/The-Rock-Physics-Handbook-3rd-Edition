"""Cracked-rock effective-medium models: Hudson family and Eshelby-Cheng.

Ports of the following RPHtools MATLAB functions:

================  ====================  =======================================
MATLAB            Python                Notes
================  ====================  =======================================
``hudson.m``      `hudson`              Single crack set -> (C, density).
``hudson1.m``     `hudson_velocities`   Same model -> velocities + Thomsen
                                        parameters (shares the core).
``hudson3.m``     `hudson3`             Three orthogonal crack sets.
``hudsonF.m``     `hudson_fisher`       Fisher-distributed crack normals.
``hudsoncone.m``  `hudson_cone`         Conical crack-normal distribution.
``echeng.m``      `eshelby_cheng`       Valid for all aspect ratios < 1.
================  ====================  =======================================

The Hudson compliance kernel (lambda, mu, U1, U3) — copied five times in the
MATLAB — is implemented once in `_hudson_kernel`, and the TI 6x6 assembly
for a symmetry axis along x1 or x3 once in `_hudson_ti_matrix`.

Behavior notes (deliberate changes from MATLAB, see PORTING_PLAN.md):

- **Bug fix**: ``hudsonF.m`` computed the crack porosity for the output
  density as ``4*pi*ar/(3*cd)`` — dividing by crack density instead of
  multiplying. `hudson_fisher` uses ``(4*pi/3)*ar*cd``, consistent with
  ``hudson.m``. (The stiffnesses were unaffected.)
- **Bug fix**: ``hudsonF.m``'s shear components ``c2323``/``c1313``/
  ``c1212`` were missing a ``mu^2`` factor in their U3 terms
  (``4*cd/mu*u3`` instead of ``4*cd*mu*u3``), violating the exact TI
  symmetry ``c66 = (c11 - c12)/2`` of an orientation-averaged medium.
  The port restores the factor; a test asserts the identity.
- ``hudsonF.m`` symmetrized only the first element of the lower triangle
  for vector inputs; the port symmetrizes correctly for stacks.
- `hudson_cone` takes the cone angle in degrees (the MATLAB took radians),
  matching the angle convention used across this package.
- Plotting/dead-code blocks are dropped.

References
----------
Hudson, J. A., 1980, 1981, 1990. Cheng, C. H., 1978, 1993.
Thomsen, L., 1986, Weak elastic anisotropy: Geophysics, 51, 1954-1966.
The Rock Physics Handbook, cracked-media sections.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

__all__ = [
    "EshelbyCheng",
    "Hudson3Result",
    "HudsonVelocities",
    "eshelby_cheng",
    "hudson",
    "hudson3",
    "hudson_cone",
    "hudson_fisher",
    "hudson_velocities",
]


def _hudson_kernel(k_min, g_min, k_fl, aspect):
    """Shared Hudson quantities: lambda, mu, and the U1, U3 crack terms."""
    lam = k_min - 2.0 / 3.0 * g_min
    mu = g_min
    kapa = k_fl * (lam + 2.0 * mu) / (np.pi * aspect * mu * (lam + mu))
    u3 = 4.0 / 3.0 * (lam + 2.0 * mu) / ((lam + mu) * (1.0 + kapa))
    u1 = 16.0 / 3.0 * (lam + 2.0 * mu) / (3.0 * lam + 4.0 * mu)
    return lam, mu, u1, u3


def _hudson_ti_matrix(c11, c13, c33, c44, c66, axis):
    """Assemble the TI 6x6 matrix from Hudson's five constants for a
    symmetry axis along x1 (``axis=1``) or x3 (``axis=3``)."""
    c11, c13, c33, c44, c66 = np.broadcast_arrays(
        *(np.asarray(a, float) for a in (c11, c13, c33, c44, c66))
    )
    c = np.zeros(c11.shape + (6, 6))
    if axis == 1:
        c[..., 0, 0] = c33
        c[..., 1, 1] = c[..., 2, 2] = c11
        c[..., 0, 1] = c[..., 1, 0] = c13
        c[..., 0, 2] = c[..., 2, 0] = c13
        c[..., 1, 2] = c[..., 2, 1] = c11 - 2.0 * c66
        c[..., 3, 3] = c66
        c[..., 4, 4] = c[..., 5, 5] = c44
    elif axis == 3:
        c[..., 0, 0] = c[..., 1, 1] = c11
        c[..., 2, 2] = c33
        c[..., 0, 1] = c[..., 1, 0] = c11 - 2.0 * c66
        c[..., 0, 2] = c[..., 2, 0] = c13
        c[..., 1, 2] = c[..., 2, 1] = c13
        c[..., 3, 3] = c[..., 4, 4] = c44
        c[..., 5, 5] = c66
    else:
        raise ValueError("axis must be 1 or 3")
    return c


def _hudson_constants(crack_density, aspect, k_fl, k_min, g_min):
    """First-order Hudson TI constants for a single aligned crack set."""
    lam, mu, u1, u3 = _hudson_kernel(k_min, g_min, k_fl, aspect)
    ec = crack_density
    c11 = lam + 2.0 * mu - lam**2 * ec * u3 / mu
    c13 = lam - lam * (lam + 2.0 * mu) * ec * u3 / mu
    c33 = lam + 2.0 * mu - (lam + 2.0 * mu) ** 2 * ec * u3 / mu
    c44 = mu - mu * ec * u1
    c66 = mu * np.ones_like(c44)
    return c11, c13, c33, c44, c66


def hudson(crack_density, aspect, k_fl, rho_fl, k_min, g_min, rho_min, axis=1):
    """Hudson stiffness and density for a single aligned crack set.

    First-order weak-inclusion theory, valid for small crack density and
    aspect ratio. For dry cracks use ``k_fl = 0``.

    Parameters
    ----------
    crack_density : array_like
        Crack density (number density times radius cubed).
    aspect : array_like
        Aspect ratio of the penny-shaped cracks (small, < 1).
    k_fl, rho_fl : array_like
        Bulk modulus and density of the fluid in the cracks.
    k_min, g_min, rho_min : array_like
        Bulk modulus, shear modulus, and density of the isotropic matrix.
    axis : {1, 3}, optional
        Direction of the aligned crack normals (= TI symmetry axis).
        Default 1, as in the MATLAB.

    Returns
    -------
    c : ndarray
        Stiffness matrix, shape ``(..., 6, 6)`` following broadcasting of
        the inputs (MATLAB grew ``6 x 6 x n``).
    rho : ndarray
        Bulk density of the cracked rock, including the crack fluid
        (crack porosity ``(4*pi/3) * aspect * crack_density``).

    See Also
    --------
    hudson_velocities : same model returning velocities and Thomsen
        parameters.
    eshelby_cheng : valid for all aspect ratios.

    Notes
    -----
    Port of ``hudson.m``.
    """
    ec, ar, kfl, rhofl, k, g, rho = (
        np.asarray(a, float) for a in (crack_density, aspect, k_fl, rho_fl, k_min, g_min, rho_min)
    )
    c11, c13, c33, c44, c66 = _hudson_constants(ec, ar, kfl, k, g)
    c = _hudson_ti_matrix(c11, c13, c33, c44, c66, axis)
    phi = (4.0 * np.pi / 3.0) * ar * ec
    rho_out = (1.0 - phi) * rho + phi * rhofl
    return c, rho_out


class HudsonVelocities(NamedTuple):
    """Hudson cracked-rock velocities, Thomsen parameters, and stiffness."""

    vp0: np.ndarray
    """P velocity along the symmetry axis."""
    vs0: np.ndarray
    """S velocity along the symmetry axis."""
    epsilon: np.ndarray
    gamma: np.ndarray
    delta: np.ndarray
    c: np.ndarray
    """Stiffness matrix, shape ``(..., 6, 6)``."""


def hudson_velocities(crack_density, aspect, k_fl, k_min, g_min, rho, axis=1):
    """Hudson single-crack-set model: velocities and Thomsen parameters.

    Parameters
    ----------
    crack_density : array_like
        Crack density.
    aspect : array_like
        Aspect ratio of the cracks.
    k_fl : array_like
        Bulk modulus of the fluid in the cracks (0 for dry).
    k_min, g_min : array_like
        Bulk and shear moduli of the isotropic matrix.
    rho : array_like
        Bulk density of the cracked rock (given directly, unlike `hudson`
        which builds it from matrix and fluid densities).
    axis : {1, 3}, optional
        Direction of the aligned crack normals. Default 1.

    Returns
    -------
    HudsonVelocities
        Named tuple ``(vp0, vs0, epsilon, gamma, delta, c)``.

    Notes
    -----
    Port of ``hudson1.m``; shares its computational core with `hudson`.
    """
    ec, ar, kfl, k, g, rho = (
        np.asarray(a, float) for a in (crack_density, aspect, k_fl, k_min, g_min, rho)
    )
    c11, c13, c33, c44, c66 = _hudson_constants(ec, ar, kfl, k, g)
    c = _hudson_ti_matrix(c11, c13, c33, c44, c66, axis)
    return HudsonVelocities(
        vp0=np.sqrt(c33 / rho),
        vs0=np.sqrt(c44 / rho),
        epsilon=(c11 - c33) / (2.0 * c33),
        gamma=(c66 - c44) / (2.0 * c44),
        delta=((c13 + c44) ** 2 - (c33 - c44) ** 2) / (2.0 * c33 * (c33 - c44)),
        c=c,
    )


class Hudson3Result(NamedTuple):
    """Hudson three-crack-set stiffness, density, and Thomsen-style
    parameters (as defined in ``hudson3.m``)."""

    c: np.ndarray
    """Orthorhombic stiffness matrix, shape ``(6, 6)``."""
    rho: float
    """Bulk density of the cracked rock."""
    epsilon_x: float
    gamma_x: float
    delta_x: float
    epsilon_y: float
    gamma_y: float
    delta_y: float
    gamma_xy: float


def hudson3(crack_densities, aspects, k_fl, rho_fl, k_min, g_min, rho_min):
    """Hudson model with three orthogonal crack sets.

    Crack normals are aligned with the three principal axes; each set has
    its own crack density and aspect ratio.

    Parameters
    ----------
    crack_densities : array_like, length 3
        Crack densities of the sets with normals along x1, x2, x3.
    aspects : array_like, length 3
        Aspect ratios of the three sets.
    k_fl, rho_fl : float
        Bulk modulus and density of the crack fluid.
    k_min, g_min, rho_min : float
        Bulk modulus, shear modulus, and density of the isotropic host.

    Returns
    -------
    Hudson3Result
        Named tuple ``(c, rho, epsilon_x, gamma_x, delta_x, epsilon_y,
        gamma_y, delta_y, gamma_xy)``.

    Notes
    -----
    Port of ``hudson3.m``.
    """
    ec = np.asarray(crack_densities, float)
    ar = np.asarray(aspects, float)
    if ec.shape != (3,) or ar.shape != (3,):
        raise ValueError("crack_densities and aspects must have length 3")

    lam, mu, u1, u3 = _hudson_kernel(k_min, g_min, k_fl, ar)
    c11cor = -(lam**2) * ec * u3 / mu
    c13cor = -lam * (lam + 2.0 * mu) * ec * u3 / mu
    c33cor = -((lam + 2.0 * mu) ** 2) * ec * u3 / mu
    c44cor = -mu * ec * u1
    c12cor = c11cor

    c11 = lam + 2.0 * mu + c33cor[0] + c11cor[1] + c11cor[2]
    c12 = lam + c13cor[0] + c13cor[1] + c12cor[2]
    c13 = lam + c13cor[0] + c12cor[1] + c13cor[2]
    c22 = lam + 2.0 * mu + c11cor[0] + c33cor[1] + c11cor[2]
    c23 = lam + c12cor[0] + c13cor[1] + c13cor[2]
    c33 = lam + 2.0 * mu + c11cor[0] + c11cor[1] + c33cor[2]
    c44 = mu + c44cor[1] + c44cor[2]
    c55 = mu + c44cor[0] + c44cor[2]
    c66 = mu + c44cor[0] + c44cor[1]

    c = np.zeros((6, 6))
    c[0, 0], c[1, 1], c[2, 2] = c11, c22, c33
    c[0, 1] = c[1, 0] = c12
    c[0, 2] = c[2, 0] = c13
    c[1, 2] = c[2, 1] = c23
    c[3, 3], c[4, 4], c[5, 5] = c44, c55, c66

    phi = 4.0 * np.pi / 3.0 * float(np.sum(ar * ec))
    rho = (1.0 - phi) * rho_min + phi * rho_fl

    return Hudson3Result(
        c=c,
        rho=float(rho),
        epsilon_x=(c22 - c33) / (2.0 * c33),
        gamma_x=(c66 - c55) / (2.0 * c55),
        delta_x=((c23 + c44) ** 2 - (c33 - c44) ** 2) / (2.0 * c33 * (c33 - c44)),
        epsilon_y=(c11 - c33) / (2.0 * c33),
        gamma_y=(c66 - c44) / (2.0 * c44),
        delta_y=((c13 + c55) ** 2 - (c33 - c55) ** 2) / (2.0 * c33 * (c33 - c55)),
        gamma_xy=(c44 - c55) / (2.0 * c55),
    )


def hudson_fisher(crack_density, aspect, k_fl, rho_fl, k_min, g_min, rho_min, sigma):
    """Hudson model for a Fisher distribution of crack normals about x3.

    Parameters
    ----------
    crack_density : array_like
        Crack density.
    aspect : array_like
        Aspect ratio of the cracks.
    k_fl, rho_fl : array_like
        Bulk modulus and density of the crack fluid.
    k_min, g_min, rho_min : array_like
        Bulk modulus, shear modulus, and density of the isotropic host.
    sigma : float
        Standard deviation of the Fisher distribution of crack normals
        about the x3 axis (radians; small sigma -> aligned cracks).

    Returns
    -------
    c : ndarray
        TI stiffness matrix, shape ``(..., 6, 6)``.
    rho : ndarray
        Bulk density of the cracked rock.

    Notes
    -----
    Port of ``hudsonF.m`` with its crack-porosity bug fixed: the MATLAB
    computed ``4*pi*ar/(3*cd)`` (dividing by crack density); the port uses
    the crack porosity ``(4*pi/3) * ar * cd`` consistent with ``hudson.m``.

    References
    ----------
    Hudson, J. A., 1990: Geophys. J. Int., 102, 465-469.
    """
    cd, ar, kfl, rhofl, k, g, rho = (
        np.asarray(a, float) for a in (crack_density, aspect, k_fl, rho_fl, k_min, g_min, rho_min)
    )
    lam, mu, u1, u3 = _hudson_kernel(k, g, kfl, ar)

    s2 = float(sigma) ** 2
    ex = np.exp(1.0 / s2)
    e11 = (-1.0 + 2.0 * s2 * ex - 2.0 * s2**2 * (ex - 1.0)) / (2.0 * (ex - 1.0))
    e1111 = (
        3.0
        / 8.0
        * (-1.0 + 4.0 * s2**2 * (2.0 * ex + 1.0) - 24.0 * s2**3 * ex + 24.0 * s2**4 * (ex - 1.0))
        / (ex - 1.0)
    )

    e22 = e11
    e33 = 1.0 - 2.0 * e11
    e1122 = e1111 / 3.0
    e1212 = e1111 / 3.0
    e3333 = 8.0 / 3.0 * e1111 - 4.0 * e11 + 1.0
    e1133 = e11 - 4.0 / 3.0 * e1111
    e1313 = e1133
    e2323 = e1133

    c1111 = 4.0 * cd * mu * u1 * (e1111 - e11) - cd / mu * u3 * (
        lam**2 + 4.0 * lam * mu * e11 + 4.0 * mu**2 * e1111
    )
    c1122 = 4.0 * cd * mu * u1 * e1122 - cd / mu * u3 * (
        lam**2 + 2.0 * lam * mu * (e22 + e11) + 4.0 * mu**2 * e1212
    )
    c1133 = 4.0 * cd * mu * u1 * e1133 - cd / mu * u3 * (
        lam**2 + 2.0 * lam * mu * (e33 + e11) + 4.0 * mu**2 * e1133
    )
    c3333 = 4.0 * cd * mu * u1 * (e3333 - e33) - cd / mu * u3 * (
        lam**2 + 4.0 * lam * mu * e33 + 4.0 * mu**2 * e3333
    )
    # Bug fix vs hudsonF.m: the U3 terms of the shear components come from
    # <M_ij M_kl> with M_ij = lam*d_ij + 2*mu*n_i*n_j, i.e. 4*mu^2*<nnnn>,
    # so the prefactor is (cd/mu)*u3*4*mu^2 = 4*cd*mu*u3. The MATLAB wrote
    # 4*cd/mu*u3 (mu^2 dropped), which breaks the exact TI symmetry that
    # any orientation-averaged medium must have (c66 = (c11-c12)/2).
    c2323 = cd * mu * u1 * (4.0 * e2323 - e22 - e33) - 4.0 * cd * mu * u3 * e2323
    c1313 = cd * mu * u1 * (4.0 * e1313 - e11 - e33) - 4.0 * cd * mu * u3 * e1313
    c1212 = cd * mu * u1 * (4.0 * e1212 - e11 - e22) - 4.0 * cd * mu * u3 * e1212

    shape = np.broadcast(cd, ar, lam).shape
    c = np.zeros(shape + (6, 6))
    c[..., 0, 0] = c[..., 1, 1] = lam + 2.0 * mu + c1111
    c[..., 0, 1] = c[..., 1, 0] = lam + c1122
    c[..., 0, 2] = c[..., 2, 0] = lam + c1133
    c[..., 1, 2] = c[..., 2, 1] = lam + c1133
    c[..., 2, 2] = lam + 2.0 * mu + c3333
    c[..., 3, 3] = mu + c2323
    c[..., 4, 4] = mu + c1313
    c[..., 5, 5] = mu + c1212

    phi = (4.0 * np.pi / 3.0) * ar * cd
    rho_out = (1.0 - phi) * rho + phi * rhofl
    return c, rho_out


def hudson_cone(crack_density, aspect, k_fl, k_min, g_min, rho, theta_deg, axis=1):
    """Hudson model for crack normals on a cone about the symmetry axis.

    Crack normals are randomly distributed at the fixed angle `theta_deg`
    from the TI symmetry axis.

    Parameters
    ----------
    crack_density : array_like
        Crack density.
    aspect : array_like
        Aspect ratio of the cracks.
    k_fl : array_like
        Bulk modulus of the fluid in the cracks (0 for dry).
    k_min, g_min : array_like
        Bulk and shear moduli of the isotropic matrix.
    rho : array_like
        Bulk density of the cracked rock.
    theta_deg : array_like
        Angle between the crack normals and the symmetry axis, in degrees
        (the MATLAB took radians).
    axis : {1, 3}, optional
        Symmetry-axis direction. Default 1, as in the MATLAB.

    Returns
    -------
    HudsonVelocities
        Named tuple ``(vp0, vs0, epsilon, gamma, delta, c)``.

    Notes
    -----
    Port of ``hudsoncone.m``. At ``theta_deg = 0`` this reduces exactly to
    the aligned-crack `hudson` model.
    """
    ec, ar, kfl, k, g, rho = (
        np.asarray(a, float) for a in (crack_density, aspect, k_fl, k_min, g_min, rho)
    )
    t = np.deg2rad(np.asarray(theta_deg, float))
    lam, mu, u1, u3 = _hudson_kernel(k, g, kfl, ar)
    st2 = np.sin(t) ** 2
    ct2 = np.cos(t) ** 2

    c11cor = (
        -ec
        / mu
        / 2.0
        * (
            u3 * (2.0 * lam**2 + 4.0 * lam * mu * st2 + 3.0 * mu**2 * st2**2)
            + u1 * mu**2 * st2 * (4.0 - 3.0 * st2)
        )
    )
    c33cor = -ec / mu * (u3 * (lam + 2.0 * mu * ct2) ** 2 + u1 * mu**2 * 4.0 * ct2 * st2)
    c13cor = (
        -ec / mu * (u3 * (lam + mu * st2) * (lam + 2.0 * mu * ct2) - u1 * mu**2 * 2.0 * st2 * ct2)
    )
    c44cor = -ec / 2.0 * mu * (u3 * 4.0 * st2 * ct2 + u1 * (st2 + 2.0 * ct2 - 4.0 * st2 * ct2))
    c66cor = -ec / 2.0 * mu * (u3 * 4.0 * st2**2 + u1 * st2 * (2.0 - st2))

    c11 = lam + 2.0 * mu + c11cor
    c13 = lam + c13cor
    c33 = lam + 2.0 * mu + c33cor
    c44 = mu + c44cor
    c66 = mu + c66cor

    # As in the MATLAB, the c12 slot of the matrix is filled with
    # c11 - 2*c66 (TI identity); the c12cor printed formula is computed but
    # never used there, and disagrees with c11 - 2*c66 at nonzero angle by
    # a term of order mu^2 sin^4(theta) U3 — one of the two originals
    # carries a typo. The port follows the MATLAB matrix assembly.
    c = _hudson_ti_matrix(c11, c13, c33, c44, c66, axis)

    return HudsonVelocities(
        vp0=np.sqrt(c33 / rho),
        vs0=np.sqrt(c44 / rho),
        epsilon=(c11 - c33) / (2.0 * c33),
        gamma=(c66 - c44) / (2.0 * c44),
        delta=((c13 + c44) ** 2 - (c33 - c44) ** 2) / (2.0 * c33 * (c33 - c44)),
        c=c,
    )


class EshelbyCheng(NamedTuple):
    """Eshelby-Cheng TI stiffnesses, packed as in ``echeng.m``."""

    c11: np.ndarray
    c13: np.ndarray
    c33: np.ndarray
    c44: np.ndarray
    c66: np.ndarray


def eshelby_cheng(c11, c13, c33, c44, c66, phi, aspect, k_fl):
    """Eshelby-Cheng TI stiffnesses for a single aligned crack set.

    Valid for all aspect ratios below 1 (unlike Hudson's small-aspect
    theory). The correction terms assume an isotropic background with
    ``lambda = c13`` and ``mu = c44``; the background constants themselves
    may be mildly TI.

    Parameters
    ----------
    c11, c13, c33, c44, c66 : array_like
        Background stiffnesses. For an isotropic background:
        ``c11 = c33 = lambda + 2*mu``, ``c13 = lambda``,
        ``c44 = c66 = mu``.
    phi : array_like
        Crack porosity (not crack density).
    aspect : array_like
        Aspect ratio of the cracks (must be < 1).
    k_fl : array_like
        Bulk modulus of the fluid in the cracks (0 for dry).

    Returns
    -------
    EshelbyCheng
        Named tuple ``(c11, c13, c33, c44, c66)`` of the cracked-rock TI
        stiffnesses (symmetry axis along x3).

    Notes
    -----
    Port of ``echeng.m`` (packed input ``[c11 c13 c33 c44 c66]`` — note
    this ordering differs from the ``[c11 c33 c44 c66 c13]`` convention of
    other RPHtools functions).

    References
    ----------
    Cheng, C. H., 1978, 1993; The Rock Physics Handbook, Eshelby-Cheng
    section.
    """
    c11, c13, c33, c44, c66, phi, a, kfl = (
        np.asarray(v, float) for v in (c11, c13, c33, c44, c66, phi, aspect, k_fl)
    )
    lam = c13
    mu = c44
    k = lam + 2.0 / 3.0 * mu
    cf = kfl / (3.0 * (k - kfl))

    sig = (3.0 * k - 2.0 * mu) / (6.0 * k + 2.0 * mu)
    r = (1.0 - 2.0 * sig) / (8.0 * np.pi * (1.0 - sig))
    q = 3.0 * r / (1.0 - 2.0 * sig)
    sa = np.sqrt(1.0 - a**2)
    ia = 2.0 * np.pi * a * (np.arccos(a) - a * sa) / sa**3
    ic = 4.0 * np.pi - 2.0 * ia
    iac = (ic - ia) / (3.0 * sa**2)
    iaa = np.pi - 0.75 * iac
    iab = iaa / 3.0

    s1313 = 0.5 * q * iac * (1.0 - a**2) + 0.5 * r * (ia + ic)
    s1212 = q * iab + r * ia
    s31 = q * iac - r * ic
    s13 = q * iac * a**2 - r * ia
    s12 = q * iab - r * ia
    s33 = q * (4.0 * np.pi / 3.0 - 2.0 * iac * a**2) + ic * r
    s11 = q * iaa + r * ia

    e = s33 * s11 - s31 * s13 - (s33 + s11 - 2.0 * cf - 1.0) + cf * (s31 + s13 - s11 - s33)
    d = (
        s33 * s11
        + s33 * s12
        - 2.0 * s31 * s13
        - (s11 + s12 + s33 - 1.0 - 3.0 * cf)
        - cf * (s11 + s12 + 2.0 * (s33 - s13 - s31))
    )

    dc11 = lam * (s31 - s33 + 1.0) + 2.0 * mu * e / (d * (s12 - s11 + 1.0))
    dc13 = (
        (lam + 2.0 * mu) * (s13 + s31) - 4.0 * mu * cf + lam * (s13 - s12 - s11 - s33 + 2.0)
    ) / (2.0 * d)
    dc33 = ((lam + 2.0 * mu) * (-s12 - s11 + 1.0) + 2.0 * lam * s13 + 4.0 * mu * cf) / d
    dc44 = mu / (1.0 - 2.0 * s1313)
    dc66 = mu / (1.0 - 2.0 * s1212)

    return EshelbyCheng(
        c11=c11 - phi * dc11,
        c13=c13 - phi * dc13,
        c33=c33 - phi * dc33,
        c44=c44 - phi * dc44,
        c66=c66 - phi * dc66,
    )
