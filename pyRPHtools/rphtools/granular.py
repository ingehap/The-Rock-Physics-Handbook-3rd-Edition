"""Granular-media models: Hertz-Mindlin, contact cement, stress-induced anisotropy.

Ports of the following RPHtools MATLAB functions:

=================  ============================  ==============================
MATLAB             Python                        Notes
=================  ============================  ==============================
``hertzmind.m``    `hertz_mindlin`               Shared core with the velocity
                                                 form.
``hertzmindv.m``   `hertz_mindlin_v`             Uses the reconstructed
                                                 ``v2ku`` from `rphtools.moduli`.
``Cem.m``          `contact_cement`              Dialog/plot stripped.
``Johnson.m``      `johnson_stress_anisotropy`   Returns the stiffness tensor
                                                 the MATLAB overwrote.
``John_Makse.m``   `johnson_makse`               Reconstructed: the MATLAB was
                                                 not runnable as shipped.
(missing)          `unconsolidated`              Reconstruction of ``Unconsol``,
                                                 absent from RPHtools.
=================  ============================  ==============================

Behavior notes (deliberate changes from MATLAB, see PORTING_PLAN.md):

- **Bug fix**: ``Johnson.m`` builds the 6x6 stiffness tensor in a variable
  named ``C``, then overwrites it with the *scalar* elastic constant
  ``C = (1/4pi)(1/mu - 1/(lambda+mu))`` used for the stress formulas.
  Its documented "Cijkl anisotropic stiffness tensor" output was therefore
  that scalar. `johnson_stress_anisotropy` returns the actual tensor.
- **Reconstruction**: ``John_Makse.m`` could not run as shipped — it used
  the coordination number ``Z`` two lines before assigning it, and assigned
  ``C(1,2,:) = C12`` without ever computing ``C12``. `johnson_makse`
  initializes the iteration at ``Z = 6`` (the value the MATLAB set) and
  takes ``C12 = C11 - 2*C66`` from ``Johnson.m``.
- `contact_cement` uses ``pi`` where the MATLAB hard-coded ``3.14`` in the
  cement stiffness parameters (the Handbook's Lambda definitions use pi);
  results differ by under 0.1%.
- The porosity-coordination number table of ``hertzmind.m`` (Handbook
  p. 150) is a module constant, `COORDINATION_TABLE`; porosities outside
  its 0.2-0.7 range interpolate to NaN, as MATLAB's ``interp1`` did.

References
----------
Mindlin, R. D., 1949. Dvorkin, J., and Nur, A., 1996, Elasticity of
high-porosity sandstones: Geophysics, 61, 1363-1370.
Norris, A. N., and Johnson, D. L., 1997: ASME J. Appl. Mech., 64, 39-49.
Johnson, D. L., et al., 1998: Trans. ASME, 65, 380-388.
Makse, H. A., et al., 1999: Phys. Rev. Lett., 83, 5070-5073.
The Rock Physics Handbook, granular-media chapter.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from .moduli import moduli_to_velocity, velocity_to_moduli

__all__ = [
    "COORDINATION_TABLE",
    "ContactCement",
    "HertzMindlin",
    "HertzMindlinVelocity",
    "JohnsonMakse",
    "JohnsonResult",
    "coordination_number",
    "contact_cement",
    "hertz_mindlin",
    "hertz_mindlin_v",
    "johnson_makse",
    "johnson_stress_anisotropy",
    "unconsolidated",
    "UnconsolidatedSand",
]

#: Porosity-coordination number relation from The Rock Physics Handbook,
#: p. 150: ``(porosity, average number of grain contacts)``.
COORDINATION_TABLE = (
    np.arange(0.2, 0.7001, 0.05),
    np.array(
        [14.007, 12.336, 10.843, 9.5078, 8.3147, 7.2517, 6.3108, 5.4878, 4.7826, 4.1988, 3.7440]
    ),
)

_DEFAULT_PHI = np.arange(0.2, 0.7001, 0.05)


def coordination_number(phi):
    """Average number of grain contacts at a given porosity.

    Linear interpolation of the Handbook's p. 150 table.

    Parameters
    ----------
    phi : array_like
        Porosity, within 0.2 to 0.7.

    Returns
    -------
    ndarray
        Coordination number; NaN outside the tabulated range (matching
        MATLAB's ``interp1``).
    """
    por, c = COORDINATION_TABLE
    return np.interp(np.asarray(phi, float), por, c, left=np.nan, right=np.nan)


def _hertz_mindlin_moduli(k_min, g_min, pressure, phi, coord):
    """Shared Hertz-Mindlin core: dry-pack bulk and shear moduli."""
    nu = (3.0 * k_min - 2.0 * g_min) / (6.0 * k_min + 2.0 * g_min)
    k = (
        (coord**2 * (1.0 - phi) ** 2 * g_min**2) / (18.0 * np.pi**2 * (1.0 - nu) ** 2) * pressure
    ) ** (1.0 / 3.0)
    g = (
        (5.0 - 4.0 * nu)
        / (5.0 * (2.0 - nu))
        * (
            (3.0 * coord**2 * (1.0 - phi) ** 2 * g_min**2)
            / (2.0 * np.pi**2 * (1.0 - nu) ** 2)
            * pressure
        )
        ** (1.0 / 3.0)
    )
    return k, g


class HertzMindlin(NamedTuple):
    """Hertz-Mindlin dry sphere-pack moduli."""

    k: np.ndarray
    """Dry-pack bulk modulus (same units as the mineral moduli)."""
    g: np.ndarray
    """Dry-pack shear modulus."""
    phi: np.ndarray
    """Porosity."""
    coord: np.ndarray
    """Coordination number used."""


def hertz_mindlin(k_min, g_min, pressure, phi=None, coord=None):
    """Bulk and shear moduli of a dry sphere pack (Hertz-Mindlin).

    Valid for a random identical-sphere pack under hydrostatic pressure.

    Parameters
    ----------
    k_min, g_min : float
        Mineral bulk and shear moduli.
    pressure : array_like
        Effective (confining) pressure, in the same units as the moduli.
    phi : array_like, optional
        Porosity. Defaults to 0.2 to 0.7 in steps of 0.05.
    coord : array_like, optional
        Coordination number. Defaults to `coordination_number(phi)`.

    Returns
    -------
    HertzMindlin
        Named tuple ``(k, g, phi, coord)``.

    Notes
    -----
    Port of ``hertzmind.m``, including its 2005 correction of the
    ``(1 - nu)^2`` denominators (Handbook p. 151).
    """
    phi = _DEFAULT_PHI if phi is None else np.asarray(phi, float)
    coord = coordination_number(phi) if coord is None else np.asarray(coord, float)
    k, g = _hertz_mindlin_moduli(k_min, g_min, np.asarray(pressure, float), phi, coord)
    return HertzMindlin(k=k, g=g, phi=phi, coord=coord)


class HertzMindlinVelocity(NamedTuple):
    """Hertz-Mindlin dry sphere-pack velocities."""

    vp: np.ndarray
    """Dry-pack P velocity."""
    vs: np.ndarray
    """Dry-pack S velocity."""
    rho: np.ndarray
    """Dry-pack bulk density, ``(1 - phi) rho_min``."""
    phi: np.ndarray
    """Porosity."""
    coord: np.ndarray
    """Coordination number used."""


def hertz_mindlin_v(vp_min, vs_min, rho_min, pressure, phi=None, coord=None):
    """P and S velocities of a dry sphere pack (Hertz-Mindlin).

    Parameters
    ----------
    vp_min, vs_min, rho_min : float
        Mineral P velocity, S velocity, and density.
    pressure : array_like
        Effective pressure, in units consistent with the mineral
        properties.
    phi : array_like, optional
        Porosity. Defaults to 0.2 to 0.7 in steps of 0.05.
    coord : array_like, optional
        Coordination number. Defaults to `coordination_number(phi)`.

    Returns
    -------
    HertzMindlinVelocity
        Named tuple ``(vp, vs, rho, phi, coord)``.

    Notes
    -----
    Port of ``hertzmindv.m``, which called the missing ``v2ku``; this uses
    `rphtools.moduli.velocity_to_moduli`, the reconstruction of it.
    """
    phi = _DEFAULT_PHI if phi is None else np.asarray(phi, float)
    coord = coordination_number(phi) if coord is None else np.asarray(coord, float)
    k_min, g_min = velocity_to_moduli(vp_min, vs_min, rho_min)
    k, g = _hertz_mindlin_moduli(k_min, g_min, np.asarray(pressure, float), phi, coord)
    rho = (1.0 - phi) * rho_min
    vp, vs = moduli_to_velocity(k, g, rho)
    return HertzMindlinVelocity(vp=vp, vs=vs, rho=rho, phi=phi, coord=coord)


class ContactCement(NamedTuple):
    """Dvorkin contact-cement model results versus porosity."""

    phi: np.ndarray
    """Porosity of the cemented rock."""
    m_sat: np.ndarray
    """Saturated P-wave modulus (Gassmann-substituted)."""
    g_frame: np.ndarray
    """Dry-frame shear modulus (unaffected by the fluid)."""
    k_frame: np.ndarray
    """Dry-frame bulk modulus."""
    k_sat: np.ndarray
    """Saturated bulk modulus."""
    k_solid: np.ndarray
    """Hill-average bulk modulus of the grain-plus-cement solid phase."""
    g_solid: np.ndarray
    """Hill-average shear modulus of the solid phase."""


def contact_cement(
    phi_c=0.38,
    coord=8.5,
    g_grain=45.0,
    nu_grain=0.064,
    g_cement=45.0,
    nu_cement=0.064,
    k_fluid=0.0,
    scheme=2,
    phi=None,
):
    """Dvorkin's contact-cement model for cemented sphere packs.

    Cement deposited at grain contacts (scheme 1) or evenly over the grain
    surface (scheme 2) stiffens the pack dramatically for small cement
    volumes.

    Parameters
    ----------
    phi_c : float, optional
        Critical (depositional) porosity of the uncemented pack.
    coord : float, optional
        Coordination number.
    g_grain, nu_grain : float, optional
        Grain shear modulus and Poisson's ratio.
    g_cement, nu_cement : float, optional
        Cement shear modulus and Poisson's ratio.
    k_fluid : float, optional
        Pore-fluid bulk modulus (0 for dry).
    scheme : {1, 2}, optional
        Cementation scheme: 1 places cement at the grain contacts, 2
        spreads it over the grain surface. Default 2, as in the MATLAB
        dialog.
    phi : array_like, optional
        Porosities to evaluate. Defaults to the MATLAB's 100-point sweep
        from `phi_c` down toward 0.15.

    Returns
    -------
    ContactCement
        Named tuple ``(phi, m_sat, g_frame, k_frame, k_sat, k_solid,
        g_solid)``; moduli are in the same units as the input moduli (GPa
        by convention).

    Notes
    -----
    Port of ``Cem.m`` with its ``inputdlg``/plotting stripped. The MATLAB
    hard-coded ``3.14`` in the two Lambda parameters; this uses ``pi``, per
    the Handbook definitions (a sub-0.1% difference).

    References
    ----------
    Dvorkin, J., and Nur, A., 1996: Geophysics, 61, 1363-1370;
    The Rock Physics Handbook, section 5.2.
    """
    if scheme not in (1, 2):
        raise ValueError("scheme must be 1 (cement at contacts) or 2 (cement on grain surface)")
    if phi is None:
        i = np.arange(1, 101)
        phi = phi_c - (i - 1) * (phi_c - 0.15) / 100.0
    else:
        phi = np.asarray(phi, float)

    k_grain = g_grain * 2.0 * (1.0 + nu_grain) / (3.0 * (1.0 - 2.0 * nu_grain))
    k_cement = g_cement * 2.0 * (1.0 + nu_cement) / (3.0 * (1.0 - 2.0 * nu_cement))

    # Hill (Voigt-Reuss) average of grain and cement -> solid-phase moduli.
    f_grain = (1.0 - phi_c) / (1.0 - phi)
    f_cement = (phi_c - phi) / (1.0 - phi)
    k_solid = (
        f_grain * k_grain + f_cement * k_cement + 1.0 / (f_grain / k_grain + f_cement / k_cement)
    ) / 2.0
    g_solid = (
        f_grain * g_grain + f_cement * g_cement + 1.0 / (f_grain / g_grain + f_cement / g_cement)
    ) / 2.0

    # Cement-layer radius, normalized by the grain radius.
    if scheme == 1:
        a = 2.0 * ((phi_c - phi) / (3.0 * coord * (1.0 - phi_c))) ** 0.25
    else:
        a = np.sqrt(2.0 * (phi_c - phi) / (3.0 * (1.0 - phi_c)))

    lam_n = (
        (2.0 / np.pi)
        * (g_cement / g_grain)
        * (1.0 - nu_grain)
        * (1.0 - nu_cement)
        / (1.0 - 2.0 * nu_cement)
    )
    lam_tau = (1.0 / np.pi) * (g_cement / g_grain)

    s_n = (
        -0.024153 * lam_n**-1.3646 * a**2
        + 0.20405 * lam_n**-0.89008 * a
        + 0.00024649 * lam_n**-1.9864
    )
    k_frame = (k_cement + 4.0 * g_cement / 3.0) * (coord * (1.0 - phi_c) / 6.0) * s_n

    nu = nu_grain
    a1t = -0.01 * (2.2606 * nu**2 + 2.0696 * nu + 2.2952)
    a2t = 0.079011 * nu**2 + 0.17539 * nu - 1.3418
    b1t = 0.05728 * nu**2 + 0.09367 * nu + 0.20162
    b2t = 0.027425 * nu**2 + 0.052859 * nu - 0.87653
    c1t = 0.0001 * (9.6544 * nu**2 + 4.9445 * nu + 3.1008)
    c2t = 0.018667 * nu**2 + 0.4011 * nu - 1.8186
    s_tau = (a1t * lam_tau**a2t) * a**2 + (b1t * lam_tau**b2t) * a + c1t * lam_tau**c2t
    g_frame = 0.6 * k_frame + g_cement * (3.0 * coord * (1.0 - phi_c) / 20.0) * s_tau

    # Gassmann saturation of the cemented frame.
    k_sat = (
        k_solid
        * (phi * k_frame - (1.0 + phi) * k_fluid * k_frame / k_solid + k_fluid)
        / ((1.0 - phi) * k_fluid + phi * k_solid - k_fluid * k_frame / k_solid)
    )
    m_sat = k_sat + 4.0 * g_frame / 3.0

    return ContactCement(
        phi=phi,
        m_sat=m_sat,
        g_frame=g_frame,
        k_frame=k_frame,
        k_sat=k_sat,
        k_solid=k_solid,
        g_solid=g_solid,
    )


def _johnson_stiffness(mu, poisson, n, phi, epsilon, e3, cn):
    """Shared Norris-Johnson TI stiffness from the contact integrals."""
    ct = 8.0 * mu / (2.0 - poisson)
    gamma = (3.0 / 32.0) * n * cn * ct * (1.0 - phi) * (-epsilon) ** 0.5
    alfa = np.sqrt(epsilon / e3)
    bw = 2.0 / (np.pi * cn)
    cw = (4.0 / np.pi) * (1.0 / ct - 1.0 / cn)

    root = np.sqrt(1.0 + alfa**2)
    i0 = 0.5 * (root + alfa**2 * np.log((1.0 + root) / alfa))
    i2 = 0.25 * ((1.0 + alfa**2) ** 1.5 - alfa**2 * i0)
    i4 = (1.0 / 6.0) * ((1.0 + alfa**2) ** 1.5 - 3.0 * alfa**2 * i0)

    scale = gamma / alfa
    c11 = scale * (2.0 * bw * (i0 - i2) + (3.0 * cw / 4.0) * (i0 - 2.0 * i2 + i4))
    c13 = scale * (cw * (i2 - i4))
    c33 = scale * (4.0 * bw * i2 + 2.0 * cw * i4)
    c44 = scale * ((bw / 2.0) * (i0 + i2) + cw * (i2 - i4))
    c66 = scale * (bw * (i0 - i2) + (cw / 4.0) * (i0 - 2.0 * i2 + i4))
    c12 = c11 - 2.0 * c66

    c11b, c12b, c13b, c33b, c44b, c66b = np.broadcast_arrays(c11, c12, c13, c33, c44, c66)
    c = np.zeros(np.shape(c11b) + (6, 6))
    c[..., 0, 0] = c[..., 1, 1] = c11b
    c[..., 2, 2] = c33b
    c[..., 0, 1] = c[..., 1, 0] = c12b
    c[..., 0, 2] = c[..., 2, 0] = c13b
    c[..., 1, 2] = c[..., 2, 1] = c13b
    c[..., 3, 3] = c[..., 4, 4] = c44b
    c[..., 5, 5] = c66b
    return c, c11, c33


def _johnson_stress_constants(mu, poisson):
    """Hertz-contact constants B and C used by the stress formulas."""
    lam = mu * (2.0 * poisson / (1.0 - 2.0 * poisson))
    b = (1.0 / (4.0 * np.pi)) * (1.0 / mu + 1.0 / (lam + mu))
    c = (1.0 / (4.0 * np.pi)) * (1.0 / mu - 1.0 / (lam + mu))
    return b, c


def _johnson_stresses(mu, poisson, n, phi, e3):
    """Axial and transverse stresses of the uniaxially strained pack."""
    b, c = _johnson_stress_constants(mu, poisson)
    common = ((-e3) ** 1.5) * (1.0 - phi) * n / (b * (2.0 * b + c))
    sigma3 = -common * (3.0 * b + c) / (6.0 * np.pi**2)
    sigma1 = -common * c / (24.0 * np.pi**2)
    return sigma1, sigma3


class JohnsonResult(NamedTuple):
    """Norris-Johnson stress-induced anisotropy of a random sphere pack."""

    vp1: np.ndarray
    """P velocity perpendicular to the applied stress."""
    vp3: np.ndarray
    """P velocity along the applied stress."""
    sigma1: np.ndarray
    """Induced stress perpendicular to the applied stress."""
    sigma3: np.ndarray
    """Applied axial stress."""
    c: np.ndarray
    """TI stiffness tensor, shape ``(..., 6, 6)``."""


def johnson_stress_anisotropy(mu, poisson, n, phi, epsilon, e3, rho, cn=None):
    """Norris-Johnson stress-induced anisotropy of a random sphere pack.

    Uniaxial strain applied to a random pack of identical spheres produces
    a transversely isotropic effective medium.

    Parameters
    ----------
    mu : float
        Grain shear modulus.
    poisson : float
        Grain Poisson's ratio.
    n : float
        Number of contacts per grain (coordination number).
    phi : float
        Initial porosity.
    epsilon : array_like
        Hydrostatic strain (negative in compression).
    e3 : array_like
        Axial strain (negative in compression).
    rho : float
        Density of the pack.
    cn : float, optional
        Normal contact stiffness, used as an adjustment parameter.
        Defaults to the theoretical ``4 mu / (1 - poisson)``.

    Returns
    -------
    JohnsonResult
        Named tuple ``(vp1, vp3, sigma1, sigma3, c)``.

    Notes
    -----
    Port of ``Johnson.m``. The MATLAB overwrote its stiffness-tensor output
    with a scalar contact constant (both were named ``C``); this returns
    the tensor. The grain size the MATLAB accepted was never used and is
    not a parameter here.

    The contact integrals are meant for a uniaxial increment comparable to
    the hydrostatic strain. When ``epsilon`` greatly exceeds ``e3`` the
    near-cancellation in the ``I2``/``I4`` integrals makes individual
    stiffnesses go negative; check that the returned tensor is positive
    definite before trusting it far from that regime.

    References
    ----------
    Norris, A. N., and Johnson, D. L., 1997: ASME J. Appl. Mech., 64,
    39-49; Johnson, D. L., et al., 1998: Trans. ASME, 65, 380-388.
    """
    epsilon = np.asarray(epsilon, float)
    e3 = np.asarray(e3, float)
    cn = 4.0 * mu / (1.0 - poisson) if cn is None else cn

    c, c11, c33 = _johnson_stiffness(mu, poisson, n, phi, epsilon, e3, cn)
    sigma1, sigma3 = _johnson_stresses(mu, poisson, n, phi, e3)
    return JohnsonResult(
        vp1=np.sqrt(c11 / rho), vp3=np.sqrt(c33 / rho), sigma1=sigma1, sigma3=sigma3, c=c
    )


class JohnsonMakse(NamedTuple):
    """Johnson-Makse pack with a stress-dependent coordination number."""

    vp1: np.ndarray
    """P velocity perpendicular to the applied stress."""
    vp3: np.ndarray
    """P velocity along the applied stress."""
    sigma1: np.ndarray
    """Induced stress perpendicular to the applied stress."""
    sigma3: np.ndarray
    """Applied axial stress."""
    n: np.ndarray
    """Converged coordination number."""
    c: np.ndarray
    """TI stiffness tensor, shape ``(..., 6, 6)``."""


def johnson_makse(
    mu,
    poisson,
    phi,
    epsilon,
    e3,
    rho,
    cn=None,
    z0=6.0,
    stress_scale=6e4,
    tol=1e-10,
    max_iter=1000,
):
    """Norris-Johnson model with the Makse stress-dependent coordination.

    Same TI stiffness as `johnson_stress_anisotropy`, but the coordination
    number is not fixed: it grows with mean stress as
    ``n = z0 + (mean_stress / stress_scale)^(1/3)``, solved self-consistently
    against the stresses that ``n`` itself produces.

    Parameters
    ----------
    mu : float
        Grain shear modulus.
    poisson : float
        Grain Poisson's ratio.
    phi : float
        Initial porosity.
    epsilon : float
        Hydrostatic strain (negative in compression).
    e3 : float
        Axial strain (negative in compression).
    rho : float
        Density of the pack.
    cn : float, optional
        Normal contact stiffness. Defaults to ``4 mu / (1 - poisson)``.
    z0 : float, optional
        Base coordination number at vanishing stress (MATLAB: 6).
    stress_scale : float, optional
        Stress scale of the Makse correction (MATLAB: 6e4, in the same
        units as the stresses).
    tol : float, optional
        Convergence tolerance on the axial stress.
    max_iter : int, optional
        Iteration cap.

    Returns
    -------
    JohnsonMakse
        Named tuple ``(vp1, vp3, sigma1, sigma3, n, c)``.

    Notes
    -----
    Reconstruction of ``John_Makse.m``, which could not run as shipped: it
    referenced ``Z`` before assigning it and never computed the ``C12``
    element it stored into the stiffness matrix. The iteration here starts
    from ``n = z0`` and repeats the MATLAB's update to convergence;
    ``C12 = C11 - 2*C66`` follows ``Johnson.m``.

    References
    ----------
    Makse, H. A., Gland, N., Johnson, D. L., and Schwartz, L. M., 1999,
    Why effective medium theory fails in granular materials: Phys. Rev.
    Lett., 83, 5070-5073.
    """
    cn = 4.0 * mu / (1.0 - poisson) if cn is None else cn
    epsilon = np.asarray(epsilon, float)
    e3 = np.asarray(e3, float)

    n = float(z0)
    sigma1, sigma3 = _johnson_stresses(mu, poisson, n, phi, e3)
    for _ in range(max_iter):
        mean_stress = (-sigma3 - 2.0 * sigma1) / 3.0
        n = z0 + (mean_stress / stress_scale) ** (1.0 / 3.0)
        sigma1_new, sigma3_new = _johnson_stresses(mu, poisson, n, phi, e3)
        if np.all(np.abs(sigma3_new - sigma3) < tol):
            sigma1, sigma3 = sigma1_new, sigma3_new
            break
        sigma1, sigma3 = sigma1_new, sigma3_new
    else:
        raise RuntimeError("coordination-number iteration did not converge")

    c, c11, c33 = _johnson_stiffness(mu, poisson, n, phi, epsilon, e3, cn)
    return JohnsonMakse(
        vp1=np.sqrt(c11 / rho),
        vp3=np.sqrt(c33 / rho),
        sigma1=sigma1,
        sigma3=sigma3,
        n=n,
        c=c,
    )


class UnconsolidatedSand(NamedTuple):
    """Dry-frame moduli of an unconsolidated (friable) sand."""

    k: np.ndarray
    """Dry-frame bulk modulus."""
    g: np.ndarray
    """Dry-frame shear modulus."""
    phi: np.ndarray
    """Porosity."""


def unconsolidated(k_min, g_min, pressure, phi=None, phi_c=0.36, coord=None):
    """Dry moduli of an unconsolidated sand (modified Hashin-Shtrikman lower bound).

    Interpolates between the Hertz-Mindlin sphere-pack moduli at the
    critical porosity and the mineral moduli at zero porosity, along the
    modified Hashin-Shtrikman *lower* bound. This is the friable-sand (or
    "unconsolidated") trend: porosity below critical is treated as
    well-sorted grains progressively packed tighter, not as cemented.

    Parameters
    ----------
    k_min, g_min : float
        Mineral bulk and shear moduli.
    pressure : float
        Effective pressure, in the same units as the moduli.
    phi : array_like, optional
        Porosities at which to evaluate, from 0 to `phi_c`. Defaults to
        50 points across that range.
    phi_c : float, optional
        Critical porosity of the uncemented pack. Defaults to 0.36.
    coord : float, optional
        Coordination number at the critical porosity. Defaults to
        `coordination_number(phi_c)`.

    Returns
    -------
    UnconsolidatedSand
        Named tuple ``(k, g, phi)``.

    Notes
    -----
    Reconstruction of ``Unconsol``, which the RPHtools ``Contents.m``
    lists but which is absent from the distribution (nothing called it).
    The model is fixed by its two endpoints, and the implementation is
    verified against both: at ``phi = phi_c`` it returns the
    Hertz-Mindlin moduli exactly, and at ``phi = 0`` the mineral moduli
    exactly. Both are asserted in the test suite.

    References
    ----------
    Dvorkin, J., and Nur, A., 1996: Geophysics, 61, 1363-1370;
    The Rock Physics Handbook, unconsolidated-sand model.
    """
    if phi is None:
        phi = np.linspace(0.0, phi_c, 50)
    phi = np.atleast_1d(np.asarray(phi, float))
    if np.any(phi < 0) or np.any(phi > phi_c):
        raise ValueError("phi must lie between 0 and phi_c")

    coord = float(coordination_number(phi_c)) if coord is None else float(coord)
    k_hm, g_hm = _hertz_mindlin_moduli(k_min, g_min, float(pressure), phi_c, coord)

    frac = phi / phi_c
    k = (
        1.0 / (frac / (k_hm + 4.0 / 3.0 * g_hm) + (1.0 - frac) / (k_min + 4.0 / 3.0 * g_hm))
        - 4.0 / 3.0 * g_hm
    )

    z = (g_hm / 6.0) * (9.0 * k_hm + 8.0 * g_hm) / (k_hm + 2.0 * g_hm)
    g = 1.0 / (frac / (g_hm + z) + (1.0 - frac) / (g_min + z)) - z
    return UnconsolidatedSand(k=k, g=g, phi=phi)
