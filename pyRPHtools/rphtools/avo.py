"""AVO reflectivity, AVO attributes, and elastic impedance.

Ports of the following RPHtools MATLAB functions:

================  =====================  ====================================
MATLAB            Python                 Notes
================  =====================  ====================================
``avopp.m``       `avo_pp`               ``approx`` number -> ``method`` name.
``avops.m``       `avo_ps`               Same.
``avo_abe.m``     `avo_attributes`       Derived from the shared coefficients.
``eimp.m``        `elastic_impedance`    ``angle="reflection"`` (default).
``eimp2.m``       `elastic_impedance`    ``angle="incidence"``.
================  =====================  ====================================

The interface averages and the Zoeppritz coefficient block are computed once
(`_interface_averages`, `_zoeppritz_coeffs`) and shared by the reflectivity
functions, and the AVO attributes are the very coefficients that `avo_pp` and
`avo_ps` multiply by their angle terms — so the curves and the attributes
cannot drift apart. The relations are asserted in the test suite:

- ``avo_pp(..., "shuey")`` is ``A + B1 sin^2 + C (tan^2 - sin^2)``
- ``avo_pp(..., "shuey-castagna")`` is ``A + B2 sin^2``
- ``avo_ps(..., "gonzalez")`` is ``E1 sin``
- ``avo_ps(..., "alejandro-reinaldo")`` is ``E2 sin``

Behavior notes (deliberate changes from MATLAB, see PORTING_PLAN.md):

- ``eimp.m`` and ``eimp2.m`` were near-duplicates differing only in the
  P-to-S branch (reflection- vs incidence-angle parameterization); they are
  one function with an ``angle`` argument.
- The full-Zoeppritz branches return **complex** reflectivity when any angle
  is at or beyond critical, exactly as MATLAB did (its ``sqrt`` of a
  negative number returns complex, promoting the whole array). Real input
  angles below critical give a real array.
- Plotting side effects are removed.

References
----------
Aki, K., and Richards, P. G., 1980, Quantitative Seismology.
Shuey, R. T., 1985, A simplification of the Zoeppritz equations:
Geophysics, 50, 609-614.
Castagna, J. P., and Backus, M. M., 1993, Offset-Dependent Reflectivity.
Whitcombe, D. N., 2002, Elastic impedance normalization: Geophysics, 67,
60-62.
The Rock Physics Handbook, AVO and elastic-impedance sections.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

__all__ = [
    "AVOAttributes",
    "ElasticImpedance",
    "avo_attributes",
    "avo_pp",
    "avo_ps",
    "elastic_impedance",
]

PP_METHODS = ("zoeppritz", "aki-richards", "shuey", "shuey-castagna")
PS_METHODS = (
    "zoeppritz",
    "aki-richards",
    "donati-quadratic",
    "donati-linear",
    "simplified",
    "gonzalez",
    "alejandro-reinaldo",
)


class _Averages(NamedTuple):
    """Averages and contrasts across an interface."""

    rho_a: np.ndarray
    d_rho: np.ndarray
    vp_a: np.ndarray
    d_vp: np.ndarray
    vs_a: np.ndarray
    d_vs: np.ndarray


def _interface_averages(vp1, vs1, rho1, vp2, vs2, rho2):
    """Layer averages and contrasts shared by every AVO approximation."""
    vp1, vs1, rho1, vp2, vs2, rho2 = (
        np.asarray(a, float) for a in (vp1, vs1, rho1, vp2, vs2, rho2)
    )
    return _Averages(
        rho_a=(rho1 + rho2) / 2.0,
        d_rho=rho2 - rho1,
        vp_a=(vp1 + vp2) / 2.0,
        d_vp=vp2 - vp1,
        vs_a=(vs1 + vs2) / 2.0,
        d_vs=vs2 - vs1,
    )


def _zoeppritz_coeffs(vp1, vs1, rho1, vp2, vs2, rho2, theta):
    """The a, b, c, d, E, F, G, H, D block of the Zoeppritz solution.

    Uses complex square roots so post-critical angles behave as in MATLAB.
    """
    p = np.sin(theta) / vp1
    ct1 = np.cos(theta)
    sin2 = np.sin(theta) ** 2
    ct2 = np.emath.sqrt(1.0 - sin2 * vp2**2 / vp1**2)
    cj1 = np.emath.sqrt(1.0 - sin2 * vs1**2 / vp1**2)
    cj2 = np.emath.sqrt(1.0 - sin2 * vs2**2 / vp1**2)

    a = rho2 * (1.0 - 2.0 * vs2**2 * p**2) - rho1 * (1.0 - 2.0 * vs1**2 * p**2)
    b = rho2 * (1.0 - 2.0 * vs2**2 * p**2) + 2.0 * rho1 * vs1**2 * p**2
    c = rho1 * (1.0 - 2.0 * vs1**2 * p**2) + 2.0 * rho2 * vs2**2 * p**2
    d = 2.0 * (rho2 * vs2**2 - rho1 * vs1**2)

    e = b * ct1 / vp1 + c * ct2 / vp2
    f = b * cj1 / vs1 + c * cj2 / vs2
    g = a - d * ct1 * cj2 / (vp1 * vs2)
    h = a - d * ct2 * cj1 / (vp2 * vs1)
    det = e * f + g * h * p**2
    return a, b, c, d, e, f, g, h, det, p, ct1, ct2, cj1, cj2


def _real_if_subcritical(x):
    """Drop a zero imaginary part, mirroring MATLAB's array-wide promotion."""
    return x.real if np.all(x.imag == 0) else x


def _shuey_terms(vp1, vs1, rho1, vp2, vs2, rho2, av):
    """Shuey's intercept, its sin^2 coefficient, and the far-angle term.

    The MATLAB computed the gradient as ``Ax * Ro`` with
    ``Bx = (dVp/Vp) / ((dVp/Vp) + (drho/rho))``, which is 0/0 when the two
    layers are identical. Since ``Bx * Ro`` is identically ``0.5 dVp/Vp``,
    that factor is substituted here: algebraically the same wherever the
    MATLAB was defined, and finite at zero contrast.
    """
    vp1, vs1, vp2, vs2 = (np.asarray(a, float) for a in (vp1, vs1, vp2, vs2))
    poi1 = (0.5 * (vp1 / vs1) ** 2 - 1.0) / ((vp1 / vs1) ** 2 - 1.0)
    poi2 = (0.5 * (vp2 / vs2) ** 2 - 1.0) / ((vp2 / vs2) ** 2 - 1.0)
    poi_a = (poi1 + poi2) / 2.0
    d_poi = poi2 - poi1

    r0 = 0.5 * (av.d_vp / av.vp_a + av.d_rho / av.rho_a)
    far = 0.5 * av.d_vp / av.vp_a  # == Bx * Ro
    ax_r0 = far - 2.0 * (r0 + far) * (1.0 - 2.0 * poi_a) / (1.0 - poi_a)

    gradient = ax_r0 + d_poi / (1.0 - poi_a) ** 2
    return r0, gradient, far


def _shuey_castagna_gradient(av):
    """Castagna's two-term form of Shuey's gradient."""
    return (
        -2.0 * av.vs_a**2 * av.d_rho / (av.vp_a**2 * av.rho_a)
        + 0.5 * av.d_vp / av.vp_a
        - 4.0 * av.vs_a * av.d_vs / av.vp_a**2
    )


def _gonzalez_ps_gradient(av):
    """Gonzalez's P-to-S gradient (the sin(theta) coefficient)."""
    r = av.vs_a / av.vp_a
    return (
        -0.5 * av.d_rho / av.rho_a
        - r * (av.d_rho / av.rho_a + 2.0 * av.d_vs / av.vs_a)
        + r**3 * (0.5 * av.d_rho / av.rho_a + av.d_vs / av.vs_a)
    )


def _alejandro_reinaldo_ps_gradient(vp1, vs1, av):
    """Alejandro and Reinaldo's P-to-S gradient."""
    vp1, vs1 = np.asarray(vp1, float), np.asarray(vs1, float)
    return (
        -2.0
        * (vs1 / vp1)
        * (av.d_rho / av.rho_a * (0.5 + 0.25 * av.vp_a / av.vs_a) + av.d_vs / av.vs_a)
    )


def avo_pp(vp1, vs1, rho1, vp2, vs2, rho2, angles_deg, method="zoeppritz"):
    """P-to-P reflectivity versus angle of incidence for a single interface.

    Parameters
    ----------
    vp1, vs1, rho1 : array_like
        P velocity, S velocity, and density of the upper layer.
    vp2, vs2, rho2 : array_like
        P velocity, S velocity, and density of the lower layer.
    angles_deg : array_like
        Angles of incidence, in degrees.
    method : str, optional
        One of `PP_METHODS`:

        ``'zoeppritz'``
            Exact Zoeppritz solution (default).
        ``'aki-richards'``
            Aki-Richards three-term linearization.
        ``'shuey'``
            Shuey's three-term form (intercept, gradient, far-angle term).
        ``'shuey-castagna'``
            Two-term ``A + B sin^2(theta)`` in Castagna's formulation.

    Returns
    -------
    ndarray
        Reflection coefficient at each angle. The ``'zoeppritz'`` result is
        complex if any angle is at or beyond critical, real otherwise.

    See Also
    --------
    avo_ps : the P-to-S counterpart.
    avo_attributes : intercept and gradient without evaluating a curve.

    Notes
    -----
    Port of ``avopp.m`` (its ``approx`` numbers 1-4 map to the method names
    in the order listed above).
    """
    method = method.lower()
    if method not in PP_METHODS:
        raise ValueError(f"method must be one of {PP_METHODS}, got {method!r}")

    theta = np.deg2rad(np.asarray(angles_deg, float))
    av = _interface_averages(vp1, vs1, rho1, vp2, vs2, rho2)
    vp1a, vs1a = np.asarray(vp1, float), np.asarray(vs1, float)

    if method == "zoeppritz":
        a, b, c, d, e, f, g, h, det, p, ct1, ct2, cj1, cj2 = _zoeppritz_coeffs(
            vp1a,
            vs1a,
            np.asarray(rho1, float),
            np.asarray(vp2, float),
            np.asarray(vs2, float),
            np.asarray(rho2, float),
            theta,
        )
        rpp = (
            (b * ct1 / vp1a - c * ct2 / np.asarray(vp2, float)) * f
            - (a + d * ct1 * cj2 / (vp1a * np.asarray(vs2, float))) * h * p**2
        ) / det
        return _real_if_subcritical(rpp)

    p = np.sin(theta) / vp1a
    if method == "aki-richards":
        return (
            0.5 * (1.0 - 4.0 * p**2 * av.vs_a**2) * av.d_rho / av.rho_a
            + av.d_vp / (2.0 * np.cos(theta) ** 2 * av.vp_a)
            - 4.0 * p**2 * av.vs_a * av.d_vs
        )

    if method == "shuey":
        r0, gradient, far = _shuey_terms(vp1, vs1, rho1, vp2, vs2, rho2, av)
        return r0 + gradient * np.sin(theta) ** 2 + far * (np.tan(theta) ** 2 - np.sin(theta) ** 2)

    # shuey-castagna
    r0 = 0.5 * (av.d_vp / av.vp_a + av.d_rho / av.rho_a)
    return r0 + _shuey_castagna_gradient(av) * np.sin(theta) ** 2


def avo_ps(vp1, vs1, rho1, vp2, vs2, rho2, angles_deg, method="zoeppritz"):
    """P-to-S reflectivity versus angle of incidence for a single interface.

    Parameters
    ----------
    vp1, vs1, rho1 : array_like
        P velocity, S velocity, and density of the upper layer.
    vp2, vs2, rho2 : array_like
        P velocity, S velocity, and density of the lower layer.
    angles_deg : array_like
        Angles of *incidence* (of the P wave), in degrees.
    method : str, optional
        One of `PS_METHODS`:

        ``'zoeppritz'``
            Exact Zoeppritz solution (default).
        ``'aki-richards'``
            Aki-Richards linearization.
        ``'donati-quadratic'``, ``'donati-linear'``
            Donati's (1998) expansions in ``cos(theta)``.
        ``'simplified'``
            The most-reduced linear form.
        ``'gonzalez'``
            Gonzalez's approximation, ``E1 sin(theta)``.
        ``'alejandro-reinaldo'``
            Alejandro and Reinaldo's approximation, ``E2 sin(theta)``.

    Returns
    -------
    ndarray
        Converted-wave reflection coefficient at each angle. The
        ``'zoeppritz'`` result is complex if any angle is at or beyond
        critical, real otherwise.

    Notes
    -----
    Port of ``avops.m`` (its ``approx`` numbers 1-7 map to the method names
    in the order listed above).
    """
    method = method.lower()
    if method not in PS_METHODS:
        raise ValueError(f"method must be one of {PS_METHODS}, got {method!r}")

    theta = np.deg2rad(np.asarray(angles_deg, float))
    av = _interface_averages(vp1, vs1, rho1, vp2, vs2, rho2)
    vp1a, vs1a = np.asarray(vp1, float), np.asarray(vs1, float)
    sin_t, ct1 = np.sin(theta), np.cos(theta)
    p = sin_t / vp1a

    if method == "zoeppritz":
        a, b, c, d, e, f, g, h, det, p, ct1, ct2, cj1, cj2 = _zoeppritz_coeffs(
            vp1a,
            vs1a,
            np.asarray(rho1, float),
            np.asarray(vp2, float),
            np.asarray(vs2, float),
            np.asarray(rho2, float),
            theta,
        )
        vp2a, vs2a = np.asarray(vp2, float), np.asarray(vs2, float)
        rps = (
            -2.0
            * (ct1 / vp1a)
            * (a * b + c * d * ct2 * cj2 / (vp2a * vs2a))
            * p
            * vp1a
            / (vs1a * det)
        )
        return _real_if_subcritical(rps)

    cj1 = np.emath.sqrt(1.0 - sin_t**2 * vs1a**2 / vp1a**2)

    if method == "aki-richards":
        rps = (-p * av.vp_a / (2.0 * cj1)) * (
            (av.d_rho / av.rho_a)
            * (1.0 - 2.0 * av.vs_a**2 * p**2 + 2.0 * av.vs_a**2 * ct1 * cj1 / (av.vp_a * av.vs_a))
            - (av.d_vs / av.vs_a)
            * (4.0 * av.vs_a**2 * p**2 - 4.0 * av.vs_a**2 * ct1 * cj1 / (av.vp_a * av.vs_a))
        )
        return _real_if_subcritical(np.asarray(rps, complex))

    if method in ("donati-quadratic", "donati-linear"):
        a0 = -0.5 * (
            (av.d_rho / av.rho_a) * (1.0 - 2.0 * av.vs_a**2 / av.vp_a**2)
            - 4.0 * av.vs_a * av.d_vs / av.vp_a**2
        )
        a1 = -0.5 * (
            (av.d_rho / av.rho_a + 2.0 * av.d_vs / av.vs_a)
            * (2.0 * av.vs_a / av.vp_a - av.vs_a**3 / av.vp_a**3)
        )
        if method == "donati-linear":
            return sin_t * (a0 + a1 * ct1)
        a2 = -(av.vs_a**2 / av.vp_a**2) * (av.d_rho / av.rho_a + 2.0 * av.d_vs / av.vs_a)
        return sin_t * (a0 + a1 * ct1 + a2 * ct1**2)

    if method == "simplified":
        return -sin_t * (
            av.d_rho / (2.0 * av.rho_a)
            - (av.d_rho / av.rho_a + 2.0 * av.d_vs / av.vs_a) * av.vs_a / av.vp_a
        )

    if method == "gonzalez":
        return sin_t * _gonzalez_ps_gradient(av)

    # alejandro-reinaldo
    return sin_t * _alejandro_reinaldo_ps_gradient(vp1, vs1, av)


class AVOAttributes(NamedTuple):
    """AVO intercept and gradients (field order follows ``avo_abe.m``)."""

    a: np.ndarray
    """Intercept: normal-incidence P-P reflectivity."""
    b1: np.ndarray
    """P-P gradient, Shuey's three-term formulation."""
    b2: np.ndarray
    """P-P gradient, Castagna's two-term formulation."""
    e1: np.ndarray
    """P-S gradient, Gonzalez's approximation."""
    e2: np.ndarray
    """P-S gradient, Alejandro and Reinaldo's approximation."""


def avo_attributes(vp1, vs1, rho1, vp2, vs2, rho2):
    """AVO intercept and gradient attributes for P-P and P-S reflections.

    Parameters
    ----------
    vp1, vs1, rho1 : array_like
        P velocity, S velocity, and density of the upper layer.
    vp2, vs2, rho2 : array_like
        P velocity, S velocity, and density of the lower layer.

    Returns
    -------
    AVOAttributes
        Named tuple ``(a, b1, b2, e1, e2)`` — the intercept, the two P-P
        gradients, and the two P-S gradients.

    See Also
    --------
    avo_pp, avo_ps : the reflectivity curves these attributes parameterize.

    Notes
    -----
    Port of ``avo_abe.m``. The MATLAB recomputed each coefficient inline;
    here they come from the same helpers `avo_pp` and `avo_ps` use, so a
    change to one cannot silently desynchronize the other.
    """
    av = _interface_averages(vp1, vs1, rho1, vp2, vs2, rho2)
    r0, b1, _ = _shuey_terms(vp1, vs1, rho1, vp2, vs2, rho2, av)
    return AVOAttributes(
        a=r0,
        b1=b1,
        b2=_shuey_castagna_gradient(av),
        e1=_gonzalez_ps_gradient(av),
        e2=_alejandro_reinaldo_ps_gradient(vp1, vs1, av),
    )


class ElasticImpedance(NamedTuple):
    """Far-offset elastic impedances, normalized and raw."""

    ipp_n: np.ndarray
    """Normalized P-P elastic impedance (scaled to zero-offset P impedance)."""
    ips_n: np.ndarray
    """Normalized P-S elastic impedance (scaled to zero-offset P impedance)."""
    isp_n: np.ndarray
    """Normalized S-P elastic impedance (scaled to zero-offset S impedance)."""
    ipp: np.ndarray
    """Raw, un-normalized P-P elastic impedance."""
    ips: np.ndarray
    """Raw P-S elastic impedance."""
    isp: np.ndarray
    """Raw S-P elastic impedance."""


def elastic_impedance(vp, vs, rho, theta_deg, k=None, angle="reflection"):
    """Far-offset elastic impedances for P-P, P-S, and S-P waves.

    Parameters
    ----------
    vp, vs, rho : array_like
        P velocity, S velocity, and bulk density (e.g. a log).
    theta_deg : array_like
        Angle in degrees; a scalar, or an array of the same length as `vp`
        for an angle varying with depth. Which angle it is depends on
        `angle` (below); for P-P, incidence and reflection angles coincide.
    k : float, optional
        Constant Vs/Vp ratio to use. Defaults to ``mean(vs / vp)``.
    angle : {'reflection', 'incidence'}, optional
        Parameterization of the P-to-S branch. ``'reflection'`` (default)
        treats `theta_deg` as the reflected S-wave angle (``eimp.m``);
        ``'incidence'`` treats it as the incident P-wave angle
        (``eimp2.m``). The P-P and S-P branches are identical either way.

    Returns
    -------
    ElasticImpedance
        Named tuple ``(ipp_n, ips_n, isp_n, ipp, ips, isp)``.

    Notes
    -----
    Merges ``eimp.m`` and ``eimp2.m``, which differed only in the P-to-S
    branch. Normalization follows Whitcombe (2002): `vp`, `vs`, and `rho`
    are divided by their means and the result is scaled by the zero-offset
    impedance, so **the normalized outputs depend on every sample passed
    in together** — pass a whole log, not one sample at a time.

    With ``angle='reflection'`` the P-S branch requires
    ``sin(theta) < k``; larger angles have no real P-wave counterpart and
    give complex results (as in MATLAB).

    References
    ----------
    Whitcombe, D. N., 2002, Elastic impedance normalization: Geophysics,
    67, 60-62.
    """
    if angle not in ("reflection", "incidence"):
        raise ValueError("angle must be 'reflection' or 'incidence'")
    vp, vs, rho = (np.asarray(a, float) for a in (vp, vs, rho))
    theta = np.deg2rad(np.asarray(theta_deg, float))
    vsvp = float(np.mean(vs / vp)) if k is None else float(k)
    vpvs = 1.0 / vsvp

    vp_n, vs_n, rho_n = vp / np.mean(vp), vs / np.mean(vs), rho / np.mean(rho)
    ip_mean = float(np.mean(vp) * np.mean(rho))
    is_mean = float(np.mean(vs) * np.mean(rho))

    sin_t, cos_t, tan_t = np.sin(theta), np.cos(theta), np.tan(theta)
    vsvp_sin2 = vsvp**2 * sin_t**2

    # --- P-P (Connolly): independent of the angle convention -------------
    x1 = 1.0 + tan_t**2
    x2 = 1.0 - 4.0 * vsvp_sin2
    x3 = -8.0 * vsvp_sin2
    ipp = vp**x1 * rho**x2 * vs**x3
    ipp_n = ip_mean * (vp_n**x1 * rho_n**x2 * vs_n**x3)

    # --- P-S ---------------------------------------------------------------
    if angle == "reflection":
        root = np.emath.sqrt(vsvp**2 - sin_t**2)
        a = tan_t * (2.0 * sin_t**2 - 1.0 - 2.0 * cos_t * root) / vsvp
        b = 4.0 * tan_t * (sin_t**2 - cos_t * root) / vsvp
    else:
        root = np.emath.sqrt(vpvs**2 - sin_t**2)
        common = sin_t / (vpvs * root)
        a = common * (2.0 * sin_t**2 - vpvs**2 - 2.0 * cos_t * root)
        b = 4.0 * common * (sin_t**2 - cos_t * root)
    ips = _real_if_subcritical(np.asarray(rho**a * vs**b, complex))
    ips_n = _real_if_subcritical(np.asarray(ip_mean * (rho_n**a * vs_n**b), complex))

    # --- S-P: same in both conventions -----------------------------------
    root_sp = np.emath.sqrt(1.0 - vsvp_sin2)
    a_sp = vsvp * tan_t * (2.0 * vsvp_sin2 - 1.0 - 2.0 * vsvp * cos_t * root_sp)
    b_sp = 4.0 * vsvp * tan_t * (vsvp_sin2 - vsvp * cos_t * root_sp)
    isp = _real_if_subcritical(np.asarray(rho**a_sp * vs**b_sp, complex))
    isp_n = _real_if_subcritical(np.asarray(is_mean * (rho_n**a_sp * vs_n**b_sp), complex))

    return ElasticImpedance(ipp_n=ipp_n, ips_n=ips_n, isp_n=isp_n, ipp=ipp, ips=ips, isp=isp)
