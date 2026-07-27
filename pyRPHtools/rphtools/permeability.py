"""Empirical and theoretical permeability models.

Ports of the following RPHtools MATLAB functions:

===================  ===============================  ======================
MATLAB               Python                           Notes
===================  ===============================  ======================
``BernabeE.m``       `bernabe_perm`                   Two pore types.
``Bloch.m``          `bloch_perm`                     Returns porosity too.
``CoatDum.m``        `coates_dumanoir_perm`
``Coates.m``         `coates_perm`
``FredrichE.m``      `fredrich_perm`
``KozCarmE.m``       `kozeny_carman_perm`
``ModKozCarm.m``     `modified_kozeny_carman_perm`
``Owolabi.m``        `owolabi_perm`                   Oil and gas sands.
``PandaLake.m``      `panda_lake_perm`
``PandaLakeKCE.m``   `panda_lake_kc_perm`
``PermMenu.m``       `PERM_MODELS`                    Registry dict, no GUI.
===================  ===============================  ======================

Every model returns permeability in millidarcy. Grain/pore sizes are in
micrometres and porosities are fractions (not percent), consistently across
the module.

Behavior notes (deliberate changes from MATLAB, see PORTING_PLAN.md):

- The MATLAB functions were GUI wrappers: with no arguments each popped an
  ``inputdlg``, ``str2num``/``evalin('base')``-ed the answers, drew a
  ``semilogy`` plot with ``hold on``, and returned a horizontally
  concatenated ``[Phi K]`` matrix (which silently became a ``1 x 2N`` row
  for row-vector inputs). The ports are plain functions returning arrays;
  the MATLAB dialog defaults survive as Python default arguments.
- **Bug fix**: ``BernabeE.m`` was unusable non-interactively — it tested
  ``nargin == 5`` for a four-argument function (so it always opened the
  dialog), called its inner ``Ber1`` without the porosity argument, and
  never assigned that call's result to its output. `bernabe_perm` is a
  plain function.
- ``Bloch.m`` returned porosity in *percent* while plotting it as a
  fraction; `bloch_perm` returns a fraction, like every other model here.
- ``PermMenu.m`` dispatched 14 model names through ``feval``, four of which
  (``RevilE``, ``Timur``, ``Tixier``, ``WylGregE``) do not exist anywhere in
  RPHtools and failed at runtime. `PERM_MODELS` contains the ten that do.

References
----------
The Rock Physics Handbook, permeability chapter; individual model
references are given in each function's docstring.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

__all__ = [
    "PERM_MODELS",
    "BernabePerm",
    "BlochPerm",
    "OwolabiPerm",
    "bernabe_perm",
    "bloch_perm",
    "coates_dumanoir_perm",
    "coates_perm",
    "fredrich_perm",
    "kozeny_carman_perm",
    "modified_kozeny_carman_perm",
    "owolabi_perm",
    "panda_lake_kc_perm",
    "panda_lake_perm",
]

_DEFAULT_PHI = np.arange(0.01, 0.3501, 0.01)


class BernabePerm(NamedTuple):
    """Bernabe two-pore-type permeability split."""

    k: np.ndarray
    """Total permeability, ``k_crack + k_tube`` (md)."""
    k_crack: np.ndarray
    """Crack contribution (md)."""
    k_tube: np.ndarray
    """Tube/nodal-pore contribution (md)."""
    phi_crack: np.ndarray
    """Porosity held in cracks."""
    phi_tube: np.ndarray
    """Porosity held in tubes."""


def bernabe_perm(phi=None, crack_fraction=0.8, crack_width=200.0, tube_radius=150.0):
    """Bernabe (1991) two-pore-type permeability model.

    Splits the pore space into strongly pressure-dependent cracks
    (parallel-plate flow) and near-constant tubes/nodal pores, with a
    tortuosity-squared of 2.5 folded into both constants:
    ``k_crack = w^2 phi_crack / 30``, ``k_tube = r^2 phi_tube / 20``.

    Parameters
    ----------
    phi : array_like, optional
        Total porosity. Defaults to ``0.01`` to ``0.35`` in steps of 0.01.
    crack_fraction : float, optional
        Fraction of the pore space held in cracks.
    crack_width : float, optional
        Crack width/aperture w (micrometres).
    tube_radius : float, optional
        Tube radius r (micrometres).

    Returns
    -------
    BernabePerm
        Named tuple ``(k, k_crack, k_tube, phi_crack, phi_tube)`` with
        permeabilities in md.

    Notes
    -----
    Port of ``BernabeE.m``. Valid for clean sandstones; inappropriate for
    rocks with complex microstructure.

    References
    ----------
    Bernabe, Y., 1991, Pore geometry and pressure dependence of the
    transport properties in sandstones: Geophysics, 56, 436-446.
    """
    phi = _DEFAULT_PHI if phi is None else np.asarray(phi, float)
    phi_crack = phi * crack_fraction
    phi_tube = phi - phi_crack
    k_crack = crack_width**2 * phi_crack / 30.0
    k_tube = tube_radius**2 * phi_tube / 20.0
    return BernabePerm(
        k=k_crack + k_tube,
        k_crack=k_crack,
        k_tube=k_tube,
        phi_crack=phi_crack,
        phi_tube=phi_tube,
    )


class BlochPerm(NamedTuple):
    """Bloch predicted porosity and permeability."""

    phi: np.ndarray
    """Predicted porosity (fraction; the MATLAB returned percent)."""
    k: np.ndarray
    """Predicted permeability (md)."""


def bloch_perm(grain_size, sorting, rigid_grain_content):
    """Bloch (1991) empirical porosity and permeability prediction.

    A statistical model for predicting sandstone reservoir quality before
    drilling, from texture alone.

    Parameters
    ----------
    grain_size : array_like
        Grain size (mm).
    sorting : array_like
        Trask sorting coefficient.
    rigid_grain_content : array_like
        Rigid-grain content (percent).

    Returns
    -------
    BlochPerm
        Named tuple ``(phi, k)`` — porosity as a fraction and permeability
        in md.

    Notes
    -----
    Port of ``Bloch.m``. Predictions are limited to the calibration data
    set: samples with less than 5-10% pore-filling cement.

    References
    ----------
    Bloch, S., 1991, Empirical prediction of porosity and permeability in
    sandstones: AAPG Bulletin, 75, 1145-1160.
    """
    size, sort, content = (np.asarray(a, float) for a in (grain_size, sorting, rigid_grain_content))
    phi_percent = -6.1 + 9.8 / sort + 0.17 * content
    k = 10.0 ** (-4.67 + 1.34 * size + 4.08 / sort + 3.42 * content / 100.0)
    return BlochPerm(phi=phi_percent / 100.0, k=k)


def coates_dumanoir_perm(phi=None, swr=0.15):
    """Coates-Dumanoir (1974) log-derived permeability.

    One of the ``k = A phi^B / Swr^C`` family: ``k = 352 phi^4 / Swr^4``.

    Parameters
    ----------
    phi : array_like, optional
        Porosity. Defaults to ``0.01`` to ``0.35`` in steps of 0.01.
    swr : array_like, optional
        Irreducible (residual) water saturation.

    Returns
    -------
    ndarray
        Permeability (md).

    Notes
    -----
    Port of ``CoatDum.m``. Empirical, calibrated for unconsolidated sands.

    References
    ----------
    Coates, G. R., and Dumanoir, J. L., 1974: Log Analyst, 15, 17-31.
    """
    phi = _DEFAULT_PHI if phi is None else np.asarray(phi, float)
    swr = np.asarray(swr, float)
    return 352.0 * phi**4 / swr**4


def coates_perm(phi=None, swr=0.15):
    """Coates et al. (1991) log-derived permeability.

    ``k = 10000 phi^4 (1 - Swr)^2 / Swr^2``.

    Parameters
    ----------
    phi : array_like, optional
        Porosity. Defaults to ``0.01`` to ``0.35`` in steps of 0.01.
    swr : array_like, optional
        Irreducible water saturation.

    Returns
    -------
    ndarray
        Permeability (md).

    Notes
    -----
    Port of ``Coates.m``. Empirical, for unconsolidated sands.

    References
    ----------
    Coates, G. R., et al., 1991: JPT, 43, 578-587.
    """
    phi = _DEFAULT_PHI if phi is None else np.asarray(phi, float)
    swr = np.asarray(swr, float)
    return 10000.0 * phi**4 * (1.0 - swr) ** 2 / swr**2


def fredrich_perm(phi=None, pore_diameter=100.0):
    """Fredrich et al. Kozeny-type permeability from porosity and grain size.

    Uses the formation factor ``F = 2.5 / phi`` and specific surface
    ``Sv = 6(1 - phi)/d``, reducing to
    ``k = (1000/450) d^2 phi^3 / (1 - phi)^2``.

    Parameters
    ----------
    phi : array_like, optional
        Porosity. Defaults to ``0.01`` to ``0.35`` in steps of 0.01.
    pore_diameter : array_like, optional
        Pore diameter d (micrometres).

    Returns
    -------
    ndarray
        Permeability (md).

    Notes
    -----
    Port of ``FredrichE.m``. Assumes the electrical and hydraulic flow
    paths are identical; calibrated on Fontainebleau sandstone and
    intended for porosity above about 10%. Numerically identical to
    `kozeny_carman_perm` — the two differ only in how the constant is
    motivated.

    References
    ----------
    Fredrich, J. T., Greaves, K. H., and Martin, J. W., 1993: Int. J. Rock
    Mech. Min. Sci., 30, 691-697.
    """
    phi = _DEFAULT_PHI if phi is None else np.asarray(phi, float)
    d = np.asarray(pore_diameter, float)
    return (1000.0 / 450.0) * d**2 * phi**3 / (1.0 - phi) ** 2


def kozeny_carman_perm(phi=None, pore_diameter=250.0):
    """Original Kozeny-Carman permeability.

    ``k = (1000/450) d^2 phi^3 / (1 - phi)^2``, i.e. the Carman form with
    the tortuosity fixed at the standard value.

    Parameters
    ----------
    phi : array_like, optional
        Porosity. Defaults to ``0.01`` to ``0.35`` in steps of 0.01.
    pore_diameter : array_like, optional
        Pore diameter d (micrometres).

    Returns
    -------
    ndarray
        Permeability (md).

    Notes
    -----
    Port of ``KozCarmE.m``.
    """
    phi = _DEFAULT_PHI if phi is None else np.asarray(phi, float)
    d = np.asarray(pore_diameter, float)
    return (1000.0 / 450.0) * d**2 * phi**3 / (1.0 - phi) ** 2


def modified_kozeny_carman_perm(phi=None, pore_diameter=60.0, b=2.0, phi_percolation=0.02):
    """Modified Kozeny-Carman with a percolation porosity.

    Only the connected porosity ``phi - phi_percolation`` carries flow:
    ``k = B d^2 phi_x^3 / (1 - phi_x)^2``.

    Parameters
    ----------
    phi : array_like, optional
        Total porosity. Defaults to ``0.01`` to ``0.35`` in steps of 0.01.
    pore_diameter : array_like, optional
        Pore diameter d (micrometres).
    b : array_like, optional
        Geometric factor B.
    phi_percolation : array_like, optional
        Percolation porosity below which the rock does not conduct.

    Returns
    -------
    ndarray
        Permeability (md). Negative connected porosity (below percolation)
        yields unphysical values; mask those yourself if needed.

    Notes
    -----
    Port of ``ModKozCarm.m``.
    """
    phi = _DEFAULT_PHI if phi is None else np.asarray(phi, float)
    d = np.asarray(pore_diameter, float)
    phi_x = phi - np.asarray(phi_percolation, float)
    return b * d**2 * phi_x**3 / (1.0 - phi_x) ** 2


class OwolabiPerm(NamedTuple):
    """Owolabi permeability for oil and gas sands."""

    k_oil: np.ndarray
    """Permeability of oil sands (md)."""
    k_gas: np.ndarray
    """Permeability of gas sands (md)."""


def owolabi_perm(phi, swi):
    """Owolabi et al. (1994) empirical permeability for Niger Delta sands.

    Separate regressions for oil and gas sands:
    ``k_oil = 307 + 26552 phi^2 - 34540 (phi Swi)^2`` and
    ``k_gas = 30.7 + 2655 phi^2 - 3454 (phi Swi)^2``.

    Parameters
    ----------
    phi : array_like
        Porosity.
    swi : array_like
        Irreducible water saturation.

    Returns
    -------
    OwolabiPerm
        Named tuple ``(k_oil, k_gas)`` in md.

    Notes
    -----
    Port of ``Owolabi.m``. Empirical, for unconsolidated Eastern Niger
    Delta sands.

    References
    ----------
    Owolabi, O. O., LongJohn, T. F., and Ajienka, J. A., 1994: Journal of
    Petroleum Geology, 17, 111-116.
    """
    phi, swi = np.asarray(phi, float), np.asarray(swi, float)
    k_oil = 307.0 + 26552.0 * phi**2 - 34540.0 * (phi * swi) ** 2
    k_gas = 30.7 + 2655.0 * phi**2 - 3454.0 * (phi * swi) ** 2
    return OwolabiPerm(k_oil=k_oil, k_gas=k_gas)


def panda_lake_perm(phi=None, tortuosity=2.0, skewness=0.25, mean_grain_size=650.0, cv=0.4):
    """Panda-Lake (1994) permeability from particle-size-distribution stats.

    A Carman-Kozeny equation corrected for the sorting and skewness of the
    particle-size distribution.

    Parameters
    ----------
    phi : array_like, optional
        Porosity. Defaults to ``0.01`` to ``0.35`` in steps of 0.01.
    tortuosity : array_like, optional
        Tortuosity.
    skewness : array_like, optional
        Skewness of the particle-size distribution.
    mean_grain_size : array_like, optional
        Mean particle size (micrometres).
    cv : array_like, optional
        Coefficient of variation of the particle-size distribution.

    Returns
    -------
    ndarray
        Permeability (md).

    Notes
    -----
    Port of ``PandaLake.m``. Assumes a homogeneous, isotropic medium; not
    valid in tight sands.

    References
    ----------
    Panda, M. N., and Lake, L. W., 1994, Estimation of single-phase
    permeability from parameters of particle-size distribution: AAPG
    Bulletin, 78, 1028-1039.
    """
    phi = _DEFAULT_PHI if phi is None else np.asarray(phi, float)
    tau, s, dpm, cdp = (np.asarray(a, float) for a in (tortuosity, skewness, mean_grain_size, cv))
    shape = ((s * cdp**3 + 3.0 * cdp**2 + 1.0) ** 2) / (1.0 + cdp**2) ** 2
    return shape * (dpm**2 * phi**3) / (72.0 * tau * (1.0 - phi) ** 2)


def panda_lake_kc_perm(phi=None, mean_particle_size=250.0):
    """Panda-Lake (1994) basic Kozeny-Carman form for unconsolidated media.

    ``k = 3.34 Dp^2 phi^3 / (1 - phi)^2``, the constant absorbing the
    particle-size-distribution parameter and the unit conversion.

    Parameters
    ----------
    phi : array_like, optional
        Porosity. Defaults to ``0.01`` to ``0.35`` in steps of 0.01.
    mean_particle_size : array_like, optional
        Mean particle size Dp (micrometres).

    Returns
    -------
    ndarray
        Permeability (md).

    Notes
    -----
    Port of ``PandaLakeKCE.m``.
    """
    phi = _DEFAULT_PHI if phi is None else np.asarray(phi, float)
    dp = np.asarray(mean_particle_size, float)
    return 3.34 * dp**2 * phi**3 / (1.0 - phi) ** 2


#: Registry of the permeability models, replacing the ``PermMenu.m`` GUI.
#: Keys are the original MATLAB names; values are the ported callables,
#: grouped as the menu grouped them.
PERM_MODELS = {
    # Porosity / grain size / shape factor inputs
    "BernabeE": bernabe_perm,
    "FredrichE": fredrich_perm,
    "KozCarmE": kozeny_carman_perm,
    "ModKozCarm": modified_kozeny_carman_perm,
    "PandaLakeKCE": panda_lake_kc_perm,
    # Porosity / irreducible-water-saturation inputs
    "CoatDum": coates_dumanoir_perm,
    "Coates": coates_perm,
    # Statistically based
    "Bloch": bloch_perm,
    "Owolabi": owolabi_perm,
    "PandaLake": panda_lake_perm,
}
