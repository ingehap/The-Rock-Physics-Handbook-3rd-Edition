"""Pore-fluid properties: Batzle-Wang relations and CO2 tables.

Ports of the following RPHtools MATLAB functions:

==============  =================  =============================================
MATLAB          Python             Notes
==============  =================  =============================================
``flprop.m``    `batzle_wang`      The ``method`` flag becomes the
                                   ``gas_index_oil`` keyword.
``flpropui.m``  (not ported)       GUI wrapper around a byte-for-byte inline
                                   copy of ``flprop.m``.
``co2prop.m``   `co2_properties`   Interpolates the packaged table converted
                                   from ``co2propdata.mat``; GUI stripped.
==============  =================  =============================================

Units follow the Batzle-Wang conventions used by the MATLAB: pressure in
MPa, temperature in degrees Celsius, salinity in ppm, velocities in km/s
(CO2 velocity in m/s), densities in g/cm^3, moduli in GPa.
"""

from __future__ import annotations

from functools import cache
from importlib.resources import files
from typing import NamedTuple

import numpy as np
from scipy.interpolate import RegularGridInterpolator

__all__ = [
    "FluidProperties",
    "batzle_wang",
    "co2_properties",
]

# Batzle-Wang pure-water velocity coefficients w[i][j] for T^i * P^j
# (Batzle & Wang, 1992, Table after Wilson; MATLAB ``matrixw`` transposed).
_WATER_VEL_COEFFS = np.array(
    [
        [1402.85, 1.524, 3.437e-3, -1.197e-5],
        [4.871, -0.0111, 1.739e-4, -1.628e-6],
        [-0.04783, 2.747e-4, -2.135e-6, 1.237e-8],
        [1.487e-4, -6.503e-7, -1.455e-8, 1.327e-10],
        [-2.197e-7, 7.987e-10, 5.230e-11, -4.614e-13],
    ]
)


class FluidProperties(NamedTuple):
    """Batzle-Wang fluid properties (field order follows ``flprop.m``)."""

    k_reuss: np.ndarray
    """Reuss average bulk modulus of the fluid mix (GPa) — homogeneous
    (uniform) saturation."""
    rho_eff: np.ndarray
    """Effective density of the fluid mix (g/cm^3)."""
    k_voigt: np.ndarray
    """Voigt average bulk modulus of the fluid mix (GPa) — patchy
    saturation."""
    vp_brine: np.ndarray
    """Brine P velocity (km/s)."""
    rho_brine: np.ndarray
    """Brine density (g/cm^3)."""
    k_brine: np.ndarray
    """Brine bulk modulus (GPa)."""
    vp_oil: np.ndarray
    """Oil P velocity (km/s)."""
    rho_oil: np.ndarray
    """Oil density (g/cm^3)."""
    k_oil: np.ndarray
    """Oil bulk modulus (GPa)."""
    vp_gas: np.ndarray
    """Gas P velocity (km/s)."""
    rho_gas: np.ndarray
    """Gas density (g/cm^3)."""
    k_gas: np.ndarray
    """Gas bulk modulus (GPa)."""
    gor: np.ndarray
    """Gas-oil ratio used (L/L) — computed from `gas_index_oil` when that
    was given, otherwise the input value."""


def batzle_wang(
    pressure,
    temperature,
    salinity=35000.0,
    oil_api=30.0,
    gas_gravity=0.6,
    gor=0.0,
    gas_index_brine=0.0,
    gas_index_oil=None,
    s_oil=0.0,
    s_gas=0.0,
):
    """Batzle-Wang properties of reservoir brine, oil, gas, and their mix.

    Parameters
    ----------
    pressure : array_like
        Pore pressure (MPa).
    temperature : array_like
        Temperature (degrees Celsius).
    salinity : array_like, optional
        Brine NaCl salinity (ppm).
    oil_api : array_like, optional
        Oil gravity (API number).
    gas_gravity : array_like, optional
        Gas specific gravity.
    gor : array_like, optional
        Gas-oil ratio (L/L). ``0`` means dead oil. Ignored when
        `gas_index_oil` is given.
    gas_index_brine : array_like, optional
        Fraction (0 to 1) of the maximum dissolvable gas actually dissolved
        in the brine.
    gas_index_oil : array_like, optional
        If given, the gas-oil ratio is computed as this fraction (0 to 1)
        of the maximum dissolvable gas in the oil, overriding `gor`
        (the MATLAB ``method = 1``).
    s_oil, s_gas : array_like, optional
        Oil and gas saturations; brine saturation is ``1 - s_oil - s_gas``.

    Returns
    -------
    FluidProperties
        Named tuple with the mixed-fluid Reuss/Voigt moduli and effective
        density, the per-phase (brine, oil, gas) velocity, density, and
        bulk modulus, and the gas-oil ratio used. Same field order as the
        13 MATLAB outputs.

    Notes
    -----
    Port of ``flprop.m``. The MATLAB scalar ``if gor == 0`` dead/live-oil
    branch is vectorized with ``np.where``, so arrays mixing dead and live
    oils are handled elementwise.

    References
    ----------
    Batzle, M., and Wang, Z., 1992, Seismic properties of pore fluids:
    Geophysics, 57, 1396-1408.
    """
    p = np.asarray(pressure, float)
    t = np.asarray(temperature, float)
    sal = np.asarray(salinity, float) / 1e6
    og = np.asarray(oil_api, float)
    gg = np.asarray(gas_gravity, float)
    gor = np.asarray(gor, float)
    giib = np.asarray(gas_index_brine, float)
    s_oil = np.asarray(s_oil, float)
    s_gas = np.asarray(s_gas, float)

    r_gas = 8.31441  # ideal gas constant

    # --- gas density and adiabatic bulk modulus ---------------------------
    pr = p / (4.892 - 0.4048 * gg)
    tr = (t + 273.15) / (94.72 + 170.75 * gg)
    e = 0.109 * (3.85 - tr) ** 2 * np.exp(-(0.45 + 8.0 * (0.56 - 1.0 / tr) ** 2) * (pr**1.2 / tr))
    z = (0.03 + 0.00527 * (3.5 - tr) ** 3) * pr + (0.642 * tr - 0.007 * tr**4 - 0.52) + e
    rho_gas = 28.8 * gg * p / (z * r_gas * (t + 273.15))

    gamma = 0.85 + 5.6 / (pr + 2.0) + 27.1 / (pr + 3.5) ** 2 - 8.7 * np.exp(-0.65 * (pr + 1.0))
    fz = e * 1.2 * (-(0.45 + 8.0 * (0.56 - 1.0 / tr) ** 2) * pr**0.2 / tr) + (
        0.03 + 0.00527 * (3.5 - tr) ** 3
    )
    k_gas = p * gamma / (1.0 - pr / z * fz) / 1000.0
    vp_gas = np.sqrt(k_gas / rho_gas)

    # --- oil density ------------------------------------------------------
    rho0 = 141.5 / (og + 131.5)

    if gas_index_oil is not None:
        gor_max = 2.03 * gg * (p * np.exp(0.02878 * og - 0.00377 * t)) ** 1.205
        gor = gor_max * np.asarray(gas_index_oil, float)

    live = gor != 0
    b0 = 0.972 + 0.00038 * (2.4 * gor * np.sqrt(gg / rho0) + t + 17.8) ** 1.175
    rho_og = np.where(live, (rho0 + 0.0012 * gg * gor) / b0, rho0)
    rho_p = rho_og + (0.00277 * p - 1.71e-7 * p**3) * (rho_og - 1.15) ** 2 + 3.49e-4 * p
    # Dead oil gets an explicit temperature correction; for live oil the
    # temperature dependence is already in the volume factor b0.
    rho_oil = np.where(live, rho_p, rho_p / (0.972 + 3.81e-4 * (t + 17.78) ** 1.175))

    # --- oil velocity (pseudo-density for live oil) -----------------------
    rho0_v = np.where(live, rho0 / b0 / (1.0 + 0.001 * gor), rho0)
    vp_oil = (
        2096.0 * np.sqrt(rho0_v / (2.6 - rho0_v))
        - 3.7 * t
        + 4.64 * p
        + 0.0115 * (4.12 * np.sqrt(1.08 / rho0_v - 1.0) - 1.0) * t * p
    ) / 1000.0
    k_oil = vp_oil**2 * rho_oil

    # --- brine density ----------------------------------------------------
    rho_w = 1.0 + 1e-6 * (
        -80.0 * t
        - 3.3 * t**2
        + 0.00175 * t**3
        + 489.0 * p
        - 2.0 * t * p
        + 0.016 * t**2 * p
        - 1.3e-5 * t**3 * p
        - 0.333 * p**2
        - 0.002 * t * p**2
    )
    rho_brine = rho_w + sal * (
        0.668
        + 0.44 * sal
        + 1e-6
        * (
            300.0 * p
            - 2400.0 * p * sal
            + t * (80.0 + 3.0 * t - 3300.0 * sal - 13.0 * p + 47.0 * p * sal)
        )
    )

    # --- brine velocity ---------------------------------------------------
    ti = t[..., None, None] ** np.arange(5)[:, None]
    pj = p[..., None, None] ** np.arange(4)[None, :]
    vel_w = np.sum(_WATER_VEL_COEFFS * ti * pj, axis=(-2, -1))

    gwr_max = 10.0 ** (
        np.log10(0.712 * p * np.abs(t - 76.71) ** 1.5 + 3676.0 * p**0.64)
        - 4.0
        - 7.786 * sal * (t + 17.78) ** (-0.306)
    )
    gwr = gwr_max * giib

    vp_b0 = (
        vel_w
        + sal
        * (
            1170.0
            - 9.6 * t
            + 0.055 * t**2
            - 8.5e-5 * t**3
            + 2.6 * p
            - 0.0029 * t * p
            - 0.0476 * p**2
        )
        + sal**1.5 * (780.0 - 10.0 * p + 0.16 * p**2)
        - 1820.0 * sal**2
    )
    vp_brine = vp_b0 / np.sqrt(1.0 + 0.0494 * gwr) / 1000.0
    k_brine = vp_brine**2 * rho_brine

    # --- fluid mix --------------------------------------------------------
    s_brine = 1.0 - s_oil - s_gas
    rho_eff = s_brine * rho_brine + s_oil * rho_oil + s_gas * rho_gas

    with np.errstate(divide="ignore", invalid="ignore"):
        terms = sum(
            np.where(s != 0, s / k, 0.0)
            for s, k in ((s_brine, k_brine), (s_oil, k_oil), (s_gas, k_gas))
        )
        k_reuss = np.where(terms > 0, 1.0 / terms, 0.0)
    k_voigt = s_brine * k_brine + s_oil * k_oil + s_gas * k_gas

    return FluidProperties(
        k_reuss=k_reuss,
        rho_eff=rho_eff,
        k_voigt=k_voigt,
        vp_brine=vp_brine,
        rho_brine=rho_brine,
        k_brine=k_brine,
        vp_oil=vp_oil,
        rho_oil=rho_oil,
        k_oil=k_oil,
        vp_gas=vp_gas,
        rho_gas=rho_gas,
        k_gas=k_gas,
        gor=gor,
    )


@cache
def _co2_interpolators():
    """Bilinear interpolators over the packaged CO2 (pressure, temperature)
    grids, NaN outside the table (matching MATLAB ``interp2``)."""
    with (files("rphtools") / "data" / "co2prop.npz").open("rb") as fh:
        data = np.load(fh)
        points = (data["pressure_mpa"], data["temperature_c"])
        grids = {name: data[name] for name in ("bulk_gpa", "rho_gcc", "vp_ms")}
    return {
        name: RegularGridInterpolator(points, grid, bounds_error=False, fill_value=np.nan)
        for name, grid in grids.items()
    }


def co2_properties(temperature, pressure):
    """CO2 properties versus temperature and pressure.

    Bilinear interpolation of Z. Wang's compiled measurements (packaged from
    ``co2propdata.mat``; temperature 17-127 degC, pressure 0.1-40 MPa).

    Parameters
    ----------
    temperature : array_like
        Temperature (degrees Celsius).
    pressure : array_like
        Pore pressure (MPa). Broadcast against `temperature`.

    Returns
    -------
    k : ndarray
        CO2 bulk modulus (GPa).
    rho : ndarray
        CO2 density (g/cm^3).
    vp : ndarray
        CO2 ultrasonic P-wave velocity (m/s).

    Notes
    -----
    Port of the computational core of ``co2prop.m`` (the GUI is dropped).
    Queries outside the tabulated range return NaN, matching MATLAB
    ``interp2``.
    """
    t, p = np.broadcast_arrays(np.asarray(temperature, float), np.asarray(pressure, float))
    pts = np.stack([p.ravel(), t.ravel()], axis=-1)
    interp = _co2_interpolators()
    k = interp["bulk_gpa"](pts).reshape(t.shape)
    rho = interp["rho_gcc"](pts).reshape(t.shape)
    vp = interp["vp_ms"](pts).reshape(t.shape)
    return k, rho, vp
