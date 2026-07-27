"""rphtools: Python port of the Rock Physics Handbook MATLAB toolbox.

A NumPy/SciPy port of the RPHtools companion software to *The Rock Physics
Handbook* (Mavko, Mukerji & Dvorkin, Cambridge University Press). See
``PORTING_PLAN.md`` at the repository root for scope, conventions, and the
full MATLAB-to-Python mapping.

Modules ported so far (Phases 1-2):

- `rphtools.moduli` — isotropic moduli/velocity conversions, critical porosity
- `rphtools.tensors` — 6x6 Voigt utilities, Thomsen parameters, Bond rotation
- `rphtools.layered` — Backus averaging (including from well logs)
- `rphtools.bounds` — Voigt-Reuss and Hashin-Shtrikman bounds
- `rphtools.fluids` — Gassmann, Brown-Korringa, Biot, squirt, patchy saturation
- `rphtools.fluid_properties` — Batzle-Wang relations, CO2 property tables
- `rphtools.effective_medium` — Berryman self-consistent, DEM
- `rphtools.cracks` — Hudson crack models, Eshelby-Cheng
"""

from .bounds import (
    ElasticBounds,
    HSBoundCurves,
    HSVelocityCurves,
    bounds,
    hashin_shtrikman,
    hashin_shtrikman_velocity,
)
from .cracks import (
    EshelbyCheng,
    Hudson3Result,
    HudsonVelocities,
    eshelby_cheng,
    hudson,
    hudson3,
    hudson_cone,
    hudson_fisher,
    hudson_velocities,
)
from .effective_medium import (
    BerrymanSCCurves,
    DEMResult,
    berryman_sc,
    berryman_sc_pressure,
    berryman_scm,
    dem,
    dem_at_fraction,
)
from .fluid_properties import (
    FluidProperties,
    batzle_wang,
    co2_properties,
)
from .fluids import (
    BiotDispersion,
    WhitePatchyResult,
    biot_dispersion,
    biot_hf,
    biot_hf_geertsma_smit,
    brown_korringa_c,
    brown_korringa_dry_to_sat,
    brown_korringa_s,
    brown_korringa_sat_to_dry,
    brown_korringa_ti,
    gassmann_k,
    gassmann_vel,
    squirt_ti,
    white_patchy,
)
from .layered import (
    BackusLogResult,
    BackusResult,
    backus_average,
    backus_average_c,
    backus_average_log,
)
from .moduli import (
    CriticalPorosity,
    critical_porosity,
    lame_to_velocity,
    moduli_to_velocity,
    velocity_to_lame,
    velocity_to_moduli,
)
from .tensors import (
    CTIVelocities,
    IsotropicCS,
    ThomsenParams,
    TICompliance5,
    TIVelocities,
    bond_matrix,
    bond_rotation,
    cti_to_velocities,
    isotropic_cs,
    thomsen_params,
    ti_c_to_s,
    ti_velocities,
    ti_voigt_matrix,
)

__version__ = "0.1.0"

__all__ = [
    "BackusLogResult",
    "BackusResult",
    "BerrymanSCCurves",
    "BiotDispersion",
    "CTIVelocities",
    "CriticalPorosity",
    "DEMResult",
    "ElasticBounds",
    "EshelbyCheng",
    "FluidProperties",
    "Hudson3Result",
    "HudsonVelocities",
    "HSBoundCurves",
    "HSVelocityCurves",
    "IsotropicCS",
    "ThomsenParams",
    "TICompliance5",
    "TIVelocities",
    "WhitePatchyResult",
    "backus_average",
    "backus_average_c",
    "backus_average_log",
    "batzle_wang",
    "berryman_sc",
    "berryman_sc_pressure",
    "berryman_scm",
    "biot_dispersion",
    "biot_hf",
    "biot_hf_geertsma_smit",
    "bond_matrix",
    "bond_rotation",
    "bounds",
    "brown_korringa_c",
    "brown_korringa_dry_to_sat",
    "brown_korringa_s",
    "brown_korringa_sat_to_dry",
    "brown_korringa_ti",
    "co2_properties",
    "critical_porosity",
    "cti_to_velocities",
    "dem",
    "dem_at_fraction",
    "eshelby_cheng",
    "gassmann_k",
    "gassmann_vel",
    "hashin_shtrikman",
    "hashin_shtrikman_velocity",
    "hudson",
    "hudson3",
    "hudson_cone",
    "hudson_fisher",
    "hudson_velocities",
    "isotropic_cs",
    "lame_to_velocity",
    "moduli_to_velocity",
    "squirt_ti",
    "thomsen_params",
    "ti_c_to_s",
    "ti_velocities",
    "ti_voigt_matrix",
    "velocity_to_lame",
    "velocity_to_moduli",
    "white_patchy",
]
