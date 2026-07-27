"""rphtools: Python port of the Rock Physics Handbook MATLAB toolbox.

A NumPy/SciPy port of the RPHtools companion software to *The Rock Physics
Handbook* (Mavko, Mukerji & Dvorkin, Cambridge University Press). See
``PORTING_PLAN.md`` at the repository root for scope, conventions, and the
full MATLAB-to-Python mapping.

Modules ported so far (Phases 1-5):

- `rphtools.moduli` — isotropic moduli/velocity conversions, critical porosity
- `rphtools.tensors` — 6x6 Voigt utilities, Thomsen parameters, Bond rotation
- `rphtools.layered` — Backus averaging (including from well logs)
- `rphtools.bounds` — Voigt-Reuss and Hashin-Shtrikman bounds
- `rphtools.fluids` — Gassmann, Brown-Korringa, Biot, squirt, patchy saturation
- `rphtools.fluid_properties` — Batzle-Wang relations, CO2 property tables
- `rphtools.effective_medium` — Berryman self-consistent, DEM
- `rphtools.cracks` — Hudson crack models, Eshelby-Cheng
- `rphtools.avo` — Zoeppritz reflectivity, AVO attributes, elastic impedance
- `rphtools.granular` — Hertz-Mindlin, contact cement, stress-induced anisotropy
- `rphtools.permeability` — empirical and theoretical permeability models
"""

from .avo import (
    AVOAttributes,
    ElasticImpedance,
    avo_attributes,
    avo_pp,
    avo_ps,
    elastic_impedance,
)
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
from .granular import (
    COORDINATION_TABLE,
    ContactCement,
    HertzMindlin,
    HertzMindlinVelocity,
    JohnsonMakse,
    JohnsonResult,
    contact_cement,
    coordination_number,
    hertz_mindlin,
    hertz_mindlin_v,
    johnson_makse,
    johnson_stress_anisotropy,
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
from .permeability import (
    PERM_MODELS,
    BernabePerm,
    BlochPerm,
    OwolabiPerm,
    bernabe_perm,
    bloch_perm,
    coates_dumanoir_perm,
    coates_perm,
    fredrich_perm,
    kozeny_carman_perm,
    modified_kozeny_carman_perm,
    owolabi_perm,
    panda_lake_kc_perm,
    panda_lake_perm,
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
    "AVOAttributes",
    "BackusLogResult",
    "BackusResult",
    "BernabePerm",
    "BerrymanSCCurves",
    "BiotDispersion",
    "BlochPerm",
    "COORDINATION_TABLE",
    "CTIVelocities",
    "ContactCement",
    "CriticalPorosity",
    "DEMResult",
    "ElasticBounds",
    "ElasticImpedance",
    "EshelbyCheng",
    "FluidProperties",
    "HSBoundCurves",
    "HSVelocityCurves",
    "HertzMindlin",
    "HertzMindlinVelocity",
    "Hudson3Result",
    "HudsonVelocities",
    "IsotropicCS",
    "JohnsonMakse",
    "JohnsonResult",
    "OwolabiPerm",
    "PERM_MODELS",
    "TICompliance5",
    "TIVelocities",
    "ThomsenParams",
    "WhitePatchyResult",
    "avo_attributes",
    "avo_pp",
    "avo_ps",
    "backus_average",
    "backus_average_c",
    "backus_average_log",
    "batzle_wang",
    "bernabe_perm",
    "berryman_sc",
    "berryman_sc_pressure",
    "berryman_scm",
    "biot_dispersion",
    "biot_hf",
    "biot_hf_geertsma_smit",
    "bloch_perm",
    "bond_matrix",
    "bond_rotation",
    "bounds",
    "brown_korringa_c",
    "brown_korringa_dry_to_sat",
    "brown_korringa_s",
    "brown_korringa_sat_to_dry",
    "brown_korringa_ti",
    "co2_properties",
    "coates_dumanoir_perm",
    "coates_perm",
    "contact_cement",
    "coordination_number",
    "critical_porosity",
    "cti_to_velocities",
    "dem",
    "dem_at_fraction",
    "elastic_impedance",
    "eshelby_cheng",
    "fredrich_perm",
    "gassmann_k",
    "gassmann_vel",
    "hashin_shtrikman",
    "hashin_shtrikman_velocity",
    "hertz_mindlin",
    "hertz_mindlin_v",
    "hudson",
    "hudson3",
    "hudson_cone",
    "hudson_fisher",
    "hudson_velocities",
    "isotropic_cs",
    "johnson_makse",
    "johnson_stress_anisotropy",
    "kozeny_carman_perm",
    "lame_to_velocity",
    "modified_kozeny_carman_perm",
    "moduli_to_velocity",
    "owolabi_perm",
    "panda_lake_kc_perm",
    "panda_lake_perm",
    "squirt_ti",
    "thomsen_params",
    "ti_c_to_s",
    "ti_velocities",
    "ti_voigt_matrix",
    "velocity_to_lame",
    "velocity_to_moduli",
    "white_patchy",
]
