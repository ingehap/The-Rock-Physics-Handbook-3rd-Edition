"""rphtools: Python port of the Rock Physics Handbook MATLAB toolbox.

A NumPy/SciPy port of the RPHtools companion software to *The Rock Physics
Handbook* (Mavko, Mukerji & Dvorkin, Cambridge University Press). See
``PORTING_PLAN.md`` at the repository root for scope, conventions, and the
full MATLAB-to-Python mapping.

Modules ported so far (Phase 1):

- `rphtools.moduli` — isotropic moduli/velocity conversions, critical porosity
- `rphtools.tensors` — 6x6 Voigt utilities, Thomsen parameters, Bond rotation
- `rphtools.layered` — Backus averaging (including from well logs)
- `rphtools.bounds` — Voigt-Reuss and Hashin-Shtrikman bounds
"""

from .bounds import (
    ElasticBounds,
    HSBoundCurves,
    HSVelocityCurves,
    bounds,
    hashin_shtrikman,
    hashin_shtrikman_velocity,
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
    "CTIVelocities",
    "CriticalPorosity",
    "ElasticBounds",
    "HSBoundCurves",
    "HSVelocityCurves",
    "IsotropicCS",
    "ThomsenParams",
    "TICompliance5",
    "TIVelocities",
    "backus_average",
    "backus_average_c",
    "backus_average_log",
    "bond_matrix",
    "bond_rotation",
    "bounds",
    "critical_porosity",
    "cti_to_velocities",
    "hashin_shtrikman",
    "hashin_shtrikman_velocity",
    "isotropic_cs",
    "lame_to_velocity",
    "moduli_to_velocity",
    "thomsen_params",
    "ti_c_to_s",
    "ti_velocities",
    "ti_voigt_matrix",
    "velocity_to_lame",
    "velocity_to_moduli",
]
