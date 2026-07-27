# pyRPHtools

Python port of **RPHtools**, the MATLAB companion software to *The Rock
Physics Handbook* (Mavko, Mukerji & Dvorkin, Cambridge University Press).
The original MATLAB lives in [`../RPHtools/`](../RPHtools/); the port plan,
conventions, and the full MATLAB → Python mapping are in
[`../PORTING_PLAN.md`](../PORTING_PLAN.md).

The package is pure NumPy/SciPy: no GUI code, plotting separated from
computation, NumPy-style docstrings with units and Handbook references, and
every function cross-referenced to its original `.m` file.

## Install

```bash
pip install ./pyRPHtools            # from the repository root
pip install "./pyRPHtools[plot]"    # with optional matplotlib helpers
```

Requires Python ≥ 3.10, NumPy, SciPy.

## Quickstart

```python
import numpy as np
import rphtools as rph

# Velocities of quartz (GPa, g/cm^3 -> km/s)
vp, vs = rph.moduli_to_velocity(k=37.0, mu=44.0, rho=2.65)

# Hashin-Shtrikman bounds for a quartz/water mixture
hs = rph.hashin_shtrikman(k1=37.0, mu1=44.0, k2=2.2, mu2=0.0)

# Backus average of a thinly layered shale/sand stack
res = rph.backus_average(f=[0.6, 0.4], vp=[3.0, 4.0], vs=[1.5, 2.4], rho=[2.4, 2.5])
print(res.vp0, res.vp90)  # slow (vertical) vs fast (horizontal) P velocity
```

Units: any *consistent* system works; the Handbook's examples typically use
GPa for moduli, g/cm³ for density, and km/s for velocities (mutually
consistent). Each docstring states its expectations.

## Status

| Phase | Modules | Status |
|---|---|---|
| 0 | packaging, CI, test harness, CO2 data conversion | done |
| 1 | `moduli`, `tensors`, `layered`, `bounds` | done |
| 2 | `fluids`, `fluid_properties` | done |
| 3 | `effective_medium`, `cracks` | planned |
| 4 | `granular`, `permeability` | planned |
| 5 | `avo` | planned |
| 6 | `seismic`, `signal` | planned |
| 7 | `stats`, `io`, `plotting` | planned |

## MATLAB → Python mapping (ported so far)

| MATLAB | Python |
|---|---|
| `ku2v.m` | `rphtools.moduli_to_velocity` |
| `lm2v.m` | `rphtools.lame_to_velocity` |
| *(missing `v2ku`)* | `rphtools.velocity_to_moduli` (reconstructed) |
| *(missing `v2lm`)* | `rphtools.velocity_to_lame` (reconstructed) |
| `critpor.m` | `rphtools.critical_porosity` |
| `CSiso.m` | `rphtools.isotropic_cs` |
| `c2anis.m` | `rphtools.thomsen_params` |
| `c2sti.m` | `rphtools.ti_c_to_s` |
| `c2vti.m` | `rphtools.ti_velocities` |
| `cti2v.m` | `rphtools.cti_to_velocities` |
| `ezbond.m` | `rphtools.bond_rotation` / `rphtools.bond_matrix` |
| `bkus.m` | `rphtools.backus_average` |
| `bkusc.m` | `rphtools.backus_average_c` |
| `bkuslog.m` | `rphtools.backus_average_log` |
| `bound.m` | `rphtools.bounds` |
| `hash.m` | `rphtools.hashin_shtrikman` |
| `hashv.m` | `rphtools.hashin_shtrikman_velocity` |
| `gassmnk.m` | `rphtools.gassmann_k` |
| `gassmnv.m` | `rphtools.gassmann_vel` |
| `patchw.m` | `rphtools.white_patchy` |
| `biot.m` | `rphtools.biot_dispersion` |
| `biothf.m` | `rphtools.biot_hf` |
| `biothfb.m` | `rphtools.biot_hf_geertsma_smit` |
| `biothfgs.m` | *(not ported — approximates `biot_hf`; test oracle only)* |
| `BKs2d.m` | `rphtools.brown_korringa_sat_to_dry` |
| `BKd2s.m` | `rphtools.brown_korringa_dry_to_sat` |
| `BKs2s.m` | `rphtools.brown_korringa_s` |
| `BKc2c.m` | `rphtools.brown_korringa_c` |
| `bkti.m` | `rphtools.brown_korringa_ti` |
| `mmti.m` | `rphtools.squirt_ti` |
| `flprop.m` | `rphtools.batzle_wang` |
| `flpropui.m` | *(not ported — GUI wrapper around flprop)* |
| `co2prop.m` | `rphtools.co2_properties` |

Deliberate behavior changes from the MATLAB (bug fixes, dropped GUI/plotting,
unified argument orders) are documented per module and in
`../PORTING_PLAN.md`, Section 7.4.

## Testing

```bash
pip install "./pyRPHtools[dev]"
pytest pyRPHtools
ruff check pyRPHtools
```

Tests are analytic-invariant based (round trips, limiting cases, symmetries,
cross-consistency between merged MATLAB twins). `tests/generate_golden.m` is
an Octave script that runs the *original* MATLAB functions to produce golden
fixtures in `tests/golden/`; fixtures are committed once generated so CI
never needs Octave. (Not yet generated — Octave is unavailable in the
current development environment; the invariant tests stand in until then.)
