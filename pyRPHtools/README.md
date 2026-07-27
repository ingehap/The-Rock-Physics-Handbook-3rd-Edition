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
| 3 | `effective_medium`, `cracks` | done |
| 4 | `granular`, `permeability` | done |
| 5 | `avo` | done |
| 6 | `seismic`, `signal` | done |
| 7 | `stats`, `io`, `plotting` | done |
| 8 | golden fixtures, reconstructions, `pdf_bayes` | done |

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
| `berrysc.m` | `rphtools.berryman_sc` |
| `berryscm.m` | `rphtools.berryman_scm` |
| `berryscp.m` | `rphtools.berryman_sc_pressure` |
| `dem.m` | `rphtools.dem` |
| `dem1.m` | `rphtools.dem_at_fraction` |
| `demyprime.m` | *(private `_dem_rhs`)* |
| `ode45m.m` | *(not ported — replaced by `scipy.integrate.solve_ivp`)* |
| `hudson.m` | `rphtools.hudson` |
| `hudson1.m` | `rphtools.hudson_velocities` |
| `hudson3.m` | `rphtools.hudson3` |
| `hudsonF.m` | `rphtools.hudson_fisher` (two bug fixes, see module notes) |
| `hudsoncone.m` | `rphtools.hudson_cone` |
| `echeng.m` | `rphtools.eshelby_cheng` |
| `hertzmind.m` | `rphtools.hertz_mindlin` |
| `hertzmindv.m` | `rphtools.hertz_mindlin_v` |
| `Cem.m` | `rphtools.contact_cement` |
| `Johnson.m` | `rphtools.johnson_stress_anisotropy` (returns the tensor the MATLAB overwrote) |
| `John_Makse.m` | `rphtools.johnson_makse` (reconstructed — the MATLAB could not run) |
| `BernabeE.m` | `rphtools.bernabe_perm` |
| `Bloch.m` | `rphtools.bloch_perm` |
| `CoatDum.m` | `rphtools.coates_dumanoir_perm` |
| `Coates.m` | `rphtools.coates_perm` |
| `FredrichE.m` | `rphtools.fredrich_perm` |
| `KozCarmE.m` | `rphtools.kozeny_carman_perm` |
| `ModKozCarm.m` | `rphtools.modified_kozeny_carman_perm` |
| `Owolabi.m` | `rphtools.owolabi_perm` |
| `PandaLake.m` | `rphtools.panda_lake_perm` |
| `PandaLakeKCE.m` | `rphtools.panda_lake_kc_perm` |
| `PermMenu.m` | `rphtools.PERM_MODELS` (registry dict, no GUI) |
| `avopp.m` | `rphtools.avo_pp` |
| `avops.m` | `rphtools.avo_ps` |
| `avo_abe.m` | `rphtools.avo_attributes` |
| `eimp.m` | `rphtools.elastic_impedance` (`angle="reflection"`) |
| `eimp2.m` | `rphtools.elastic_impedance` (`angle="incidence"`) |
| `kennet.m` | `rphtools.kennett` |
| `kennett_aux.m` | *(merged into `kennett` — its taper fix adopted)* |
| `pgator.m` | `rphtools.propagator_seis` |
| `kenfdisp.m` | `rphtools.kennett_frazer_dispersion` |
| `kenfrtt.m` | `rphtools.kennett_frazer_traveltimes` |
| `ezseis.m` | `rphtools.quick_seismic_section` |
| *(missing `sourcewvlt`)* | `rphtools.ricker` (replacement default wavelet) |
| `fftplot.m` | `rphtools.spectrum` |
| `iatrib.m` | `rphtools.instantaneous_attributes` |
| `blockav.m` | `rphtools.block_average` |
| `ft1axis.m` | `rphtools.fft_axis` (`axis=0`) |
| `ft2axis.m` | `rphtools.fft_axis` (`axis=1`) |
| *(missing `v2cti`)* | `rphtools.ti_from_velocities` (reconstructed) |
| *(missing `Unconsol`)* | `rphtools.unconsolidated` (reconstructed) |
| `pdfbayes.m` | `rphtools.pdf_bayes` (reconstructed — engines missing) |
| `hist2d.m` | `rphtools.hist2d` |
| `hist3d.m` | `rphtools.hist3d` (1-3 columns; weighted path fixed) |
| `bayesclass.m` | `rphtools.bayes_classify` |
| `private/bayesclass.m` | *(not ported — a different, cruder implementation)* |
| `monte.m` | `rphtools.monte_carlo_cdf` |
| `monteccdf.m` | `rphtools.monte_carlo_ccdf` |
| `loadlas.m` | `rphtools.load_las` |
| `logax.m` | `rphtools.plotting.set_depth_limits` |
| *(plot side effects of `hash`/`hashv`)* | `rphtools.plotting.plot_bounds` |
| *(plot side effect of `fftplot`)* | `rphtools.plotting.plot_spectrum` |
| *(plot side effect of `hist2d`)* | `rphtools.plotting.plot_hist2d` |

Deliberate behavior changes from the MATLAB (bug fixes, dropped GUI/plotting,
unified argument orders) are documented per module and in
`../PORTING_PLAN.md`, Section 7.4.

## Testing

```bash
pip install "./pyRPHtools[dev]"
pytest pyRPHtools
ruff check pyRPHtools
```

Two complementary layers:

**Golden values from the original MATLAB.** `tests/generate_golden.m` runs
the RPHtools `.m` files themselves under GNU Octave and writes
`tests/golden/phase1.json`; `tests/test_golden.py` asserts the port
reproduces them. Most quantities match to 1e-10 relative or better.
The fixture is committed, so CI needs no Octave — regenerate it with:

```bash
apt-get install -y octave                       # plus gnuplot-nox if desired
octave --no-gui --quiet pyRPHtools/tests/generate_golden.m
```

`tests/octave_shims/` supplies what modern Octave lacks: the legacy
`bessel`, the toolbox functions `harmmean`/`nanmean`/`hilbert`, the
reconstruction of the missing `v2ku`, and no-op plotting stubs (many
RPHtools functions draw figures unconditionally).

**Analytic invariants.** Round trips, limiting cases, symmetries, and
cross-consistency between merged MATLAB twins — these cover the functions
whose originals cannot run at all, and catch classes of error that
matching one set of numbers would not.
