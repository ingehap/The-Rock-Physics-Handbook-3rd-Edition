# Porting Plan: RPHtools (MATLAB) → pyRPHtools (Python)

This document is the implementation plan for a Python port of the Rock Physics
Handbook MATLAB toolbox in [`RPHtools/`](RPHtools/) (Mavko, Mukerji & Dvorkin,
Cambridge University Press). The port will live in a new top-level folder
`pyRPHtools/`, parallel to `RPHtools/`, and aims to be **Pythonic: clear,
simple, and well documented** — pure NumPy/SciPy functions with NumPy-style
docstrings, no GUI code, and plotting separated from computation.

The plan is grounded in a full review of every `.m` file in the folder
(92 source files, ~6,000 lines), including a cross-check of every internal
dependency. Findings that shape the plan are listed first.

---

## 1. What is actually in `RPHtools/`

**Inventory:** 92 MATLAB source files (86 in the root folder, 6 in `private/`),
two `.mat` data files, plus `Contents.m` (the index), two editor backup files
(`Contents.asv`, `hudson1.m_old`), and two HTML files.

**Difficulty profile** (per-file assessment): 32 trivial, 35 easy, 23 moderate,
2 hard (`John_Makse.m`, `pdfbayes.m`).

**Dependency structure:** remarkably flat. Roughly 80 of the 92 files are pure
leaves — self-contained formulas with no internal calls — so most functions can
be ported in any order. There are only 7 small dependency clusters, none deeper
than two levels (Brown–Korringa, Backus, Berryman SC, DEM, Gassmann→patchy,
Hertz–Mindlin, and the statistics stack).

Findings that the plan must handle:

1. **The distribution is incomplete relative to its own index.** 15 functions
   listed in `Contents.m` do not exist in the folder: `Rorsym`, `Rruger`,
   `v2cti`, `v2ku`, `v2lm`, `squirt`, `stdlin`, `Unconsol`, `walton`,
   `waltonv`, `RevilE`, `Timur`, `Tixier`, `WylGregE`, `sourcewvlt`.
   Four of these are actually *called* by existing code:
   `v2ku` (by `hertzmindv.m`), `sourcewvlt` (default wavelet in `kennet.m` and
   `pgator.m`), and `Timur`/`Tixier`/`RevilE`/`WylGregE` (dispatched by
   `PermMenu.m`, so 4 of its 14 menu entries are broken at runtime).
2. **The `pdfbayes` stack cannot be translated, only reconstructed.** Its two
   computational engines (`pdfgendraw`, `pdfstat`), ten GUI plotters, and three
   primitives used by the `private/` helpers (`histnd`, `entropdf`, `str2cell`)
   are all missing. The six `private/` functions are its stranded support
   library — nothing else in the toolbox reaches them.
3. **GUI and computation are entangled** in ~20 files. All ten permeability
   models share an `inputdlg`/`evalin('base')`/`semilogy` boilerplate around a
   one-line formula; `flpropui.m` contains a byte-for-byte inline copy of
   `flprop.m`'s Batzle–Wang math; `Cem.m`, `co2prop.m`, `ezseis.m`, `monte.m`,
   and `monteccdf.m` mix dialogs/plots with computation. The port extracts the
   formulas and drops the dialogs.
4. **Near-duplicate files abound** and should collapse to single Python cores
   with thin wrappers: `bkus`/`bkusc`, `hudson`/`hudson1`, `eimp`/`eimp2`,
   `dem`/`dem1`, `hertzmind`/`hertzmindv`, `gassmnk`/`gassmnv`,
   `kennet`/`kennett_aux` (the latter is a classic-Mac CR-line-ending duplicate
   that even defines `function kennet`), `berrysc`/`berryscm`/`demyprime`
   (spheroid P,Q algebra copied three times), the Zoeppritz algebra shared by
   `avopp`/`avops`/`avo_abe`, and ~80% shared code in `Johnson`/`John_Makse`.
5. **Known MATLAB bugs / traps to fix deliberately** (each fix logged in
   Section 7.4): `hist3d.m` calls `hist2d` with 4 arguments but `hist2d.m`
   accepts 3 (broken weighted path); `berryscp.m` has a `find(k==max(k))` tie
   bug; `bkus.m` uses unit-dependent absolute tolerances and a blocking
   `pause` as a warning; `bkuslog.m` silently mishandles non-monotonic depth;
   `dem.m`/`dem1.m` couple to their ODE right-hand side through a
   `global DEMINPT` and a `feval` string; `biot.m` calls the legacy `bessel`
   (removed from modern MATLAB); `outputdlg.m` defines a mismatched function
   name and has no callers.
6. **Data files:** `co2propdata.mat` (CO2 velocity/density/bulk modulus on a
   10×12 pressure–temperature grid) is required by `co2prop.m` and must be
   converted to Python-native package data. `countsegy.mat` is an orphan
   scalar counter — dropped.

---

## 2. Goals and design principles

1. **Faithful numerics, modern interface.** Same physics, same numbers (to
   documented tolerances), but Pythonic APIs: snake_case names, explicit
   parameters, no output-count magic (`nargin`/`nargout` branching becomes
   keyword arguments and small result dataclasses).
2. **Pure functions.** No dialogs, no `evalin('base')`, no global state, no
   plotting inside numerical code. Interactive menus become plain data
   (e.g. a `PERM_MODELS` registry dict).
3. **NumPy-first.** Vectorized with broadcasting where the MATLAB is
   elementwise; honest loops where the algorithm is iterative (self-consistent
   solvers, ODE integration).
4. **Documented like a handbook.** Every public function gets a NumPy-style
   docstring with parameter units, the Handbook section it implements, the
   original literature reference, and the original MATLAB file name (for users
   coming from the book).
5. **Tested against the original.** Golden reference values generated by
   running the original `.m` files in GNU Octave, plus analytic invariants
   (limiting cases, round-trips, symmetries) as property tests.
6. **Behavior changes are explicit.** Anything that deliberately differs from
   MATLAB (bug fixes, dropped GUI, changed argument order) is listed in this
   plan and in the module docstrings.

---

## 3. Package layout

```
pyRPHtools/                      # parallel to RPHtools/, self-contained project
├── pyproject.toml               # package name: rphtools; deps: numpy, scipy
├── README.md                    # quickstart + full MATLAB → Python mapping table
├── rphtools/                    # the importable package  (import rphtools as rph)
│   ├── __init__.py              # re-exports the public API, __version__
│   ├── moduli.py                # moduli/velocity conversions, critical porosity
│   ├── tensors.py               # 6x6 Voigt utilities, Thomsen params, Bond rotation
│   ├── layered.py               # Backus averaging (incl. from logs)
│   ├── bounds.py                # Voigt-Reuss & Hashin-Shtrikman bounds
│   ├── effective_medium.py      # Berryman self-consistent, DEM
│   ├── cracks.py                # Hudson family, Eshelby-Cheng
│   ├── fluids.py                # Gassmann, Brown-Korringa, Biot, squirt, patchy
│   ├── fluid_properties.py      # Batzle-Wang, CO2 property tables
│   ├── granular.py              # Hertz-Mindlin, cementation, stress-induced anisotropy
│   ├── permeability.py          # 10+ permeability models + PERM_MODELS registry
│   ├── avo.py                   # Zoeppritz/approximations, elastic impedance
│   ├── seismic.py               # 1-D synthetics (Kennett, propagator), wavelets
│   ├── signal.py                # spectra, instantaneous attributes, block averaging
│   ├── stats.py                 # Bayes classification, histograms, Monte Carlo
│   ├── io.py                    # LAS reader
│   ├── plotting.py              # optional matplotlib helpers (imported lazily)
│   └── data/
│       └── co2prop.npz          # converted from co2propdata.mat
└── tests/
    ├── golden/                  # Octave-generated reference values (JSON/CSV)
    ├── generate_golden.m        # Octave script that produced them (committed)
    └── test_<module>.py         # one test file per module
```

Notes:
- The **folder** is `pyRPHtools/` as requested; the **package** inside it is
  `rphtools` (lowercase, per PEP 8), installable with `pip install ./pyRPHtools`.
- `numpy` and `scipy` are required dependencies; `matplotlib` is an optional
  extra (`pip install "rphtools[plot]"`) used only by `plotting.py`.
- Supported Python: 3.10+.

---

## 4. API conventions

Illustrative signatures (final signatures reviewed per function at
implementation time; representative of the conventions used throughout):

```python
def gassmann_k(k_sat1, k_fl1, k_fl2, k_min, phi):
    """Gassmann fluid substitution on bulk modulus (MATLAB: gassmnk.m)."""

def backus_average(f, vp, vs, rho):
    """Backus average of thin isotropic layers -> BackusResult (MATLAB: bkus.m/bkusc.m)."""

def berryman_scm(k, mu, aspect, fraction, tol=1e-6, max_iter=3000):
    """N-phase Berryman self-consistent moduli -> (k_eff, mu_eff) (MATLAB: berryscm.m)."""

def dem(k1, mu1, k2, mu2, aspect, phi_max=1.0):
    """Differential effective medium -> (k, mu, phi) arrays (MATLAB: dem.m)."""

def avo_pp(vp1, vs1, rho1, vp2, vs2, rho2, angles_deg, method="zoeppritz"):
    """P-P reflectivity vs angle; method in {'zoeppritz','aki-richards','shuey', ...}."""

def batzle_wang(pressure, temperature, salinity, oil_api, gas_gravity, gor=0.0):
    """Batzle-Wang fluid properties -> FluidProperties dataclass (MATLAB: flprop.m)."""

def kennett(layers, wavelet=None, dt=0.001, multiples="all"):
    """Normal-incidence synthetic via invariant imbedding -> KennettResult (MATLAB: kennet.m)."""
```

Rules applied consistently:

- **Names:** snake_case, spelled out (`hertz_mindlin`, not `hm`). Model names
  keep their literature attribution (`berryman_scm`, `eshelby_cheng`,
  `panda_lake_perm`). Every docstring cross-references the original `.m` file,
  and the README carries the full mapping table, so book users can find things.
- **Inputs:** positional for physics arguments in a consistent order
  (moduli → geometry → fractions), keywords with defaults for options. MATLAB's
  `nargin`-dependent GUI fallbacks are removed, not replicated.
- **Outputs:** one value or a tuple for ≤3 closely-related outputs; a small
  frozen `@dataclass` (e.g. `FluidProperties`, `BiotDispersion`,
  `KennettResult`) where MATLAB returned 4+ values.
- **Arrays:** inputs accepted as scalars or array_like, broadcast with NumPy
  rules; 6×6 stiffness matrices are `(6, 6)` ndarrays, stacked as `(n, 6, 6)`
  where the MATLAB grew `6×6×n` arrays. Two competing MATLAB conventions for
  packing the five TI constants (`[c11 c33 c44 c66 c13]` vs
  `(a11,a12,a13,a33,a44)`) are replaced by named parameters / a tiny
  `TIStiffness` dataclass with converters to and from the 6×6 Voigt matrix.
- **Units:** identical to the original functions (the toolbox mixes km/s +
  g/cm³ and SI by function); every docstring states units explicitly.
  Unit-dependent absolute tolerances in the MATLAB become relative tolerances.
- **Errors:** `disp`+`pause` "warnings" become `ValueError` /
  `warnings.warn`; singular cases (e.g. `mu = 0` fluid in `isotropic_cs`) are
  guarded and documented.
- **Randomness:** Monte-Carlo functions take an optional
  `rng: np.random.Generator` for reproducibility.

---

## 5. Module-by-module mapping

Difficulty is the per-file porting assessment (trivial / easy / moderate /
hard). "MERGE" marks the near-duplicate consolidations from Section 1.

### `moduli.py` — moduli/velocity conversions

| MATLAB | Python | Difficulty | Notes |
|---|---|---|---|
| `ku2v.m` | `moduli_to_velocity` | trivial | Isotropic elasticity conversion: velocities from bulk modulus, shear modulus and density — Vp = sqrt((K + 4/3 mu)/rho), Vs = sqrt(mu/rho). |
| `lm2v.m` | `lame_to_velocity` | trivial | Isotropic elasticity conversion: velocities from Lame parameters and density — Vp = sqrt((lambda + 2 mu)/rho), Vs = sqrt(mu/rho). |
| `(missing) v2ku.m` | `velocity_to_moduli` | n/a | Reconstruct: `k = rho*(vp**2 - 4/3*vs**2)`, `mu = rho*vs**2`. Needed by `hertz_mindlin_v`; listed in Contents.m but absent. |
| `(missing) v2lm.m` | `velocity_to_lame` | n/a | Reconstruct from the Handbook (inverse of `lm2v`). |
| `critpor.m` | `critical_porosity` | trivial | Nur's critical-porosity model: moduli, density, and Vp/Vs at critical porosity from the Reuss average of the end members. |

### `tensors.py` — stiffness/compliance utilities

| MATLAB | Python | Difficulty | Notes |
|---|---|---|---|
| `CSiso.m` | `isotropic_cs` | trivial | Prefer closed-form compliance over `inv()`; guard mu=0 (fluid) singularity. |
| `c2anis.m` | `thomsen_params` | trivial | Computes Thomsen (1986) anisotropy parameters epsilon, gamma, exact delta, and delta_sv for a TI medium from the five elastic stiffnesses. |
| `c2sti.m` | `ti_c_to_s` | trivial | Same routine converts both directions; document clearly. |
| `c2vti.m` | `ti_velocities` | easy | Phase velocities vs angle for TI media; needed by `backus_average`. |
| `cti2v.m` | `cti_to_velocities` | easy | Fast/slow P and S velocities plus Thomsen parameters from full 6x6 TI stiffness matrices (handles VTI and HTI orientations). |
| `ezbond.m` | `bond_rotation` | easy | Generalize to rotation about any axis if simple; else keep z-rotation and document. |

### `layered.py` — Backus averaging

| MATLAB | Python | Difficulty | Notes |
|---|---|---|---|
| `bkus.m` | `backus_average` | easy | MERGE with bkusc.m: one core computing (c11,c12,c13,c33,c44,c66); replace `disp`/`pause` warnings with exceptions; make unit-dependent absolute tolerances relative. |
| `bkusc.m` | `backus_average_c` | trivial | Thin formatter over the shared core (returns full 6x6). Beware differing MATLAB argument orders between bkus/bkusc — pick one Python order `(f, vp, vs, rho)`. |
| `bkuslog.m` | `backus_average_log` | trivial | Depth-to-thickness preprocessing wrapper; raise on non-monotonic depth (silent garbage in MATLAB). |

### `bounds.py` — elastic bounds

| MATLAB | Python | Difficulty | Notes |
|---|---|---|---|
| `bound.m` | `bounds` | easy | Voigt-Reuss and Hashin-Shtrikman bounds; strip plotting. |
| `hash.m` | `hashin_shtrikman` | trivial | Compute-only; plot moves to `plotting.py`. |
| `hashv.m` | `hashin_shtrikman_velocity` | trivial | Same; delegates to `hashin_shtrikman` + `moduli_to_velocity`. |

### `effective_medium.py` — inclusion models

| MATLAB | Python | Difficulty | Notes |
|---|---|---|---|
| `berrysc.m` | `berryman_sc` | moderate | Implement as a fraction sweep over the `berryman_scm` core. |
| `berryscm.m` | `berryman_scm` | moderate | Core N-phase self-consistent solver; factor spheroid P,Q helpers shared with the DEM right-hand side. |
| `berryscp.m` | `berryman_sc_pressure` | easy | Pressure loop + crack-closure preprocessing; fix `find(k==max(k))` tie bug deliberately. |
| `dem.m` | `dem` | moderate | Replace `ode45m` + `global DEMINPT` + `feval` with `scipy.integrate.solve_ivp` and a closure. |
| `dem1.m` | `dem_at_fraction` | easy | Single-fraction variant; shares the RHS with `dem`. |
| `demyprime.m` | `_dem_rhs` | moderate | Becomes a private function taking explicit arguments instead of globals. |
| `ode45m.m` | `(drop)` | moderate | Modified classic ode45; superseded by `scipy.integrate.solve_ivp`. |

### `cracks.py` — cracked-rock models

| MATLAB | Python | Difficulty | Notes |
|---|---|---|---|
| `echeng.m` | `eshelby_cheng` | moderate | Eshelby-Cheng model (Cheng 1978, 1993) for the effective TI stiffness of a rock with a single set of aligned fluid-filled ellipsoidal cracks, valid for all aspect ratios (unlike Hudson). |
| `hudson.m` | `hudson` | easy | MERGE with hudson1.m: one core (first- and second-order corrections), two output views (Cij vs velocities/Thomsen). Factor the shared U1/U3 kernel used by the whole family. |
| `hudson1.m` | `hudson_velocities` | easy | Thin wrapper over the merged core. |
| `hudson3.m` | `hudson3` | easy | Three orthogonal crack sets. |
| `hudsonF.m` | `hudson_fisher` | moderate | Fisher-distributed crack normals. |
| `hudsoncone.m` | `hudson_cone` | easy | Conical crack distributions; reuse the VTI 6x6 assembly helper (copy-pasted 3x in MATLAB). |

### `fluids.py` — fluid substitution, poroelasticity, dispersion

| MATLAB | Python | Difficulty | Notes |
|---|---|---|---|
| `gassmnk.m` | `gassmann_k` | trivial | Preserve the phi==0 pass-through guard. |
| `gassmnv.m` | `gassmann_vel` | trivial | Delegate to `gassmann_k` (MATLAB duplicates the formula inline, without the phi==0 guard — unify and document). |
| `patchw.m` | `white_patchy` | moderate | White's patchy-saturation model with Dutta-Ode correction; calls `gassmann_k`. |
| `biot.m` | `biot_dispersion` | moderate | Full frequency-dependent Biot velocities and attenuation. Legacy `bessel` → `scipy.special.jv`; complex arithmetic throughout. |
| `biothf.m` | `biot_hf` | easy | High-frequency limit; share the dry-moduli/density preamble across the Biot family. |
| `biothfb.m` | `biot_hf_geertsma_smit` | easy | Approximate HF limit. |
| `biothfgs.m` | `(test oracle)` | trivial | Not a re-derivation of `biot_hf` as it first appears: it computes `sqrt(vp1^2 + vp2^2)` (the root-sum of the HF quadratic), an approximation neglecting the slow wave. Kept as a test-oracle identity. |
| `bkti.m` | `brown_korringa_ti` | easy | Brown and Korringa (1975) anisotropic fluid substitution specialized to TI (or isotropic) symmetry: computes saturated-rock compliances from dry-rock and mineral compliances given as 5-element rows [s11 s12 s13 s33 s44]. |
| `BKc2c.m` | `brown_korringa_c` | trivial | Stiffness-domain wrapper over s2d/d2s. |
| `BKd2s.m` | `brown_korringa_dry_to_sat` | easy | Brown-Korringa (1975) dry-to-saturated substitution on the 6x6 compliance matrix, isotropic mineral. |
| `BKs2d.m` | `brown_korringa_sat_to_dry` | easy | Brown and Korringa saturated-to-dry fluid substitution (exact inverse of BKd2s) for a general anisotropic rock on the 6x6 Voigt compliance matrix, with an isotropic mineral: Sdry = Ssat + factor * outer(Svect, Svect). |
| `BKs2s.m` | `brown_korringa_s` | trivial | Compliance-domain fluid-to-fluid wrapper. |
| `mmti.m` | `squirt_ti` | moderate | Mavko-Jizba style unrelaxed wet-frame TI compliances. |

### `fluid_properties.py` — pore-fluid properties

| MATLAB | Python | Difficulty | Notes |
|---|---|---|---|
| `flprop.m` | `batzle_wang` | moderate | Batzle-Wang brine/oil/gas properties. Return a small dataclass per fluid; strip any UI. |
| `flpropui.m` | `(drop)` | easy | GUI wrapper containing a byte-for-byte inline copy of flprop — the port must have exactly one Batzle-Wang core. |
| `co2prop.m` | `co2_properties` | moderate | Tabulated CO2 properties: ship `co2propdata.mat` converted to package data (.npz/.csv) + `scipy.interpolate.RegularGridInterpolator`; strip GUI/plots. |

### `granular.py` — granular media

| MATLAB | Python | Difficulty | Notes |
|---|---|---|---|
| `Cem.m` | `contact_cement` | moderate | Dvorkin cementation model; strip `inputdlg`/plot UI. |
| `hertzmind.m` | `hertz_mindlin` | easy | One core + module-level coordination-number table (RPH p.150); strip UI. |
| `hertzmindv.m` | `hertz_mindlin_v` | easy | Thin wrapper via `velocity_to_moduli`/`moduli_to_velocity` (MATLAB calls the missing `v2ku`). |
| `Johnson.m` | `johnson_stress_anisotropy` | moderate | Norris-Johnson stress-induced TI; factor ~80% shared code with John_Makse into a private helper. |
| `John_Makse.m` | `johnson_makse` | hard | Hard: dense shared tensor algebra; port after `johnson_stress_anisotropy` using the shared helper. |

### `permeability.py` — permeability models

| MATLAB | Python | Difficulty | Notes |
|---|---|---|---|
| `BernabeE.m` | `bernabe_perm` | easy | Bernabe (1991) two-pore-type model: pressure-sensitive cracks + stiff tubes, `K = Kcrack + Ktube`. Clean sandstones. |
| `Bloch.m` | `bloch_perm` | trivial | Bloch (1991) empirical regression: porosity and permeability from sorting, grain size, and cement content. |
| `CoatDum.m` | `coates_dumanoir_perm` | trivial | Coates-Dumanoir (1974) log-derived estimator, `K(md) = 352*phi^4/Swr^4`. |
| `Coates.m` | `coates_perm` | trivial | Coates et al. (1991) log-derived estimator, `K(md) = 10000*phi^4*(1-Swr)^2/Swr^2`. |
| `FredrichE.m` | `fredrich_perm` | trivial | Fredrich et al. Kozeny-type model from porosity and grain diameter (Fontainebleau sandstone calibration). |
| `KozCarmE.m` | `kozeny_carman_perm` | trivial | Original Kozeny-Carman: `K ~ d^2*phi^3/(1-phi)^2` with fixed tortuosity, d in micrometers. |
| `ModKozCarm.m` | `modified_kozeny_carman_perm` | trivial | Includes percolation porosity. |
| `Owolabi.m` | `owolabi_perm` | trivial | Owolabi et al. (1994) empirical regressions for unconsolidated Niger Delta sands (separate oil/gas fits). |
| `PandaLake.m` | `panda_lake_perm` | trivial | Panda-Lake (1994) modified Kozeny-Carman from particle-size-distribution statistics. |
| `PandaLakeKCE.m` | `panda_lake_kc_perm` | trivial | Panda-Lake (1994) basic Kozeny-Carman form, `K(md) = 3.34*Dp^2*phi^3/(1-phi)^2`. |
| `PermMenu.m` | `PERM_MODELS registry` | easy | GUI menu replaced by a plain dict `{name: callable}`; 4 of its 14 menu entries (RevilE, Timur, Tixier, WylGregE) are broken in MATLAB — the .m files do not exist (reconstruct from the Handbook as stretch work). |

### `seismic.py` — 1-D wave propagation & synthetics

| MATLAB | Python | Difficulty | Notes |
|---|---|---|---|
| `kennet.m` | `kennett` | moderate | Invariant-imbedding synthetic seismogram. Adopt the odd-length taper `round()` fix from kennett_aux; `hanning(m)` == `np.hanning(m+2)[1:-1]` (endpoint trap); share wavelet/frequency-axis/taper helpers with `propagator_seis`. |
| `kennett_aux.m` | `(drop)` | easy | CR-line-ending near-duplicate of kennet.m (defines `function kennet`!); zero callers; drop after adopting its taper fix; drop its `save omega1D.mat` side effect. |
| `pgator.m` | `propagator_seis` | moderate | Propagator-matrix twin of `kennett`. |
| `kenfdisp.m` | `kennett_frazer_dispersion` | easy | Scattering (stratigraphic) velocity dispersion in 1-D layered media via the Kennett-Frazer recursion. |
| `kenfrtt.m` | `kennett_frazer_traveltimes` | easy | `harmmean` → `scipy.stats.hmean` (or explicit 1/mean(1/x)). |
| `ezseis.m` | `quick_seismic_section` | moderate | Strip GUI; `fir1`/`filtfilt`/`decimate`/`boxcar`/`mean2` → `scipy.signal` equivalents. |
| `(missing) sourcewvlt.m` | `ricker` | n/a | Called by kennett/pgator as default wavelet but absent from the distribution. Provide a documented `ricker()` default and require explicit wavelets elsewhere. |

### `avo.py` — reflectivity & impedance

| MATLAB | Python | Difficulty | Notes |
|---|---|---|---|
| `avopp.m` | `avo_pp` | easy | Full Zoeppritz + approximations; factor shared interface-average and Zoeppritz-coefficient helpers used by all three AVO files. |
| `avops.m` | `avo_ps` | easy | P-to-S converted-wave reflectivity Rps(theta) at an interface between two isotropic half-spaces. |
| `avo_abe.m` | `avo_attributes` | easy | Intercept/gradient attributes; derive from the shared coefficient helpers so curves and attributes cannot drift apart (MATLAB duplicates the algebra inline). |
| `eimp.m` | `elastic_impedance` | moderate | MERGE with eimp2.m: same math, reflection- vs incidence-angle parameterization as a mode argument or two thin wrappers. |
| `eimp2.m` | `elastic_impedance_inc` | moderate | Same as eimp. |

### `signal.py` — signal utilities

| MATLAB | Python | Difficulty | Notes |
|---|---|---|---|
| `fftplot.m` | `spectrum` | trivial | Split: `spectrum()` returns (f, amplitude, phase); plotting helper lives in `plotting.py`. |
| `iatrib.m` | `instantaneous_attributes` | easy | `scipy.signal.hilbert`; strip plotting. |
| `blockav.m` | `block_average` | easy | Block-averaging (upscaling) of well logs: replaces each consecutive block of nb samples with its NaN-ignoring arithmetic mean, output resampled back to the original length. |
| `ft1axis.m` | `fft_axis` | trivial | MERGE ft1axis/ft2axis into one frequency-axis helper. |
| `ft2axis.m` | `(merged)` | trivial | Exact twin of ft1axis. |

### `stats.py` — statistics & classification

| MATLAB | Python | Difficulty | Notes |
|---|---|---|---|
| `bayesclass.m` | `bayes_classify` | easy | Port the ROOT version (bin-edges-from-centers logic). NOTE: `private/bayesclass.m` is a DIFFERENT, cruder implementation — do not conflate. |
| `hist2d.m` | `hist2d` | easy | Reimplement on `np.histogram2d` preserving center-based bin semantics. |
| `hist3d.m` | `hist3d` | easy | Reimplement on `np.histogramdd`; MATLAB version has a broken weighted-2D fallback (calls hist2d with 4 args) and calls a missing `hist1d` — the numpy rewrite fixes both for free. |
| `monte.m` | `monte_carlo_cdf` | easy | Non-parametric marginal-CDF draws; strip plotting; `rng` parameter for reproducibility. |
| `monteccdf.m` | `monte_carlo_ccdf` | moderate | Conditional-CDF variant; 3 internal subfunctions become private module functions. |
| `pdfbayes.m` | `pdf_bayes` | hard | Its computational engines (`pdfgendraw`, `pdfstat`) and ten GUI plotters are MISSING from the distribution — approximate with SciPy (`gaussian_kde` + `histogramdd`) in Phase 8; see Section 8. |

### `io.py` — data I/O

| MATLAB | Python | Difficulty | Notes |
|---|---|---|---|
| `loadlas.m` | `load_las` | easy | Simple LAS 2.0 reader returning header + numpy array (or thin optional wrapper over `lasio`). |

### `plotting.py` — optional matplotlib helpers

| MATLAB | Python | Difficulty | Notes |
|---|---|---|---|
| `logax.m` | `depth_axis` | trivial | Tiny matplotlib helper (reversed depth axis). |
| `(from hash/hashv/fftplot)` | `plot_bounds, plot_spectrum` | n/a | Plot companions to the compute functions, kept out of the numerics. |

### `Dropped` — MATLAB GUI / orphans (no port)

| MATLAB | Python | Difficulty | Notes |
|---|---|---|---|
| `bwoutdlg.m` | `(drop)` | trivial | Batzle-Wang output dialog; pure GUI. |
| `outputdlg.m` | `(drop)` | trivial | Generic output dialog (actually defines `function outdlg`); zero callers. |
| `private/begin.m` | `(drop)` | trivial | Orphan script; loads `vpvsrhodata.mat` which is absent from the repo. |
| `private/bayes.m` | `(stretch)` | moderate | Confusion-matrix helper for the missing pdfbayes engine. |
| `private/bayesclass.m` | `(drop)` | easy | Cruder shadow of root bayesclass.m (see stats.py note). |
| `private/centropy.m` | `(stretch)` | moderate | Conditional entropy; calls missing `entropdf`. |
| `private/colormarkerset.m` | `(drop)` | trivial | Plot styling; calls missing `str2cell`. |
| `private/cpdf.m` | `(stretch)` | moderate | Class-conditional pdf; calls missing `histnd`. |
| `countsegy.mat` | `(drop)` | — | Orphan scalar counter; nothing in RPHtools reads it. |
| `Contents.asv, hudson1.m_old` | `(drop)` | — | Editor backup files. |

---

## 6. Missing functions: reconstruction policy

| Missing | Called by | Policy |
|---|---|---|
| `v2ku`, `v2lm` | `hertzmindv.m`; Contents | **Reconstruct now** (one-line inverses of `ku2v`/`lm2v`). |
| `sourcewvlt` | `kennet.m`, `pgator.m` | **Replace**: provide a documented `ricker()` default; the original wavelet is unknown, so the default is new behavior (noted in docstrings). |
| `hist1d` | `hist3d.m` | **Obsolete**: `np.histogram` covers it; the numpy rewrite of `hist3d` removes the call. |
| `Timur`, `Tixier`, `RevilE`, `WylGregE` | `PermMenu.m` menu | **Stretch**: reconstruct from the Handbook's permeability formulas (each is a one-liner); until then the registry ships without them and the README says so. |
| `walton`, `waltonv`, `squirt`, `stdlin`, `Unconsol`, `v2cti`, `Rorsym`, `Rruger` | nothing (Contents only) | **Stretch**: optional reconstructions from the Handbook to restore parity with the book's index; not needed for any existing code path. |
| `pdfgendraw`, `pdfstat`, `histnd`, `entropdf`, `str2cell`, ten `figure_*` plotters | `pdfbayes.m`, `private/` | **Approximate with SciPy** (see Section 8), decided by owner; scheduled in Phase 8. |

---

## 7. Cross-cutting porting decisions

### 7.1 GUI policy — no GUIs
All dialogs (`inputdlg`, `listdlg`, `uicontrol`, `menu`, `evalin('base')`) are
removed. `PermMenu` becomes a `PERM_MODELS: dict[str, Callable]` registry;
`flpropui`/`bwoutdlg`/`outputdlg` are dropped outright (their computation
already lives in `flprop`). Nothing interactive survives in the numerics.

### 7.2 Plotting policy — compute returns data, plotting is separate
Functions that optionally plotted (36 files) return arrays only. Files whose
plot *was* the product (`fftplot`, `logax`, the `hash`/`hashv` bound plots)
split into a compute function plus a small helper in `plotting.py`, which
imports matplotlib lazily so the core package never requires it.

### 7.3 MATLAB → Python equivalences to standardize
| MATLAB | Python |
|---|---|
| `ode45m` + `global DEMINPT` + `feval('demyprime')` | `scipy.integrate.solve_ivp` with a closure passing parameters explicitly |
| legacy `bessel(0/1, x)` in `biot.m` | `scipy.special.jv(0/1, x)` |
| `hanning(m)` | `np.hanning(m + 2)[1:-1]` — **endpoint semantics differ; single most likely silent numerical discrepancy in the seismic module** |
| `fir1`/`filtfilt`/`decimate`/`boxcar`/`mean2` (`ezseis`) | `scipy.signal.firwin`/`filtfilt`/`decimate`, `np.mean` |
| `harmmean` (`kenfrtt`) | `scipy.stats.hmean` |
| `hilbert` (`iatrib`) | `scipy.signal.hilbert` |
| `interp1`/`interp2` (`co2prop` et al.) | `np.interp` / `scipy.interpolate.RegularGridInterpolator` |
| `hist`-family center-based bins | `np.histogram`/`histogram2d`/`histogramdd` with explicit center→edge conversion (preserve MATLAB bin semantics; test) |
| `ifft`/`fft` analysis–synthesis pairing (`kennet`, `pgator`) | ports directly — NumPy's 1/n placement matches MATLAB's |
| `load co2propdata.mat` | packaged `data/co2prop.npz` + loader |
| `save omega1D.mat` side effect (`kennett_aux`) | dropped |

### 7.4 Deliberate behavior changes (the change log)
1. `hist3d` weighted-2D path works (broken call into `hist2d` in MATLAB).
2. `berryscp` pressure-step selection uses a deterministic tie-break
   (`find(k==max(k))` bug).
3. `backus_average` always normalizes the fractions by their sum (as
   `bkusc.m` did; raw thicknesses are accepted) and raises on non-finite or
   negative fractions instead of `bkus.m`'s `pause`. Its runtime
   `c66 == (c11-c12)/2` self-check is dropped: the equality is a mathematical
   identity of the Backus average (both sides reduce to `sum(f*mu)`), so the
   check could never fire; it is asserted in the test suite instead.
4. `backus_average_log` raises on non-monotonic depth.
5. `gassmann_vel` inherits `gassmann_k`'s `phi == 0` pass-through guard
   (MATLAB's inline copy lacked it).
6. Biot-family functions drop the accepted-but-unused mineral shear modulus
   argument (documented).
7. All argument-order inconsistencies between MATLAB twins (`bkus(f,r,vp,vs)`
   vs `bkusc(f,vp,vs,den)`) are unified to one order: `(f, vp, vs, rho)`.
8. Permeability functions return arrays/dataclasses, never the MATLAB
   `[Phi K]` horizontal concat (which silently produced a `1×2N` row for row
   inputs).
9. `hudson_fisher` fixes two bugs found in `hudsonF.m` during Phase 3:
   (a) the output-density crack porosity was `4*pi*ar/(3*cd)` — dividing by
   crack density instead of multiplying (`(4*pi/3)*ar*cd` restored); (b) the
   shear components `c2323`/`c1313`/`c1212` were missing a `mu^2` factor in
   their U3 terms, violating the exact TI identity `c66 = (c11-c12)/2` that
   any orientation-averaged medium must satisfy (verified against Hudson's
   `<M_ij M_kl>` structure; a test asserts the identity).
10a. Phase 4 findings in the granular and permeability files:
    (a) `Johnson.m` overwrites its 6x6 stiffness-tensor output with a *scalar*
    contact constant — both are named `C`, so the documented "Cijkl
    anisotropic stiffness tensor" output was that scalar;
    `johnson_stress_anisotropy` returns the tensor.
    (b) `John_Makse.m` cannot run as shipped: it uses the coordination
    number `Z` two lines before assigning it, and stores an undefined `C12`
    into its stiffness matrix. `johnson_makse` is a reconstruction that
    starts the iteration at `Z = 6` and takes `C12 = C11 - 2*C66` from
    `Johnson.m`.
    (c) `BernabeE.m` is unusable non-interactively: it tests `nargin == 5`
    for a four-argument function, calls its inner `Ber1` without the
    porosity argument, and never assigns that call's result to its output.
    (d) `Bloch.m` returns porosity in percent while plotting it as a
    fraction; the port returns a fraction throughout.
    (e) `Cem.m` hard-codes `3.14` for pi in the two cement Lambda
    parameters; the port uses `pi` per the Handbook definition (<0.1%).
10. `hudson_cone` follows the MATLAB matrix assembly (`c12` slot filled with
    `c11 - 2*c66`); the `c12cor` formula printed in `hudsoncone.m` is
    computed there but never used, and disagrees with `c11 - 2*c66` at
    nonzero cone angle — one of the two carries a typo in the original.
    The cone angle is taken in degrees (MATLAB: radians).

---

## 8. The `pdfbayes` statistics stack — approximate with SciPy

`pdfbayes.m` (non-parametric PDF estimation, Bayes error, information) is the
one file whose core is unrecoverable from this repository: `pdfgendraw` and
`pdfstat` are missing, as are the primitives under `private/`'s call graph.
Porting it means **re-deriving** the pipeline (class-conditional histogram PDFs
→ Gaussian kernel smoothing → Bayes confusion/error → conditional entropy),
for which `scipy.stats.gaussian_kde` and `np.histogramdd` are the natural
tools, with `private/bayes.m`, `private/centropy.m`, and `private/cpdf.m`
serving as partial specifications. This is planned as a separate, final work
item — it must not block the 90 portable functions. **Decision (owner,
2026-07-27): approximate with SciPy.** The port will land in Phase 8 as
`stats.pdf_bayes()` built on `scipy.stats.gaussian_kde` and `np.histogramdd`,
with its differences from the original MATLAB clearly documented in the
docstring.

---

## 9. Testing and validation

**Golden-value tests (primary).** A committed Octave script
(`tests/generate_golden.m`) runs the original `.m` files (computational cores
only — GUI-stripped where needed) over a grid of physically meaningful inputs
and writes JSON/CSV fixtures to `tests/golden/`. Python tests assert agreement
to `rtol=1e-10` for algebraic functions and documented looser tolerances for
iterative solvers (SC fixed points, ODE integration, FFT-based synthetics).
Fixtures are committed so CI never needs Octave; regeneration is documented.

**Analytic invariants (property tests), for example:**
- Round-trips: `ti_c_to_s` twice = identity; `moduli_to_velocity` ∘
  `velocity_to_moduli` = identity; Bond rotation by 0° = identity, by 90°
  four times = identity; `C @ S = I` for `isotropic_cs`.
- Degenerate/limiting cases: Backus average of identical layers is isotropic
  with `c11 = c33 = rho*vp²`; Hashin–Shtrikman bounds coincide for a single
  phase; Gassmann with `k_fl1 == k_fl2` is the identity; Hudson with zero
  crack density returns the isotropic background; DEM at zero porosity returns
  the mineral; `dem` and `dem_at_fraction` agree at shared fractions;
  `biot_hf` matches the `biothfgs.m` re-derivation and `biot_dispersion`'s
  high-frequency limit; Zoeppritz at normal incidence equals the acoustic
  reflection coefficient; elastic impedance at 0° reduces to acoustic
  impedance.
- Self-consistency: `berryman_sc` (sweep) vs `berryman_scm` (core) agree;
  `avo_attributes` intercept/gradient match a fit to the `avo_pp` curve;
  merged twins (`backus_average` vs `backus_average_c`, `hudson` vs
  `hudson_velocities`) are mutually consistent by construction and by test.
- Published values: spot checks against tables/figures in the Handbook
  (e.g. quartz `isotropic_cs`, Hertz–Mindlin coordination table p. 150).

**Infrastructure:** `pytest` (+`numpy.testing`), `ruff` for lint/format,
GitHub Actions CI on Python 3.10–3.13. Every function ships with at least one
golden test or one invariant test; most get both.

---

## 10. Implementation phases

Each phase is independently shippable and ends green (tests + lint). Counts
are ported public functions.

| Phase | Scope | Modules | ~Count | Effort |
|---|---|---|---|---|
| 0 | Scaffolding: package skeleton, `pyproject.toml`, CI, test harness, golden-value generator, convert `co2propdata.mat` | — | — | S |
| 1 | Foundations: conversions, tensor utilities, Backus, bounds (incl. reconstructed `v2ku`/`v2lm`) | `moduli`, `tensors`, `layered`, `bounds` | 17 | M |
| 2 | Fluids: Gassmann, Brown–Korringa, Biot family, squirt-TI, patchy, Batzle–Wang, CO2 | `fluids`, `fluid_properties` | 15 | M–L |
| 3 | Effective media: Berryman SC, DEM (SciPy ODE), Hudson family, Eshelby–Cheng | `effective_medium`, `cracks` | 11 | L |
| 4 | Granular & permeability: Hertz–Mindlin, cementation, Johnson/Johnson–Makse, 10 perm models + registry | `granular`, `permeability` | 16 | M–L |
| 5 | AVO & impedance: Zoeppritz + approximations, attributes, elastic impedance | `avo` | 5 | M |
| 6 | Seismic & signal: Kennett, propagator, dispersion/traveltimes, quick sections, wavelets, spectra, attributes | `seismic`, `signal` | 10 | L |
| 7 | Statistics & I/O: Bayes classification, histograms, Monte Carlo, LAS reader, plotting helpers | `stats`, `io`, `plotting` | 8 | M |
| 8 | Polish & stretch: README mapping table, examples, missing-function reconstructions (perm four, `walton`/`squirt`/…), `pdf_bayes` SciPy approximation | — | 0–12 | M |

Rationale for the order: Phase 1 unblocks every cluster (Brown–Korringa needs
`isotropic_cs`; Backus needs `ti_velocities`; Hertz–Mindlin needs the
conversions); fluids (Phase 2) is the highest-value, most-used part of the
toolbox; everything after is dependency-free and ordered by expected use.

---

## 11. Risks and open questions

1. **License.** The repository has no LICENSE file and the MATLAB code is the
   book's companion software. Before publishing the port beyond this fork, the
   redistribution terms should be confirmed. *(User decision.)*
2. **`sourcewvlt` is unknowable.** Any default wavelet for `kennett`/
   `propagator_seis` is new behavior; mitigated by requiring/encouraging an
   explicit wavelet argument.
3. **Golden values need Octave-compatible originals.** A few files need
   temporary GUI-stripping or legacy-function shims (`bessel`) to run in
   Octave; the shimmed copies live only in the golden-generation script.
4. **Hanning/taper endpoint semantics** in the seismic module are the most
   likely source of silent numerical drift — covered by dedicated
   fixture tests on the taper itself, not just end-to-end synthetics.
5. **`pdf_bayes` reconstruction scope** — resolved: the owner chose
   approximation with modern SciPy tools (see Section 8); scheduled for
   Phase 8 and does not block the rest of the port.

---

*Prepared from a file-by-file analysis of all 92 MATLAB sources in
`RPHtools/`, with every internal dependency cross-checked against the code.*
