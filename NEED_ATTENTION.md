# Needs Your Attention

Open items that require a decision or action from the repository owner.
Resolved items move to the bottom with their outcome.

## Open

### 1. No LICENSE file in the repository
The repository has no LICENSE file, and the MATLAB code in `RPHtools/` is the
companion software to *The Rock Physics Handbook* (Mavko, Mukerji & Dvorkin,
Cambridge University Press). Before the Python port in `pyRPHtools/` is
published or shared beyond this fork, the redistribution terms of the original
code should be confirmed (e.g. with the authors or the publisher), and a
LICENSE file added stating the terms for both the original MATLAB and the
ported Python code.

**Needed from you:** confirm what license (if any) applies to the original
RPHtools code, and choose a license for the port.

### 2. Nine functions still missing — reconstruction needs the Handbook text
`Contents.m` advertises fifteen functions RPHtools never shipped. Six have
now been reconstructed and verified (`v2ku`, `v2lm`, `v2cti`, `Unconsol`,
`sourcewvlt` as a Ricker replacement, and `pdfbayes` — see
`PORTING_PLAN.md` §6). The remaining nine were **deliberately not
guessed**, because each depends on published empirical constants or a
specific formulation that I could not verify from the code alone, and a
wrong constant in a scientific library is worse than an absent function:

| Missing | What it needs |
|---|---|
| `walton`, `waltonv` | Walton's (1987) rough- and smooth-sphere coefficients |
| `squirt` | The Mavko-Jizba high-frequency unrelaxed-moduli formulation |
| `stdlin` | Which standard-linear-solid parameterization the book uses |
| `Rorsym`, `Rruger` | Rüger's and the orthorhombic symmetry-plane reflectivity forms |
| `Timur`, `Tixier`, `RevilE`, `WylGregE` | Their regression constants and the porosity units they assume (sources disagree) |

**Needed from you:** the relevant Handbook sections (or confirmation of the
constants), and I can implement and test these the same way as the rest.
Everything else in RPHtools that runs is ported.

## Resolved

### `pdf_bayes` reconstruction scope — approximate with SciPy (2026-07-27)
`pdfbayes.m` cannot be translated because its computational engines
(`pdfgendraw`, `pdfstat`) are missing from this distribution. **Decision:**
approximate the pipeline with modern SciPy tools rather than reconstructing
the original engines or dropping the feature. Recorded in `PORTING_PLAN.md`,
Section 8.

**Implemented as `stats.pdf_bayes`.** The surviving helpers
(`private/cpdf.m`, `private/bayes.m`, `private/centropy.m`) showed the
original smoothed a *multidimensional histogram* with a Gaussian kernel, so
the port uses `scipy.ndimage.gaussian_filter` over `np.histogramdd` rather
than the `scipy.stats.gaussian_kde` named when the decision was taken —
`gaussian_filter` reproduces the described algorithm more closely and keeps
the output on the same grid `bayes_classify` consumes. All differences from
the original MATLAB are listed in the `stats.pdf_bayes` Notes section.
