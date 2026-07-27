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

## Resolved

### `pdf_bayes` reconstruction scope — approximate with SciPy (2026-07-27)
`pdfbayes.m` cannot be translated because its computational engines
(`pdfgendraw`, `pdfstat`) are missing from this distribution. **Decision:**
approximate the pipeline with modern SciPy tools
(`scipy.stats.gaussian_kde` + `np.histogramdd`) rather than reconstructing
the original engines or dropping the feature. Differences from the original
MATLAB will be documented in the `stats.pdf_bayes` docstring. Recorded in
`PORTING_PLAN.md`, Section 8.
