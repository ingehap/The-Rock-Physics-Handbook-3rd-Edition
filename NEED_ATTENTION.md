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

### 2. `pdf_bayes` reconstruction scope (decide by Phase 8)
`pdfbayes.m` cannot be translated because its computational engines
(`pdfgendraw`, `pdfstat`) are missing from this distribution (see
`PORTING_PLAN.md`, Section 8). Options: reconstruct the pipeline from the
surviving `private/` helpers, approximate it with modern SciPy tools
(`gaussian_kde` + `histogramdd`), or drop it.

**Needed from you:** a preference — reconstruct, approximate, or drop. This
does not block any other part of the port.

## Resolved

*(none yet)*
