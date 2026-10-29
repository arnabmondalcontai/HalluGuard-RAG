════════════════════════════════════════════════════
 HalluGuard Paper — Compilation Instructions
════════════════════════════════════════════════════

Prerequisites:
  - pdflatex (from TeX Live, MacTeX, or MiKTeX)
  - bibtex

Compilation (4 passes):

  cd paper/
  pdflatex halluguard_paper.tex
  bibtex halluguard_paper
  pdflatex halluguard_paper.tex
  pdflatex halluguard_paper.tex

The first pdflatex pass generates .aux files.
bibtex resolves \cite{} references from halluguard_paper.bib.
The second pdflatex pass incorporates the bibliography.
The third pdflatex pass resolves all cross-references.

Output: halluguard_paper.pdf

Note: The IEEEtran.cls file is required. It is typically
included in standard TeX distributions (TeX Live, MacTeX).
If missing, download from:
  https://ctan.org/pkg/ieeetran

Troubleshooting:
  - If tikz errors occur, ensure tikz and pgf packages are
    installed in your TeX distribution.
  - The paper uses figure* (two-column spanning figure).
    This is standard IEEEtran behaviour.
