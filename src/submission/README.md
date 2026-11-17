# Borrow Khan technical submission

Four-page English technical report using the IEEEtran conference template:
two columns, 10 pt body text, and eight explanatory/data figures. Author names
and registered team name are in `team.tex`.
The described algorithm is frozen at code revision `188b66d` on `taufiq` (macro v10).
The report explains team ownership before alive/dead refinement, then validation,
error tradeoffs and match-statistic denominators. Four full pages retain the
IEEE 10 pt body, eight figures and two explanatory tables.

The separately requested evaluation tables, time-fold results and measured
learning curve are in [bench/EVALUATION_BUNDLE.md](../bench/EVALUATION_BUNDLE.md).
They export to `build/Borrow_Khan_Evaluation.zip` and are not required by the
IEEE document's build or Overleaf archive.

Build from the repository root with Python 3, Matplotlib and either Tectonic or pdfLaTeX:

```bash
python3 src/submission/build.py
# Or select a compiler explicitly:
python3 src/submission/build.py --engine /path/to/tectonic
# Or use a separate Python environment for figure generation:
python3 src/submission/build.py --plot-python .venv-docs/bin/python
```

Tectonic installation: <https://tectonic-typesetting.github.io/install.html>.
The local build can use `cache/document_tools/tectonic` if already installed.
Standard TeX Live/Overleaf provides IEEEtran and the other document packages. Install the plotting
dependency from `src/submission/requirements.txt` in the chosen Python environment.
The local build also detects the existing `.venv-cv` plotting environment.

The outputs are `build/Borrow_Khan.pdf` and `build/Borrow_Khan_Overleaf.zip`.
Upload the PDF as the short technical document; the ZIP is an editable source
project with `Borrow_Khan.tex` as its main file. It does not contain match assets.
Generated PDFs, archives and auxiliary files remain outside Git.

`figures.py` generates vector PDFs and PNG previews for ablation, paired fold
sensitivity, and predicted match statistics. All chart values are read directly
from the committed aggregate report and copied to `build/figures/data.json`.
The architecture, validation protocol and two labelled illustrative possession
timelines are native LaTeX/TikZ. Illustrations are not presented as match evidence.
The Overleaf archive includes every required vector figure and its aggregate data.

`build.py` regenerates `metrics.tex` from the committed aggregate checkpoint
report. If the local `evaluation/results/barca.csv` exists, its SHA-256 must match
the frozen report. Typesetting otherwise needs no restricted data or models.
The report's reproduction section distinguishes the final inference command
from the preprocessing/model caches that must be prepared first.
