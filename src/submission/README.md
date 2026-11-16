# Borrow Khan technical submission

English technical report using `IEEEtran` conference layout, with no publication
or affiliation claims. Author names and registered team name are in `team.tex`.
The described algorithm is frozen at code revision `b8e4ad0` on `taufiq`.

Build from the repository root with Python 3 and either Tectonic or pdfLaTeX:

```bash
python3 src/submission/build.py
# Or select a compiler explicitly:
python3 src/submission/build.py --engine /path/to/tectonic
```

Tectonic installation: <https://tectonic-typesetting.github.io/install.html>.
The local build can use `cache/document_tools/tectonic` if already installed.
Standard TeX Live and Overleaf provide the `IEEEtran` class.

The outputs are `build/Borrow_Khan.pdf` and `build/Borrow_Khan_Overleaf.zip`.
Upload the PDF as the short technical document; the ZIP is an editable source
project with `Borrow_Khan.tex` as its main file. It does not contain match assets.
Generated PDFs, archives and auxiliary files remain outside Git.

`build.py` regenerates `metrics.tex` from the committed aggregate checkpoint
report. If the local `evaluation/results/barca.csv` exists, its SHA-256 must match
the frozen report. Typesetting otherwise needs no restricted data or models.
The report's reproduction section distinguishes the final inference command
from the preprocessing/model caches that must be prepared first.
