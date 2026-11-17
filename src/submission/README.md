# Technical report

Four pages in IEEE conference format: two columns, 10 pt text, eight figures.
The report covers the final method, approach comparisons, validation, errors and
predicted match statistics. Authors are listed in `team.tex`.

From the repository root:

```bash
python src/submission/build.py
```

Requires Matplotlib (`src/submission/requirements.txt`) and either Tectonic or
pdfLaTeX. Use `--plot-python` or `--engine` to select an installed interpreter or
compiler. The build uses aggregate metrics and does not require match data.

Outputs:

- `build/Borrow_Khan.pdf`: submission document.
- `build/Borrow_Khan_Overleaf.zip`: editable LaTeX source and vector figures.

The main Overleaf file is `Borrow_Khan.tex`. Generated files stay outside Git.
[Evaluation tables and learning curve](../bench/EVALUATION_BUNDLE.md) are exported
separately.
