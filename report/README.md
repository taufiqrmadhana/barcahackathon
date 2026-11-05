# Submission report

`TECHNICAL_REPORT.md` is the scaffold for the technical document required by
[SUBMISSION.md](../SUBMISSION.md). Fill the `TODO`s; keep it to a few pages.

## Before submitting

- [ ] Pick the submitted pipeline and fill the header (repo URL, pipeline path)
- [ ] Every number in §4 regenerated from the exact submitted file
- [ ] Reproduction command in §7 run from a fresh clone + empty cache → byte-identical CSV
- [ ] External data and models listed with licences (§2)
- [ ] Rename to `<team_name>.md` (or export `<team_name>.pdf`) for the form

## Export to PDF (optional)

```bash
pandoc report/TECHNICAL_REPORT.md -o <team_name>.pdf
```

Existing material to draw from: `src/TECHNICAL.md` (Taufiq's pipeline), `src_s2/README.md` (S4).
