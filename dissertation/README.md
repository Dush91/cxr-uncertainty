# Dissertation sources

The MSc dissertation is written as Markdown here and built into the University
of Surrey template by `build_docx.py`. The Word file is a build artifact; edit
the Markdown, not the `.docx`.

## Layout

| Path | What |
|---|---|
| `meta.json` | title, author, pathway, supervisor, dates |
| `abstract.md` | the abstract (200–300 words) |
| `NN-*.md` | the ten chapters, built in filename order |
| `appendix.md` | unnumbered appendix, rendered after the references |
| `references.md` | `@key :: full text`, one per line |
| `figures/` | every figure referenced by the chapters |
| `build_docx.py` | Markdown + template → `Dissertation.docx` |
| `make_figures.py` | regenerates `figures/` from saved run artifacts |
| `verify_numbers.py` | checks every headline number against its source JSON |

## Build

```bash
pip install python-docx                 # only extra dependency
python dissertation/make_figures.py     # regenerate figures (optional)
python dissertation/verify_numbers.py   # 85 checks against runs/*.json
python dissertation/build_docx.py       # -> dissertation/Dissertation.docx
```

`build_docx.py` reports the chapter list, figure and table counts, the number of
references cited against defined, the body word count, and any unresolved
cross-reference or missing reference. It exits non-zero if anything is
unresolved, so it is safe to run in a check.

## After building — the one manual step

Open `Dissertation.docx` in Word and **right-click the Table of Contents →
Update Field → Update entire table**. The heading TOC is a live Word field and
still shows the template's placeholder chapters until it is refreshed. The List
of Figures is written statically and needs no update.

Also before submission: fill in the supervisor's name and signature date in
`meta.json`, and replace the repository URL placeholder in `appendix.md`.

## Markdown dialect

Deliberately small, so the mapping to Word styles stays auditable.

| Markdown | Word style |
|---|---|
| `#` `##` `###` `####` | Heading 1–4 (Word supplies the numbers) |
| paragraph after a heading | Body First |
| other paragraphs | Body Text |
| `- item` | List Bullet |
| `> quote` | Body Text, italic |
| fenced code | Teletype character style |
| `![caption](figures/f.png){#fig:id}` | centred picture + numbered caption |
| `Table: {#tab:id} caption` then a GFM table | numbered caption + Table Grid |
| a bare GFM table | Table Grid, no caption |
| `$$ … $$ {#eq:id}` | Body Centre + right-aligned equation number |
| `{{fig:id}}` `{{tab:id}}` `{{eq:id}}` | resolved to "Figure 3-1" etc. |
| `[@key]`, `[@k1;@k2]` | resolved to `[1]`, `[1, 2]` in citation order |
| `**bold**` `*italic*` `` `code` `` | inline formatting |

Figures, tables and equations are numbered per chapter and written literally,
so the document is correct as soon as it opens. References are numbered in
order of first citation and emitted in the template's self-numbering
`Reference` style; an uncited entry is dropped from the list and reported.
