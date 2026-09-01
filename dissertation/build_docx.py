"""Build Dissertation.docx from the Markdown sources + the Surrey template.

    python dissertation/build_docx.py [--out dissertation/Dissertation.docx]

Sources (this directory):
    meta.json          title / author / pathway / supervisor / date
    abstract.md        the abstract (Unnumbered 1 section of the template)
    NN-<slug>.md       chapters, built in filename order
    references.md      one reference per line, "@key :: full text"
    figures/*.png      referenced by the chapter Markdown

The template's own styles are used throughout -- Heading 1-4 (self-numbering
via numId 4), Body First / Body Text, caption, Reference (self-numbering),
List Bullet, Body Centre, Table Grid, and the Teletype character style for
inline code -- so the output is the school's format, not a lookalike.

Markdown dialect (deliberately small, so the mapping stays auditable):

    # / ## / ### / ####        Heading 1..4        (Word numbers them)
    paragraph                  Body First after a heading, else Body Text
    - item                     List Bullet
    > quote                    Body Text, italic
    ```fenced```               Teletype, one paragraph per line
    $$ x = y $$ {#eq:id}       Body Centre + right-aligned equation number
    ![caption](figures/f.png){#fig:id}
    Table: {#tab:id} caption   followed immediately by a GFM table
    {{fig:id}} {{tab:id}} {{eq:id}}   cross-references, resolved to numbers
    [@key] [@key1;@key2]       citations, numbered in order of first appearance
    **bold**  *italic*  `code`

Figures, tables and equations are numbered per chapter (Figure 3-1, ...) and
the numbers are written literally, so the document is correct the moment it
opens. The heading Table of Contents stays a live Word field: after opening,
right-click it and Update Field (the List of Figures is written statically).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import docx
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
TEMPLATE = os.path.join(REPO, "MScDissertationTemplate2026.docx")
W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"

MAX_FIG_IN = 6.0


# ---------------------------------------------------------------------------
# source loading
# ---------------------------------------------------------------------------
def chapter_files():
    return sorted(f for f in os.listdir(HERE)
                  if re.match(r"^\d\d-.*\.md$", f))


def load_references():
    """references.md -> {key: text}, preserving file order as a tiebreak."""
    path = os.path.join(HERE, "references.md")
    refs, order = {}, []
    if not os.path.exists(path):
        return refs, order
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "::" not in line:
            continue
        key, text = line.split("::", 1)
        key = key.strip().lstrip("-").strip().lstrip("@").strip()
        refs[key] = text.strip()
        order.append(key)
    return refs, order


# ---------------------------------------------------------------------------
# pass 1 -- assign numbers to figures / tables / equations, and to citations
# ---------------------------------------------------------------------------
FIG_RE = re.compile(r"^!\[(?P<cap>.*?)\]\((?P<src>[^)]+)\)\{#(?P<id>fig:[\w.-]+)\}\s*$")
TAB_RE = re.compile(r"^Table:\s*\{#(?P<id>tab:[\w.-]+)\}\s*(?P<cap>.*)$")
EQ_RE = re.compile(r"^\$\$(?P<body>.+?)\$\$(?:\s*\{#(?P<id>eq:[\w.-]+)\})?\s*$")
XREF_RE = re.compile(r"\{\{((?:fig|tab|eq):[\w.-]+)\}\}")
CITE_RE = re.compile(r"\[@([^\]]+)\]")


def scan_numbers(files):
    """{anchor_id: 'C-N'} for every figure, table and equation."""
    nums, ch = {}, 0
    for fn in files:
        ch += 1
        f_i = t_i = e_i = 0
        for line in open(os.path.join(HERE, fn)):
            m = FIG_RE.match(line)
            if m:
                f_i += 1
                nums[m.group("id")] = "%d-%d" % (ch, f_i)
                continue
            m = TAB_RE.match(line)
            if m:
                t_i += 1
                nums[m.group("id")] = "%d-%d" % (ch, t_i)
                continue
            m = EQ_RE.match(line)
            if m and m.group("id"):
                e_i += 1
                nums[m.group("id")] = "%d-%d" % (ch, e_i)
    return nums


class Citations:
    """IEEE-style: numbered in order of first appearance in the text."""

    def __init__(self, refs):
        self.refs = refs
        self.order = []
        self.missing = set()

    def number(self, key):
        if key not in self.refs:
            self.missing.add(key)
        if key not in self.order:
            self.order.append(key)
        return self.order.index(key) + 1

    def render(self, text):
        def sub(m):
            keys = [k.strip().lstrip("@") for k in m.group(1).split(";")]
            return "[" + ", ".join(str(self.number(k)) for k in keys) + "]"
        return CITE_RE.sub(sub, text)


# ---------------------------------------------------------------------------
# inline formatting
# ---------------------------------------------------------------------------
INLINE_RE = re.compile(r"(\*\*.+?\*\*|(?<!\*)\*[^*]+?\*(?!\*)|`[^`]+?`)")


def add_runs(par, text, doc):
    """**bold**, *italic*, `code` (Teletype character style)."""
    for piece in INLINE_RE.split(text):
        if not piece:
            continue
        if piece.startswith("**") and piece.endswith("**"):
            par.add_run(piece[2:-2]).bold = True
        elif piece.startswith("`") and piece.endswith("`"):
            r = par.add_run(piece[1:-1])
            try:
                r.style = doc.styles["Teletype"]
            except KeyError:
                r.font.name = "Courier New"
        elif piece.startswith("*") and piece.endswith("*"):
            par.add_run(piece[1:-1]).italic = True
        else:
            par.add_run(piece)


# ---------------------------------------------------------------------------
# template surgery
# ---------------------------------------------------------------------------
def set_text(par, text):
    """Replace a paragraph's text, keeping its style, dropping any field."""
    for r in list(par.runs):
        r._element.getparent().remove(r._element)
    for child in list(par._p):
        if child.tag in (W + "hyperlink", W + "fldSimple"):
            par._p.remove(child)
    par.add_run(text)


def prepare_front_matter(doc, meta, abstract_paras):
    paras = doc.paragraphs
    # Title page (field-backed placeholders -> literal text)
    set_text(paras[0], meta["title"])
    set_text(paras[2], meta["author"])
    if meta.get("student_id"):
        set_text(paras[4], "Student ID: " + meta["student_id"])
    set_text(paras[5], "Master of Science in " + meta["pathway"])
    set_text(paras[17], meta["date"])
    set_text(paras[18], "Supervised by: " + meta["supervisor"])
    set_text(paras[20], "© " + meta["author"] + " " + meta["year"])
    # Declaration page
    set_text(paras[27], meta["title"])
    set_text(paras[31], meta["author"])
    if meta.get("student_id"):
        set_text(paras[33], "Student ID: " + meta["student_id"])
    set_text(paras[36], "Author Signature: ______________________"
                        "\t\tDate: " + meta["date_signed"])
    set_text(paras[40], "Supervisor's name: " + meta["supervisor"])
    # Instruction paragraphs that must not survive into a submission
    drop = [1, 3, 25, 28, 32, 47, 58, 89]
    # Abstract: replace the template's guidance block (50..56)
    anchor = paras[50]
    set_text(anchor, abstract_paras[0])
    anchor.style = doc.styles["Body First"]
    prev = anchor
    for extra in abstract_paras[1:]:
        new = copy_paragraph_after(prev, doc, "Body Text")
        set_text(new, extra)
        prev = new
    drop += list(range(51, 57))
    for i in sorted(set(drop), reverse=True):
        p = paras[i]._p
        p.getparent().remove(p)
    return paras


def copy_paragraph_after(par, doc, style):
    import copy as _copy
    new_p = _copy.deepcopy(par._p)
    par._p.addnext(new_p)
    new = docx.text.paragraph.Paragraph(new_p, par._parent)
    for r in list(new.runs):
        r._element.getparent().remove(r._element)
    new.style = doc.styles[style]
    return new


def truncate_body(doc):
    """Delete everything from the first Heading 1 ('Introduction') onward,
    leaving the front matter and the trailing sectPr."""
    body = doc.element.body
    start = None
    for i, el in enumerate(body):
        if el.tag != W + "p":
            continue
        par = docx.text.paragraph.Paragraph(el, doc)
        if par.style.name == "Heading 1":
            start = i
            break
    if start is None:
        raise SystemExit("template shape changed: no Heading 1 found")
    for el in list(body)[start:]:
        if el.tag == W + "sectPr":
            continue
        body.remove(el)


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------
def para(doc, style, text=None):
    p = doc.add_paragraph(style=doc.styles[style])
    if text:
        add_runs(p, text, doc)
    return p


def add_table(doc, header, rows, caption, number):
    if caption is not None:
        cap = para(doc, "caption", "Table %s – %s" % (number, caption))
        cap.paragraph_format.keep_with_next = True
    t = doc.add_table(rows=1, cols=len(header))
    t.style = doc.styles["Table Grid"]
    t.autofit = True
    for j, h in enumerate(header):
        cell = t.rows[0].cells[j]
        cell.text = ""
        add_runs(cell.paragraphs[0], h, doc)
        for r in cell.paragraphs[0].runs:
            r.bold = True
            r.font.size = Pt(9)
    for row in rows:
        cells = t.add_row().cells
        for j, v in enumerate(row[:len(header)]):
            cells[j].text = ""
            add_runs(cells[j].paragraphs[0], v, doc)
            for r in cells[j].paragraphs[0].runs:
                r.font.size = Pt(9)
    doc.add_paragraph(style=doc.styles["Body Text"])
    return t


def add_figure(doc, src, caption, number, lof):
    path = src if os.path.isabs(src) else os.path.join(HERE, src)
    holder = doc.add_paragraph(style=doc.styles["Body Centre"])
    holder.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if os.path.exists(path):
        run = holder.add_run()
        try:
            from PIL import Image
            with Image.open(path) as im:
                w_in = min(MAX_FIG_IN, im.width / 150.0)
        except Exception:                                # noqa: BLE001
            w_in = MAX_FIG_IN
        run.add_picture(path, width=Inches(max(2.5, w_in)))
    else:
        holder.add_run("[missing figure: %s]" % src).italic = True
    text = "Figure %s - %s" % (number, caption)
    cap = para(doc, "caption", text)
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    lof.append(text)


def add_equation(doc, body, number):
    p = doc.add_paragraph(style=doc.styles["Body Centre"])
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(body.strip())
    r.italic = True
    if number:
        p.add_run("\t\t(%s)" % number)


def split_row(line):
    return [c.strip() for c in line.strip().strip("|").split("|")]


def render_chapter(doc, path, nums, cites, lof, unnumbered=False):
    lines = open(path).read().split("\n")
    i, first_after_heading = 0, False
    while i < len(lines):
        raw = lines[i]
        line = raw.rstrip()
        s = line.strip()

        if not s:
            i += 1
            continue

        # --- headings -------------------------------------------------
        m = re.match(r"^(#{1,4})\s+(.*)$", s)
        if m:
            lvl, text = len(m.group(1)), m.group(2).strip()
            if unnumbered:
                # The appendix must not take a chapter number, and its
                # subsections must not number under the last chapter.
                if lvl == 1:
                    para(doc, "Unnumbered 1", resolve(text, nums, cites))
                else:
                    q = para(doc, "Body First",
                             resolve(text, nums, cites))
                    for r in q.runs:
                        r.bold = True
            else:
                para(doc, "Heading %d" % lvl, resolve(text, nums, cites))
            first_after_heading = True
            i += 1
            continue

        # --- figure ---------------------------------------------------
        m = FIG_RE.match(s)
        if m:
            add_figure(doc, m.group("src"),
                       resolve(m.group("cap"), nums, cites),
                       nums[m.group("id")], lof)
            first_after_heading = False
            i += 1
            continue

        # --- equation -------------------------------------------------
        m = EQ_RE.match(s)
        if m:
            add_equation(doc, m.group("body"),
                         nums.get(m.group("id")) if m.group("id") else None)
            first_after_heading = False
            i += 1
            continue

        # --- table ----------------------------------------------------
        m = TAB_RE.match(s)
        if m:
            j = i + 1
            while j < len(lines) and not lines[j].strip().startswith("|"):
                j += 1
            header = split_row(lines[j])
            j += 2                                   # skip the |---| rule
            body = []
            while j < len(lines) and lines[j].strip().startswith("|"):
                body.append([resolve(c, nums, cites) for c in split_row(lines[j])])
                j += 1
            add_table(doc, [resolve(h, nums, cites) for h in header], body,
                      resolve(m.group("cap"), nums, cites), nums[m.group("id")])
            first_after_heading = False
            i = j
            continue

        # --- bare table (no caption line) ------------------------------
        if (s.startswith("|") and i + 1 < len(lines)
                and set(lines[i + 1].strip()) <= set("|-: ")
                and "-" in lines[i + 1]):
            header = split_row(s)
            j = i + 2
            body = []
            while j < len(lines) and lines[j].strip().startswith("|"):
                body.append([resolve(c, nums, cites)
                             for c in split_row(lines[j])])
                j += 1
            add_table(doc, [resolve(h, nums, cites) for h in header], body,
                      None, None)
            first_after_heading = False
            i = j
            continue

        # --- fenced code ----------------------------------------------
        if s.startswith("```"):
            j = i + 1
            while j < len(lines) and not lines[j].strip().startswith("```"):
                p = doc.add_paragraph(style=doc.styles["Body Text"])
                r = p.add_run(lines[j])
                try:
                    r.style = doc.styles["Teletype"]
                except KeyError:
                    r.font.name = "Courier New"
                p.paragraph_format.space_after = Pt(0)
                j += 1
            i = j + 1
            continue

        # --- bullets ---------------------------------------------------
        if s.startswith("- "):
            para(doc, "List Bullet", resolve(s[2:], nums, cites))
            first_after_heading = False
            i += 1
            continue

        # --- block quote -----------------------------------------------
        if s.startswith("> "):
            p = para(doc, "Body Text", resolve(s[2:], nums, cites))
            for r in p.runs:
                r.italic = True
            first_after_heading = False
            i += 1
            continue

        # --- body paragraph (may wrap over several source lines) --------
        buf = [s]
        j = i + 1
        while j < len(lines):
            nxt = lines[j].strip()
            if (not nxt or nxt.startswith(("#", "- ", "> ", "|", "```", "!["))
                    or TAB_RE.match(nxt) or EQ_RE.match(nxt)):
                break
            buf.append(nxt)
            j += 1
        style = "Body First" if first_after_heading else "Body Text"
        para(doc, style, resolve(" ".join(buf), nums, cites))
        first_after_heading = False
        i = j
    return


def resolve(text, nums, cites):
    def xref(m):
        key = m.group(1)
        label = {"fig": "Figure", "tab": "Table", "eq": ""}[key.split(":")[0]]
        n = nums.get(key)
        if n is None:
            return "[[unresolved %s]]" % key
        return ("(%s)" % n) if not label else "%s %s" % (label, n)
    return cites.render(XREF_RE.sub(xref, text))


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "Dissertation.docx"))
    args = ap.parse_args()

    meta = json.load(open(os.path.join(HERE, "meta.json")))
    abstract = [p.strip().replace("\n", " ") for p in
                open(os.path.join(HERE, "abstract.md")).read().split("\n\n")
                if p.strip() and not p.strip().startswith("#")]
    refs, _ = load_references()
    files = chapter_files()
    if not files:
        raise SystemExit("no chapter files (NN-*.md) in %s" % HERE)

    nums = scan_numbers(files)
    cites = Citations(refs)

    doc = docx.Document(TEMPLATE)
    prepare_front_matter(doc, meta, abstract)
    truncate_body(doc)

    lof: list[str] = []
    for fn in files:
        render_chapter(doc, os.path.join(HERE, fn), nums, cites, lof)

    # ---- References (Unnumbered 1 + self-numbering Reference style) ----
    para(doc, "Unnumbered 1", "References")
    for key in cites.order:
        para(doc, "Reference", refs.get(key, "MISSING REFERENCE: @" + key))

    # ---- Appendix (unnumbered, after the references) -------------------
    appendix = os.path.join(HERE, "appendix.md")
    if os.path.exists(appendix):
        render_chapter(doc, appendix, nums, cites, lof, unnumbered=True)

    # ---- static List of Figures, injected after the template's heading --
    inject_list_of_figures(doc, lof)

    words = body_word_count(doc)
    stamp_word_count(doc, words)
    force_field_update_on_open(doc)
    doc.save(args.out)

    # ---- report ---------------------------------------------------------
    missing_figs = [s for s in lof if s.startswith("[missing")]
    unresolved = 0
    for p in doc.paragraphs:
        unresolved += p.text.count("[[unresolved")
    print("chapters      : %d (%s)" % (len(files), ", ".join(files)))
    print("figures       : %d   tables/eqs numbered: %d"
          % (len(lof), len(nums)))
    print("references    : %d cited of %d defined" % (len(cites.order), len(refs)))
    print("word count    : ~%d (chapters + tables; excludes front matter, "
          "captions and references)" % words)
    print("unresolved refs: %d" % unresolved)
    if cites.missing:
        print("MISSING REFERENCES: %s" % ", ".join(sorted(cites.missing)))
    uncited = [k for k in refs if k not in cites.order]
    if uncited:
        print("uncited (dropped from the list): %s" % ", ".join(sorted(uncited)))
    if missing_figs:
        print("MISSING FIGURE FILES: %d" % len(missing_figs))
    print("wrote %s" % args.out)
    print("\nWord will refresh the Table of Contents and page count when the "
          "file is opened (w:updateFields is set). If your Word build asks "
          "\"update fields?\", answer YES. If the TOC still shows template "
          "chapters, right-click it -> Update Field -> Update entire table.")
    return 1 if (cites.missing or unresolved) else 0


def inject_list_of_figures(doc, lof):
    """Replace the template's List-of-Figures field with a static list."""
    body = doc.element.body
    target = None
    for i, el in enumerate(body):
        if el.tag != W + "p":
            continue
        p = docx.text.paragraph.Paragraph(el, doc)
        if p.style.name == "Unnumbered 1" and p.text.strip() == "List of Figures":
            target = (i, p)
            break
    if target is None:
        return
    idx, heading = target
    # drop the stale table-of-figures entries that follow the heading
    for el in list(body)[idx + 1:]:
        if el.tag != W + "p":
            break
        p = docx.text.paragraph.Paragraph(el, doc)
        if p.style.name in ("table of figures", "Body Text") and \
                (p.text.strip().startswith("Figure") or not p.text.strip()
                 or p.text.strip().startswith("<")):
            body.remove(el)
            continue
        break
    prev = heading
    for entry in lof:
        new = copy_paragraph_after(prev, doc, "table of figures")
        new.add_run(entry)
        prev = new


def body_word_count(doc):
    """Chapters only: from the first Heading 1 up to the References heading.
    Front matter, the TOC, the reference list and captions are excluded."""
    total, started = 0, False
    for p in doc.paragraphs:
        style = p.style.name
        if not started:
            started = style == "Heading 1"
            if not started:
                continue
        if style == "Unnumbered 1" and p.text.strip() in ("References",
                                                          "Appendix"):
            break
        if style in ("caption", "table of figures"):
            continue
        total += len(p.text.split())
    for t in doc.tables:
        for row in t.rows:
            for c in row.cells:
                total += len(c.text.split())
    return total


def add_field(par, instr):
    """Insert a real Word field (begin / instrText / separate / end) so Word
    computes the value itself."""
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement
    r1 = OxmlElement("w:r"); fc = OxmlElement("w:fldChar")
    fc.set(qn("w:fldCharType"), "begin"); r1.append(fc); par._p.append(r1)
    r2 = OxmlElement("w:r"); it = OxmlElement("w:instrText")
    it.set(qn("xml:space"), "preserve"); it.text = instr
    r2.append(it); par._p.append(r2)
    r3 = OxmlElement("w:r"); fc2 = OxmlElement("w:fldChar")
    fc2.set(qn("w:fldCharType"), "separate"); r3.append(fc2); par._p.append(r3)
    r4 = OxmlElement("w:r"); t = OxmlElement("w:t"); t.text = "1"
    r4.append(t); par._p.append(r4)
    r5 = OxmlElement("w:r"); fc3 = OxmlElement("w:fldChar")
    fc3.set(qn("w:fldCharType"), "end"); r5.append(fc3); par._p.append(r5)


def force_field_update_on_open(doc):
    """Set w:updateFields so Word refreshes the Table of Contents (and the page
    count) when the document is opened, instead of leaving the template's stale
    placeholder chapters on display."""
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement
    settings = doc.settings.element
    for existing in settings.findall(qn("w:updateFields")):
        settings.remove(existing)
    el = OxmlElement("w:updateFields")
    el.set(qn("w:val"), "true")
    settings.insert(0, el)


def stamp_word_count(doc, words):
    for p in doc.paragraphs:
        t = p.text.strip()
        if t.startswith("Number of Words"):
            set_text(p, "Number of Words:\t%d (body text; excludes front "
                        "matter, references and appendices)" % words)
        elif t.startswith("Number of Pages"):
            set_text(p, "Number of Pages:\t")
            add_field(p, " NUMPAGES  \\* MERGEFORMAT ")


if __name__ == "__main__":
    sys.exit(main())
