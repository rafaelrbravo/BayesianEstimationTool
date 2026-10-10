"""Shared deterministic Markdown and PDF rendering for BET documentation."""
from pathlib import Path
import re
from xml.sax.saxutils import escape

from reportlab import rl_config
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Preformatted, HRFlowable

rl_config.invariant = 1


def render(markdown, stem):
    """Write matching .md and .pdf files beside docs/generators/."""
    out = Path(__file__).resolve().parents[1]
    out.mkdir(parents=True, exist_ok=True)
    markdown = markdown.strip() + "\n"
    (out / (stem + ".md")).write_text(markdown, encoding="utf-8", newline="\n")
    _pdf(markdown, out / (stem + ".pdf"))


def _inline(s):
    s = escape(s)
    s = re.sub(r"`([^`]+)`", r'<font face="Courier">\1</font>', s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", s)
    return s


def _pdf(markdown, path):
    compact = path.stem.lower() == "cheatsheet"
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="BETTitle", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=19, leading=23, spaceAfter=14))
    styles.add(ParagraphStyle(name="BETHeading", parent=styles["Heading2"], fontSize=13, leading=17, spaceBefore=13, spaceAfter=5))
    styles.add(ParagraphStyle(name="BETSubheading", parent=styles["Heading3"], fontSize=10.5, leading=14, spaceBefore=10, spaceAfter=4))
    styles.add(ParagraphStyle(name="BETBody", parent=styles["BodyText"], fontSize=9.2, leading=13.5, spaceAfter=7))
    styles.add(ParagraphStyle(name="BETCode", fontName="Courier", fontSize=7.1, leading=9.5, leftIndent=8, backColor=colors.HexColor("#f2f4f7"), borderPadding=6, spaceAfter=9))
    styles.add(ParagraphStyle(name="BETList", parent=styles["BETBody"], leftIndent=14, firstLineIndent=-10))
    if compact:
        # Dense reference layout; retain normal typography for the manual.
        styles["BETTitle"].fontSize = 14
        styles["BETTitle"].leading = 16
        styles["BETTitle"].spaceAfter = 5
        styles["BETHeading"].fontSize = 9.5
        styles["BETHeading"].leading = 11.5
        styles["BETHeading"].spaceBefore = 5
        styles["BETHeading"].spaceAfter = 2
        styles["BETBody"].fontSize = 8.2
        styles["BETBody"].leading = 10.1
        styles["BETBody"].spaceAfter = 3.5
        styles["BETCode"].fontSize = 6.8
        styles["BETCode"].leading = 8.2
        styles["BETCode"].spaceAfter = 3
        styles["BETCode"].borderPadding = 3
        styles["BETList"].fontSize = 8.2
        styles["BETList"].leading = 10.1
    margin = 29 if compact else 51
    doc = SimpleDocTemplate(str(path), pagesize=(612,792), rightMargin=margin, leftMargin=margin, topMargin=28 if compact else 48, bottomMargin=28 if compact else 49,
                            title=path.stem, author="BayesianEstimationTool", pageCompression=1, invariant=1)
    story = []
    paragraph = []
    code = []
    in_code = False

    def flush_paragraph():
        if paragraph:
            story.append(Paragraph(_inline(" ".join(paragraph)), styles["BETBody"]))
            paragraph.clear()

    def flush_code():
        if code:
            story.append(Preformatted("\n".join(code), styles["BETCode"], maxLineLength=95, splitChars=" ,"))
            code.clear()

    for line in markdown.splitlines():
        if line.startswith("```"):
            flush_paragraph()
            if in_code:
                flush_code()
            in_code = not in_code
            continue
        if in_code:
            code.append(line)
            continue
        if not line.strip():
            flush_paragraph()
        elif line.startswith("# "):
            flush_paragraph()
            story.append(Paragraph(_inline(line[2:]), styles["BETTitle"]))
        elif line.startswith("## "):
            flush_paragraph()
            story.append(Paragraph(_inline(line[3:]), styles["BETHeading"]))
        elif line.startswith("### "):
            flush_paragraph()
            story.append(Paragraph(_inline(line[4:]), styles["BETSubheading"]))
        elif line.startswith("- "):
            flush_paragraph()
            story.append(Paragraph("&#8226; " + _inline(line[2:]), styles["BETList"]))
        elif line.startswith("|"):
            flush_paragraph()
            story.append(Paragraph(_inline(line), styles["BETBody"]))
        elif line.strip() == "---":
            flush_paragraph()
            story.append(HRFlowable(width="100%", thickness=.5, color=colors.grey, spaceBefore=8, spaceAfter=8))
        else:
            paragraph.append(line.strip())
    flush_paragraph()
    flush_code()
    doc.build(story)
