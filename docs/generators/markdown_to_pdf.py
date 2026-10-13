"""Convert a Markdown document directly to a print-ready PDF.

Usage: python docs/generators/markdown_to_pdf.py docs/Manual.md
Markdown remains authoritative; this script never writes or changes it.
"""
import argparse
from pathlib import Path

import markdown
from weasyprint import HTML, CSS

CSS_TEXT = """
@page { size: Letter; margin: 0.72in 0.76in; }
html { font-family: -apple-system, BlinkMacSystemFont, "DejaVu Sans", sans-serif; color: #1f2328; font-size: 10pt; line-height: 1.55; }
body { margin: 0; overflow-wrap: anywhere; }
h1, h2, h3, h4 { line-height: 1.25; break-after: avoid; font-weight: 650; }
h1 { font-size: 2em; padding-bottom: .3em; border-bottom: 1px solid #d1d9e0; }
h2 { font-size: 1.5em; margin-top: 1.4em; padding-bottom: .25em; border-bottom: 1px solid #d1d9e0; }
h3 { font-size: 1.25em; margin-top: 1.2em; }
p, ul, ol, pre, table, blockquote { margin: 0 0 1em; }
ul, ol { padding-left: 2em; }
li + li { margin-top: .2em; }
code, pre { font-family: "DejaVu Sans Mono", monospace; font-size: .88em; }
code { background: #eff1f3; border-radius: 3px; padding: .12em .3em; }
pre { padding: 1em; background: #f6f8fa; border-radius: 6px; white-space: pre-wrap; overflow-wrap: anywhere; break-inside: avoid; }
pre code { padding: 0; background: transparent; }
table { border-collapse: collapse; width: 100%; font-size: .94em; }
th, td { border: 1px solid #d1d9e0; padding: .4em .7em; text-align: left; }
tr:nth-child(even) { background: #f6f8fa; }
thead { background: #eff1f3; }
blockquote { padding: 0 1em; border-left: .25em solid #d1d9e0; color: #59636e; }
a { color: #0969da; text-decoration: none; }
img { max-width: 100%; }
hr { border: 0; border-top: 1px solid #d1d9e0; }
"""

def convert(source: Path, destination: Path | None = None) -> Path:
    source = source.resolve()
    destination = destination or source.with_suffix(".pdf")
    md = source.read_text(encoding="utf-8")
    html = markdown.markdown(md, extensions=["extra", "sane_lists", "fenced_code", "tables", "toc"])
    HTML(string=f"<!doctype html><html><meta charset='utf-8'><body>{html}</body></html>", base_url=str(source.parent)).write_pdf(str(destination), stylesheets=[CSS(string=CSS_TEXT)])
    return destination

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(convert(args.source, args.output))
