"""Render Markdown to PDF using GitHub's Markdown API and GitHub Markdown CSS.

Requires GITHUB_TOKEN for private repositories, and Playwright Chromium.
Markdown is read-only and remains the sole source of documentation content.
"""
import argparse
import os
from pathlib import Path

import requests
from playwright.sync_api import sync_playwright

REPO = "rafaelrbravo/BayesianEstimationTool"
CSS_URL = "https://raw.githubusercontent.com/sindresorhus/github-markdown-css/main/github-markdown.css"
PRINT_CSS = """
@page { size: Letter; margin: 0.65in; }
html, body { background: white !important; }
body { margin: 0; }
.markdown-body { box-sizing: border-box; min-width: 0; max-width: none; padding: 0; font-size: 12px; }
.markdown-body pre, .markdown-body table { break-inside: auto; }
.markdown-body h1, .markdown-body h2, .markdown-body h3,
.markdown-body h4 { break-after: avoid; }
.markdown-body pre { white-space: pre-wrap; overflow-wrap: anywhere; }
.markdown-body img { max-width: 100%; }
"""

def convert(source: Path, destination: Path | None = None) -> Path:
    source = source.resolve()
    destination = (destination or source.with_suffix(".pdf")).resolve()
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise RuntimeError("GITHUB_TOKEN is required to render private-repository Markdown through GitHub.")
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
    response = requests.post("https://api.github.com/markdown",
        headers=headers, json={"text": source.read_text(encoding="utf-8"), "mode": "gfm", "context": REPO}, timeout=60)
    response.raise_for_status()
    css_response = requests.get(CSS_URL, timeout=60)
    css_response.raise_for_status()
    html = ("<!doctype html><html><head><meta charset='utf-8'></head>"
            "<body><article class='markdown-body'>" + response.text + "</article></body></html>")
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page()
        page.set_content(html, wait_until="load")
        page.add_style_tag(content=css_response.text)
        page.add_style_tag(content=PRINT_CSS)
        page.emulate_media(media="print")
        page.pdf(path=str(destination), format="Letter", print_background=True, prefer_css_page_size=True)
        browser.close()
    return destination

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(convert(args.source, args.output))
