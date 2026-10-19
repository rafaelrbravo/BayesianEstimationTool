# BayesianEstimationTool (BET)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23431103.svg)](https://doi.org/10.5281/zenodo.23431103)

BayesianEstimationTool fits population-level parameters and individual-specific latent variables with NumPyro inference methods.

## Documentation

Start here:

- **[Manual (Markdown)](docs/Manual.md)** — model contracts, likelihoods, masking, JAX modeling, inference, persistence, and testing.
- **[Manual (PDF)](docs/Manual.pdf)** — printable manual. If GitHub cannot preview the PDF, download it to view locally.
- **[Cheatsheet (Markdown)](docs/Cheatsheet.md)** — compact API reference and common workflows.
- **[Cheatsheet (PDF)](docs/Cheatsheet.pdf)** — one-page printable reference. If GitHub cannot preview the PDF, download it to view locally.

**Manual.md is the authoritative manual.** Its PDF is rendered directly from that file using GitHub-like styling and print pagination. The cheatsheet is a separate compact, Python-generated reference. To regenerate locally, install `requests`, `playwright`, and `reportlab`, install Playwright Chromium (`python -m playwright install chromium`), and set `GITHUB_TOKEN` to a token permitted to call GitHub's Markdown API for this repository. The manual generator uses GitHub's Markdown rendering API and Playwright; the cheatsheet generator uses ReportLab:

```bash
python docs/generators/markdown_to_pdf.py docs/Manual.md
python docs/generators/cheatsheet.py
```

The GitHub Actions test workflow regenerates the PDFs (and the Python-authored cheatsheet Markdown), checks that the cheatsheet is exactly one PDF page, uploads the outputs as a `bet-docs` artifact, and runs the test suite.

## Installation

BET requires Python 3.10 or newer. Install the latest version from [PyPI](https://pypi.org/project/BayesianEstimationTool/):

```bash
python -m pip install BayesianEstimationTool
```

To install the specific first release:

```bash
python -m pip install BayesianEstimationTool==1.0.1
```

Alternatively, install directly from the tagged GitHub source:

```bash
python -m pip install "git+https://github.com/rafaelrbravo/BayesianEstimationTool.git@v1.0.1"
```

You can also download a wheel (`.whl`) or source archive (`.tar.gz`) from [GitHub Releases](https://github.com/rafaelrbravo/BayesianEstimationTool/releases) and install it with `python -m pip install PATH_TO_DOWNLOADED_FILE`. These release assets become available after publishing.

Verify installation:

```bash
python -c "import BayesianEstimationTool as bet; print(bet.Estimator)"
```

For editable development from a source checkout, run `python -m pip install -e .` in the repository root. The conda recipe in `conda-recipe/meta.yaml` is for local builds; BET is not yet published on conda-forge.

See the manual for the minimal model and training examples.

## Bundled examples and PDF documentation

After installing BET, copy its bundled runnable examples and PDF manual/cheatsheet to an editable folder in the current directory (no GitHub download needed):

```bash
python -m BayesianEstimationTool.ExampleInstaller
```

To select a different parent directory:

```bash
python -m BayesianEstimationTool.ExampleInstaller ~/Documents
```

This creates `BET-Examples/` with the six Python examples and a `Documentation/` subfolder containing `Manual.pdf` and `Cheatsheet.pdf`. Run a script with `python BET-Examples/1_SimpleExampleNuts.py`. The installer refuses to overwrite an existing `BET-Examples/` directory. The PDFs and examples are included in new package builds; the currently published 1.0.1 wheel predates this change.

## How to cite

If you use BET in your research, please cite:

Bravo, R., Tercan, B., Lima, E., Aguilar, B., & Yankeelov, T. (2026). *BayesianEstimationTool* (Version 1.0.1) [Computer software]. https://doi.org/10.5281/zenodo.23431103

This DOI identifies all versions of BET. GitHub also provides a **Cite this repository** option using [CITATION.cff](CITATION.cff).
