# BayesianEstimationTool (BET)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23431103.svg)](https://doi.org/10.5281/zenodo.23431103)

BayesianEstimationTool fits population-level parameters and individual-specific latent variables with NumPyro inference methods.

## Installation and examples

BET requires Python 3.10 or newer. Install from [PyPI](https://pypi.org/project/BayesianEstimationTool/):

```bash
python -m pip install BayesianEstimationTool
```

Copy the bundled runnable examples and PDF/Markdown documentation into an editable `BET-Examples/` folder in your current directory:

```bash
python -m BayesianEstimationTool.ExampleInstaller
```

Run an example with `python BET-Examples/1_SimpleExampleNuts.py`. The installer creates a `Documentation/` subfolder with the manual and cheatsheet in Markdown and PDF; it does not download from GitHub or overwrite an existing `BET-Examples/` folder.

## Documentation

Start here:

- **[Manual (Markdown)](docs/Manual.md)** — model contracts, likelihoods, masking, JAX modeling, inference, persistence, and testing.
- **[Manual (PDF)](docs/Manual.pdf)** — printable manual. If GitHub cannot preview the PDF, download it to view locally.
- **[Cheatsheet (Markdown)](docs/Cheatsheet.md)** — compact API reference and common workflows.
- **[Cheatsheet (PDF)](docs/Cheatsheet.pdf)** — one-page printable reference. If GitHub cannot preview the PDF, download it to view locally.

## Installation details

To copy examples to a different parent directory:

```bash
python -m BayesianEstimationTool.ExampleInstaller ~/Documents
```

## Alternative installation methods

Alternatively, install the latest source directly from GitHub:

```bash
python -m pip install "git+https://github.com/rafaelrbravo/BayesianEstimationTool.git"
```

You can also download a wheel (`.whl`) or source archive (`.tar.gz`) from [GitHub Releases](https://github.com/rafaelrbravo/BayesianEstimationTool/releases) and install it with `python -m pip install PATH_TO_DOWNLOADED_FILE`.

Verify installation:

```bash
python -c "import BayesianEstimationTool as bet; print(bet.Estimator)"
```

For editable development from a source checkout, run `python -m pip install -e .` in the repository root. The conda recipe in `conda-recipe/meta.yaml` is for local builds; BET is not yet published on conda-forge.

See the manual for the minimal model and training examples.

## How to cite

If you use BET in your research, please cite:

Bravo, R., Tercan, B., Lima, E., Aguilar, B., & Yankeelov, T. (2026). *BayesianEstimationTool* [Computer software]. https://doi.org/10.5281/zenodo.23431103

This DOI identifies all versions of BET. GitHub also provides a **Cite this repository** option using [CITATION.cff](CITATION.cff).
