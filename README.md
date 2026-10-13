# BayesianEstimationTool (BET)

BayesianEstimationTool fits population-level parameters and individual-specific latent variables with NumPyro inference methods.

## Documentation

Start here:

- **[Manual (Markdown)](docs/Manual.md)** — model contracts, likelihoods, masking, JAX modeling, inference, persistence, and testing.
- **[Manual (PDF)](docs/Manual.pdf)** — printable manual. If GitHub cannot preview the PDF, download it to view locally.
- **[Cheatsheet (Markdown)](docs/Cheatsheet.md)** — compact API reference and common workflows.
- **[Cheatsheet (PDF)](docs/Cheatsheet.pdf)** — one-page printable reference. If GitHub cannot preview the PDF, download it to view locally.

**Manual.md is the authoritative manual.** Its PDF is rendered directly from that file using GitHub-like styling and print pagination. The cheatsheet is a separate compact, Python-generated reference. To regenerate locally (install `markdown`, `weasyprint`, and `reportlab` first):

```bash
python docs/generators/markdown_to_pdf.py docs/Manual.md
python docs/generators/cheatsheet.py
```

The GitHub Actions test workflow regenerates the PDFs (and the Python-authored cheatsheet Markdown), checks that the cheatsheet is exactly one PDF page, uploads the outputs as a `bet-docs` artifact, and runs the test suite.

## Installation

Install the package and its dependencies in your environment, then import it as:

```python
import BayesianEstimationTool as bet
```

See the manual for the minimal model and training examples.
