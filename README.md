# BayesianEstimationTool (BET)

BayesianEstimationTool fits population-level parameters and individual-specific latent variables with NumPyro inference methods.

## Documentation

Start here:

- **[Manual (Markdown)](docs/Manual.md)** — model contracts, likelihoods, masking, JAX modeling, inference, persistence, and testing.
- **[Manual (PDF)](docs/Manual.pdf)** — printable manual. If GitHub cannot preview the PDF, download it to view locally.
- **[Cheatsheet (Markdown)](docs/Cheatsheet.md)** — compact API reference and common workflows.
- **[Cheatsheet (PDF)](docs/Cheatsheet.pdf)** — one-page printable reference. If GitHub cannot preview the PDF, download it to view locally.

The Markdown and PDF files are generated from shared Python sources in `docs/generators/`. To regenerate locally:

```bash
python docs/generators/manual.py
python docs/generators/cheatsheet.py
```

The GitHub Actions test workflow regenerates the documents, checks that the cheatsheet is exactly one PDF page, uploads the outputs as a `bet-docs` artifact, and runs the test suite.

## Installation

Install the package and its dependencies in your environment, then import it as:

```python
import BayesianEstimationTool as bet
```

See the manual for the minimal model and training examples.
