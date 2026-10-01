"""Generate BayesianEstimationTool_CheatSheet.pdf.

The cheat-sheet content is intentionally kept at the top of this file so this
source is useful as a readable reference even without generating the PDF.

Dependency:
    pip install reportlab

Run:
    python BayesianEstimationTool_CheatSheet.py

The PDF is written beside this file. The generator uses only ReportLab's
built-in Helvetica/Courier PDF fonts, so it does not depend on system fonts.
It automatically chooses the largest body font that keeps the cheat sheet on
one US Letter page.
"""


# -----------------------------------------------------------------------------
# Cheat-sheet content
# -----------------------------------------------------------------------------

TITLE = "BayesianEstimationTool (BET) Cheat Sheet"

PURPOSE = (
    "`Estimator` organizes hierarchical Bayesian estimation for datasets containing multiple related "
    "entries. Models may contain global parameters shared across entries and standardized local parameters that vary by "
    "entry. `Train` infers the population/global posterior, training locals, and local-parameter correlation. `Test` carries "
    "the learned population state to new entries and can either sample locals from the trained population or infer them from "
    "new observations with a method. `ScoreTrain` and `ScoreTest` evaluate predictions. `Save` and `Load` preserve the "
    "learned state and the settings needed to reconstruct how estimation was performed. Inference is supplied through an "
    "Method interface; built-in methods include NUTS and SVI."
)

DEPENDENCIES = "jax, numpyro, numpy"

INSTALLATION = (
    "Install in editable mode with `python -m pip install -e \"/path/to/BayesianEstimationTool\"` then import with "
    "`import BayesianEstimationTool as bet`. Editable installation means "
    "repository code changes are used without reinstalling. For repository examples, the parent folder may instead be added "
    "to `sys.path`."
)

CONVENTIONS = [
    ("`import BayesianEstimationTool as bet`", "Public import; construct the main object with `bet.Estimator(...)` and choose inference with `Method=bet.Nuts(...)` or `Method=bet.Svi(...)`."),
    ("`arg{}`", "dictionary whose values have no entry dimension; used for one entry's data, local parameters, model output, globals, or other named values."),
    ("`arg{[]}`", "dictionary containing arrays spanning multiple entries or posterior samples."),
    ("`arg[]`", "1D sequence."),
    ("`Fn(...) -> value`", "function argument or method together with the value it returns."),
    ("*Italic arguments*", "accept `None` as a valid input."),
]

FUNCTIONS = [
    (
        'Estimator(modelDataFull{[]}, ModelFn(globalParams{}, localParams{}, data{}) -> modelOut{}, *localParamNames[]*, *GlobalParamFn(dataShapes{}) -> globalParams{}*, choleskyConcentration=2.0, *rngSeed*) -> Estimator',
        "Creates an Estimator. **localParamNames** defines standardized entry-level parameters. With multiple locals, the Estimator learns their correlation with an LKJ Cholesky prior controlled by **choleskyConcentration**. **GlobalParamFn** defines NumPyro global parameters and receives the per-entry shapes of the full data. **rngSeed** initializes the Estimator's persistent RNG.",
    ),
    (
        'Train(trainIndices[], TrainLikelihoodFn(globalParams{}, localParams{[]}, data{[]}, modelOut{[]}), Method=Nuts(...) | Svi(...), printSummary=False, *rngSeed*, saveLocals=True, saveModelOut=False) -> methodOutput',
        "Fits **trainIndices** using **Method**, typically `Nuts(...)` or `Svi(...)`. The Estimator stores global posterior samples and the learned Cholesky state; **saveLocals** stores transformed training-local samples and **saveModelOut** stores model-output samples. **printSummary** uses the Estimator's generic posterior summary unless the estimator replaces or augments it.",
    ),
    (
        'Test(testIndices[], *TestLikelihoodFn(globalParams{}, localParams{[]}, data{[]}, modelOut{[]})*, *Method=Nuts(...) | Svi(...)*, *nGlobalSamples*, nLocalSamples=2000, printSummary=False, *rngSeed*, saveModelOut=False) -> None | methodOutput[]',
        'Tests on new entries. **nGlobalSamples=None** uses the median trained global/Cholesky state; a positive integer samples that many trained states. With **TestLikelihoodFn=None**, no method is needed and the Estimator draws **nLocalSamples** locals from the trained population. With a likelihood, **Method** infers test locals separately for each selected global state.',
    ),
    (
        'ScoreTrain(ScoreFn(modelOut{}, data{}) -> score{}, *nSamples*, *rngSeed*) -> scores{[]}',
        'Scores training predictions. **nSamples=None** uses all available joint training-posterior samples; otherwise the Estimator selects that many. Saved model outputs are reused when available, otherwise they are regenerated from saved parameters.',
    ),
    (
        'ScoreTest(ScoreFn(modelOut{}, data{}) -> score{}, *nSamples*, *rngSeed*) -> scores{[]}',
        'Scores the latest test state. **nSamples=None** uses all stored local samples; otherwise that many are selected for each stored global state. Saved model outputs are reused when available, otherwise they are regenerated.',
    ),
    (
        'Save(*fileName*, saveTrainLocals=True, saveTestLocals=True, saveTrainModelOut=False, saveTestModelOut=False, saveAllData=False) -> None | record{}',
        'Writes compressed `.npz` state, or returns a nested record when **fileName=None**. Saved metadata includes method class names and constructor arguments so the estimation setup can be reconstructed without serializing method machinery. Functions are not serialized.',
    ),
    (
        'BayesianEstimationTool.Load(fileName, *ModelFn*, *GlobalParamFn*, *modelDataFull*) -> Estimator | record{}',
        'Loads saved state. With **ModelFn=None**, returns the saved record without reconstructing an Estimator. Otherwise model functions are supplied explicitly; **modelDataFull** is required unless the data were saved with `saveAllData=True`.',
    ),
    (
        'Nuts(numWarmup=2000, numSamples=2000, numChains=4, acceptProb=0.95, denseMass=False, medianSamples=50) -> Nuts',
        'Creates a NUTS inference method to pass as `Method=Nuts(...)` to `Train` or likelihood-based `Test`. **medianSamples** controls `init_to_median`; **acceptProb** is the target acceptance probability. NUTS retains posterior samples from the MCMC trajectory and prints NumPyro convergence diagnostics including n_eff and r_hat.',
    ),
    (
        'Svi(numSteps=10000, numSamples=2000, stepSize=0.01, numParticles=1, progressBar=True, medianSamples=50, multivariate=False) -> Svi',
        'Creates an SVI inference method to pass as `Method=Svi(...)` to `Train` or likelihood-based `Test`. Uses `init_to_median`; **numSamples** is the number of posterior samples drawn from the fitted guide for the Estimator. By default the guide is mean-field `AutoNormal`; **multivariate=True** uses `AutoMultivariateNormal`, which can represent posterior covariance but scales less favorably for large latent spaces.',
    ),
]


# -----------------------------------------------------------------------------
# PDF generation
# -----------------------------------------------------------------------------

from pathlib import Path
import html
import re

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch
    from reportlab.platypus import BaseDocTemplate, Frame, KeepTogether, PageTemplate, Paragraph
    from reportlab.lib.styles import ParagraphStyle
except ImportError as exc:
    raise SystemExit(
        "Generating the PDF requires ReportLab. Install it with: pip install reportlab"
    ) from exc


OUTPUT_FILE = Path(__file__).with_name("BayesianEstimationTool_CheatSheet.pdf")
MIN_BODY_FONT = 7.0
MAX_BODY_FONT = 11.5
FONT_STEP = 0.1


def _markup(text):
    """Convert the tiny readable markup used above to ReportLab paragraph markup."""
    text = html.escape(text, quote=False).replace("->", "-&gt;")
    text = re.sub(r"`([^`]+)`", r"<font name='Courier'>\1</font>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\*([^*]+)\*", r"<i>\1</i>", text)
    return text


def _signature_markup(text):
    """Render signatures bold overall, with None-accepting arguments italicized."""
    text = html.escape(text, quote=False).replace("->", "-&gt;")
    text = re.sub(r"\*([^*]+)\*", r"<i>\1</i>", text)
    return text


def _build_pdf(body_font):
    """Build the PDF at one candidate font size and return its page count."""
    page_width, page_height = letter
    margin = 0.38 * inch
    header_height = 0.38 * inch
    usable_height = page_height - 2 * margin - header_height

    frame = Frame(
        margin,
        margin,
        page_width - 2 * margin,
        usable_height,
        leftPadding=4,
        rightPadding=4,
        topPadding=2,
        bottomPadding=2,
    )

    def draw_header(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica-Bold", 16)
        canvas.drawString(margin, page_height - margin - 10, TITLE)
        canvas.setStrokeColor(colors.HexColor("#888888"))
        canvas.setLineWidth(0.5)
        canvas.line(margin, page_height - margin - 16, page_width - margin, page_height - margin - 16)
        canvas.restoreState()

    purpose_style = ParagraphStyle(
        "Purpose",
        fontName="Helvetica",
        fontSize=body_font + 0.8,
        leading=(body_font + 0.8) * 1.18,
        spaceAfter=4,
    )
    heading_style = ParagraphStyle(
        "Heading",
        fontName="Helvetica-Bold",
        fontSize=body_font + 2.5,
        leading=(body_font + 2.5) * 1.12,
        spaceBefore=2.5,
        spaceAfter=4.0,
    )
    conventions_style = ParagraphStyle(
        "Conventions",
        fontName="Helvetica",
        fontSize=body_font,
        leading=body_font * 1.18,
        spaceAfter=2.5,
        backColor=colors.HexColor("#F4F4F4"),
        borderPadding=4,
    )
    entry_style = ParagraphStyle(
        "Entry",
        fontName="Helvetica",
        fontSize=body_font,
        leading=body_font * 1.17,
        spaceBefore=2.2,
        spaceAfter=1.6,
        leftIndent=14,
        firstLineIndent=-14,
    )

    convention_lines = [f"<b>{_markup(symbol)}</b> - {_markup(description)}" for symbol, description in CONVENTIONS]

    story = [
        Paragraph("<b>Purpose:</b> " + _markup(PURPOSE) + "<br/><b>Dependencies:</b> " + _markup(DEPENDENCIES) + "<br/><b>Installation:</b> " + _markup(INSTALLATION), purpose_style),
        Paragraph("Conventions", heading_style),
        Paragraph("<br/>".join(convention_lines), conventions_style),
        Paragraph("Main Functions", heading_style),
    ]

    for signature, description in FUNCTIONS:
        story.append(
            KeepTogether([
                Paragraph(f"<b>{_signature_markup(signature)}:</b>  {_markup(description)}", entry_style)
            ])
        )

    doc = BaseDocTemplate(
        str(OUTPUT_FILE),
        pagesize=letter,
        leftMargin=margin,
        rightMargin=margin,
        topMargin=margin,
        bottomMargin=margin,
    )
    doc.addPageTemplates(PageTemplate(id="Main", frames=[frame], onPage=draw_header))
    doc.build(story)
    return doc.page


def generate_pdf():
    """Use the largest body font that keeps the cheat sheet on one page."""
    steps = round((MAX_BODY_FONT - MIN_BODY_FONT) / FONT_STEP)
    candidate_sizes = [round(MAX_BODY_FONT - i * FONT_STEP, 1) for i in range(steps + 1)]

    for body_font in candidate_sizes:
        if _build_pdf(body_font) == 1:
            print(f"Wrote {OUTPUT_FILE.name} at {body_font:.1f} pt body font.")
            return

    raise RuntimeError(
        f"Cheat sheet does not fit on one page even at {MIN_BODY_FONT:.1f} pt. "
        "Shorten the content or lower MIN_BODY_FONT."
    )


if __name__ == "__main__":
    generate_pdf()
