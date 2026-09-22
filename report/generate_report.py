# -*- coding: utf-8 -*-
"""
generate_report.py  (report/ folder)
=====================================
Builds the Bachelor of Engineering research report (.docx) for:

   "Explainable AI for Diabetes Risk Prediction and Actionable Insights"

Final experiment: train + validate on BRFSS 2017 (US), test once on Cambodia
STEPS (873 records). Final model = Soft Voting ensemble (XGBoost + Logistic
Regression + CatBoost), threshold = 0.43.

The script also generates the data figures (matplotlib) it can build from the
aggregate numbers, saves them in report/figures/, and embeds them. Figures that
need per-record data (SHAP, threshold sweep, probability distribution, dashboard)
are left as labelled placeholders for export from the notebook.

Run:  python report/generate_report.py
Out:  report/Diabetes_Project_Report.docx
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import (WD_ALIGN_PARAGRAPH, WD_LINE_SPACING,
                            WD_TAB_ALIGNMENT, WD_TAB_LEADER)
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

# --------------------------------------------------------------------------- #
#  Paths and constants
# --------------------------------------------------------------------------- #
REPORT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR    = os.path.join(REPORT_DIR, "figures")
OUT_PATH   = os.path.join(REPORT_DIR, "Diabetes_Project_Report.docx")
os.makedirs(FIG_DIR, exist_ok=True)

TNR  = "Times New Roman"
SIZE = 12
PH   = RGBColor(0xC0, 0x00, 0x00)   # red for placeholders / verify notes
BLACK = RGBColor(0, 0, 0)

# Colours for figures
C_NEG = "#4C72B0"   # no diabetes
C_POS = "#DD8452"   # diabetes
C_LOW, C_MOD, C_HIGH = "#4CAF50", "#FF9800", "#F44336"

TITLE = "Explainable AI for Diabetes Risk Prediction and Actionable Insights"

# --------------------------------------------------------------------------- #
#  Real numbers from the final notebook
# --------------------------------------------------------------------------- #
BRFSS_ROWS, BRFSS_RATE = 275746, 16.0
STEPS_ROWS, STEPS_RATE = 873, 14.0

SPLITS = [  # name, rows, no_diab, diab, pct_pos
    ("Train — BRFSS 2017 (80%)", 220596, 185404, 35192, 15.95),
    ("Validation — BRFSS 2017 (20%)", 55150, 46352, 8798, 15.95),
    ("Test — Cambodia STEPS", 873, 751, 122, 13.98),
]
SMOTE_BEFORE = (185404, 35192)
SMOTE_AFTER  = (182354, 182354)

# Validation @ 0.50 per model: acc, prec, rec, f1 (percentages)
VAL_MODELS = [
    ("XGBoost",             77.45, 37.07, 59.24, 45.60),
    ("Logistic Regression", 70.44, 31.79, 74.45, 44.55),
    ("CatBoost",            80.08, 40.04, 49.99, 44.46),
    ("Random Forest",       78.50, 35.48, 42.49, 38.67),
    ("LightGBM",            80.41, 38.49, 38.12, 38.31),
]

BEST_MODEL = "Soft Voting ensemble"
THR = 0.43
# Soft Voting @ 0.43 validation: rec, prec, acc, f1
VAL_FINAL = dict(Recall=70.46, Precision=33.22, Accuracy=72.69, F1=45.15)
# Soft Voting @ 0.43 STEPS test
TEST_FINAL = dict(Recall=76.23, Precision=20.58, Accuracy=55.56, F1=32.40)
# Confusion matrix on STEPS: [[TN, FP],[FN, TP]]
CM = [[392, 359], [29, 93]]
TIERS = dict(Low=299, Moderate=401, High=173)

FEATURES = ["Sex", "Age", "Education", "HighBP", "HighChol", "BMI",
            "Smoker", "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump"]


def p(x):
    return "%.2f%%" % x


# =========================================================================== #
#  PART 1 — GENERATE FIGURES
# =========================================================================== #
plt.rcParams.update({"font.size": 11, "axes.titleweight": "bold",
                     "figure.dpi": 150, "savefig.bbox": "tight"})


def _save(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_survey_dist():
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    data = [("BRFSS 2017 (train + validation)", 231756, 43990),
            ("Cambodia STEPS (independent test)", 751, 122)]
    for ax, (name, neg, pos) in zip(axes, data):
        tot = neg + pos
        bars = ax.bar(["No Diabetes", "Diabetes"], [neg, pos],
                      color=[C_NEG, C_POS], edgecolor="white")
        for b, v in zip(bars, [neg, pos]):
            ax.text(b.get_x() + b.get_width() / 2, v,
                    "%s\n(%.1f%%)" % (f"{v:,}", v / tot * 100),
                    ha="center", va="bottom", fontsize=9)
        ax.set_title(name, fontsize=10)
        ax.set_ylabel("Number of respondents")
        ax.margins(y=0.18)
    fig.tight_layout()
    return _save(fig, "fig_survey_dist.png")


def fig_split_dist():
    fig, ax = plt.subplots(figsize=(8, 4))
    names = [s[0].split(" — ")[0] for s in SPLITS]
    neg = [s[2] for s in SPLITS]
    pos = [s[3] for s in SPLITS]
    x = np.arange(len(names))
    w = 0.38
    b1 = ax.bar(x - w / 2, neg, w, label="No Diabetes", color=C_NEG, edgecolor="white")
    b2 = ax.bar(x + w / 2, pos, w, label="Diabetes", color=C_POS, edgecolor="white")
    for bars in (b1, b2):
        ax.bar_label(bars, fmt="%d", fontsize=8, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Number of records")
    ax.set_title("Class Distribution per Data Split")
    ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    return _save(fig, "fig_split_dist.png")


def fig_smotetomek():
    fig, ax = plt.subplots(figsize=(7, 4))
    groups = ["Before SMOTETomek", "After SMOTETomek"]
    neg = [SMOTE_BEFORE[0], SMOTE_AFTER[0]]
    pos = [SMOTE_BEFORE[1], SMOTE_AFTER[1]]
    x = np.arange(len(groups))
    w = 0.38
    b1 = ax.bar(x - w / 2, neg, w, label="No Diabetes", color=C_NEG, edgecolor="white")
    b2 = ax.bar(x + w / 2, pos, w, label="Diabetes", color=C_POS, edgecolor="white")
    for bars in (b1, b2):
        ax.bar_label(bars, fmt="%d", fontsize=8, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel("Number of training records")
    ax.set_title("BRFSS 2017 Training Set Before and After SMOTETomek")
    ax.legend()
    ax.margins(y=0.15)
    fig.tight_layout()
    return _save(fig, "fig_smotetomek.png")


def fig_val_models():
    fig, ax = plt.subplots(figsize=(9, 4.3))
    metrics = ["Accuracy", "Precision", "Recall", "F1-score"]
    names = [m[0] for m in VAL_MODELS]
    vals = np.array([[m[1], m[2], m[3], m[4]] for m in VAL_MODELS])
    x = np.arange(len(names))
    w = 0.2
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    for i, met in enumerate(metrics):
        ax.bar(x + (i - 1.5) * w, vals[:, i], w, label=met, color=colors[i],
               edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Individual Model Performance on Validation Set (threshold 0.50)")
    ax.legend(ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.tight_layout()
    return _save(fig, "fig_val_models.png")


def fig_confusion():
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    cm = np.array(CM)
    im = ax.imshow(cm, cmap="Blues")
    labels = ["No Diabetes", "Diabetes"]
    ax.set_xticks([0, 1]); ax.set_xticklabels(labels)
    ax.set_yticks([0, 1]); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix — Cambodia STEPS Test\n(Soft Voting, threshold 0.43)")
    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=14, fontweight="bold")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return _save(fig, "fig_confusion.png")


def fig_val_vs_test():
    fig, ax = plt.subplots(figsize=(8, 4))
    metrics = ["Recall", "F1-score", "Accuracy", "Precision"]
    val = [VAL_FINAL["Recall"], VAL_FINAL["F1"], VAL_FINAL["Accuracy"], VAL_FINAL["Precision"]]
    tst = [TEST_FINAL["Recall"], TEST_FINAL["F1"], TEST_FINAL["Accuracy"], TEST_FINAL["Precision"]]
    x = np.arange(len(metrics))
    w = 0.38
    b1 = ax.bar(x - w / 2, val, w, label="Validation (BRFSS 2017)", color=C_NEG, edgecolor="white")
    b2 = ax.bar(x + w / 2, tst, w, label="Test (Cambodia STEPS)", color=C_POS, edgecolor="white")
    for bars in (b1, b2):
        ax.bar_label(bars, fmt="%.1f", fontsize=8, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Validation vs Independent Test — Soft Voting Ensemble")
    ax.legend()
    fig.tight_layout()
    return _save(fig, "fig_val_vs_test.png")


def fig_tiers():
    fig, ax = plt.subplots(figsize=(6.5, 4))
    order = ["Low", "Moderate", "High"]
    vals = [TIERS[k] for k in order]
    bars = ax.bar(order, vals, color=[C_LOW, C_MOD, C_HIGH], edgecolor="white")
    ax.bar_label(bars, fmt="%d", fontsize=10, padding=2)
    ax.set_ylabel("Number of STEPS respondents")
    ax.set_title("Risk-Tier Distribution — Cambodia STEPS")
    ax.margins(y=0.15)
    fig.tight_layout()
    return _save(fig, "fig_tiers.png")


def fig_cm_structure():
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    grid = np.array([[1, 0], [0, 1]])
    ax.imshow(grid, cmap="Blues", vmin=0, vmax=2)
    cells = [("True Positive (TP)", 0, 0), ("False Negative (FN)\nType II error", 1, 0),
             ("False Positive (FP)\nType I error", 0, 1), ("True Negative (TN)", 1, 1)]
    for txt, r, c in cells:
        ax.text(c, r, txt, ha="center", va="center", fontsize=10, fontweight="bold",
                color="black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Predicted Positive", "Predicted Negative"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Actual Positive", "Actual Negative"])
    ax.set_title("Confusion Matrix Structure")
    fig.tight_layout()
    return _save(fig, "fig_cm_structure.png")


def fig_roc_illustration():
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    fpr = np.linspace(0, 1, 200)
    tpr = fpr ** 0.35
    ax.plot(fpr, tpr, color="#7B3FA0", lw=2)
    ax.fill_between(fpr, tpr, alpha=0.15, color="#7B3FA0")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random (AUC = 0.5)")
    ax.text(0.55, 0.30, "AUC", fontsize=13, color="#7B3FA0", fontweight="bold")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve and Area Under the Curve (illustration)")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    return _save(fig, "fig_roc.png")


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.axis("off")
    steps = ["BRFSS 2017\n+ Cambodia STEPS", "Feature Alignment\n(11 features)",
             "BRFSS 80/20\nTrain / Validation", "Scaling +\nSMOTETomek (train)",
             "Train 5 models\n+ Soft Voting", "Select model &\nthreshold (val F1)",
             "Independent test\nCambodia STEPS", "SHAP +\nRisk tiers + App"]
    positions = [(0.06, 0.62), (0.30, 0.62), (0.54, 0.62), (0.78, 0.62),
                 (0.78, 0.18), (0.54, 0.18), (0.30, 0.18), (0.06, 0.18)]
    boxes = []
    for (x, y), txt in zip(positions, steps):
        b = ax.annotate(txt, (x, y), ha="center", va="center", fontsize=8.5,
                        bbox=dict(boxstyle="round,pad=0.4", fc="#E8EEF7",
                                  ec="#4C72B0", lw=1.4))
        boxes.append((x, y))
    # arrows along the snake path
    seq = positions
    for (x1, y1), (x2, y2) in zip(seq, seq[1:]):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#888888", lw=1.5))
    ax.set_xlim(0, 0.92); ax.set_ylim(0, 0.85)
    ax.set_title("Research Design and Pipeline Workflow")
    fig.tight_layout()
    return _save(fig, "fig_pipeline.png")


print("Generating figures ...")
IMG = {}
for key, fn in [
    ("pipeline", fig_pipeline), ("cm_structure", fig_cm_structure),
    ("roc", fig_roc_illustration), ("survey_dist", fig_survey_dist),
    ("split_dist", fig_split_dist), ("smotetomek", fig_smotetomek),
    ("val_models", fig_val_models), ("confusion", fig_confusion),
    ("val_vs_test", fig_val_vs_test), ("tiers", fig_tiers),
]:
    try:
        IMG[key] = fn()
        print("  ok:", os.path.basename(IMG[key]))
    except Exception as e:                       # pragma: no cover
        print("  FAILED", key, e)
        IMG[key] = None


# =========================================================================== #
#  PART 2 — BUILD DOCUMENT
#  The front matter (cover + RUPP logo, committee, supervisor and candidate
#  statements, Khmer + English abstracts, acknowledgements) is taken directly
#  from the user's sample report so the formatting and logo are preserved.
# =========================================================================== #
SAMPLE = os.path.join(REPORT_DIR, "Ketya_Report.docx")


def load_base_with_frontmatter():
    """Open the sample report, keep its front matter, and delete everything from
    its old 'TABLE OF CONTENTS' to the end so a fresh body can be appended."""
    d = Document(SAMPLE)
    body_el = d.element.body
    cut = None
    for par in d.paragraphs:
        if par.text.strip().upper() == "TABLE OF CONTENTS":
            cut = par._p
            break
    if cut is not None:
        final_sectPr = body_el.find(qn("w:sectPr"))   # document-level section
        el = cut
        while el is not None and el is not final_sectPr:
            nxt = el.getnext()
            body_el.remove(el)
            el = nxt
    return d


doc = load_base_with_frontmatter()


def _margins(section):
    section.top_margin = Inches(1)
    section.bottom_margin = Inches(1)
    section.left_margin = Inches(1.25)
    section.right_margin = Inches(1.0)


def _enable_update_fields():
    settings = doc.settings.element
    el = OxmlElement("w:updateFields")
    el.set(qn("w:val"), "true")
    settings.append(el)


# ---- paragraph helpers ---------------------------------------------------- #
def para(text="", *, align=None, bold=False, italic=False, size=SIZE,
         space_before=0, space_after=6, first_indent=0.5, color=None):
    pp = doc.add_paragraph()
    pf = pp.paragraph_format
    pf.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    pf.space_before = Pt(space_before)
    pf.space_after = Pt(space_after)
    if first_indent:
        pf.first_line_indent = Inches(first_indent)
    if align is not None:
        pp.alignment = align
    if text:
        r = pp.add_run(text)
        r.bold = bold; r.italic = italic; r.font.size = Pt(size); r.font.name = TNR
        if color is not None:
            r.font.color.rgb = color
    return pp


def body(*texts):
    for t in texts:
        para(t, align=WD_ALIGN_PARAGRAPH.JUSTIFY)


def rich(segments, *, align=WD_ALIGN_PARAGRAPH.JUSTIFY, first_indent=0.5, space_after=6):
    pp = doc.add_paragraph()
    pf = pp.paragraph_format
    pf.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    pf.space_after = Pt(space_after)
    pf.first_line_indent = Inches(first_indent)
    pp.alignment = align
    for text, opts in segments:
        r = pp.add_run(text)
        r.bold = opts.get("bold", False); r.italic = opts.get("italic", False)
        r.font.size = Pt(opts.get("size", SIZE)); r.font.name = TNR
        if "color" in opts:
            r.font.color.rgb = opts["color"]
    return pp


def ph_para(text):
    return para("[" + text + "]", align=WD_ALIGN_PARAGRAPH.CENTER, color=PH,
                italic=True, first_indent=0)


def centre(text, bold=True, size=SIZE, space_after=12):
    return para(text, align=WD_ALIGN_PARAGRAPH.CENTER, bold=bold, size=size,
                first_indent=0, space_after=space_after)


def _style_heading_run(pp, size, align, page_break):
    pf = pp.paragraph_format
    pf.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    pf.space_before = Pt(12)
    pf.space_after = Pt(6)
    pf.keep_with_next = True
    pf.page_break_before = page_break
    pp.alignment = align
    for r in pp.runs:
        r.bold = True
        r.font.name = TNR
        r.font.size = Pt(size)
        r.font.color.rgb = BLACK


def H1(text, new_page=True):
    # "CHAPTER 1  INTRODUCTION" -> two centred lines: "CHAPTER 1" / "INTRODUCTION"
    pp = doc.add_paragraph(style="Heading 1")
    parts = text.split("  ", 1)
    r1 = pp.add_run(parts[0])
    if len(parts) > 1:
        r1.add_break()
        pp.add_run(parts[1])
    _style_heading_run(pp, 14, WD_ALIGN_PARAGRAPH.CENTER, new_page)
    return pp


def H2(text):
    pp = doc.add_paragraph(text, style="Heading 2")
    _style_heading_run(pp, 13, WD_ALIGN_PARAGRAPH.LEFT, False)
    return pp


def H3(text):
    pp = doc.add_paragraph(text, style="Heading 3")
    _style_heading_run(pp, 12, WD_ALIGN_PARAGRAPH.LEFT, False)
    return pp


def bullet(text, lead=None):
    pp = doc.add_paragraph()
    pf = pp.paragraph_format
    pf.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    pf.space_after = Pt(3)
    pf.left_indent = Inches(0.5)
    pf.first_line_indent = Inches(-0.25)
    rb = pp.add_run("•  "); rb.font.size = Pt(SIZE); rb.font.name = TNR
    if lead:
        r = pp.add_run(lead); r.bold = True; r.font.size = Pt(SIZE); r.font.name = TNR
    r2 = pp.add_run(text); r2.font.size = Pt(SIZE); r2.font.name = TNR
    return pp


# ---- SEQ-numbered captions (feed the auto List of Tables/Figures) --------- #
def _seq_run(label):
    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), ' SEQ %s \\* ARABIC ' % label)
    r = OxmlElement("w:r")
    rpr = OxmlElement("w:rPr")
    it = OxmlElement("w:i"); rpr.append(it)
    sz = OxmlElement("w:sz"); sz.set(qn("w:val"), "20"); rpr.append(sz)
    rf = OxmlElement("w:rFonts"); rf.set(qn("w:ascii"), TNR); rf.set(qn("w:hAnsi"), TNR)
    rpr.append(rf)
    r.append(rpr)
    t = OxmlElement("w:t"); t.text = "1"; r.append(t)
    fld.append(r)
    return fld


def caption(label, text, before=2, after=10):
    pp = doc.add_paragraph()
    pp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pp.paragraph_format.space_before = Pt(before)
    pp.paragraph_format.space_after = Pt(after)
    r = pp.add_run(label + " "); r.italic = True; r.font.size = Pt(10); r.font.name = TNR
    pp._p.append(_seq_run(label))
    r2 = pp.add_run(": " + text); r2.italic = True; r2.font.size = Pt(10); r2.font.name = TNR
    return pp


# ---- figures: embed if image exists, else placeholder box ----------------- #
def figure(img_key, caption_text, width=5.7):
    path = IMG.get(img_key) if img_key else None
    if path and os.path.exists(path):
        doc.add_picture(path, width=Inches(width))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        doc.paragraphs[-1].paragraph_format.space_before = Pt(6)
        doc.paragraphs[-1].paragraph_format.space_after = Pt(0)
    caption("Figure", caption_text)


def figure_placeholder(caption_text, source):
    pp = doc.add_paragraph()
    pp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pp.paragraph_format.space_before = Pt(6)
    pp.paragraph_format.space_after = Pt(0)
    r = pp.add_run("[ FIGURE PLACEHOLDER ]\n%s" % source)
    r.italic = True; r.font.size = Pt(11); r.font.color.rgb = PH
    pPr = pp._p.get_or_add_pPr()
    shd = OxmlElement("w:shd"); shd.set(qn("w:val"), "clear"); shd.set(qn("w:fill"), "F2F2F2")
    pPr.append(shd)
    pbdr = OxmlElement("w:pBdr")
    for edge in ("top", "left", "bottom", "right"):
        e = OxmlElement("w:" + edge)
        e.set(qn("w:val"), "dashed"); e.set(qn("w:sz"), "6")
        e.set(qn("w:space"), "8"); e.set(qn("w:color"), "C00000")
        pbdr.append(e)
    pPr.append(pbdr)
    caption("Figure", caption_text)


# ---- tables --------------------------------------------------------------- #
def _shade(cell, fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd"); shd.set(qn("w:val"), "clear"); shd.set(qn("w:fill"), fill)
    tcPr.append(shd)


def _add_borders(t):
    """Apply single-line borders to all edges (sample template has no Table Grid)."""
    tblPr = t._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        e = OxmlElement("w:" + edge)
        e.set(qn("w:val"), "single"); e.set(qn("w:sz"), "4")
        e.set(qn("w:space"), "0"); e.set(qn("w:color"), "808080")
        borders.append(e)
    tblPr.append(borders)


def table(headers, rows, widths, caption_text):
    t = doc.add_table(rows=1, cols=len(headers))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    _add_borders(t)
    hdr = t.rows[0].cells
    for i, htext in enumerate(headers):
        hdr[i].text = ""
        run = hdr[i].paragraphs[0].add_run(htext)
        run.bold = True; run.font.size = Pt(11); run.font.name = TNR
        hdr[i].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        _shade(hdr[i], "D9E1F2")
    for row in rows:
        cells = t.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = ""
            run = cells[i].paragraphs[0].add_run(str(val))
            run.font.size = Pt(11); run.font.name = TNR
            cells[i].paragraphs[0].alignment = (
                WD_ALIGN_PARAGRAPH.LEFT if i == 0 else WD_ALIGN_PARAGRAPH.CENTER)
    for i, w in enumerate(widths):
        for r in t.rows:
            r.cells[i].width = Inches(w)
    caption("Table", caption_text)
    return t


# ---- TOC / List fields ---------------------------------------------------- #
def toc_field(instr, placeholder):
    pp = doc.add_paragraph()
    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), instr)
    r = OxmlElement("w:r")
    t = OxmlElement("w:t"); t.text = placeholder
    r.append(t); fld.append(r)
    pp._p.append(fld)
    return pp


def update_note():
    para("(Right-click this list and choose “Update Field”, or press F9, to "
         "build it with correct page numbers.)", align=WD_ALIGN_PARAGRAPH.CENTER,
         italic=True, size=9, color=PH, first_indent=0, space_after=10)


# ---- page numbers --------------------------------------------------------- #
def footer_pagenum(section, fmt):
    section.footer.is_linked_to_previous = False
    pp = section.footer.paragraphs[0]
    pp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = pp.add_run()
    f1 = OxmlElement("w:fldChar"); f1.set(qn("w:fldCharType"), "begin")
    it = OxmlElement("w:instrText"); it.set(qn("xml:space"), "preserve"); it.text = "PAGE"
    f2 = OxmlElement("w:fldChar"); f2.set(qn("w:fldCharType"), "end")
    run._r.append(f1); run._r.append(it); run._r.append(f2)
    run.font.name = TNR; run.font.size = Pt(SIZE)
    sectPr = section._sectPr
    pg = sectPr.find(qn("w:pgNumType"))
    if pg is None:
        pg = OxmlElement("w:pgNumType"); sectPr.append(pg)
    pg.set(qn("w:fmt"), fmt); pg.set(qn("w:start"), "1")


# =========================================================================== #
#  Assemble: front matter (cover, logo, committee, statements, abstracts,
#  acknowledgements) is inherited from the sample; we append the rest below.
# =========================================================================== #
_enable_update_fields()

# Table of Contents (auto field)
doc.add_page_break()
centre("TABLE OF CONTENTS", bold=True, size=13)
toc_field(' TOC \\o "1-3" \\h \\z \\u ',
          "Right-click and choose “Update Field” to build the Table of Contents.")
update_note()

# List of Tables
doc.add_page_break()
centre("LIST OF TABLES", bold=True, size=13)
toc_field(' TOC \\h \\z \\c "Table" ',
          "Right-click and choose “Update Field” to build the List of Tables.")
update_note()

# List of Figures
doc.add_page_break()
centre("LIST OF FIGURES", bold=True, size=13)
toc_field(' TOC \\h \\z \\c "Figure" ',
          "Right-click and choose “Update Field” to build the List of Figures.")
update_note()

# List of Abbreviations
doc.add_page_break()
centre("LIST OF ABBREVIATIONS", bold=True, size=13)
ABBR = [
    ("AUC", "Area Under the (ROC) Curve"),
    ("BMI", "Body Mass Index"),
    ("BRFSS", "Behavioral Risk Factor Surveillance System"),
    ("CatBoost", "Categorical Boosting"),
    ("CDC", "Centers for Disease Control and Prevention"),
    ("F1", "F1-Score (harmonic mean of precision and recall)"),
    ("LightGBM", "Light Gradient Boosting Machine"),
    ("LIME", "Local Interpretable Model-agnostic Explanations"),
    ("LR", "Logistic Regression"),
    ("ML", "Machine Learning"),
    ("NCD", "Non-Communicable Disease"),
    ("RF", "Random Forest"),
    ("ROC", "Receiver Operating Characteristic"),
    ("SHAP", "SHapley Additive exPlanations"),
    ("SMOTE", "Synthetic Minority Oversampling Technique"),
    ("SMOTETomek", "SMOTE combined with Tomek-links cleaning"),
    ("STEPS", "WHO STEPwise Approach to NCD Risk-Factor Surveillance"),
    ("WHO", "World Health Organization"),
    ("XAI", "Explainable Artificial Intelligence"),
    ("XGBoost", "Extreme Gradient Boosting"),
]
for ab, full in ABBR:
    pp = doc.add_paragraph()
    pp.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    pp.paragraph_format.space_after = Pt(4)
    pp.paragraph_format.left_indent = Inches(0.3)
    pp.paragraph_format.tab_stops.add_tab_stop(Inches(1.9), WD_TAB_ALIGNMENT.LEFT)
    r = pp.add_run(ab); r.bold = True; r.font.size = Pt(SIZE)
    r2 = pp.add_run("\t" + full); r2.font.size = Pt(SIZE)

# ===== BODY (arabic numerals) ============================================== #
bsec = doc.add_section(WD_SECTION.NEW_PAGE)
_margins(bsec)

# --------------------------------------------------------------------------- #
#  CHAPTER 1 — INTRODUCTION
# --------------------------------------------------------------------------- #
H1("CHAPTER 1  INTRODUCTION", new_page=False)

H2("1.1 Background to the Study")
body(
    "Diabetes is one of the most common chronic diseases in the world today. It happens "
    "when the body cannot control the level of sugar in the blood, and over time it can "
    "damage the heart, blood vessels, kidneys, eyes, and nerves. Most cases are type 2 "
    "diabetes, which is closely linked to age, body weight, diet, physical activity, and "
    "other lifestyle factors. Many of these factors can be changed, so people who are found "
    "to be at risk early can take action before the disease becomes serious.",
    "In Cambodia, the number of people with diabetes has been rising as the country becomes "
    "more urban and diets change. National surveys carried out under the WHO STEPwise "
    "approach (STEPS) have shown that a large share of Cambodian adults have diabetes or "
    "are at risk, and that many of them do not know it. Because screening services are "
    "limited, people often discover the disease only after complications have already "
    "started, when treatment is harder and more expensive.",
    "Finding people at high risk early is therefore important for public health. If we can "
    "flag people who are likely to have diabetes using simple and low-cost information such "
    "as age, body mass index, blood pressure history, and lifestyle habits, those people "
    "can be sent for proper testing and given advice in time. Machine learning is a useful "
    "tool for this, because it can learn the relationship between many risk factors and the "
    "disease, and it can give a risk estimate quickly.",
    "However, two problems make this difficult in Cambodia. The first is data. Large, "
    "well-labelled diabetes datasets are common in rich countries but rare in Cambodia, "
    "where collecting confirmed labels at a large scale is costly. The second is trust. A "
    "model that only outputs a number, without explaining why, is a black box, and it is "
    "hard for health workers or patients to trust a result they cannot understand. This "
    "study deals with both problems by testing whether a model trained on foreign data can "
    "work on Cambodian data, and by making the predictions explainable.")

H2("1.2 Problem Statement")
body(
    "The main problem in this study is how to identify people at risk of type 2 diabetes in "
    "Cambodia in an early, low-cost, and trustworthy way, when local labelled data are "
    "limited. Collecting a large Cambodian dataset would be ideal, but it is expensive and "
    "slow, and the result would still be small compared with the data available "
    "internationally.",
    "One option is to use a large foreign dataset, the United States BRFSS survey, which "
    "has hundreds of thousands of labelled records, and to test the model on the Cambodian "
    "population through the STEPS survey. This raises a question that is not well answered "
    "in the local context: how well does a diabetes model trained on one population work on "
    "a different population, with different diets, body types, and survey methods?",
    "There is also the problem of class imbalance. People with diabetes are the minority in "
    "both surveys, so a model that simply predicts “no diabetes” for everyone would look "
    "accurate but would be useless. In screening, a false negative, which means telling a "
    "person at risk that they are fine, is more harmful than a false positive, which only "
    "leads to an extra test. For this reason, the model in this study is built to give high "
    "recall rather than just high accuracy, and it is designed to be explainable so that "
    "its results can be trusted and acted on.")

H2("1.3 Aim and Objectives")
body("The aim of this study is to build and honestly evaluate an explainable machine "
     "learning pipeline that screens for diabetes risk in Cambodia by transferring a model "
     "trained on a large foreign survey, while keeping recall high and the results easy to "
     "understand.")
body("The specific objectives are:")
bullet("align the BRFSS (United States) and STEPS (Cambodia) surveys into a shared set of "
       "common, low-cost risk factors;", lead="Objective 1: ")
bullet("build a clear preprocessing pipeline with scaling and class-imbalance handling "
       "applied only to the training data;", lead="Objective 2: ")
bullet("train and compare five machine learning models and combine the best ones into a "
       "Soft Voting ensemble;", lead="Objective 3: ")
bullet("select the final model and decision threshold using validation F1-score while "
       "keeping recall high, instead of using accuracy alone;", lead="Objective 4: ")
bullet("test the final model once on the Cambodia STEPS data, which is never used during "
       "training or selection, to measure honest cross-population performance;", lead="Objective 5: ")
bullet("explain the model with SHAP, both overall and for individual patients, and present "
       "the results in a simple dashboard with actionable lifestyle insights.", lead="Objective 6: ")

H2("1.4 Rationale of the Study")
body(
    "There are three main reasons for this work. First, the number of undiagnosed diabetes "
    "cases in Cambodia is large and growing, and the cost of late diagnosis, in terms of "
    "complications such as kidney failure, blindness, and heart disease, is high. A "
    "screening tool that uses only simple information could help raise awareness beyond the "
    "limited number of clinics.",
    "Second, the shortage of local labelled data will not be solved quickly, so it is worth "
    "finding out now, in an honest way, how far a transfer learning approach can go using "
    "free international data. A clear benchmark, including its weaknesses, is more useful "
    "than an over-optimistic claim of high accuracy. For this reason, this study reports "
    "its weaker numbers together with its strong recall.",
    "Third, any tool used in health care needs to be trusted. Explainability methods such "
    "as SHAP can show which factors increase or decrease a person’s risk. Building "
    "explainability into the pipeline from the start, rather than adding it at the end, is "
    "what makes the difference between a research experiment and a tool that a health worker "
    "or patient could actually use.")

H2("1.5 Limitation and Scope")
body(
    "This study focuses on non-invasive, survey-based screening for type 2 diabetes risk. "
    "It does not try to give a medical diagnosis, which needs laboratory tests such as "
    "fasting blood glucose or HbA1c. The output is a risk estimate that should lead to "
    "follow-up, not a final medical decision.",
    "The most important limitation is the data. The training labels come from a United "
    "States survey, whose population, diet, and health system are quite different from "
    "Cambodia’s, and the Cambodian test set is fairly small (873 records). The two surveys "
    "also ask and record questions differently, so only eleven common features could be "
    "aligned, and some useful variables had to be dropped. Because of this, the "
    "cross-population results should be read as a lower-bound benchmark for transfer, not as "
    "the best a fully local model could do. These limitations are discussed again in "
    "Chapter 5 and lead to the future-work suggestions in Chapter 6.")

H2("1.6 Structure of the Study")
body(
    "This report has six chapters. Chapter 1 has introduced the background, the problem, "
    "the aim and objectives, the rationale, and the scope.",
    "Chapter 2 reviews related work on diabetes risk prediction with survey data, machine "
    "learning models, class imbalance, the importance of recall, explainable AI, and the "
    "cross-population gap, and then states the research gap.",
    "Chapter 3 explains the methodology and tools, including the research design, the "
    "software used, the BRFSS and STEPS data, feature alignment, preprocessing, the models "
    "and the Soft Voting ensemble, threshold tuning, the evaluation metrics, and the "
    "explainability method.",
    "Chapter 4 presents the results, including the data distributions, the validation "
    "comparison, the final model and threshold selection, the independent Cambodia STEPS "
    "test, the explainability results, and the risk tiers.",
    "Chapter 5 discusses what the results mean, and Chapter 6 gives the conclusion and "
    "future work. References and appendices are at the end.")

# --------------------------------------------------------------------------- #
#  CHAPTER 2 — LITERATURE REVIEW
# --------------------------------------------------------------------------- #
H1("CHAPTER 2  LITERATURE REVIEW")

H2("2.1 Diabetes Risk Prediction Using Survey Data")
body(
    "Predicting diabetes risk from routine information has been studied for many years. "
    "Early methods used simple risk scores that added up the weights of a few factors such "
    "as age, body mass index, and blood pressure. These scores are easy to use and "
    "transparent, but their fixed form does not capture complex relationships between "
    "factors, and they may not work well on populations that are different from the one used "
    "to build them.",
    "Large health surveys have made it possible to build data-driven models instead. The "
    "Behavioral Risk Factor Surveillance System (BRFSS), run by the United States CDC, is "
    "one of the most widely used sources. It is useful for machine learning because it "
    "contains many behavioural, demographic, and health-status variables, such as smoking, "
    "physical activity, diet, education, age, blood pressure, and cholesterol, collected "
    "from hundreds of thousands of people. Xie et al. (2019) used BRFSS data to build "
    "machine learning models for type 2 diabetes and showed that such survey data can "
    "support useful risk prediction, while also pointing out the importance of sensitivity "
    "(recall) for screening, because the goal is to catch as many true cases as possible.")

H2("2.2 Machine Learning Models for Diabetes Prediction")
body(
    "Many studies have compared machine learning models on diabetes data. Logistic "
    "Regression is a common and easy-to-explain baseline. Tree-based models such as Random "
    "Forest and gradient boosting methods like XGBoost, LightGBM, and CatBoost are often "
    "among the best on this kind of tabular data, because they can capture non-linear "
    "patterns and handle different types of features. Katiyar et al. (2024) review recent "
    "machine learning and deep learning approaches for diabetes detection and show that "
    "ensemble and boosting methods are widely used and usually perform well, while deep "
    "learning is also being explored. These findings support the choice in this study to "
    "test Logistic Regression, Random Forest, XGBoost, LightGBM, and CatBoost, and to "
    "combine the strongest models into an ensemble.")

H2("2.3 Class Imbalance and Resampling")
body(
    "Diabetes datasets are usually imbalanced, because people with diabetes are the "
    "minority. If this is ignored, a model can become biased toward the majority class and "
    "miss many real cases. To deal with this, studies often use resampling methods such as "
    "SMOTE, which creates synthetic minority examples, or SMOTETomek, which adds SMOTE "
    "oversampling and then removes confusing borderline points. Khokhar et al. (2025) "
    "combine SMOTE-type resampling with ensemble learning and explainable AI for diabetes "
    "prediction, which is close to the approach used in this study. An important rule, "
    "followed here, is that resampling must be applied only to the training data, so that "
    "the validation and test results are not affected by synthetic data.")

H2("2.4 The Importance of Recall in Screening")
body(
    "In diabetes screening, recall (also called sensitivity) is especially important. Recall "
    "measures how many of the real diabetes cases the model is able to find. A false "
    "negative, where a person who really has diabetes is told they are fine, is more "
    "dangerous than a false positive, where a healthy person is asked to do an extra test. "
    "For this reason, several studies, including Xie et al. (2019), highlight recall as a "
    "key metric for screening, rather than focusing only on accuracy. This study follows "
    "the same idea by choosing the final model and threshold based on a recall requirement.")

H2("2.5 Explainable AI and SHAP in Healthcare")
body(
    "As models become more complex, they also become harder to understand, which is a "
    "problem in healthcare where decisions affect people. Explainable AI (XAI) tries to "
    "solve this. SHAP (SHapley Additive exPlanations) is one of the most popular methods. "
    "It gives each feature a value that shows how much it pushed a prediction up or down, "
    "both overall and for a single patient. LIME is another method that explains single "
    "predictions using a simple local model. Khokhar et al. (2025), in their work on "
    "transparent and accurate diabetes prediction, show that combining machine learning "
    "with SHAP makes models more transparent and easier to trust. Their systematic review "
    "(Khokhar et al., 2025) also reports that explainability is becoming an important part "
    "of AI for diabetes prediction, alongside datasets, algorithms, and evaluation metrics. "
    "This supports the use of SHAP in this study.")

H2("2.6 Cross-Population Generalisation")
body(
    "Most diabetes prediction studies train and test their models on the same population, "
    "usually with a random split of one dataset. This measures how well the model works on "
    "data similar to the training data, but it does not show how well the model works on a "
    "different population. Li et al. (2025) discuss the performance of diabetes risk "
    "prediction models and the need for external validation, which means testing a model on "
    "data from a different source. When a model is used on a new population, it faces "
    "“domain shift”, where the relationship between the features and the disease is "
    "different. For example, the link between body mass index and diabetes risk is known to "
    "differ between Western and Asian populations. Studies that honestly test "
    "cross-population transfer, especially into low-resource settings such as Cambodia, are "
    "still rare.")

H2("2.7 Research Gap and Contribution")
body(
    "From this review, three gaps stand out. First, most studies test on the same "
    "population and do not measure transfer to a new one. Second, most work is based in "
    "high-income countries, with little focus on low-resource settings such as Cambodia. "
    "Third, even though explainability is now discussed often, many applied studies still "
    "report only accuracy-type metrics and do not include patient-level explanations or a "
    "deployed tool.",
    "This study addresses these gaps. It trains a model on BRFSS 2017 and tests it once, "
    "independently, on Cambodia STEPS, to measure honest cross-population performance. It "
    "uses a recall-focused selection rule that fits the needs of screening. And it uses "
    "SHAP to explain the model and presents the results in a simple dashboard with risk "
    "tiers and lifestyle advice. In this way, the study tests whether a model trained on "
    "BRFSS 2017 can generalise to Cambodia STEPS, and uses explainability to support the "
    "interpretation of the results.")

# --------------------------------------------------------------------------- #
#  CHAPTER 3 — METHODOLOGY AND TOOLS
# --------------------------------------------------------------------------- #
H1("CHAPTER 3  METHODOLOGY AND TOOLS")

H2("3.1 Research Design")
body(
    "This study follows a clear, repeatable pipeline that goes from raw survey data to an "
    "explainable screening tool. The key design choice is to keep the training population "
    "and the final test population separate. The model is trained and validated only on the "
    "United States BRFSS 2017 survey, and is then tested once on the independent Cambodia "
    "STEPS dataset. This design measures how well the model transfers to a new population, "
    "instead of only measuring performance on data similar to the training data.",
    "The main steps, shown in Figure 1, are: load the two surveys; align them into eleven "
    "common features; split the BRFSS data into training and validation sets; scale the "
    "features and balance the training data with SMOTETomek; train five models; choose the "
    "three best on validation and combine them into a Soft Voting ensemble; select the "
    "final model and threshold using validation F1-score with a recall requirement; test "
    "the final model once on Cambodia STEPS; explain the predictions with SHAP; group the "
    "results into risk tiers; and present everything in a simple dashboard with lifestyle "
    "advice.")
figure("pipeline", "Research design and pipeline workflow, from data loading and feature "
       "alignment to training, ensemble selection, threshold tuning, independent testing, "
       "explainability, and the dashboard.")

H2("3.2 Tools and Technologies Used")
body("All work was done in Python using common open-source libraries.")
H3("3.2.1 Python and Visual Studio Code")
body("Python was used as the main language because it has many machine learning libraries "
     "and supports Jupyter notebooks, which keep the work organised and repeatable. Visual "
     "Studio Code was used as the editor for the notebooks and the application.")
H3("3.2.2 pandas, NumPy, and Matplotlib")
body("pandas and NumPy were used to load, align, clean, and transform the survey data. "
     "Matplotlib was used to draw the figures in this report.")
H3("3.2.3 scikit-learn")
body("scikit-learn provided the train/validation split, the StandardScaler, the Logistic "
     "Regression and Random Forest models, the VotingClassifier used for the Soft Voting "
     "ensemble, the RandomizedSearchCV tuning, and the evaluation metrics (accuracy, "
     "precision, recall, F1-score, ROC and AUC, and the confusion matrix).")
H3("3.2.4 XGBoost, LightGBM, and CatBoost")
body("XGBoost, LightGBM, and CatBoost are gradient boosting libraries that work well on "
     "tabular data. They were used as three of the five candidate models, and they support "
     "ways to handle class imbalance. XGBoost was also used later as the tree-based model "
     "for the SHAP analysis.")
H3("3.2.5 imbalanced-learn (SMOTETomek)")
body("The imbalanced-learn library provided SMOTETomek, which combines SMOTE oversampling "
     "with Tomek-links cleaning. It was applied only to the training data.")
H3("3.2.6 SHAP")
body("The SHAP library was used to explain the model. Its tree explainer computes feature "
     "contributions for tree-based models efficiently, for both overall and patient-level "
     "explanations.")
H3("3.2.7 Streamlit and the Gemini API")
body("The pipeline was deployed as a web application using Streamlit, which shows each "
     "prediction as a patient dashboard. The Google Gemini API was used to turn the model "
     "explanation into simple lifestyle advice, with a fixed fallback message when the API "
     "is not available, so the tool still works offline.")

H2("3.3 Data Sources")
H3("3.3.1 BRFSS 2017 (Training and Validation)")
rich([("The BRFSS is a large yearly health survey carried out by the United States CDC. "
       "This study uses a cleaned binary version of the 2017 survey with ", {}),
      ("%s records" % f"{BRFSS_ROWS:,}", {"bold": True}),
      (", of which " + p(BRFSS_RATE) + " are labelled as having diabetes. Its large size "
       "and many risk-factor variables make it a good source for training.", {})])
H3("3.3.2 Cambodia STEPS (Independent Test)")
rich([("The WHO STEPS survey is a standard method used in many countries to measure NCD "
       "risk factors. This study uses a cleaned Cambodia STEPS dataset of ", {}),
      ("%s records" % f"{STEPS_ROWS:,}", {"bold": True}),
      (", of which " + p(STEPS_RATE) + " are labelled as having diabetes. This dataset is "
       "used only once, as an independent test set, and is never used for training, tuning, "
       "or model selection.", {})])

H2("3.4 Data Preparation and Feature Alignment")
body(
    "Because the two surveys were designed separately, their variables do not match exactly. "
    "The main preparation task was to align them into a shared set of risk factors that are "
    "present in both, are low-cost and non-invasive, and make clinical sense. Eleven common "
    "features remained after this step, grouped into three categories, as shown in Table 1. "
    "The target variable is Diabetes_binary, which is 1 for diabetes and 0 for no diabetes.")
table(["Group", "Aligned Features"],
      [["Demographic", "Sex, Age, Education"],
       ["Health Status", "HighBP (high blood pressure), HighChol (high cholesterol), BMI"],
       ["Lifestyle", "Smoker, PhysActivity, Fruits, Veggies, HvyAlcoholConsump"]],
      [1.6, 4.4],
      "Aligned feature set common to BRFSS and STEPS (11 features).")
body(
    "Two alignment notes are important. First, age is recorded as a group from 1 to 13 in "
    "BRFSS but in actual years in STEPS, so the STEPS ages were converted into the matching "
    "BRFSS age groups. Second, income, which is in BRFSS, was dropped because it is not "
    "consistently available in STEPS. Keeping only features with a clear match in both "
    "surveys makes the transfer fair. Basic cleaning of missing values and consistent "
    "encoding was also applied so that the feature columns mean the same thing in both "
    "datasets.")

H2("3.5 Feature Engineering and Imbalance Handling")
H3("3.5.1 Standardisation")
body("The features are on different scales, for example BMI compared with a yes/no smoking "
     "value. A StandardScaler was fit on the BRFSS training data to give each feature zero "
     "mean and unit variance, and the same scaler was then applied to the BRFSS validation "
     "set and the Cambodia STEPS test set. Fitting the scaler only on the training data "
     "prevents information from the validation or test data from leaking into the model.")
H3("3.5.2 SMOTETomek and Data Leakage")
body(
    "Because people with diabetes are the minority, the training data are imbalanced. "
    "SMOTETomek was used to fix this. SMOTE creates new synthetic minority examples by "
    "interpolating between real minority points, and the Tomek-links step removes pairs of "
    "opposite-class points that sit on the border, which cleans up overlapping cases.",
    "SMOTETomek was applied only to the training set. The validation set and the Cambodia "
    "STEPS test set were kept as real, untouched data. This is important because it avoids "
    "data leakage: if synthetic data were mixed into validation or test, the results would "
    "look better than they really are and would not reflect performance on real people. By "
    "balancing only the training data, the reported validation and test numbers come from "
    "genuine cases.")

H2("3.6 Models Used")
body("Five models were trained on the balanced BRFSS 2017 training data. The four tree-based "
     "and boosting models were tuned with RandomizedSearchCV using three-fold "
     "cross-validation and F1-score.")
H3("3.6.1 Logistic Regression")
body("Logistic Regression is a linear model for binary classification. It is fast and easy "
     "to explain, and is used here as a baseline. It was set with balanced class weights.")
H3("3.6.2 Random Forest")
body("Random Forest is a group of decision trees trained on random samples of the data, "
     "whose results are averaged. It captures non-linear patterns and reduces overfitting "
     "compared with a single tree.")
H3("3.6.3 XGBoost")
body("XGBoost builds trees one after another, where each new tree corrects the errors of "
     "the previous ones. It is a strong method for tabular data and can handle imbalance "
     "through weighting.")
H3("3.6.4 LightGBM")
body("LightGBM is another gradient boosting method that grows trees leaf-wise and uses "
     "histogram-based splits, which makes it fast and memory-efficient on large datasets.")
H3("3.6.5 CatBoost")
body("CatBoost is a gradient boosting method that is robust and works well with default "
     "settings. It was one of the three models chosen for the final ensemble.")

H2("3.7 Soft Voting Ensemble")
body(
    "Instead of trusting one model, the pipeline combines the strongest models. After all "
    "five models were checked on the validation set, the three best by validation F1-score, "
    "which were XGBoost, Logistic Regression, and CatBoost, were combined into a Soft "
    "Voting ensemble.",
    "Soft Voting works by averaging the predicted diabetes probabilities from the three "
    "models. The averaged probability is then compared with the chosen threshold to decide "
    "the final class. Soft Voting is useful because it balances the strengths of the "
    "different models and usually gives more stable predictions than any single model on "
    "its own.")

H2("3.8 Threshold Tuning")
body(
    "A model gives a probability, which must be turned into a yes/no decision using a "
    "threshold. The default of 0.50 is often not the best choice for imbalanced data. The "
    "selection rule used here is to choose the model-and-threshold pair with the highest "
    "validation F1-score among those where diabetes recall is at least 70%. This keeps "
    "recall high, which is the priority for screening, while still keeping a reasonable "
    "balance with precision. Using this rule, the Soft Voting ensemble with a threshold of "
    + str(THR) + " was selected as the final model.")

H2("3.9 Evaluation Metrics")
body("Because accuracy can be misleading on imbalanced data, several metrics were used, all "
     "based on the confusion matrix.")
H3("3.9.1 Confusion Matrix")
body("The confusion matrix shows four values: true positives (diabetes correctly found), "
     "true negatives (no diabetes correctly cleared), false positives (no diabetes wrongly "
     "flagged, a Type I error), and false negatives (diabetes wrongly cleared, a Type II "
     "error). In screening, false negatives are the most harmful, which is why recall is "
     "the priority.")
figure("cm_structure", "The structure of a confusion matrix, showing true positives, true "
       "negatives, false positives, and false negatives.")
H3("3.9.2 Accuracy, Precision, Recall, and F1-Score")
body("Accuracy is the share of all predictions that are correct. Precision is the share of "
     "flagged people who really have diabetes. Recall is the share of real diabetes cases "
     "that the model finds, and it is the main clinical priority here. F1-score is the "
     "harmonic mean of precision and recall and balances the two, and it is the main metric "
     "used for model selection.")
H3("3.9.3 ROC Curve and AUC")
body("The ROC curve plots the true positive rate against the false positive rate as the "
     "threshold changes, and the AUC (Area Under the Curve) summarises how well the model "
     "separates the two classes across all thresholds. An AUC of 1.0 is perfect and 0.5 is "
     "random guessing.")
figure("roc", "Illustration of an ROC curve and the area under the curve (AUC).")

H2("3.10 Explainability Method")
body(
    "Explainability is treated as part of the method, not an extra step. After the final "
    "model is selected, SHAP is used on the Cambodia STEPS test set. SHAP gives each "
    "feature a value showing how much it increased or decreased the predicted risk, for "
    "each person. The absolute SHAP values can be averaged over all patients to give an "
    "overall importance ranking, while the values for one patient explain a single "
    "prediction.",
    "One point must be made clearly. The final model is a Soft Voting ensemble, so a "
    "tree-based SHAP explainer cannot explain the full ensemble exactly. For this reason, "
    "XGBoost was used as a tree-based proxy model for SHAP. This choice is reasonable "
    "because XGBoost was the strongest individual tree-based model by validation F1-score, "
    "it is one of the three models inside the Soft Voting ensemble, and TreeSHAP is "
    "efficient and well suited to tree-based models. The SHAP results therefore explain the "
    "strongest tree-based part of the ensemble and help us understand the model’s "
    "behaviour, but they are not a perfect explanation of the full Soft Voting model.")
body("Finally, the predicted probabilities are grouped into three risk tiers, Low, "
     "Moderate, and High, to make the result easier to read in the dashboard. The meaning "
     "of these tiers is explained carefully in Section 4.7.")

# --------------------------------------------------------------------------- #
#  CHAPTER 4 — RESULTS
# --------------------------------------------------------------------------- #
H1("CHAPTER 4  RESULTS")

H2("4.1 Experimental Setup")
body("All experiments were run on a personal laptop using the Python tools described in "
     "Chapter 3, inside Jupyter notebooks. A fixed random seed of 42 was used for the data "
     "split, the resampling, the model setup, and the hyperparameter search, so that the "
     "whole pipeline can be reproduced. Hyperparameters were tuned with RandomizedSearchCV "
     "using three-fold cross-validation and F1-score.")

H2("4.2 Data Distribution")
rich([("The two surveys differ in size and slightly in diabetes rate. BRFSS 2017 has "
       "%s records with a " % f"{BRFSS_ROWS:,}", {}),
      (p(BRFSS_RATE), {"bold": True}),
      (" diabetes rate, while Cambodia STEPS has %s records with a " % f"{STEPS_ROWS:,}", {}),
      (p(STEPS_RATE), {"bold": True}),
      (" rate. Figure 2 compares the two class distributions.", {})])
figure("survey_dist", "Class distribution of BRFSS 2017 (training source) and Cambodia "
       "STEPS (independent test).")
body("The figure shows that both datasets are imbalanced, with far more non-diabetes cases "
     "than diabetes cases. The diabetes rate is close in the two surveys (16.0% versus "
     "14.0%), but the BRFSS dataset is much larger, which is why it is used for training.")

H2("4.3 Data Splits")
body("The BRFSS 2017 data were split into 80% for training and 20% for validation using "
     "stratified sampling, so the diabetes rate stays the same in both parts. The whole "
     "Cambodia STEPS dataset was kept as the test set. Table 2 and Figure 3 show the sizes "
     "and class balance of each split.")
table(["Split", "Total Records", "No Diabetes", "Diabetes", "Diabetes Rate"],
      [[s[0], f"{s[1]:,}", f"{s[2]:,}", f"{s[3]:,}", p(s[4])] for s in SPLITS],
      [2.5, 1.1, 1.1, 0.9, 1.0],
      "Class distribution per data split. Cambodia STEPS is used only for the final test.")
figure("split_dist", "Class distribution across the training, validation, and test splits "
       "(log scale, because BRFSS is much larger than STEPS).")
body("Table 2 confirms that the training and validation sets keep almost the same diabetes "
     "rate (about 16%), which is the effect of stratified splitting. The Cambodia STEPS test "
     "set is much smaller and comes from a different population, which is the main challenge "
     "of this study.")

H2("4.4 Class Balancing with SMOTETomek")
body("SMOTETomek was applied to the training set only, to balance the two classes before "
     "training. Table 3 and Figure 4 show the class counts before and after balancing.")
table(["Training Set", "No Diabetes", "Diabetes"],
      [["Before SMOTETomek", f"{SMOTE_BEFORE[0]:,}", f"{SMOTE_BEFORE[1]:,}"],
       ["After SMOTETomek", f"{SMOTE_AFTER[0]:,}", f"{SMOTE_AFTER[1]:,}"]],
      [2.4, 1.8, 1.8],
      "Training-set class counts before and after SMOTETomek (validation and test untouched).")
figure("smotetomek", "BRFSS 2017 training set before and after SMOTETomek balancing.")
body("After SMOTETomek, the two classes in the training set are equal, with " +
     f"{SMOTE_AFTER[0]:,}" + " records each. This helps the models learn the minority "
     "diabetes class better. The validation and test sets were not changed, so the results "
     "reported later come from real cases and are not affected by synthetic data.")

H2("4.5 Validation Results of Individual Models")
body("The five models were checked on the BRFSS 2017 validation set using a default "
     "threshold of 0.50. Table 4 reports accuracy, precision, recall, and F1-score for each "
     "model, and Figure 5 shows the same results as a bar chart.")
table(["Model", "Accuracy", "Precision", "Recall", "F1-score"],
      [[m[0], p(m[1]), p(m[2]), p(m[3]), p(m[4])] for m in VAL_MODELS],
      [2.3, 1.1, 1.1, 1.0, 1.1],
      "Individual model performance on the BRFSS 2017 validation set (threshold 0.50).")
figure("val_models", "Comparison of the five models on the validation set "
       "(threshold 0.50).")
body("The results show a trade-off between recall and precision. Logistic Regression has "
     "the highest recall (" + p(74.45) + ") but lower precision, while LightGBM and "
     "CatBoost have higher accuracy but lower recall. XGBoost gives the best F1-score (" +
     p(45.60) + "), with a good balance between recall and precision. Based on validation "
     "F1-score, the three best models are XGBoost, Logistic Regression, and CatBoost, which "
     "were chosen for the Soft Voting ensemble. The neural network was not used; the final "
     "design is based on these tree-based and linear models combined by Soft Voting.")

H2("4.6 Final Model and Threshold Selection")
body("The three best models were combined into a Soft Voting ensemble. The final model and "
     "threshold were chosen using the rule of highest validation F1-score among options "
     "where diabetes recall is at least 70%. This rule selected the Soft Voting ensemble "
     "with a threshold of " + str(THR) + ". Table 5 shows the validation results of the "
     "selected model.")
table(["Metric", "Soft Voting ensemble (Validation, threshold 0.43)"],
      [["Recall (clinical priority)", p(VAL_FINAL["Recall"])],
       ["F1-score (selection metric)", p(VAL_FINAL["F1"])],
       ["Accuracy", p(VAL_FINAL["Accuracy"])],
       ["Precision", p(VAL_FINAL["Precision"])]],
      [3.0, 3.2],
      "Validation performance of the selected Soft Voting ensemble at threshold 0.43.")
body("At threshold 0.43, the Soft Voting ensemble reaches a validation recall of " +
     p(VAL_FINAL["Recall"]) + ", which meets the recall requirement, with an F1-score of " +
     p(VAL_FINAL["F1"]) + ". The accuracy is " + p(VAL_FINAL["Accuracy"]) + " and the "
     "precision is " + p(VAL_FINAL["Precision"]) + ". The lower threshold (0.43 instead of "
     "0.50) makes the model flag more people as at risk, which raises recall at the cost of "
     "precision. This is the correct trade-off for a screening tool.")
figure_placeholder("Threshold tuning result on the validation set, showing how recall, "
                   "precision, F1-score, and accuracy change with the threshold, with the "
                   "chosen threshold of 0.43 marked.",
                   "Source: export the threshold-sweep plot from the final notebook "
                   "(threshold tuning cell).")

H2("4.7 Risk Tiers and the Classification Threshold")
body(
    "It is important to separate two different things: the classification threshold and the "
    "risk tiers. The classification threshold is the value used to make the yes/no "
    "decision. In this study the final threshold is " + str(THR) + ". If the predicted "
    "probability is " + str(THR) + " or higher, the person is classified as diabetes risk; "
    "if it is below " + str(THR) + ", the person is classified as no diabetes risk.",
    "The risk tiers are different. They are only a way to communicate the probability level "
    "in the dashboard, so the result is easier to understand. The tiers are based on the "
    "predicted probabilities from the Soft Voting model on Cambodia STEPS, using these "
    "ranges: Low risk is a predicted probability below 30%, Moderate risk is from 30% up to "
    "below 70%, and High risk is 70% or above.",
    "These tiers are model-based communication categories. They are not official medical "
    "diagnostic cut-offs and should not be presented as clinically validated thresholds. "
    "They simply help a reader see whether a predicted probability is low, medium, or high. "
    "Table 6 and Figure 6 show how the Cambodia STEPS respondents fall into the three "
    "tiers.")
table(["Risk Tier", "Predicted Probability", "STEPS Respondents"],
      [["Low", "below 30%", str(TIERS["Low"])],
       ["Moderate", "30% to below 70%", str(TIERS["Moderate"])],
       ["High", "70% or above", str(TIERS["High"])]],
      [1.8, 2.4, 1.8],
      "Risk-tier distribution of the Cambodia STEPS respondents under the Soft Voting model.")
figure("tiers", "Risk-tier distribution for the Cambodia STEPS test set.")
body("Most respondents fall into the Low and Moderate tiers, with " + str(TIERS["High"]) +
     " in the High tier. These tiers are only for communication and interpretation; the "
     "actual classification still uses the 0.43 threshold described above.")

H2("4.8 Independent Test on Cambodia STEPS")
body("The selected Soft Voting model and the 0.43 threshold were then applied, for the "
     "first time, to the Cambodia STEPS test set. This data had no part in training, "
     "validation, or selection, so it gives an honest measure of how the model transfers to "
     "a new population. Table 7 reports the results.")
table(["Metric", "Cambodia STEPS (independent test)"],
      [["Recall (clinical priority)", p(TEST_FINAL["Recall"])],
       ["F1-score", p(TEST_FINAL["F1"])],
       ["Accuracy", p(TEST_FINAL["Accuracy"])],
       ["Precision", p(TEST_FINAL["Precision"])]],
      [3.0, 3.2],
      "Independent test results on Cambodia STEPS (Soft Voting ensemble, threshold 0.43).")
body("The most important result is recall. At " + p(TEST_FINAL["Recall"]) + ", the model "
     "correctly found most of the real diabetes cases in a population it had never seen, "
     "which is what a screening tool needs to do. At the same time, accuracy (" +
     p(TEST_FINAL["Accuracy"]) + ") and precision (" + p(TEST_FINAL["Precision"]) + ") were "
     "lower than on validation, and the F1-score dropped to " + p(TEST_FINAL["F1"]) + ". In "
     "simple terms, the model flags many people to make sure it does not miss real cases, "
     "but this also produces many false alarms when it is used on a different population.")
body("Figure 7 shows the confusion matrix on Cambodia STEPS. Out of 122 real diabetes "
     "cases, the model correctly found 93 and missed 29. Out of 751 non-diabetes cases, it "
     "correctly cleared 392 but wrongly flagged 359. This matches the pattern of high recall "
     "and low precision.")
figure("confusion", "Confusion matrix on the Cambodia STEPS test set (Soft Voting, "
       "threshold 0.43).")

H2("4.9 Validation vs Test Comparison")
body("Table 8 and Figure 8 compare the validation and test results of the Soft Voting "
     "model side by side, which shows the cross-population gap clearly.")
table(["Metric", "Validation (BRFSS 2017)", "Test (Cambodia STEPS)"],
      [["Recall", p(VAL_FINAL["Recall"]), p(TEST_FINAL["Recall"])],
       ["F1-score", p(VAL_FINAL["F1"]), p(TEST_FINAL["F1"])],
       ["Accuracy", p(VAL_FINAL["Accuracy"]), p(TEST_FINAL["Accuracy"])],
       ["Precision", p(VAL_FINAL["Precision"]), p(TEST_FINAL["Precision"])]],
      [1.9, 2.1, 2.1],
      "Validation versus independent-test results for the Soft Voting ensemble.")
figure("val_vs_test", "Validation versus Cambodia STEPS test results for the Soft Voting "
       "ensemble.")
body("Recall actually went up slightly from validation (" + p(VAL_FINAL["Recall"]) +
     ") to test (" + p(TEST_FINAL["Recall"]) + "), because the low threshold flags many "
     "cases. But accuracy and precision dropped on the Cambodia data. This is the effect of "
     "cross-population domain shift: the link between the risk factors and diabetes in "
     "Cambodia is not the same as the one the model learned from United States data. This is "
     "an honest result and is expected when transferring a model between very different "
     "populations.")

H2("4.10 Explainability Results")
body(
    "SHAP was used on the Cambodia STEPS predictions, with XGBoost as the tree-based proxy "
    "model explained in Section 3.10. Figure 9 shows the overall feature importance, based "
    "on the average absolute SHAP values, and Figure 10 shows a local explanation for one "
    "patient. In line with medical knowledge and earlier studies, the strongest factors "
    "were body mass index, high blood pressure, high cholesterol, and age, while lifestyle "
    "factors such as physical activity and fruit and vegetable intake tended to lower the "
    "predicted risk.")
figure_placeholder("SHAP global feature importance (mean absolute SHAP value) on the "
                   "Cambodia STEPS test set, using XGBoost as the tree-based proxy model.",
                   "Source: export the SHAP summary/importance plot from the final notebook "
                   "(SHAP cell).")
figure_placeholder("SHAP local explanation for a single Cambodia STEPS patient, showing "
                   "which features increased or decreased the predicted risk.",
                   "Source: export the SHAP waterfall/force plot for one patient from the "
                   "final notebook.")
body("These explanations are useful in two ways. Overall, they show that the model relies "
     "on sensible health factors and not on strange patterns, which helps build trust. For "
     "a single patient, they give a short and clear reason for the prediction, which is the "
     "basis of the dashboard. As noted before, these SHAP results explain the strongest "
     "tree-based part of the ensemble, not the full Soft Voting model exactly.")
figure_placeholder("Distribution of predicted probabilities on Cambodia STEPS, separated "
                   "by true class, with the 30% and 70% tier cut-offs marked.",
                   "Source: export the predicted-probability distribution plot from the "
                   "final notebook (risk-level cell).")

H2("4.11 Patient Dashboard and Actionable Insights")
body("The final model is shown to the user through a Streamlit dashboard. For each person, "
     "the dashboard shows the predicted probability and risk tier, the main SHAP reasons "
     "coloured by whether they raise or lower risk, the contribution of the demographic, "
     "health-status, and lifestyle groups, and simple lifestyle advice generated through "
     "the Gemini API with a safe offline fallback. The application can also accept a CSV "
     "file to screen many people at once. This turns a single number into an explanation "
     "and a set of actionable insights that a non-expert can understand.")
figure_placeholder("Patient dashboard from the deployed application, showing the risk "
                   "gauge, the main SHAP reasons, the group contributions, and the "
                   "lifestyle advice.",
                   "Source: take a screenshot of the running Streamlit app and insert it "
                   "here.")
body("It is important to repeat that the dashboard is a screening support tool, not a "
     "medical diagnosis system. A High risk result is a reason to do a proper medical test, "
     "not a confirmed diagnosis.")

# --------------------------------------------------------------------------- #
#  CHAPTER 5 — DISCUSSION
# --------------------------------------------------------------------------- #
H1("CHAPTER 5  DISCUSSION")

H2("5.1 Overview")
body("This chapter explains what the results mean for diabetes screening in Cambodia. It "
     "looks at how well the models worked, the cross-population gap, the value of "
     "explainability and the dashboard, and the main challenges of the study.")

H2("5.2 Effectiveness of the Models")
body(
    "On the validation data, the approach worked as expected. The tree-based and boosting "
    "models performed well, and the Soft Voting ensemble combined the strengths of XGBoost, "
    "Logistic Regression, and CatBoost. Choosing the model and threshold by validation "
    "F1-score with a recall requirement made sure the final model fits the needs of "
    "screening rather than just chasing accuracy.",
    "The choice to focus on recall was the most important design decision. If the project "
    "had focused on accuracy, it could have reported a higher accuracy by leaning toward the "
    "majority “no diabetes” class, but the tool would then miss many real cases. By "
    "keeping recall high, the model is more careful, which is the safer choice for "
    "screening. The cost is lower precision and more false alarms, but a false alarm only "
    "leads to an extra test, while a missed case can lead to untreated disease.")

H2("5.3 The Cross-Population Gap")
body(
    "The clearest finding is the gap between validation and test. On the BRFSS validation "
    "data the model worked reasonably, but when used on Cambodia STEPS its accuracy and "
    "precision dropped, even though recall stayed high. This is the effect of domain shift, "
    "and it is exactly what single-population studies cannot show, because they test on data "
    "similar to the training data.",
    "There are several likely reasons. The link between body mass index and diabetes risk "
    "is known to differ between Western and Asian populations, so a value learned on United "
    "States data may not fit Cambodia well. Diet, physical activity, and the way survey "
    "questions are asked also differ, so even aligned features are not perfectly equal. The "
    "eleven common features, while needed for a fair test, also leave out information that a "
    "local model could use.",
    "The honest reading is not that the model failed, but that transferring a diabetes model "
    "from one population to a very different one is hard. A transferred model like this "
    "should be treated as a first-pass screening tool with high recall, whose positive "
    "results need local confirmation, and not as a diagnostic system.")

H2("5.4 Value of Explainability and the Dashboard")
body(
    "Even with its limits as a predictor, the system has real value because it is "
    "explainable and deployed. The SHAP analysis showed that the model uses sensible health "
    "factors such as body mass index, blood pressure, cholesterol, and age, which is "
    "important for trust. The patient-level explanations turn a probability into a short "
    "reason, and the lifestyle advice turns that reason into action. The dashboard puts all "
    "of this in a form that a non-expert can use, and the batch mode supports screening many "
    "people. In a setting where trust and clarity matter as much as raw accuracy, these are "
    "important features and not just extras. The honesty of using XGBoost as a proxy for "
    "SHAP, rather than claiming a perfect explanation of the ensemble, also keeps the "
    "interpretation realistic.")

H2("5.5 Challenges and Findings")
body(
    "Several challenges shaped this work. The main one was data: the lack of a large local "
    "labelled dataset is the reason transfer learning was needed, and the small size of the "
    "STEPS set limits how precise the test results can be. Aligning two different surveys "
    "into a fair common set needed careful choices, such as the age-group mapping and "
    "dropping income. Class imbalance needed careful handling with SMOTETomek and threshold "
    "tuning, applied only to the training data to avoid leakage. Balancing performance with "
    "explainability and ease of use was also a constant concern.",
    "The main findings are clear. Tree-based and boosting models, combined by Soft Voting, "
    "are a good fit for this tabular task. A recall-focused threshold is essential for "
    "screening and changes how the model behaves. Cross-population transfer has a real cost "
    "that single-population studies hide, and reporting this honestly is more useful than "
    "an inflated accuracy claim. And explainability is what makes the tool trustworthy and "
    "usable in practice.")

# --------------------------------------------------------------------------- #
#  CHAPTER 6 — CONCLUSION
# --------------------------------------------------------------------------- #
H1("CHAPTER 6  CONCLUSION")

H2("6.1 Summary of Findings")
body(
    "This study built and tested an explainable machine learning pipeline for diabetes risk "
    "screening in Cambodia, using a model trained on the large United States BRFSS 2017 "
    "survey and tested once on the independent Cambodia STEPS dataset. Eleven common risk "
    "factors were aligned into a shared set; the training data were scaled and balanced with "
    "SMOTETomek; five models were trained; and the three best (XGBoost, Logistic Regression, "
    "and CatBoost) were combined into a Soft Voting ensemble. The final model and threshold "
    "were chosen by validation F1-score while keeping recall at least 70%.",
    "The selected Soft Voting ensemble, at a threshold of " + str(THR) + ", reached a recall "
    "of " + p(TEST_FINAL["Recall"]) + " on the Cambodia STEPS test set, finding most of the "
    "real diabetes cases in a population it had never seen. Accuracy (" +
    p(TEST_FINAL["Accuracy"]) + ") and precision (" + p(TEST_FINAL["Precision"]) + ") were "
    "lower on the Cambodian data, which honestly reflects the difficulty of cross-population "
    "transfer rather than a fault of the method. SHAP, using XGBoost as a proxy, showed that "
    "the model relies on sensible health factors and allowed clear patient-level "
    "explanations, which were shown together with risk tiers and lifestyle advice in a "
    "dashboard.",
    "The contribution of this study is an honest benchmark for cross-population diabetes "
    "risk transfer to a low-resource setting, a screening pipeline focused on recall and "
    "explainability instead of headline accuracy, and a working, explainable tool that gives "
    "actionable insights. The main message is that this kind of transferred, recall-focused "
    "screening model can be a useful early-awareness and research tool, but it cannot "
    "replace medical diagnosis, and its positive results should always be confirmed by a "
    "proper medical test.")

H2("6.2 Future Work")
body("Several directions could build on this study. The most important is to collect local "
     "labelled Cambodian data. Even a moderate, confirmed dataset would allow the model to "
     "be retrained or fine-tuned for the local population, which the results suggest would "
     "improve precision and calibration.")
bullet("apply domain-adaptation or transfer-learning methods, such as fine-tuning on a "
       "small local sample, to reduce the cross-population gap.", lead="Domain adaptation: ")
bullet("add more locally useful features that were left out of the common set, such as "
       "waist circumference, sedentary time, and more detailed diet information, where they "
       "can be collected reliably.", lead="Richer features: ")
bullet("calibrate the predicted probabilities so that the risk tiers better match the true "
       "probabilities in the Cambodian population.", lead="Calibration: ")
bullet("check the model for fairness across sex, age, and income groups using the SHAP "
       "tools already in place.", lead="Fairness: ")
bullet("test the tool in a real setting with community health workers, and move toward "
       "real-time and offline use.", lead="Field testing: ")
body("Together, these steps would move the system from an honest research benchmark toward a "
     "locally validated and fair screening support tool that could help with early diabetes "
     "awareness in Cambodia.")

# --------------------------------------------------------------------------- #
#  REFERENCES
# --------------------------------------------------------------------------- #
H1("REFERENCES")
ph_para("Check and reformat all references to your institution’s required citation style "
        "before submission.")
REFS = [
    ("Xie, Z., Nikolayeva, O., Luo, J., & Li, D. (2019). Building Risk Prediction Models "
     "for Type 2 Diabetes Using Machine Learning Techniques. Preventing Chronic Disease, "
     "16, E130. https://www.cdc.gov/pcd/issues/2019/19_0109.htm", False),
    ("Khokhar, P. B., et al. (2025). Towards Transparent and Accurate Diabetes Prediction "
     "Using Machine Learning and Explainable Artificial Intelligence. arXiv preprint "
     "arXiv:2501.18071. https://arxiv.org/abs/2501.18071", False),
    ("Katiyar, et al. (2024). Recent Advancements Using Machine Learning and Deep Learning "
     "Approaches for Diabetes Detection. ScienceDirect. "
     "https://www.sciencedirect.com/science/article/pii/S2772671124002419  "
     "[Verify the exact citation details before submission — the provided link appears to "
     "overlap with another listed paper.]", True),
    ("Li, et al. (2025). Performance of Diabetes Risk Prediction Models. ScienceDirect.  "
     "[Verify the exact citation details and DOI before submission — the provided link is "
     "the same as the Katiyar et al. (2024) link and must be corrected.]", True),
    ("Khokhar, P. B., et al. (2025). Advances in Artificial Intelligence for Diabetes "
     "Prediction: Insights from a Systematic Literature Review. Artificial Intelligence in "
     "Medicine. https://www.sciencedirect.com/science/article/pii/S0933365725000673", False),
    ("Centers for Disease Control and Prevention. (2017). Behavioral Risk Factor "
     "Surveillance System (BRFSS) Survey Data and Documentation. U.S. Department of Health "
     "and Human Services.", False),
    ("World Health Organization. (2010). Cambodia STEPS Survey: WHO STEPwise Approach to "
     "Noncommunicable Disease Risk Factor Surveillance. WHO Regional Office.", False),
    ("Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. "
     "Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and "
     "Data Mining, 785–794.", False),
    ("Ke, G., et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. "
     "Advances in Neural Information Processing Systems, 30.", False),
    ("Prokhorenkova, L., et al. (2018). CatBoost: Unbiased Boosting with Categorical "
     "Features. Advances in Neural Information Processing Systems, 31.", False),
    ("Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5–32.", False),
    ("Lundberg, S. M., & Lee, S.-I. (2017). A Unified Approach to Interpreting Model "
     "Predictions. Advances in Neural Information Processing Systems, 30, 4765–4774.", False),
    ("Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: "
     "Synthetic Minority Over-sampling Technique. Journal of Artificial Intelligence "
     "Research, 16, 321–357.", False),
    ("Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. Journal of "
     "Machine Learning Research, 12, 2825–2830.", False),
    ("International Diabetes Federation. (2021). IDF Diabetes Atlas, 10th edition. Brussels, "
     "Belgium.", False),
]
for text, verify in REFS:
    pp = doc.add_paragraph()
    pp.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    pp.paragraph_format.space_after = Pt(8)
    pp.paragraph_format.left_indent = Inches(0.5)
    pp.paragraph_format.first_line_indent = Inches(-0.5)
    run = pp.add_run(text); run.font.size = Pt(SIZE)
    if verify:
        run.font.color.rgb = PH

# --------------------------------------------------------------------------- #
#  APPENDICES
# --------------------------------------------------------------------------- #
H1("APPENDICES")
H2("Appendix A — Deployed Application")
rich([("Link to the deployed diabetes risk-screening web application: ", {}),
      ("[insert deployment URL]", {"color": PH, "italic": True})])
H2("Appendix B — Source Code and Notebooks")
body("The complete source code, including the final training notebook, the data-preparation "
     "notebooks, and the Streamlit application (app.py), is available in the project "
     "repository.")
rich([("Repository link: ", {}), ("[insert repository URL]", {"color": PH, "italic": True})])
H2("Appendix C — Reproducibility")
body("All experiments use a fixed random seed of 42. The final model and its preprocessing "
     "files are saved and loaded directly by the application, so that the model used in the "
     "app is the same as the one evaluated in this report.")

# --------------------------------------------------------------------------- #
#  Page numbering: roman for front matter (no number on the cover), arabic body
# --------------------------------------------------------------------------- #
front_sec = doc.sections[0]
front_sec.different_first_page_header_footer = True   # cover page shows no number
_margins(front_sec)
footer_pagenum(front_sec, "lowerRoman")
footer_pagenum(doc.sections[-1], "decimal")           # body restarts at 1

# --------------------------------------------------------------------------- #
#  Save
# --------------------------------------------------------------------------- #
doc.save(OUT_PATH)
print("Saved report to:", OUT_PATH)
print("Paragraphs:", len(doc.paragraphs), "| Tables:", len(doc.tables))
words = sum(len(par.text.split()) for par in doc.paragraphs)
for t in doc.tables:
    for row in t.rows:
        for c in row.cells:
            words += len(c.text.split())
print("Word count:", words)
