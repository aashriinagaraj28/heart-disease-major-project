"""Build the formatted PDF report from the project artifacts."""

from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.lib.utils import ImageReader


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
OUT = ROOT / "Heart_Disease_Prediction_Report.pdf"
NAVY = colors.HexColor("#16324F")
BLUE = colors.HexColor("#2B6CB0")
TEAL = colors.HexColor("#2C7A7B")
INK = colors.HexColor("#243447")
MUTED = colors.HexColor("#5B6B7A")
PALE = colors.HexColor("#EAF1F7")
GRID = colors.HexColor("#D7E0E8")

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(
    name="CoverKicker", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=10,
    leading=14, textColor=TEAL, alignment=TA_CENTER, spaceAfter=12,
))
styles.add(ParagraphStyle(
    name="CoverTitle", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=27,
    leading=33, textColor=NAVY, alignment=TA_CENTER, spaceAfter=18,
))
styles.add(ParagraphStyle(
    name="CoverSub", parent=styles["Normal"], fontName="Helvetica", fontSize=14,
    leading=20, textColor=MUTED, alignment=TA_CENTER,
))
styles.add(ParagraphStyle(
    name="Section", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=17,
    leading=21, textColor=NAVY, spaceBefore=3, spaceAfter=9, keepWithNext=True,
))
styles.add(ParagraphStyle(
    name="Subsection", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=11.5,
    leading=15, textColor=TEAL, spaceBefore=7, spaceAfter=5, keepWithNext=True,
))
styles.add(ParagraphStyle(
    name="BodyProject", parent=styles["BodyText"], fontName="Helvetica", fontSize=9.3,
    leading=13.2, textColor=INK, spaceAfter=7,
))
styles.add(ParagraphStyle(
    name="SmallProject", parent=styles["BodyText"], fontName="Helvetica", fontSize=8,
    leading=10.5, textColor=MUTED, spaceAfter=4,
))
styles.add(ParagraphStyle(
    name="CaptionProject", parent=styles["BodyText"], fontName="Helvetica-Oblique", fontSize=8,
    leading=10, textColor=MUTED, alignment=TA_CENTER, spaceBefore=2, spaceAfter=7,
))
styles.add(ParagraphStyle(
    name="CellProject", parent=styles["BodyText"], fontName="Helvetica", fontSize=8.1,
    leading=10, textColor=INK,
))
styles.add(ParagraphStyle(
    name="HeaderCellProject", parent=styles["BodyText"], fontName="Helvetica-Bold", fontSize=8,
    leading=10, textColor=colors.white,
))


def para(text: str, style: str = "BodyProject") -> Paragraph:
    return Paragraph(text, styles[style])


def table(rows: list[list[object]], widths: list[float], repeat: int = 1) -> Table:
    converted = []
    for row_index, row in enumerate(rows):
        converted.append([
            cell if isinstance(cell, (Paragraph, Image, Table)) else
            para(str(cell), "HeaderCellProject" if row_index == 0 else "CellProject")
            for cell in row
        ])
    t = Table(converted, colWidths=widths, repeatRows=repeat, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.45, GRID),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE]),
    ]))
    return t


def chart(filename: str, max_width: float, max_height: float) -> Image:
    path = ARTIFACTS / filename
    iw, ih = ImageReader(str(path)).getSize()
    scale = min(max_width / iw, max_height / ih)
    img = Image(str(path), width=iw * scale, height=ih * scale)
    img.hAlign = "CENTER"
    return img


def on_page(canvas, doc) -> None:
    canvas.saveState()
    width, height = A4
    if doc.page > 1:
        canvas.setStrokeColor(GRID)
        canvas.setLineWidth(0.6)
        canvas.line(doc.leftMargin, height - 34, width - doc.rightMargin, height - 34)
        canvas.setFont("Helvetica-Bold", 7.5)
        canvas.setFillColor(MUTED)
        canvas.drawString(doc.leftMargin, height - 26, "HEART DISEASE PREDICTION  |  MAJOR PROJECT")
        canvas.line(doc.leftMargin, 29, width - doc.rightMargin, 29)
        canvas.setFont("Helvetica", 7.5)
        canvas.drawString(doc.leftMargin, 17, "Prepared from supplied brief and dataset  |  25 October 2026")
        page_label = f"{doc.page}"
        canvas.drawRightString(width - doc.rightMargin, 17, page_label)
    canvas.restoreState()


def main() -> None:
    metrics = pd.read_csv(ARTIFACTS / "model_comparison.csv")
    summary = pd.read_json(ARTIFACTS / "project_summary.json", typ="series")
    importance = pd.read_csv(ARTIFACTS / "feature_importance.csv")
    target_counts = summary["target_counts_after_deduplication"]
    selected = str(summary["selected_model_by_cv_roc_auc"])
    cv_auc = float(summary["selected_model_cv_roc_auc"])
    cv_metrics = summary["selected_model_test_metrics"]
    test_cm = summary["selected_model_confusion_matrix"]

    doc = SimpleDocTemplate(
        str(OUT), pagesize=A4, rightMargin=18 * mm, leftMargin=18 * mm,
        topMargin=17 * mm, bottomMargin=17 * mm,
        title="Heart Disease Prediction Using Machine Learning Classification Models",
        author="Major Project Report",
    )
    width = doc.width
    story = []

    # Cover page
    story.extend([Spacer(1, 42 * mm), para("MAJOR PROJECT REPORT", "CoverKicker")])
    story.append(para("Heart Disease Prediction Using Machine Learning Classification Models", "CoverTitle"))
    story.append(para("A comparison of four supervised classification approaches", "CoverSub"))
    story.extend([Spacer(1, 23 * mm)])
    cover_data = [
        [para("PROJECT FOCUS", "HeaderCellProject"), para("DATASET", "HeaderCellProject")],
        [para("Binary classification, model comparison, feature interpretation, and an optional prediction demo.", "CellProject"),
         para("303 supplied rows; 302 unique rows used; 13 input features and a binary target.", "CellProject")],
    ]
    cover_table = Table(cover_data, colWidths=[width * 0.5, width * 0.5])
    cover_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY), ("BACKGROUND", (0, 1), (-1, 1), PALE),
        ("GRID", (0, 0), (-1, -1), 0.6, GRID), ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 11), ("RIGHTPADDING", (0, 0), (-1, -1), 11),
        ("TOPPADDING", (0, 0), (-1, -1), 10), ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))
    story.append(cover_table)
    story.extend([Spacer(1, 18 * mm), para("Prepared from the supplied project brief and dataset", "CoverSub"),
                  Spacer(1, 4 * mm), para("25 October 2026", "CoverKicker"), PageBreak()])

    # Abstract and data overview
    story.append(para("Abstract", "Section"))
    story.append(para(
        "This project compares Logistic Regression, Decision Tree, Random Forest, and a small "
        "feedforward neural network for predicting the supplied binary heart-disease target from "
        "13 patient attributes. The worksheet contains 303 records and 14 columns. It has no "
        "missing cells and one exact duplicate row; after removing that row, 302 records were used. "
        "Hyperparameters were tuned with five-fold stratified cross-validation on the training "
        "partition. Logistic Regression had the strongest mean cross-validation ROC-AUC (0.911) "
        "and was selected. On the held-out test set it achieved 0.820 accuracy, 0.844 precision, "
        "0.818 recall, 0.831 F1-score, and 0.893 ROC-AUC. The Random Forest had the highest test "
        "ROC-AUC (0.904), but its cross-validation score was slightly lower. Results on this small "
        "sample are educational and do not establish clinical performance.",
    ))
    story.append(para("Dataset summary after duplicate removal", "Subsection"))
    summary_rows = [
        ["Item", "Value"],
        ["Rows used for analysis", "302 (303 supplied; one exact duplicate removed)"],
        ["Input features / target", "13 input features / binary target"],
        ["Target 0", f"{target_counts['0']} ({target_counts['0'] / 302:.1%})"],
        ["Target 1", f"{target_counts['1']} ({target_counts['1'] / 302:.1%})"],
        ["Missing cells", "0"],
        ["Original file format", "Excel workbook saved with a .csv filename; exported to a proper CSV for this package"],
    ]
    story.append(table(summary_rows, [width * 0.34, width * 0.66]))
    story.extend([Spacer(1, 3 * mm), para("Target class counts", "CaptionProject"),
                  chart("target_distribution.png", width * 0.83, 66 * mm),
                  PageBreak()])

    # EDA and preparation
    story.append(para("Exploratory analysis and data preparation", "Section"))
    story.append(para(
        "The two target classes are represented at similar rates in the supplied records. "
        "The age distributions overlap. Among the five continuous measurements, the largest "
        "simple correlations with the binary target are 0.42 for maximum heart rate (`thalach`) "
        "and -0.43 for `oldpeak`; these are descriptive relationships and not causal findings. "
        "The correlation chart includes continuous measurements only, because the dataset's "
        "category codes are not treated as a numeric scale.",
    ))
    story.append(para("Age by target class", "Subsection"))
    story.append(chart("age_by_target.png", width * 0.78, 58 * mm))
    story.append(para("Numeric measurement correlations", "Subsection"))
    story.append(chart("numeric_correlation.png", width * 0.68, 72 * mm))
    story.append(para(
        "Five numeric columns were median-imputed and standardized. Eight coded columns "
        "(`sex`, `cp`, `fbs`, `restecg`, `exang`, `slope`, `ca`, `thal`) were most-frequent "
        "imputed and one-hot encoded. There were no missing cells, but imputation is retained "
        "inside the pipelines for future input files. All supplied code values were kept; the "
        "workbook does not include a complete verified codebook.",
        "SmallProject",
    ))
    story.append(PageBreak())

    # Method and comparison table
    story.append(para("Modeling methodology", "Section"))
    story.append(para(
        "A stratified 80/20 split produced 241 training rows and 61 test rows (`random_state=42`). "
        "Each model uses the same preprocessing pipeline. Grid search with five-fold stratified "
        "cross-validation selected hyperparameters using mean ROC-AUC on training data only. "
        "The final model family was selected by the highest training cross-validation ROC-AUC; "
        "the test partition was held out for performance comparison. Precision, recall, and F1 "
        "use target 1 as the positive class at the 0.50 probability threshold.",
    ))
    param_rows = [
        ["Model", "Tuned settings"],
        ["Logistic Regression", "C = 0.1, 1, 10; class weight = none or balanced"],
        ["Decision Tree", "Depth = 3 or unlimited; leaf size = 2 or 5; class weight = none or balanced"],
        ["Random Forest", "300 trees; depth = 6 or unlimited; leaf size = 1 or 3; class weight = none or balanced"],
        ["Feedforward Neural Network", "Hidden layers = (16), (32), or (32, 16); alpha = 0.0001 or 0.01"],
    ]
    story.append(table(param_rows, [width * 0.29, width * 0.71]))
    story.append(para("Held-out test results", "Subsection"))
    result_rows = [["Model", "CV AUC", "Accuracy", "Precision", "Recall", "F1", "Test AUC"]]
    for _, row in metrics.iterrows():
        result_rows.append([
            str(row["Model"]), f"{row['CV ROC-AUC']:.3f}", f"{row['Accuracy']:.3f}",
            f"{row['Precision']:.3f}", f"{row['Recall']:.3f}", f"{row['F1']:.3f}",
            f"{row['ROC-AUC']:.3f}",
        ])
    story.append(table(result_rows, [width * x for x in [0.30, 0.12, 0.12, 0.12, 0.10, 0.09, 0.15]]))
    story.append(para(
        f"{selected} was selected because it had the highest mean cross-validation ROC-AUC ({cv_auc:.3f}). "
        "Random Forest performed slightly better on the single test ROC-AUC. Since the scores are "
        "close and the test set is small, this difference is not enough to establish a reliable winner.",
    ))
    story.append(PageBreak())

    # Evaluation visuals
    story.append(para("Evaluation visuals", "Section"))
    story.append(para("All four tuned models were evaluated on the same held-out test records.", "BodyProject"))
    story.append(chart("model_comparison.png", width, 68 * mm))
    story.append(para("ROC curves on the held-out set", "Subsection"))
    story.append(chart("roc_curves.png", width * 0.82, 78 * mm))
    story.append(para(
        f"Selected model confusion matrix (actual rows x predicted columns): TN={test_cm[0][0]}, "
        f"FP={test_cm[0][1]}, FN={test_cm[1][0]}, TP={test_cm[1][1]}. "
        "At the chosen threshold, the model correctly classified 23 target-0 and 27 target-1 records.",
        "SmallProject",
    ))
    story.append(chart("confusion_matrix.png", width * 0.55, 53 * mm))
    story.append(PageBreak())

    # Interpretation and conclusion
    story.append(para("Feature interpretation", "Section"))
    top_imp = importance.head(5)
    top_list = ", ".join(
        f"{row['Feature']} ({row['Mean ROC-AUC decrease']:.3f})"
        for _, row in top_imp.iterrows()
    )
    story.append(para(
        "Permutation importance was calculated by shuffling one feature at a time on the held-out "
        f"test data and measuring the change in ROC-AUC. The largest mean decreases were: {top_list}. "
        "These values describe predictive contribution for this model and split; they do not mean "
        "that a feature causes disease. Near-zero or negative scores can occur on a small test set, "
        "especially when predictors contain overlapping information.",
    ))
    story.append(chart("feature_importance.png", width * 0.92, 72 * mm))
    story.append(para("Conclusion", "Subsection"))
    story.append(para(
        "The project completed the requested data review, exploratory analysis, model tuning, "
        "evaluation, and feature-importance steps. Logistic Regression had the strongest mean "
        "cross-validation score and reached 0.893 test ROC-AUC. Random Forest reached 0.904 test "
        "ROC-AUC, with a slightly lower cross-validation score. Repeated or nested cross-validation "
        "and validation on independent data would be needed for more stable estimates. The package "
        "also includes a Streamlit interface that loads the selected model and demonstrates a "
        "class-1 probability output.",
    ))
    story.append(para("Limitations", "Subsection"))
    limitations = [
        "The analysis uses 302 unique records from one supplied dataset; the test set has 61 records.",
        "A single holdout can produce variable estimates, and the exact source provenance and code mappings are not confirmed.",
        "Performance on these records does not establish generalization to other populations or clinical settings.",
        "Calibration, clinical utility, fairness, and the consequences of errors were not assessed.",
        "The model is an educational demonstration, not a diagnostic or treatment tool.",
    ]
    for item in limitations:
        story.append(para("- " + item, "SmallProject"))
    story.append(PageBreak())

    # References
    story.append(para("References", "Section"))
    references = [
        "1. User-supplied project brief, <i>Heart Disease Prediction Using Machine Learning Classification Models</i>.",
        "2. UCI Machine Learning Repository. (n.d.). <i>Heart Disease</i>. "
        "<link href='https://archive.ics.uci.edu/dataset/45/heart+disease' color='#2B6CB0'>"
        "https://archive.ics.uci.edu/dataset/45/heart+disease</link>. The supplied workbook's exact provenance is not confirmed.",
        "3. Scikit-learn developers. (n.d.). <i>GridSearchCV</i>. "
        "<link href='https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html' color='#2B6CB0'>"
        "scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html</link>.",
        "4. Scikit-learn developers. (n.d.). <i>MLPClassifier</i>. "
        "<link href='https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html' color='#2B6CB0'>"
        "scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html</link>.",
    ]
    for ref in references:
        story.append(para(ref, "BodyProject"))
    story.append(Spacer(1, 7 * mm))
    callout = Table([[para(
        "Educational scope: The model's probability output describes how this fitted classifier "
        "responds to values similar to the supplied dataset. It is not an individual's clinical "
        "risk estimate and must not be used for diagnosis, screening, or treatment.", "CellProject"
    )]], colWidths=[width])
    callout.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PALE), ("BOX", (0, 0), (-1, -1), 0.8, TEAL),
        ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 10), ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))
    story.append(callout)

    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    print(f"Created {OUT}")


if __name__ == "__main__":
    main()
