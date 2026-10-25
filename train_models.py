"""Train and compare heart disease classification models.

Run from the project folder with: python train_models.py
The source dataset is data/heart.csv. Results and figures are written to artifacts/.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


SEED = 42
TARGET = "target"
FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]
NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
EXPECTED_COLUMNS = FEATURES + [TARGET]


def make_pipeline(estimator: object) -> Pipeline:
    numeric_pipe = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    prep = ColumnTransformer(
        [("numeric", numeric_pipe, NUMERIC), ("categorical", categorical_pipe, CATEGORICAL)],
        remainder="drop",
    )
    return Pipeline([("preprocess", prep), ("model", estimator)])


def get_searches() -> dict[str, tuple[Pipeline, dict[str, list[object]]]]:
    return {
        "Logistic Regression": (
            make_pipeline(LogisticRegression(max_iter=3000, random_state=SEED)),
            {"model__C": [0.1, 1.0, 10.0], "model__class_weight": [None, "balanced"]},
        ),
        "Decision Tree": (
            make_pipeline(DecisionTreeClassifier(random_state=SEED)),
            {
                "model__max_depth": [3, None],
                "model__min_samples_leaf": [2, 5],
                "model__class_weight": [None, "balanced"],
            },
        ),
        "Random Forest": (
            make_pipeline(RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=1)),
            {
                "model__max_depth": [None, 6],
                "model__min_samples_leaf": [1, 3],
                "model__class_weight": [None, "balanced"],
                "model__max_features": ["sqrt"],
            },
        ),
        "Feedforward Neural Network": (
            make_pipeline(
                MLPClassifier(
                    solver="lbfgs", max_iter=2500, random_state=SEED,
                    early_stopping=False,
                )
            ),
            {
                "model__hidden_layer_sizes": [(16,), (32,), (32, 16)],
                "model__alpha": [0.0001, 0.01],
            },
        ),
    }


def save_eda_figures(data: pd.DataFrame, out_dir: Path) -> None:
    sns.set_theme(style="whitegrid", palette="deep")

    counts = data[TARGET].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bars = ax.bar(["No disease (0)", "Disease (1)"], [counts.get(0, 0), counts.get(1, 0)],
                  color=["#718096", "#2B6CB0"], width=0.58)
    ax.set_title("Target distribution", loc="left", fontsize=15, weight="bold")
    ax.set_ylabel("Records")
    ax.set_ylim(0, max(counts.values) * 1.18)
    for bar, value in zip(bars, [counts.get(0, 0), counts.get(1, 0)]):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 2, f"{value} ({value / len(data):.1%})",
                ha="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "target_distribution.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.boxplot(data=data, x=TARGET, y="age", hue=TARGET, legend=False,
                palette={0: "#718096", 1: "#2B6CB0"}, ax=ax)
    ax.set_title("Age by target class", loc="left", fontsize=15, weight="bold")
    ax.set_xlabel("Target (0 = no disease, 1 = disease)")
    ax.set_ylabel("Age (years)")
    fig.tight_layout()
    fig.savefig(out_dir / "age_by_target.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    corr_cols = NUMERIC + [TARGET]
    corr = data[corr_cols].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, square=True,
                linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Correlation among numeric measurements", loc="left", fontsize=14, weight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "numeric_correlation.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_comparison_chart(metrics: pd.DataFrame, out_dir: Path) -> None:
    metric_cols = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    chart = metrics.set_index("Model")[metric_cols]
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#2B6CB0", "#63B3ED", "#38A169", "#D69E2E", "#805AD5"]
    centers = np.arange(len(chart.index))
    bar_height = 0.14
    offsets = (np.arange(len(metric_cols)) - (len(metric_cols) - 1) / 2) * bar_height
    for metric, color, offset in zip(metric_cols, colors, offsets):
        ax.barh(centers + offset, chart[metric].values, height=bar_height,
                label=metric, color=color)
    ax.set_yticks(centers, chart.index)
    ax.invert_yaxis()
    ax.set_title("Holdout performance by model", loc="left", fontsize=15, weight="bold")
    ax.set_xlabel("Score")
    ax.set_xlim(0, 1.05)
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.23), frameon=False)
    ax.grid(axis="x", alpha=0.35)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(out_dir / "model_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/heart.csv"), help="Path to the cleaned CSV dataset")
    parser.add_argument("--out", type=Path, default=Path("artifacts"), help="Output directory")
    args = parser.parse_args()
    data_path = args.data.resolve()
    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(data_path)
    missing_cols = sorted(set(EXPECTED_COLUMNS) - set(data.columns))
    extra_cols = sorted(set(data.columns) - set(EXPECTED_COLUMNS))
    if missing_cols or extra_cols:
        raise ValueError(f"Unexpected columns. Missing: {missing_cols}; extra: {extra_cols}")
    data = data[EXPECTED_COLUMNS].copy()
    if not set(data[TARGET].dropna().unique()).issubset({0, 1}):
        raise ValueError("The target column must contain binary values 0 and 1.")

    source_rows = len(data)
    missing_cells = int(data.isna().sum().sum())
    duplicate_rows = int(data.duplicated().sum())
    data = data.drop_duplicates().reset_index(drop=True)
    X = data[FEATURES]
    y = data[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    results = []
    best_estimators: dict[str, Pipeline] = {}

    for name, (pipeline, param_grid) in get_searches().items():
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring="roc_auc",
            cv=cv,
            n_jobs=1,
            refit=True,
            return_train_score=False,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            search.fit(X_train, y_train)
        fitted = search.best_estimator_
        probabilities = fitted.predict_proba(X_test)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        metrics = {
            "Model": name,
            "CV ROC-AUC": float(search.best_score_),
            "Accuracy": float(accuracy_score(y_test, predictions)),
            "Precision": float(precision_score(y_test, predictions, zero_division=0)),
            "Recall": float(recall_score(y_test, predictions, zero_division=0)),
            "F1": float(f1_score(y_test, predictions, zero_division=0)),
            "ROC-AUC": float(auc(*roc_curve(y_test, probabilities)[:2])),
            "Best parameters": json.dumps(search.best_params_, sort_keys=True),
        }
        results.append(metrics)
        best_estimators[name] = fitted

    metrics_df = pd.DataFrame(results).sort_values("CV ROC-AUC", ascending=False).reset_index(drop=True)
    metrics_df.to_csv(out_dir / "model_comparison.csv", index=False)
    selected_name = str(metrics_df.iloc[0]["Model"])
    selected_model = best_estimators[selected_name]
    selected_probs = selected_model.predict_proba(X_test)[:, 1]
    selected_preds = (selected_probs >= 0.5).astype(int)
    cm = confusion_matrix(y_test, selected_preds, labels=[0, 1])

    # Reproducible project metadata for the report and app.
    dataset_summary = {
        "source_file": str(data_path.name),
        "rows_in_source": source_rows,
        "exact_duplicate_rows_removed": duplicate_rows,
        "rows_used": len(data),
        "columns": len(EXPECTED_COLUMNS),
        "missing_cells_before_imputation": missing_cells,
        "target_counts_after_deduplication": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "seed": SEED,
        "cv_folds": 5,
        "selected_model_by_cv_roc_auc": selected_name,
        "selected_model_cv_roc_auc": float(metrics_df.iloc[0]["CV ROC-AUC"]),
        "selected_model_test_metrics": metrics_df.iloc[0][["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]].to_dict(),
        "selected_model_confusion_matrix": cm.tolist(),
        "python_version": __import__("sys").version.split()[0],
        "scikit_learn_version": sklearn.__version__,
    }
    (out_dir / "project_summary.json").write_text(json.dumps(dataset_summary, indent=2), encoding="utf-8")

    save_eda_figures(data, out_dir)
    save_comparison_chart(metrics_df, out_dir)

    # ROC plot for all tuned pipelines.
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, fitted in best_estimators.items():
        probs = fitted.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, probs)
        area = auc(fpr, tpr)
        ax.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC {area:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#718096", linewidth=1)
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="False positive rate", ylabel="True positive rate")
    ax.set_title("ROC curves on the held-out test set", loc="left", fontsize=14, weight="bold")
    ax.legend(loc="lower right", fontsize=8, frameon=True)
    fig.tight_layout()
    fig.savefig(out_dir / "roc_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Confusion matrix for the model selected by training cross-validation.
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, square=True,
                xticklabels=["No disease (0)", "Disease (1)"],
                yticklabels=["No disease (0)", "Disease (1)"], ax=ax)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    ax.set_title(f"Confusion matrix: {selected_name}", loc="left", fontsize=13, weight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    perm = permutation_importance(
        selected_model, X_test, y_test, scoring="roc_auc", n_repeats=30,
        random_state=SEED, n_jobs=1,
    )
    importance = pd.DataFrame(
        {"Feature": FEATURES, "Mean ROC-AUC decrease": perm.importances_mean,
         "Standard deviation": perm.importances_std}
    ).sort_values("Mean ROC-AUC decrease", ascending=False)
    importance.to_csv(out_dir / "feature_importance.csv", index=False)
    top = importance.head(10).sort_values("Mean ROC-AUC decrease")
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(top["Feature"], top["Mean ROC-AUC decrease"], xerr=top["Standard deviation"],
            color="#2B6CB0", alpha=0.9, capsize=2)
    ax.axvline(0, color="#4A5568", linewidth=0.8)
    ax.set_title(f"Permutation importance: {selected_name}", loc="left", fontsize=14, weight="bold")
    ax.set_xlabel("Decrease in test ROC-AUC after shuffling feature")
    fig.tight_layout()
    fig.savefig(out_dir / "feature_importance.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    bundle = {
        "model": selected_model,
        "model_name": selected_name,
        "features": FEATURES,
        "categorical_features": CATEGORICAL,
        "numeric_features": NUMERIC,
        "metrics": dataset_summary["selected_model_test_metrics"],
        "dataset_summary": dataset_summary,
        "training_data_min": {c: float(data[c].min()) for c in NUMERIC},
        "training_data_max": {c: float(data[c].max()) for c in NUMERIC},
        "category_values": {c: sorted(int(v) for v in data[c].dropna().unique()) for c in CATEGORICAL},
    }
    joblib.dump(bundle, out_dir / "best_model.joblib")

    print(f"Rows: {source_rows} supplied; {duplicate_rows} exact duplicate(s) removed; {len(data)} used")
    print(f"Train/test: {len(X_train)}/{len(X_test)}; selected by CV ROC-AUC: {selected_name}")
    print(metrics_df[["Model", "CV ROC-AUC", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"Artifacts saved to: {out_dir}")


if __name__ == "__main__":
    main()
