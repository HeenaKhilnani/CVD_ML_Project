"""
Stage 4
=======
Evaluate the three trained models on the held-out test set.

Everything here is computed from real model outputs - no values are
hard-coded. Outputs written to ``results/``:

* ``metrics.csv`` / ``metrics.json``   - accuracy, precision, recall, F1, ROC-AUC
* ``predictions.csv``                  - per-patient true label, prediction and probability
* ``confusion_matrices.png``           - heatmap for each model
* ``metric_comparison.png``            - grouped bar chart of the metrics
* ``roc_curves.png``                   - ROC curves (all models)
* ``xgb_feature_importance.png``       - XGBoost feature importances

Run from the project root::

    .venv/bin/python -m src.evaluate_models
"""

from __future__ import annotations

import json

# Compatibility shim for sklvq (see src/compat.py) - must be imported before
# any LVQ model is used or unpickled.
from src import compat  # noqa: F401

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # save figures without needing a display

import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config import (
    FEATURE_NAMES,
    MODEL_FILES,
    MODEL_NAMES,
    MODELS_DIR,
    RESULTS_DIR,
    ensure_dirs,
)
from src.data_preparation import prepare_data

sns.set_theme(style="whitegrid", context="talk")
COLORS = {"XGBoost": "#2a6f97", "LVQ (GLVQ)": "#e07a5f", "Gaussian Naive Bayes": "#3d9970"}


def load_pipelines() -> dict:
    """Load the saved pipelines produced by ``src.train_models``."""
    return {name: joblib.load(MODELS_DIR / MODEL_FILES[name]) for name in MODEL_NAMES}


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
    cm = confusion_matrix(y_true, y_pred)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "confusion_matrix": cm.tolist(),
    }


def plot_confusion_matrices(y_true, results: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    for ax, name in zip(axes, MODEL_NAMES):
        cm = np.array(results[name]["confusion_matrix"])
        ConfusionMatrixDisplay(cm, display_labels=["No disease", "Disease"]).plot(
            ax=ax, colorbar=False, cmap="Blues"
        )
        ax.set_title(name, fontsize=13)
        ax.set_xlabel("Predicted label", fontsize=11)
        ax.set_ylabel("True label", fontsize=11)
    fig.suptitle(f"Confusion matrices on the held-out test set (n={len(y_true)})", fontsize=15)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "confusion_matrices.png", dpi=150)
    plt.close(fig)


def plot_metric_comparison(results: dict) -> None:
    metric_names = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    rows = []
    for name in MODEL_NAMES:
        for metric in metric_names:
            rows.append({"Model": name, "Metric": metric.upper(), "Score": results[name][metric]})
    long_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=long_df, x="Metric", y="Score", hue="Model", palette=COLORS, ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_title("Model performance comparison (held-out test set)", fontsize=14)
    ax.set_ylabel("Score")
    ax.legend(title="", loc="lower right")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", fontsize=8, padding=2)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "metric_comparison.png", dpi=150)
    plt.close(fig)


def plot_roc_curves(y_true, probas: dict) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    for name in MODEL_NAMES:
        fpr, tpr, _ = roc_curve(y_true, probas[name])
        auc = roc_auc_score(y_true, probas[name])
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})", color=COLORS[name], linewidth=2.2)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random classifier")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves - held-out test set")
    ax.legend(loc="lower right", fontsize=11)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "roc_curves.png", dpi=150)
    plt.close(fig)


def plot_xgb_importance(xgb_pipeline) -> None:
    booster = xgb_pipeline.named_steps["model"]
    importances = pd.Series(booster.feature_importances_, index=FEATURE_NAMES)
    importances = importances.sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(9, 7))
    importances.plot.barh(ax=ax, color="#2a6f97")
    ax.set_title("XGBoost feature importance (gain-based)", fontsize=14)
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "xgb_feature_importance.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    X_train, X_test, y_train, y_test, _ = prepare_data()
    pipelines = load_pipelines()

    print("=" * 66)
    print("STAGE 4 - EVALUATION ON THE HELD-OUT TEST SET")
    print("=" * 66)
    print(f"Test samples: {len(y_test)} (never used for training)\n")

    results = {}
    probas = {}
    predictions = pd.DataFrame({"y_true": y_test.values})

    for name in MODEL_NAMES:
        pipeline = pipelines[name]
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        results[name] = compute_metrics(y_test, y_pred, y_proba)
        probas[name] = y_proba
        predictions[f"{name}_pred"] = y_pred
        predictions[f"{name}_proba"] = y_proba

    # ---------------- metrics table ----------------
    table = pd.DataFrame(
        {
            name: {
                "Accuracy": results[name]["accuracy"],
                "Precision": results[name]["precision"],
                "Recall": results[name]["recall"],
                "F1-score": results[name]["f1"],
                "ROC-AUC": results[name]["roc_auc"],
            }
            for name in MODEL_NAMES
        }
    ).T

    print(table.round(4).to_string())
    print("\nConfusion matrices (rows = true, columns = predicted):")
    for name in MODEL_NAMES:
        print(f"  {name}: {results[name]['confusion_matrix']}")

    table.round(6).to_csv(RESULTS_DIR / "metrics.csv")
    predictions.round(6).to_csv(RESULTS_DIR / "predictions.csv", index=True)

    best_by = {
        "accuracy": max(MODEL_NAMES, key=lambda m: results[m]["accuracy"]),
        "f1": max(MODEL_NAMES, key=lambda m: results[m]["f1"]),
        "roc_auc": max(MODEL_NAMES, key=lambda m: results[m]["roc_auc"]),
    }
    metrics_json = {
        "test_set_size": int(len(y_test)),
        "test_class_balance": {str(k): int(v) for k, v in y_test.value_counts().sort_index().items()},
        "models": results,
        "best_by": best_by,
        "note": "All values computed from actual model predictions on the held-out test set.",
    }
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics_json, indent=2))

    # ---------------- plots ----------------
    plot_confusion_matrices(y_test, results)
    plot_metric_comparison(results)
    plot_roc_curves(y_test, probas)
    plot_xgb_importance(pipelines["XGBoost"])

    print("\nSaved to results/:")
    for fname in [
        "metrics.csv",
        "metrics.json",
        "predictions.csv",
        "confusion_matrices.png",
        "metric_comparison.png",
        "roc_curves.png",
        "xgb_feature_importance.png",
    ]:
        print(f"  - {RESULTS_DIR / fname}")

    print("\nBest model by F1-score      :", best_by["f1"])
    print("Best model by ROC-AUC       :", best_by["roc_auc"])
    print("Best model by accuracy      :", best_by["accuracy"])


if __name__ == "__main__":
    main()
