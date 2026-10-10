"""
Stage 6
=======
SHAP explanations for the XGBoost model.

SHAP (SHapley Additive exPlanations) answers "how much did each feature push
this prediction?" using cooperative-game-theory values. We generate:

* ``results/shap_summary_bar.png``      - global mean |SHAP| importance
* ``results/shap_summary_beeswarm.png`` - global impact + direction per feature
* ``results/shap_local_example.png``    - explanation of one single test patient
* ``results/shap_top_features.json``    - machine-readable global ranking

The local example is the test patient with the highest predicted disease
probability (chosen programmatically, not hand-picked).

Run from the project root::

    .venv/bin/python -m src.shap_explain
"""

from __future__ import annotations

import json

# Compatibility shim for sklvq, imported so unpickling the pipelines is safe.
from src import compat  # noqa: F401

import joblib
import matplotlib
import numpy as np
import pandas as pd
import shap

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from src.config import FEATURE_NAMES, MODELS_DIR, RESULTS_DIR, ensure_dirs  # noqa: E402
from src.data_preparation import prepare_data  # noqa: E402


def main() -> None:
    ensure_dirs()

    print("=" * 66)
    print("STAGE 6 - SHAP EXPLANATIONS (XGBoost)")
    print("=" * 66)

    X_train, X_test, y_train, y_test, _ = prepare_data()
    pipeline = joblib.load(MODELS_DIR / "xgb_pipeline.joblib")

    # Apply the pipeline's preprocessing (imputer) manually so SHAP sees the
    # same feature matrix the XGBoost model was trained on.
    preprocessor = pipeline[:-1]
    booster = pipeline.named_steps["model"]
    X_test_prep = preprocessor.transform(X_test)

    explainer = shap.TreeExplainer(booster)
    explanation = explainer(X_test_prep)
    explanation.feature_names = FEATURE_NAMES

    shap_values = np.array(explanation.values)
    print(f"SHAP values computed: {shap_values.shape} (test patients x features)")

    # ---------------- global ranking ----------------
    mean_abs = np.abs(shap_values).mean(axis=0)
    ranking = (
        pd.Series(mean_abs, index=FEATURE_NAMES)
        .sort_values(ascending=False)
        .round(5)
    )
    print("\nGlobal feature importance (mean |SHAP value|):")
    for feat, val in ranking.items():
        print(f"  {feat:>10}: {val:.4f}")

    (RESULTS_DIR / "shap_top_features.json").write_text(
        json.dumps(ranking.to_dict(), indent=2)
    )

    # ---------------- global bar plot ----------------
    plt.figure()
    shap.plots.bar(explanation, show=False)
    plt.title("SHAP global importance - XGBoost")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "shap_summary_bar.png", dpi=150, bbox_inches="tight")
    plt.close("all")

    # ---------------- beeswarm (direction of effect) ----------------
    plt.figure()
    shap.plots.beeswarm(explanation, show=False)
    plt.title("SHAP beeswarm - XGBoost")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "shap_summary_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close("all")

    # ---------------- local explanation ----------------
    probas = pipeline.predict_proba(X_test)[:, 1]
    idx = int(np.argmax(probas))  # most confident positive case
    print(
        f"\nLocal example: test patient index {idx} "
        f"(predicted probability = {probas[idx]:.3f}, true label = {y_test.iloc[idx]})"
    )

    plt.figure()
    shap.plots.waterfall(explanation[idx], show=False)
    plt.title(f"SHAP local explanation - test patient {idx} (p(disease)={probas[idx]:.2f})")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "shap_local_example.png", dpi=150, bbox_inches="tight")
    plt.close("all")

    print("\nSaved to results/:")
    for fname in [
        "shap_top_features.json",
        "shap_summary_bar.png",
        "shap_summary_beeswarm.png",
        "shap_local_example.png",
    ]:
        print(f"  - {RESULTS_DIR / fname}")


if __name__ == "__main__":
    main()
