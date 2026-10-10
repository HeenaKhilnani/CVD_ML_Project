"""
Stage 3
=======
Train the three models on the same training set and save them to disk.

Models
------
1. XGBoost                  - gradient-boosted decision trees (tree ensemble)
2. LVQ (GLVQ)               - genuine Learning Vector Quantization from the
                              ``sklvq`` library (Generalized LVQ, Sato & Yamada)
3. Gaussian Naive Bayes     - probabilistic baseline

Each model is a scikit-learn ``Pipeline`` containing its own imputation /
scaling steps, so the saved artifact can take a raw 13-feature patient row and
produce a prediction. Because every pipeline is fitted on the training split
only, there is no data leakage.

Run from the project root::

    .venv/bin/python -m src.train_models
"""

from __future__ import annotations

import json
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# The compatibility import must happen before sklvq is imported.
from src import compat  # noqa: F401
import sklvq  # noqa: E402
from sklvq import GLVQ  # noqa: E402

from src.config import (
    FEATURE_NAMES,
    MODEL_FILES,
    MODEL_NAMES,
    MODELS_DIR,
    RANDOM_STATE,
    RESULTS_DIR,
    ensure_dirs,
)
from src.data_preparation import prepare_data


def build_pipelines() -> dict:
    """
    Build one fresh preprocessing+model pipeline per algorithm.

    * XGBoost: median imputation only. Trees do not need feature scaling
      (XGBoost can even handle missing values natively).
    * Gaussian Naive Bayes: median imputation only. GaussianNB estimates a
      mean and variance per feature, so scaling is not required.
    * LVQ (GLVQ): median imputation + standardisation. Learning Vector
      Quantization is distance-based, so scaling is essential.
    """
    return {
        "XGBoost": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    xgboost.XGBClassifier(
                        n_estimators=300,
                        max_depth=3,
                        learning_rate=0.05,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        reg_lambda=1.0,
                        eval_metric="logloss",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "LVQ (GLVQ)": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    GLVQ(
                        prototype_init="class-conditional-mean",
                        prototype_n_per_class=2,
                        solver_type="adam",
                        solver_params={"max_runs": 300, "step_size": 0.01},
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Gaussian Naive Bayes": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("model", GaussianNB()),
            ]
        ),
    }


def main() -> None:
    ensure_dirs()
    X_train, X_test, y_train, y_test, _ = prepare_data()

    print("=" * 66)
    print("STAGE 3 - TRAINING MODELS")
    print("=" * 66)
    print(f"Training rows: {X_train.shape[0]} | features: {X_train.shape[1]}")
    print(f"Test rows (held out, untouched during training): {X_test.shape[0]}\n")

    pipelines = build_pipelines()
    train_accuracies = {}

    for name in MODEL_NAMES:
        pipeline = pipelines[name]
        print(f"Training {name} ...", end=" ", flush=True)
        pipeline.fit(X_train, y_train)

        train_pred = pipeline.predict(X_train)
        train_acc = float((train_pred == y_train).mean())
        train_accuracies[name] = train_acc

        model_file = MODELS_DIR / MODEL_FILES[name]
        joblib.dump(pipeline, model_file)
        print(f"done. Train accuracy: {train_acc:.4f} -> saved {model_file.name}")

    metadata = {
        "feature_names": FEATURE_NAMES,
        "random_state": RANDOM_STATE,
        "test_size": len(y_test) / (len(y_train) + len(y_test)),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "train_accuracies": train_accuracies,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "library_versions": {
            "python": "3.12",
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "sklvq": sklvq.__version__,
        },
        "model_files": MODEL_FILES,
        "notes": (
            "Each .joblib file is a full scikit-learn Pipeline (imputer [+ scaler] "
            "+ model). Pass a raw 13-column dataframe in config.FEATURE_NAMES order."
        ),
    }
    metadata_file = MODELS_DIR / "metadata.json"
    metadata_file.write_text(json.dumps(metadata, indent=2))

    print(f"\nSaved metadata -> {metadata_file}")
    print("\nSaved model files:")
    for name in MODEL_NAMES:
        print(f"  - {MODELS_DIR / MODEL_FILES[name]}")

    # Guarantee the results folder exists for the evaluation stage.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    main()
