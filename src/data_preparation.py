"""
Stage 1 & Stage 2
=================
Inspect the UCI Cleveland Heart Disease dataset and prepare it for modelling.

What this module does
---------------------
Stage 1 (inspection)
    * Loads ``data/processed.cleveland.data``.
    * Assigns the 13 standard feature names + the ``num`` target.
    * Treats the original ``?`` missing markers as ``NaN``.
    * Reports row/column counts, missing values and the class distribution.

Stage 2 (preparation)
    * Converts the 5-level target into a binary label (0 stays 0, 1-4 become 1).
    * Splits into 80% train / 20% test using stratification and a fixed seed.

Preprocessing (imputing + scaling) is deliberately NOT done here. It is wrapped
inside each model's scikit-learn ``Pipeline`` so it is fitted on the training
data only - that is what prevents data leakage.
"""

from __future__ import annotations

import json
from typing import Dict, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    DATA_FILE,
    FEATURE_NAMES,
    RANDOM_STATE,
    RESULTS_DIR,
    TARGET_NAME,
    TEST_SIZE,
    ensure_dirs,
)


def load_dataset(path=DATA_FILE) -> pd.DataFrame:
    """Load the Cleveland data with correct columns and ``?`` treated as NaN."""
    df = pd.read_csv(path, header=None, na_values="?")
    df.columns = FEATURE_NAMES + [TARGET_NAME]
    return df


def binarize_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the original 0-4 target into a binary label.

    The original Cleveland target ``num`` is the diameter narrowing of the
    major vessels, scaled 0-4. Following the standard convention used for this
    dataset, we treat 0 as "no disease" and any value 1-4 as "disease present".
    """
    df = df.copy()
    df["target"] = (df[TARGET_NAME] > 0).astype(int)
    return df


def dataset_report(df: pd.DataFrame) -> Dict:
    """Return a JSON-serialisable summary of the raw (un-binarised) dataset."""
    features = df[FEATURE_NAMES]
    return {
        "source_file": str(DATA_FILE),
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "n_features": len(FEATURE_NAMES),
        "feature_names": FEATURE_NAMES,
        "target_name": TARGET_NAME,
        "missing_values": {
            "total_missing_cells": int(features.isna().sum().sum()),
            "per_column": {c: int(features[c].isna().sum()) for c in FEATURE_NAMES},
        },
        "original_target_counts": {
            str(k): int(v) for k, v in df[TARGET_NAME].value_counts().sort_index().items()
        },
    }


def prepare_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    """
    Load, binarise and split the dataset.

    Returns
    -------
    X_train, X_test, y_train, y_test, raw_df
        ``raw_df`` is the fully loaded dataframe (useful for reporting).
    """
    raw_df = load_dataset()
    df = binarize_target(raw_df)

    X = df[FEATURE_NAMES]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test, raw_df


def class_distribution(df: pd.DataFrame) -> Dict:
    """Binary class counts after binarising the target."""
    binary = binarize_target(df)
    counts = binary["target"].value_counts().sort_index()
    total = int(counts.sum())
    return {
        "counts": {str(k): int(v) for k, v in counts.items()},
        "percentages": {str(k): round(100.0 * v / total, 2) for k, v in counts.items()},
    }


def main() -> None:
    """Print the Stage 1 report and save it for the app / documentation."""
    ensure_dirs()
    raw_df = load_dataset()

    print("=" * 66)
    print("STAGE 1 - DATASET INSPECTION")
    print("=" * 66)
    print(f"File            : {DATA_FILE}")
    print(f"Shape           : {raw_df.shape[0]} rows x {raw_df.shape[1]} columns")
    print(f"Features (13)   : {FEATURE_NAMES}")
    print(f"Target          : {TARGET_NAME}")

    report = dataset_report(raw_df)

    print("\nMissing values ('?' -> NaN):")
    print(f"  Total missing cells: {report['missing_values']['total_missing_cells']}")
    for col, n in report["missing_values"]["per_column"].items():
        if n:
            print(f"    - {col}: {n}")

    print("\nOriginal target value counts (0-4):")
    for k, v in report["original_target_counts"].items():
        print(f"    - num={k}: {v}")

    dist = class_distribution(raw_df)
    print("\nBinarised class distribution (0 = no disease, 1 = disease):")
    for k in ("0", "1"):
        print(f"    - class {k}: {dist['counts'][k]} ({dist['percentages'][k]}%)")

    report["binary_class_distribution"] = dist
    report["random_state"] = RANDOM_STATE
    report["test_size"] = TEST_SIZE

    out = RESULTS_DIR / "dataset_summary.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nSaved dataset summary -> {out}")

    # Stage 2 quick verification
    X_train, X_test, y_train, y_test, _ = prepare_data()
    print("\n" + "=" * 66)
    print("STAGE 2 - TRAIN / TEST SPLIT")
    print("=" * 66)
    print(f"Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")
    print(f"Train class balance: {y_train.value_counts().sort_index().to_dict()}")
    print(f"Test  class balance: {y_test.value_counts().sort_index().to_dict()}")


if __name__ == "__main__":
    main()
