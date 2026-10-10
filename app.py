"""
Stage 5
=======
Interactive Streamlit application for the CVD Machine Learning project.

Run from the project root::

    .venv/bin/streamlit run app.py

The app uses the already-trained pipelines in ``models/`` and the real
evaluation numbers in ``results/``. It never trains anything on the fly.
"""

from __future__ import annotations

import json

# Compatibility shim for sklvq - must be imported before any LVQ model loads.
from src import compat  # noqa: F401

import joblib
import pandas as pd
import streamlit as st

from src.config import (
    FEATURE_LABELS,
    FEATURE_NAMES,
    MODEL_FILES,
    MODEL_NAMES,
    MODELS_DIR,
    RESULTS_DIR,
)

st.set_page_config(
    page_title="CVD Prediction using Machine Learning",
    page_icon="🫀",
    layout="wide",
)

DISCLAIMER = (
    "**Educational use only.** This application is a college PBL (project-based "
    "learning) demo trained on 303 historical patient records from the UCI "
    "Cleveland Heart Disease dataset. It is **not** a clinically validated "
    "diagnostic tool and must never be used for real medical decisions. "
    "Always consult a qualified healthcare professional."
)


# ---------------------------------------------------------------------------
# Cached loaders (re-load only when the app restarts)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_pipelines():
    return {name: joblib.load(MODELS_DIR / MODEL_FILES[name]) for name in MODEL_NAMES}


@st.cache_data(show_spinner=False)
def load_json(path):
    return json.loads(path.read_text())


def read_text(path):
    return path.read_text() if path.exists() else None


def show_image(path, caption=None):
    if path.exists():
        st.image(str(path), caption=caption, width="stretch")
    else:
        st.info(f"Chart not found yet: run the pipeline scripts to generate `{path.name}`.")


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🫀 Prediction of Cardiovascular Disease Using Machine Learning")
st.caption(
    "College PBL project - XGBoost, Learning Vector Quantization (GLVQ) and "
    "Gaussian Naive Bayes on the UCI Cleveland Heart Disease dataset."
)
st.warning(DISCLAIMER, icon="⚠️")

# ---------------------------------------------------------------------------
# Overview + methodology
# ---------------------------------------------------------------------------
with st.expander("📖 Project overview & methodology", expanded=True):
    st.markdown(
        """
**Goal.** Compare three machine-learning algorithms at predicting the presence
of heart disease from 13 routine clinical measurements.

**Dataset.** UCI Cleveland Heart Disease (303 patients, 13 features). The
original 0-4 severity target is converted to a binary label:
`0 = no disease`, `1-4 = disease present`.

**Methodology.**
1. `?` missing values are handled inside each model pipeline (median imputation).
2. Data is split 80/20 with **stratification** and a fixed random seed (42), so
   every model sees the same 242 training and 61 test patients.
3. Imputers/scalers are fitted on **training data only** (no data leakage).
4. LVQ uses standardised features because it is distance-based.
5. All reported metrics come from the untouched held-out test set.

**Models.**
- **XGBoost** - gradient-boosted decision trees.
- **LVQ (GLVQ)** - genuine Generalized Learning Vector Quantization from the
  `sklvq` library: it learns one or more "prototype" vector per class and
  classifies new patients by nearest-prototype distance.
- **Gaussian Naive Bayes** - probabilistic baseline assuming feature
  independence.
        """
    )

# ---------------------------------------------------------------------------
# Dataset summary
# ---------------------------------------------------------------------------
summary_file = RESULTS_DIR / "dataset_summary.json"
if summary_file.exists():
    summary = load_json(summary_file)
    with st.expander("📊 Dataset summary", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Patients", summary["n_rows"])
        c2.metric("Features", summary["n_features"])
        c3.metric("Missing cells", summary["missing_values"]["total_missing_cells"])
        dist = summary["binary_class_distribution"]["counts"]
        c4.metric("Disease / No disease", f"{dist['1']} / {dist['0']}")

        st.markdown(
            "Original 5-level target counts: "
            + ", ".join(
                f"`num={k}`: {v}" for k, v in summary["original_target_counts"].items()
            )
        )
        st.markdown(
            "Missing values occur only in `ca` (4) and `thal` (2); both are "
            "imputed with the training-set median inside each pipeline."
        )
else:
    st.info("Run `python -m src.data_preparation` to generate the dataset summary.")

# ---------------------------------------------------------------------------
# Prediction section
# ---------------------------------------------------------------------------
st.header("🩺 Patient prediction")
left, right = st.columns([1.6, 1])

with left:
    st.subheader("Model")
    selected_model = st.radio(
        "Choose a trained model",
        MODEL_NAMES,
        horizontal=True,
        help="All three models were trained on the same 242 patients.",
    )

    st.subheader("Patient attributes (13 features)")
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.slider("Age (years)", 29, 77, 54, help="Dataset range: 29-77")
        sex = st.selectbox("Sex", options=[1, 0], format_func=lambda v: "Male (1)" if v else "Female (0)")
        cp = st.selectbox(
            "Chest pain type",
            options=[1, 2, 3, 4],
            format_func=lambda v: {
                1: "1 - Typical angina",
                2: "2 - Atypical angina",
                3: "3 - Non-anginal pain",
                4: "4 - Asymptomatic",
            }[v],
        )
        trestbps = st.slider("Resting blood pressure (mm Hg)", 94, 200, 130)
        chol = st.slider("Serum cholesterol (mg/dl)", 126, 564, 246)
    with c2:
        fbs = st.selectbox("Fasting blood sugar > 120 mg/dl", options=[0, 1], format_func=lambda v: "Yes (1)" if v else "No (0)")
        restecg = st.selectbox(
            "Resting ECG",
            options=[0, 1, 2],
            format_func=lambda v: {
                0: "0 - Normal",
                1: "1 - ST-T abnormality",
                2: "2 - LV hypertrophy",
            }[v],
        )
        thalach = st.slider("Max heart rate achieved", 71, 202, 150)
        exang = st.selectbox("Exercise induced angina", options=[0, 1], format_func=lambda v: "Yes (1)" if v else "No (0)")
    with c3:
        oldpeak = st.slider("ST depression (oldpeak)", 0.0, 6.2, 1.0, step=0.1)
        slope = st.selectbox(
            "Slope of peak ST segment",
            options=[1, 2, 3],
            format_func=lambda v: {1: "1 - Upsloping", 2: "2 - Flat", 3: "3 - Downsloping"}[v],
        )
        ca = st.selectbox("Major vessels coloured by fluoroscopy", options=[0, 1, 2, 3])
        thal = st.selectbox(
            "Thalassemia",
            options=[3, 6, 7],
            format_func=lambda v: {3: "3 - Normal", 6: "6 - Fixed defect", 7: "7 - Reversible defect"}[v],
        )

    predict_clicked = st.button("🔍 Predict", type="primary", width="stretch")

with right:
    st.subheader("Prediction result")
    if predict_clicked:
        patient = pd.DataFrame(
            [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]],
            columns=FEATURE_NAMES,
        )
        pipeline = load_pipelines()[selected_model]
        prediction = int(pipeline.predict(patient)[0])
        probability = float(pipeline.predict_proba(patient)[0, 1])

        if prediction == 1:
            st.error(f"**{selected_model}: heart disease likely**")
        else:
            st.success(f"**{selected_model}: no heart disease indicated**")

        st.metric("Probability of heart disease (class 1)", f"{probability:.1%}")
        st.progress(min(max(probability, 0.0), 1.0))
        st.caption(
            f"Model used: **{selected_model}** with its own preprocessing "
            "pipeline (median imputation"
            + (" + standardisation" if selected_model == "LVQ (GLVQ)" else "")
            + ")."
        )

        with st.expander("Patient row passed to the model"):
            st.dataframe(patient, width="stretch")
    else:
        st.info("Fill in the patient attributes and press **Predict**.")

    st.caption(DISCLAIMER.replace("**", ""))

# ---------------------------------------------------------------------------
# Model comparison (real saved results)
# ---------------------------------------------------------------------------
st.header("📈 Model comparison - actual test-set results")

metrics_file = RESULTS_DIR / "metrics.json"
if metrics_file.exists():
    metrics = load_json(metrics_file)
    table = pd.DataFrame(
        {
            name: {
                "Accuracy": m["accuracy"],
                "Precision": m["precision"],
                "Recall": m["recall"],
                "F1-score": m["f1"],
                "ROC-AUC": m["roc_auc"],
            }
            for name, m in metrics["models"].items()
        }
    ).T.round(4)

    st.markdown(
        f"All numbers below are computed on the untouched held-out test set "
        f"(**{metrics['test_set_size']} patients**) and stored in `results/metrics.json`."
    )
    st.dataframe(
        table.style.highlight_max(axis=0, color="#c6efce"),
        width="stretch",
    )
    best = metrics["best_by"]
    st.success(
        f"Best model on this test split - F1: **{best['f1']}**, "
        f"ROC-AUC: **{best['roc_auc']}**, Accuracy: **{best['accuracy']}**."
    )

    chart_tabs = st.tabs(
        ["Confusion matrices", "Metric comparison", "ROC curves", "XGBoost importance"]
    )
    with chart_tabs[0]:
        show_image(RESULTS_DIR / "confusion_matrices.png", "Confusion matrices (test set)")
    with chart_tabs[1]:
        show_image(RESULTS_DIR / "metric_comparison.png", "Metric comparison")
    with chart_tabs[2]:
        show_image(RESULTS_DIR / "roc_curves.png", "ROC curves with AUC")
    with chart_tabs[3]:
        show_image(RESULTS_DIR / "xgb_feature_importance.png", "XGBoost feature importance")

    with st.expander("🔎 SHAP explanations for XGBoost (global + one patient)"):
        st.markdown(
            "SHAP shows how much each feature pushed XGBoost's prediction. "
            "`ca` (major vessels), `thal`, `cp` (chest pain type) and `oldpeak` "
            "are the strongest contributors on the test set."
        )
        show_image(RESULTS_DIR / "shap_summary_bar.png", "Global SHAP importance")
        show_image(RESULTS_DIR / "shap_summary_beeswarm.png", "SHAP beeswarm (direction of effects)")
        show_image(RESULTS_DIR / "shap_local_example.png", "Local SHAP explanation for one test patient")
else:
    st.info("Run `python -m src.evaluate_models` to generate the comparison results.")

st.divider()
st.caption(
    "Project structure: `src/` training & evaluation code · `models/` saved pipelines · "
    "`results/` metrics & charts · `app.py` this dashboard. See README.md for the full guide."
)
