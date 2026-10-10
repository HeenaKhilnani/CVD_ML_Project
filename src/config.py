"""
Central configuration for the CVD Machine Learning project.

Every path, feature name, and experiment setting lives here so the whole
project stays consistent and reproducible.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project folders
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# The ML-ready Cleveland file from the UCI Heart Disease distribution.
DATA_FILE = DATA_DIR / "processed.cleveland.data"

# ---------------------------------------------------------------------------
# Dataset schema (the 13 standard Cleveland features + target)
# ---------------------------------------------------------------------------
FEATURE_NAMES = [
    "age",       # age in years
    "sex",       # 1 = male, 0 = female
    "cp",        # chest pain type (1..4)
    "trestbps",  # resting blood pressure (mm Hg)
    "chol",      # serum cholesterol (mg/dl)
    "fbs",       # fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
    "restecg",   # resting ECG result (0,1,2)
    "thalach",   # maximum heart rate achieved
    "exang",     # exercise induced angina (1 = yes, 0 = no)
    "oldpeak",   # ST depression induced by exercise
    "slope",     # slope of peak exercise ST segment (1,2,3)
    "ca",        # number of major vessels coloured by fluoroscopy (0..3)
    "thal",      # thalassemia / defect type (3,6,7)
]
TARGET_NAME = "num"

# Human-readable labels used in the Streamlit app.
FEATURE_LABELS = {
    "age": "Age (years)",
    "sex": "Sex (1 = male, 0 = female)",
    "cp": "Chest pain type (1-4)",
    "trestbps": "Resting blood pressure (mm Hg)",
    "chol": "Serum cholesterol (mg/dl)",
    "fbs": "Fasting blood sugar > 120 mg/dl (0/1)",
    "restecg": "Resting ECG (0-2)",
    "thalach": "Maximum heart rate achieved",
    "exang": "Exercise induced angina (0/1)",
    "oldpeak": "ST depression (oldpeak)",
    "slope": "Slope of peak ST segment (1-3)",
    "ca": "Major vessels coloured (0-3)",
    "thal": "Thalassemia (3, 6, 7)",
}

# ---------------------------------------------------------------------------
# Reproducible experiment settings
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20

# Display names for the three models, in a fixed order.
MODEL_NAMES = ["XGBoost", "LVQ (GLVQ)", "Gaussian Naive Bayes"]

# Saved-model filenames keyed by display name.
MODEL_FILES = {
    "XGBoost": "xgb_pipeline.joblib",
    "LVQ (GLVQ)": "lvq_pipeline.joblib",
    "Gaussian Naive Bayes": "gnb_pipeline.joblib",
}


def ensure_dirs() -> None:
    """Create the output folders if they do not exist yet."""
    for folder in (MODELS_DIR, RESULTS_DIR):
        folder.mkdir(parents=True, exist_ok=True)
