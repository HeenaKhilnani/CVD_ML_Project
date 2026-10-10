# Prediction of Cardiovascular Disease Using Machine Learning Techniques

A complete, tested college PBL project comparing **XGBoost**, **Learning Vector
Quantization (GLVQ)** and **Gaussian Naive Bayes** on the UCI Cleveland Heart
Disease dataset, with an interactive Streamlit dashboard and SHAP explanations.

> ⚠️ **Educational use only.** This is a student project trained on 303
> historical records. It is **not** a clinically validated diagnostic tool and
> must never be used for real medical decisions.

---

## Results (actual, held-out test set)

61 patients were held out from training and used once for evaluation
(stratified 80/20 split, `random_state=42`):

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---|---|---|---|---|
| XGBoost | 0.8852 | 0.8182 | 0.9643 | 0.8852 | 0.9556 |
| **LVQ (GLVQ)** | **0.9016** | **0.9231** | 0.8571 | **0.8889** | **0.9621** |
| Gaussian Naive Bayes | 0.8689 | 0.7941 | 0.9643 | 0.8710 | 0.9524 |

**Best model on this split: genuine LVQ (GLVQ)** by accuracy, F1 and ROC-AUC.
These numbers are produced by `src/evaluate_models.py` from real model
predictions — see `results/metrics.json` and `results/predictions.csv`.

Charts: `results/confusion_matrices.png`, `results/metric_comparison.png`,
`results/roc_curves.png`, `results/xgb_feature_importance.png`, plus SHAP plots
`results/shap_summary_bar.png`, `shap_summary_beeswarm.png`, `shap_local_example.png`.

Top XGBoost features by mean |SHAP|: `ca` > `thal` > `cp` > `sex` > `age` >
`oldpeak` — consistent with known cardiology risk factors.

---

## Project structure

```
CVD_ML_Project/
├── data/
│   ├── processed.cleveland.data   # the file used (303 patients x 13 features + target)
│   └── ...                        # other original UCI distribution files
├── src/
│   ├── config.py                  # paths, feature names, seed, model names
│   ├── compat.py                  # NumPy 2.x shim so the genuine sklvq library runs
│   ├── data_preparation.py        # Stage 1-2: inspect, clean, stratify, split
│   ├── train_models.py            # Stage 3: train + save all three pipelines
│   ├── evaluate_models.py         # Stage 4: metrics, confusion matrices, charts
│   ├── shap_explain.py            # Stage 6: SHAP for XGBoost
│   └── run_all.py                 # runs everything in one command
├── models/                        # saved pipelines (.joblib) + metadata.json
├── results/                       # metrics, predictions and charts (actual outputs)
├── app.py                         # Stage 5: Streamlit dashboard
├── requirements.txt
├── README.md
└── BEGINNER_GUIDE.md              # stage-by-stage explanation + viva Q&A
```

---

## Setup (macOS)

Requirements: Python 3.12 (Homebrew) and Homebrew. The virtual environment is
**already created** in `.venv/` — you can skip straight to "Run".

### If you need to rebuild the environment from scratch

```bash
# 1. XGBoost needs Apple's OpenMP runtime (one-time system package)
brew install libomp

# 2. Create and activate a virtual environment
cd /Users/heenakhilnani/Desktop/CVD_ML_Project
python3.12 -m venv .venv
source .venv/bin/activate

# 3. Install exactly the tested package versions
pip install -r requirements.txt
```

Why these pins?
* `scikit-learn==1.5.2` — the genuine LVQ library (`sklvq 0.1.2`) uses a
  scikit-learn API removed in 1.6.
* The XGBoost import will fail without `libomp` on macOS — that is an
  Apple/Homebrew dependency, not a Python package. On Linux (including
  Streamlit Community Cloud) the OpenMP runtime is bundled inside the
  XGBoost wheel, so no system package is needed there.

---

## Run the project (exact commands)

```bash
cd /Users/heenakhilnani/Desktop/CVD_ML_Project
source .venv/bin/activate

# Option A - run every stage (inspect, train, evaluate, SHAP) in order
python -m src.run_all

# Option B - run stages individually
python -m src.data_preparation    # Stage 1-2: dataset report + split
python -m src.train_models        # Stage 3:   train + save the 3 models
python -m src.evaluate_models     # Stage 4:   metrics + charts -> results/
python -m src.shap_explain        # Stage 6:   SHAP plots -> results/

# Launch the interactive dashboard (Stage 5)
streamlit run app.py
```

Streamlit opens `http://localhost:8501` in your browser. Press `Ctrl+C` in the
terminal to stop it.

---

## Deploy the dashboard (Streamlit Community Cloud)

The project is ready to deploy for free. See **[DEPLOYMENT.md](DEPLOYMENT.md)**
for the full walkthrough. Short version:

1. Push this folder to a GitHub repository (`.venv` and the dataset ZIP are
   excluded by `.gitignore`).
2. On <https://share.streamlit.io> create an app from that repo with main file
   path `app.py`.
3. In **Advanced settings**, set **Python version = 3.12** (required), then
   deploy. No `packages.txt` or secrets are needed.

---

## Methodology (short version)

1. **Data.** `data/processed.cleveland.data`: 303 patients × 13 features.
   `?` values (4 in `ca`, 2 in `thal`) become `NaN`.
2. **Target.** Original severity 0–4 → binary: `0` = no disease,
   `1–4` = disease (164 vs 139 patients).
3. **Split.** 80/20 stratified, `random_state=42` → 242 train / 61 test,
   same split for all models.
4. **No leakage.** Imputation and scaling live *inside each model's Pipeline*
   and are fitted on the training split only.
5. **Per-model preprocessing.**
   * XGBoost: median imputation (trees do not need scaling).
   * Gaussian Naive Bayes: median imputation (models each feature's Gaussian).
   * LVQ (GLVQ): median imputation + standardisation (distance-based).
6. **Models.**
   * **XGBoost** — gradient-boosted trees (`n_estimators=300, max_depth=3, lr=0.05`).
   * **LVQ** — genuine *Generalized Learning Vector Quantization* from the
     [`sklvq`](https://pypi.org/project/sklvq/) library: learns 2 prototype
     vectors per class, classifies by nearest-prototype distance (Adam solver,
     300 runs, `random_state=42`).
   * **Gaussian Naive Bayes** — probabilistic baseline.
7. **Explainability.** SHAP `TreeExplainer` for XGBoost gives global and
   per-patient feature contributions.

---

## Tested environment

| Item | Version |
|---|---|
| Python | 3.12.14 (Homebrew) |
| numpy / pandas / scipy | 2.5.3 / 3.0.6 / 1.18.1 |
| scikit-learn | 1.5.2 (pinned for sklvq) |
| xgboost / sklvq / shap | 3.4.1 / 0.1.2 / 0.53.0 |
| streamlit | 1.65.0 |
| libomp (Homebrew) | 23.1.3 |

---

## Limitations (important for the viva)

* Only **303 patients** from one hospital (Cleveland, 1988) — results have wide
  confidence intervals and may not generalise to other populations.
* The dataset has just **6 missing values** (4 `ca`, 2 `thal`), so imputation
  choice barely matters here.
* 13 features cannot capture all cardiovascular risk factors (smoking history,
  family history, BMI, blood lipids, etc.).
* **Accuracy alone is insufficient in medicine.** Missing a sick patient
  (false negative) is usually far worse than a false alarm, which is why
  recall and ROC-AUC matter — and why XGBoost/GaussianNB, with higher recall
  but lower precision than LVQ, are not automatically worse.
* This project is a comparison exercise, not evidence of clinical value.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `XGBoostError ... libomp.dylib` | `brew install libomp` |
| `AttributeError: module 'numpy' has no attribute 'int'` | You imported sklvq without the shim: make sure `from src import compat` comes first (all scripts already do this). |
| `cannot import name '_validate_data'` (or similar) from sklearn | You have scikit-learn ≥ 1.6 installed: `pip install scikit-learn==1.5.2`. |
| Streamlit says port 8501 in use | `streamlit run app.py --server.port 8502` |
| Model file not found in the app | Run `python -m src.train_models` first. |
