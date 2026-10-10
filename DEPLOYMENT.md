# Deploying to Streamlit Community Cloud

This guide deploys the **dashboard** (`app.py`) to
[share.streamlit.io](https://share.streamlit.io). The models are already
trained and committed, so **nothing is retrained on the cloud**.

Verified facts (see the deployment checklist at the end):

* The app needs only `app.py`, `src/__init__.py`, `src/config.py`,
  `src/compat.py`, the three `models/*.joblib` files, and the `results/*`
  JSON + PNG files — all committed.
* Every pinned dependency has a **prebuilt Linux (x86_64) wheel** for
  **Python 3.12** — no compilation on the build server.
* **No `packages.txt` is required.** The Linux XGBoost wheel bundles its own
  OpenMP runtime (`xgboost.libs/libgomp-*.so`), unlike macOS which needed
  `brew install libomp`.

---

## 1. Push the project to GitHub

The repository has been initialised locally and everything is staged. From the
project folder:

```bash
cd /Users/heenakhilnani/Desktop/CVD_ML_Project

# Set your identity once (no global config existed on this machine)
git config user.name "Your Name"
git config user.email "you@example.com"

# First commit
git commit -m "CVD prediction: XGBoost + LVQ + GaussianNB with Streamlit app"
git branch -M main
```

Create an **empty** repository on GitHub named `CVD_ML_Project` (no README),
then connect and push:

```bash
git remote add origin https://github.com/<your-username>/CVD_ML_Project.git
git push -u origin main
```

What is uploaded is small (~1.7 MB): `.venv/`, the dataset ZIP and
`__pycache__` are all excluded by `.gitignore`.

---

## 2. Deploy on Streamlit Community Cloud

1. Go to <https://share.streamlit.io> and sign in with GitHub.
2. Click **Create app** → **Deploy a public app from GitHub**.
3. Fill in:
   * **Repository:** `<your-username>/CVD_ML_Project`
   * **Branch:** `main`
   * **Main file path:** `app.py`
4. Open **Advanced settings** and set **Python version = 3.12**.
   > ⚠️ This is important. `runtime.txt` is **not** supported by Community
   > Cloud; the version is chosen here. Python 3.12 is the version all
   > dependencies were verified against. (Python 3.11 cannot install
   > `numpy==2.5.3`, which has no cp311 wheel.)
5. **Secrets:** none are needed — leave empty.
6. Click **Deploy**.

The first build installs the pinned packages; expect it to take a few minutes
(see the risk note about NCCL below).

---

## 3. Verify the deployment

Once the app is running, check that:

* The header, disclaimer and methodology expander render.
* The **Dataset summary** expander shows 303 patients, 13 features,
  6 missing cells, 139 / 164.
* Selecting each model and pressing **Predict** returns a probability for all
  three (XGBoost, LVQ, Gaussian Naive Bayes).
* The comparison table and all four charts (confusion matrices, metric
  comparison, ROC curves, XGBoost importance) display.
* The SHAP expander shows the bar, beeswarm and local plots.

---

## 4. Deployment checks performed locally

| Check | Result |
|---|---|
| App runs and predicts for all 3 models (`streamlit.testing` AppTest) | ✅ no exceptions |
| Paths are deployment-safe (`src/config.py` uses `Path(__file__).resolve()`) | ✅ cwd-independent |
| Full dependency graph resolves to Linux cp312 wheels | ✅ 51 wheels, 0 source builds |
| scikit-learn 1.5.2 Linux cp312 wheel exists | ✅ |
| `sklvq` pure-Python wheel (Linux-safe) | ✅ |
| `shap` Linux cp312-abi3 wheel exists | ✅ |
| XGBoost Linux x86_64 wheel exists (57.6 MB) | ✅ |
| XGBoost Linux wheel bundles `libgomp` → no `packages.txt` | ✅ confirmed inside wheel |
| `.venv`, `*.zip`, `__pycache__` excluded from git | ✅ `git check-ignore` |
| Model files + result images/JSON committed | ✅ 3 `.joblib` + 10 result files |

---

## 5. Known deployment risks (read before presenting)

1. **XGBoost pulls a 305 MB CUDA package on Linux.**
   `xgboost==3.4.1` declares `nvidia-nccl-cu13; platform_system == "Linux"`.
   On Community Cloud this downloads ~305 MB even though the app is CPU-only.
   It does **not** affect correctness or accuracy, but it makes the first build
   slower/heavier. We keep this version deliberately because the committed
   XGBoost model was pickled with it; downgrading risks failing to load the
   model and would force a retrain. If the build ever fails on size/time,
   the fallback is to retrain XGBoost with a version that omits NCCL (e.g.
   3.0.5 with the arch marker removed) — but only if genuinely needed.
2. **Python version must be 3.12.** If the app is left on the default and it
   is not 3.12, dependency resolution can fail. Change it in Advanced settings.
3. **scikit-learn is pinned to 1.5.2** on purpose: the genuine LVQ library
   (`sklvq 0.1.2`) uses a scikit-learn API removed in 1.6. Do not "helpfully"
   upgrade it.
4. **Unpinned transitive dependencies.** Only the 11 direct dependencies are
   pinned; transitive packages (e.g. `altair`, `pyarrow`, `protobuf`) resolve
   to the latest compatible versions at build time. This matched the local
   build when verified, but a future release could change them. If you need
   byte-for-byte reproducibility, commit a `pip freeze` as a lock file.
5. **Local vs cloud differs only by OS** (macOS vs Debian Linux). All packages
   ship Linux wheels, so behaviour should match; the model maths is identical.
