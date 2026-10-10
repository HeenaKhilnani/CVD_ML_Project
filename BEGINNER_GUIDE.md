# Beginner Guide — CVD Prediction Project

Written for a second-year CS student who has never built an ML model before.
Read this once and you can explain every part of the project in your viva.

---

## 1. The big picture

Machine learning (ML) means: instead of writing rules by hand ("if cholesterol >
240 then…"), we show a program many **examples** and let it learn the rules.

Our examples are **303 real patients** from the UCI Cleveland Heart Disease
dataset. For each patient we have 13 medical measurements (age, cholesterol,
chest pain type, …) and a label saying whether they had heart disease. The
program learns to map measurements → label, and we then test it on patients it
has never seen.

Two ground rules that examiners love:

* **Never test on training data.** A student who memorises the textbook gets
  100% on that textbook and 0% on the exam. Models are the same.
* **Never mix test data into training.** Even tiny leaks (e.g. computing an
  average over all data before splitting) inflate results. Our code keeps all
  preprocessing *inside* model pipelines fitted on training data only.

---

## 2. Stage 1 — Inspecting the dataset (`src/data_preparation.py`)

**File used:** `data/processed.cleveland.data` (from the original UCI zip in
your project).

**What we found (verify by running `python -m src.data_preparation`):**

* 303 rows × 14 columns = 13 features + 1 target.
* The target column `num` runs from 0 to 4 (severity of vessel narrowing).
* Missing values are stored as the character `?`: **6 cells total** —
  4 in `ca` (major vessels) and 2 in `thal` (thalassemia).
* Original target counts: 0→164, 1→55, 2→36, 3→35, 4→13.

**Two decisions we made:**

1. `?` → `NaN` (Python's "missing"), because a `?` is not a number and any
   arithmetic on it would be wrong.
2. Severity 1–4 → all become `1` ("disease present"), 0 stays `0`. This turns
   a 5-class problem into a clean **binary classification** problem
   (164 no-disease vs 139 disease). This is the standard convention for this
   dataset.

---

## 3. Stage 2 — Preparing data correctly

### Why split before anything else

We split 303 patients into:

* **Training set (80%)**: 242 patients — the model learns from these.
* **Test set (20%)**: 61 patients — locked away until final evaluation.

We use `stratify=y`, which keeps the same disease/no-disease ratio in both
sets. Without stratification, random chance could give the test set very few
sick patients and make the scores meaningless. `random_state=42` freezes the
random split so every teammate and every rerun gets the same 61 patients.

### What is data leakage, in one example

If we computed the average cholesterol over all 303 patients and used it to
fill missing values *before* splitting, the training set would contain a tiny
bit of information from test patients. The model would look slightly better
than it deserves. The fix: **fit the imputer on the training set only**, then
apply it to the test set. In our code this happens automatically because
`SimpleImputer` lives inside each model's `Pipeline`.

### Why scaling matters for LVQ but not the others

* LVQ compares distances between points, so a feature like cholesterol
  (range 126–564) would crush a feature like `oldpeak` (range 0–6.2) unless we
  standardise every feature to mean 0, standard deviation 1.
* XGBoost splits on thresholds; rescaling changes nothing for trees.
* Gaussian Naive Bayes estimates the mean and variance **per feature**, so
  scaling changes nothing mathematically (we keep it simple and skip it).

---

## 4. Stage 3 — The three models

### XGBoost (eXtreme Gradient Boosting)

Builds hundreds of small decision trees one after another; each new tree is
trained to fix the mistakes of the trees before it ("boosting"). It is the
strongest general-purpose tabular model today, which is why it is our
performance benchmark. We use 300 trees, depth 3, learning rate 0.05.

### LVQ — Learning Vector Quantization (the star of your project)

Instead of building rules, LVQ keeps a small set of **prototype vectors**
(imaginary "typical patients") and classifies a new patient by asking: *which
prototype is closest?* During training, prototypes move: the closest correct
prototype is pulled toward the patient, and the closest wrong prototype is
pushed away.

Our implementation is the **genuine `sklvq` library** — specifically
**GLVQ** (Generalized LVQ, Sato & Yamada), which replaces the old
"move if correct" rule with a smooth cost function optimised by **Adam**
gradient descent. We use 2 prototypes per class, standardised features and a
fixed seed.

> Engineering story worth telling in the viva: `sklvq 0.1.2` is old software.
> Two real incompatibilities had to be solved honestly:
> 1. It needs scikit-learn < 1.6 → we pinned `scikit-learn==1.5.2`.
> 2. It uses the removed NumPy 1.x alias `np.int` → `src/compat.py` restores
>    that one alias so the unmodified library runs on NumPy 2.x.
> We did **not** fake LVQ with KNN or any other algorithm.

### Gaussian Naive Bayes

A probabilistic baseline. It assumes each feature follows a bell curve per
class and that features are independent given the class ("naive"). It is
extremely fast and surprisingly decent, but the independence assumption is
clearly false for medical data (e.g. age and max heart rate correlate).

### What "training" produced (verify with `python -m src.train_models`)

| Model | Training accuracy |
|---|---|
| XGBoost | 0.9959 |
| LVQ (GLVQ) | 0.8347 |
| Gaussian Naive Bayes | 0.8512 |

XGBoost nearly memorises the training set (99.6%) — a warning sign of
overfitting, which is exactly why we judge everything on the test set.

---

## 5. Stage 4 — Understanding the metrics

**Confusion matrix** — the 2×2 table behind every metric:

|  | Predicted healthy | Predicted sick |
|---|---|---|
| **Actually healthy** | True Negative (TN) | False Positive (FP) |
| **Actually sick** | False Negative (FN) | True Positive (TP) |

* **Accuracy** = (TP+TN)/everything. Good overall summary, misleading when
  classes are imbalanced.
* **Precision** = TP/(TP+FP). "When the model says sick, how often is it right?"
  Low precision = many false alarms.
* **Recall (sensitivity)** = TP/(TP+FN). "Of all truly sick patients, how many
  did we catch?" In medicine this is usually the metric you care about most —
  a false negative can cost a life.
* **F1-score** = harmonic mean of precision and recall; balances both.
* **ROC-AUC** = probability that the model ranks a random sick patient above a
  random healthy one. 0.5 = coin flip, 1.0 = perfect. It uses predicted
  probabilities, not just labels.

### Our actual test-set results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| XGBoost | 0.8852 | 0.8182 | 0.9643 | 0.8852 | 0.9556 |
| **LVQ (GLVQ)** | **0.9016** | **0.9231** | 0.8571 | **0.8889** | **0.9621** |
| Gaussian Naive Bayes | 0.8689 | 0.7941 | 0.9643 | 0.8710 | 0.9524 |

**How to read this honestly:**

* LVQ wins on accuracy, precision, F1 and ROC-AUC on this particular 61-patient
  split. It made only 6 mistakes: 4 false negatives, 2 false positives
  (`[[31, 2], [4, 24]]`).
* XGBoost and GaussianNB have higher **recall** (0.9643 — only 1 missed sick
  patient each) but more false alarms, so their precision is lower.
* Differences of 1–3 patients are **not** statistically meaningful with n=61.
  The correct claim is: *"all three models performed comparably (ROC-AUC
  0.95–0.96); LVQ was slightly ahead on this split."* — never "LVQ is the best
  heart disease model".

### Why accuracy alone is not enough for medicine

Imagine a lazy model that predicts "healthy" for everyone: it would be 54%
accurate here without learning anything. If only 5% of patients were sick, it
would be 95% accurate while catching zero sick people. That is why we report
precision, recall, F1 and ROC-AUC, and why class balance and the cost of each
error type must always be discussed.

---

## 6. Stage 5 — The Streamlit app (`app.py`)

Streamlit turns a Python script into a web page. Our app:

1. Shows the overview, methodology and disclaimer.
2. Summarises the dataset (numbers loaded from `results/dataset_summary.json`).
3. Lets you pick a model, enter all 13 patient values with labelled dropdowns
   and sliders, and press **Predict**.
4. Loads the saved pipeline for that model — including **its own correct
   preprocessing** — and shows the predicted class plus probability.
5. Displays the real comparison metrics and all charts from `results/`.
6. Shows the SHAP plots in an expander.

Run it with:

```bash
source .venv/bin/activate
streamlit run app.py
```

---

## 7. Stage 6 — SHAP explanations (`src/shap_explain.py`)

SHAP answers: *"how much did each feature push this prediction up or down?"*
It comes from game theory (Shapley values): imagine removing one feature at a
time and averaging its effect across all possible orderings.

* **Bar plot** — global importance: mean absolute SHAP value per feature.
* **Beeswarm plot** — one dot per patient; colour shows whether the feature
  value was high or low, position shows whether it pushed the prediction up
  (right) or down (left).
* **Waterfall plot** — one single patient's explanation: start from the average
  prediction and add each feature's contribution.

**Our real result:** the most influential features are `ca` (major vessels),
`thal`, `cp` (chest pain type), `sex`, `age`, `oldpeak`. This matches
established cardiology knowledge — a strong sanity check that the model learned
something medically sensible rather than noise.

---

## 8. Viva questions & answers (short version)

**Q1. What problem are you solving?**
Binary classification: predict presence/absence of heart disease from 13
clinical features, and compare three algorithms fairly.

**Q2. Why did you convert the target 1–4 into 1?**
The original `num` is disease severity on a 0–4 scale. For a yes/no clinical
question we need binary labels; grouping 1–4 as "disease present" is the
standard convention. (We show the original distribution so the choice is
transparent.)

**Q3. How did you handle missing values?**
The six `?` cells become NaN and are filled with the **training-set median**
inside each pipeline, so no test information leaks into training.

**Q4. What is data leakage and how did you prevent it?**
Using test data (or all data) while preparing/training. We split first, and
fit every imputer/scaler inside a scikit-learn Pipeline on training data only.

**Q5. Why 80/20 and why stratified?**
80% gives enough patients to learn from; 20% (61) is enough to evaluate.
`stratify=y` preserves the class ratio in both sets; a fixed `random_state=42`
makes the split reproducible.

**Q6. How does XGBoost work?**
Gradient boosting: many shallow decision trees are added sequentially, each
trained on the errors of the current ensemble; predictions are combined.
Regularisation (max depth, lambda) controls overfitting.

**Q7. How does LVQ work?**
It learns prototype vectors per class. A new sample is classified by the
nearest prototype. In GLVQ a cost function pulls the correct prototype closer
and pushes the wrong one away, optimised with gradient descent (we used Adam).

**Q8. Did you really implement LVQ or substitute something?**
Genuine LVQ via the `sklvq` library (GLVQ model). Two real compatibility
issues with modern NumPy/scikit-learn were fixed by pinning scikit-learn to
1.5.2 and adding one NumPy alias shim (`src/compat.py`) — documented in the
README, not hidden.

**Q9. Why does LVQ need feature scaling?**
It is distance-based: unscaled large-range features (cholesterol) would
dominate distance computations. Standardisation puts all 13 features on the
same footing.

**Q10. Why doesn't XGBoost need scaling?**
Trees split on thresholds ("is chol > 240?"), and monotonic rescaling does not
change the ordering, so the same splits are available.

**Q11. Why Gaussian Naive Bayes?**
It is a classic, fast probabilistic baseline for the comparison. Its
independence assumption is unrealistic, so it shows what "simplest possible
model" can do on this data.

**Q12. Which model won and by how much?**
On the held-out 61 patients, LVQ (GLVQ) led: accuracy 0.9016, F1 0.8889,
ROC-AUC 0.9621, versus XGBoost 0.8852/0.8852/0.9556 and GaussianNB
0.8689/0.8710/0.9524. The gaps are 1–3 patients wide, so we describe the models
as comparable rather than claiming a decisive winner.

**Q13. Why is accuracy alone insufficient in medicine?**
Class imbalance and unequal error costs. A model can be "accurate" by never
predicting disease while missing every sick patient. We therefore report
precision, recall, F1 and ROC-AUC; recall reflects missed patients, which is
the dangerous error.

**Q14. What is ROC-AUC?**
The probability that a random positive patient is ranked above a random
negative patient by the model's probability score. Threshold-independent;
0.5 is random guessing.

**Q15. Why do XGBoost and GaussianNB have higher recall but lower precision?**
They flag more patients as sick (more false positives), so they miss fewer
sick patients but raise more false alarms. It is a precision–recall trade-off.

**Q16. What does SHAP tell us?**
How much each feature contributed to a specific prediction and, averaged over
patients, which features matter globally. Top features were `ca`, `thal`, `cp`,
`sex`, `age` — medically plausible.

**Q17. What are the limitations of your project?**
Small single-centre dataset (303 patients, 1988, Cleveland); wide uncertainty
on 61 test patients; only 13 features; no external validation; not
clinically validated; probabilities are not calibrated for real triage.

**Q18. How would you improve it with more time?**
Cross-validation instead of a single split; hyperparameter tuning inside CV;
probability calibration; combining the other UCI centres (Hungarian,
Switzerland, VA) for external validation; recruiting more features
(smoking, family history, BMI); fairness analysis across sex and age groups.

**Q19. Why is the target so imbalanced between classes (164 vs 139)?**
It is not severely imbalanced (54%/46%); stratification keeps that ratio in
both splits, and ROC-AUC/F1 are appropriate metrics. We would revisit if one
class were under ~20%.

**Q20. What exactly does your app do when I press Predict?**
It builds a one-row dataframe with the 13 values in the trained order, loads
the selected model's full pipeline (imputer [+ scaler] + classifier), and
returns the predicted class and the probability of class 1. No retraining
happens in the app.
