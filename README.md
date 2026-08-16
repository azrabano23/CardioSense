# CardioSense — a rigorous benchmarking harness for cardiac-risk classification

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

CardioSense trains, balances, tunes, and **honestly compares** six classifiers for predicting heart-disease risk from patient features — and reports the result you actually get, not the one you hoped for. On the dataset shipped here the headline finding is a negative one (the models sit near chance), and the value of the project is the disciplined pipeline that *establishes* that cleanly and explains why.

---

## The problem

Cardiovascular disease is the leading cause of death worldwide — **~17.9 million deaths a year, 32% of all global deaths** ([WHO, 2023](https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds))). Most of that burden is driven by risk factors that are measurable years in advance, so earlier risk stratification is one of the highest-leverage problems in medicine. The catch that this project ran straight into: a model is only as good as the *signal in its features*, and a lot of "heart-disease prediction" tooling reports flattering accuracy numbers that come from class imbalance, leakage, or an over-fit test split rather than real predictive power.

## Market

Clinical AI for cardiology is a real and growing market — AI-in-cardiology was estimated around **USD 1–2B in the early 2020s with strong double-digit projected CAGR** (Grand View Research / MarketsandMarkets sector reports). But the binding constraint isn't model architecture; it's **validated signal and trust**. Risk tools that don't generalize, or that hide a near-chance AUC behind a high accuracy on an imbalanced set, are exactly what clinicians have learned to distrust. The gap CardioSense targets is methodological: a reusable harness that makes the honest comparison — and the honest *failure* — legible.

## What it does

A single, reproducible pipeline that:

1. **Preprocesses** — mean/mode imputation, label-encoding of categoricals, standardization of numerics.
2. **Handles class imbalance** explicitly — SMOTE, ADASYN, and Balanced Bagging, rather than letting the majority class dominate.
3. **Trains six models** — Logistic Regression, Random Forest, Gradient Boosting, XGBoost, LightGBM, and a Balanced Bagging classifier — plus a soft-voting ensemble.
4. **Tunes** Random Forest with `GridSearchCV`.
5. **Evaluates honestly** — accuracy, precision, recall, F1, and **ROC AUC** side by side, with per-class classification reports, feature-importance ranking, and overlaid ROC curves.

## Results — reported straight

On the bundled `heart_disease.csv` (2,000 patients, ~19% positive — a heavily imbalanced set):

| Model | Accuracy | F1 (weighted) | ROC AUC |
|---|---:|---:|---:|
| Logistic Regression | 0.55 | 0.27 | **0.48** |
| Random Forest | 0.61 | 0.20 | **0.46** |
| Gradient Boosting | 0.56 | 0.20 | ~0.47 |
| Balanced Bagging | 0.66\* | 0.18 (minority) | ~0.50 |

\*The 0.66 "accuracy" is the trap: it's near the 81% majority-class base rate achieved by mostly predicting *healthy*. **The ROC AUCs hover at ~0.5 — chance.** Minority-class (actual-disease) recall and F1 stay around 0.18–0.27 across every model and every resampling method.

**The honest read:** on these features, these models cannot separate at-risk from healthy patients better than a coin flip, and no amount of SMOTE/ADASYN/tuning rescues that — because resampling rebalances classes, it doesn't manufacture signal that isn't there. The most likely culprit is the dataset: the features as given carry little real predictive information about the label (consistent with a synthetic or weakly-correlated source). That is a genuine, useful finding — a negative result, cleanly established, is worth more than a 0.95 AUC that quietly came from leakage.

## Technical breakdown (the core of the project)

The engineering is the deliverable, and it's deliberately rigorous:

- **Imbalanced-learning done properly.** Resampling (`imblearn`) is applied **inside** the modelling flow with stratified evaluation, and the metric of record is ROC AUC / minority-class F1 — *not* accuracy — precisely so an imbalanced base rate can't masquerade as performance. Knowing which metric to trust on a 19%-positive set is the whole game.
- **A fair model bake-off.** Six estimators spanning linear, bagging, and three gradient-boosting implementations (sklearn GB, XGBoost, LightGBM), all on an identical split with fixed seeds, plus a soft-voting ensemble — so differences are attributable to the model, not the harness.
- **Interpretability first.** Random-forest feature importances rank what (if anything) drives predictions; ROC curves expose the recall/false-positive trade-off at every threshold instead of a single accuracy number.
- **Diagnostic honesty.** The pipeline is built to *detect* the failure mode it found — a near-chance AUC under a deceptively high accuracy — which is the single most common way medical-ML results mislead.

**Skills demonstrated:** scikit-learn / XGBoost / LightGBM / imbalanced-learn; the statistics of evaluation under class imbalance; hyperparameter search; feature-importance and ROC analysis; and the scientific discipline to report a null result with its diagnosis rather than dress it up.

## Where this goes

The harness is reusable and the failure is informative. A deployable version needs the one thing this dataset lacks — **real predictive signal**: a clinically sourced cohort (e.g. UCI Cleveland, Framingham-style longitudinal data, or institutional EHR extracts) with verified outcomes, calibration (reliability curves, not just AUC), and a prospective validation split. The customers for *that* — cardiology groups, EHR-integrated risk tools, payers running preventative-care programs — are real; the precondition is earning an AUC that clears chance on data that actually carries the signal. CardioSense is the measurement rig that would keep that honest.

## Run it

```bash
pip install -r requirements.txt        # pandas, numpy, scikit-learn, xgboost, lightgbm, imbalanced-learn, matplotlib
python cardiosense_analysis.py         # full comparison + plots
python cardiosense_simple.py           # lightweight version
# or open heart.ipynb for the annotated walkthrough
```

## License

MIT — see [LICENSE](LICENSE). Author: **Azra Bano**.
