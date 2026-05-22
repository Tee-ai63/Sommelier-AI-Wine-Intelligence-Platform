# 🍷 Sommelier AI — Wine Quality Prediction with Explainable ML

> *Predict the quality of a red wine from its chemical properties — and understand exactly why.*

[![Live App](https://sommelier-ai-wine-intelligence-platform-csmsbyvkrkto9kwnvfclje.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-RandomForest-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![SHAP](https://img.shields.io/badge/Explainability-SHAP-c9846a?style=for-the-badge)](https://shap.readthedocs.io)

---

## What This Project Is

Sommelier AI is an end-to-end machine learning application that predicts the quality score of a red wine (scale 3–8) from 11 chemical properties — and crucially, explains *why* that score was assigned using SHAP (SHapley Additive exPlanations).

Most prediction apps stop at the number. This one goes further: every prediction is accompanied by a per-wine SHAP breakdown showing which chemical properties pushed the score up or down, plus a percentile comparison against 1,143 reference wines in the dataset.

Built on the [UCI Wine Quality Dataset](https://archive.ics.uci.edu/dataset/186/wine+quality), this project demonstrates a complete ML pipeline from raw data through to a deployed, explainable application.

---

## Live Demo

**[→ Try Sommelier AI on Streamlit](https://your-app-link.streamlit.app)**

Adjust the sliders to match a wine's chemistry profile, click **Analyse This Wine**, and get:
- A predicted quality score with confidence level
- A SHAP chart explaining the prediction feature by feature
- A percentile ranking for key features against the full dataset
- A full model transparency panel

---

## Project Structure

```
sommelier-ai/
├── sommelier_ai.ipynb      # Full analysis notebook (EDA → preprocessing → modelling → SHAP)
├── app.py                  # Streamlit application
├── WineQT.csv              # Dataset (1,143 red wines, UCI)
├── sommelier_model.pkl     # Trained Random Forest (300 trees)
├── sommelier_scaler.pkl    # Fitted StandardScaler
├── model_metrics.json      # Accuracy, feature list, global SHAP importances
└── requirements.txt        # Python dependencies
```

---

## The Full Pipeline

### 1. Exploratory Data Analysis
- Quality score distribution — identified severe class imbalance (quality 5 & 6 = 83% of dataset)
- Correlation heatmap — alcohol (+0.48), volatile acidity (−0.39), and sulphates (+0.25) flagged as top correlates
- Box plots and distribution analysis per feature
- Outlier detection using IQR method

### 2. Preprocessing
- Removed 240 duplicate rows (1,599 → 1,143 clean samples)
- Log transformation (`np.log1p`) applied to 5 right-skewed features: residual sugar, chlorides, free SO₂, total SO₂, sulphates
- Quality tier labels created for human-readable output: Poor / Average / Good / Excellent
- **Stratified** 80/20 train-test split — preserves class proportions across both sets
- StandardScaler applied: fit on train only, transform on test (no data leakage)

### 3. Modelling

| Model | Accuracy |
|-------|----------|
| Random Forest (300 trees, balanced) | **59.8%** |
| Gradient Boosting (200 estimators) | 50.5% |

Random Forest was selected as the final model. `class_weight='balanced'` was used to compensate for the minority classes (quality 3: 6 samples, quality 8: 16 samples).

**Why 59.8% is meaningful:** On a 6-class classification problem, random guessing achieves ~17%. A 59.8% weighted accuracy — with weighted F1 of 0.58 — represents strong performance on this imbalanced dataset.

### 4. Explainability — SHAP

SHAP (SHapley Additive exPlanations) was applied at two levels:

**Global importance** — averaged across all test predictions:

| Rank | Feature | Mean |SHAP| |
|------|---------|---------|
| 1 | alcohol | 0.0424 |
| 2 | sulphates | 0.0380 |
| 3 | volatile acidity | 0.0330 |
| 4 | total sulfur dioxide | 0.0217 |
| 5 | chlorides | 0.0191 |

**Per-prediction importance** — shown live in the Streamlit app for every wine analysed. Each prediction comes with a bar chart showing exactly how much each feature pushed the score up (green) or down (red) for that specific wine.

---

## Key Findings

- **Alcohol is the strongest predictor of quality.** Wines with alcohol content above the 75th percentile (>11.2% vol) score significantly higher on average.
- **Volatile acidity is the primary negative driver.** Above ~0.7 g/dm³, the model consistently predicts lower quality — high volatile acidity produces a vinegar-like taste.
- **Sulphates at the right level signal quality winemaking.** They act as a preservative and antimicrobial agent; their presence in balanced concentrations correlates with better scores.
- **The model is honest about its limitations.** Classes 3 and 8 (very poor / very exceptional wines) are nearly unpredictable from only 6–16 training examples. The app communicates this directly.

---

## Running Locally

```bash
git clone https://github.com/Tee-ai63/Sommelier-AI.git
cd Sommelier-AI
pip install -r requirements.txt
streamlit run app.py
```

All model artifacts (`.pkl` files) are included in the repo — no retraining required.

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| Data manipulation | pandas, numpy |
| Modelling | scikit-learn (RandomForestClassifier) |
| Explainability | SHAP (TreeExplainer) |
| Visualisation | matplotlib |
| Application | Streamlit |
| Environment | Python 3.10+ |

---

## What I Learned

This project pushed me to think beyond model accuracy. The most valuable skill I practised here was **communicating model behaviour honestly** — why stratified splitting matters on imbalanced data, why the scaler must be fit on train only, and why a model that can't predict quality-3 wines isn't a failure if you explain it clearly. SHAP turned a black-box classifier into something a non-technical user can actually interrogate. That gap — between a model that runs and a model that explains itself — is where most of the real work in ML lives.

---

## Dataset

[UCI Wine Quality Dataset](https://archive.ics.uci.edu/dataset/186/wine+quality) — P. Cortez et al., 2009.
Red wine variant. 1,599 original samples, 1,143 after deduplication.

---

*Built by [Tess Kamau](https://linkedin.com/in/tesskamau) · [GitHub](https://github.com/Tee-ai63) · [LinkedIn](https://linkedin.com/in/tesskamau)*
