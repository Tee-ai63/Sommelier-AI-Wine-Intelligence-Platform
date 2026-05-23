# 🍷 Sommelier AI — Wine Quality Prediction with Explainable ML

> *Predict the quality of a red wine from its chemical properties — and understand exactly why.*

[![Live App](https://img.shields.io/badge/Live%20App-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://sommelier-ai-wine-intelligence-platform-csmsbyvkrkto9kwnvfclje.streamlit.app/)
[![Medium Article](https://img.shields.io/badge/Read%20on-Medium-000000?style=for-the-badge&logo=medium&logoColor=white)](https://medium.com/@wanjirutee4/i-taught-a-machine-to-think-like-a-sommelier-heres-everything-i-learned-8f862b75434f)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-RandomForest-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![SHAP](https://img.shields.io/badge/Explainability-SHAP-c9846a?style=for-the-badge)](https://shap.readthedocs.io)

---

## What This Project Is

Most wine quality apps stop at a number. This one explains the number.

Sommelier AI is an end-to-end machine learning application that predicts the quality score of a red wine (scale 3–8) from 11 chemical properties, and uses **SHAP (SHapley Additive exPlanations)** to show exactly which chemicals drove that score — up or down — for each individual prediction.

Built on the [UCI Wine Quality Dataset](https://archive.ics.uci.edu/dataset/186/wine+quality), this project covers the full pipeline: data cleaning, EDA, feature engineering, model training, explainability, and deployment as a Streamlit web app.

The deeper goal was to practise something I think matters more than accuracy: building a model that communicates honestly — about what it knows, what it doesn't, and why it made each decision.

📖 **Full technical walkthrough on Medium →** [I Taught a Machine to Think Like a Sommelier](https://medium.com/@wanjirutee4/i-taught-a-machine-to-think-like-a-sommelier-heres-everything-i-learned-8f862b75434f)

---

## Live Demo

**[→ Try Sommelier AI](https://sommelier-ai-wine-intelligence-platform-csmsbyvkrkto9kwnvfclje.streamlit.app/)**

Use the sliders to set a wine's chemical profile, hit **Analyse This Wine**, and get:

- A quality score (3–8) with a tier label: Poor / Average / Good / Excellent
- Model confidence percentage
- A per-prediction SHAP chart — green bars push the score up, red bars pull it down
- Plain-English explanation of the top contributing features
- Percentile ranking for 5 key features against 1,143 reference wines
- A model transparency panel showing training details and known limitations

---

## Project Structure

```
sommelier-ai/
├── sommelier_ai.ipynb      # Full notebook: EDA → preprocessing → modelling → SHAP
├── app.py                  # Streamlit application
├── WineQT.csv              # Dataset (1,143 red wines, UCI)
├── sommelier_model.pkl     # Trained Random Forest (300 trees, balanced weights)
├── sommelier_scaler.pkl    # Fitted StandardScaler (fit on train only)
├── model_metrics.json      # Accuracy, feature list, global SHAP importances
├── shap_importance.png     # Global SHAP feature importance chart
└── requirements.txt        # Python dependencies
```

---

## The Pipeline

### Data Cleaning
The raw dataset has 1,599 rows. After dropping the uninformative `Id` column and removing 456 duplicate rows, I was left with **1,143 unique wines**. No missing values. Deduplication was non-negotiable — duplicates appearing in both train and test sets would silently inflate evaluation metrics.

### Exploratory Data Analysis
Three signals stood out from the correlation analysis and box plots:

- **Alcohol (+0.48 correlation)** — the strongest positive predictor. Wines above 11.2% ABV score significantly higher on average.
- **Volatile acidity (−0.39)** — the primary negative driver. Above ~0.7 g/dm³, the model reliably expects lower quality. Chemically, this is the compound that makes wine taste like vinegar.
- **Sulphates (+0.25)** — a preservative and antimicrobial agent. Balanced levels signal careful winemaking.

The class distribution revealed the central challenge: quality scores 5 and 6 account for 83% of the dataset. Scores 3 and 8 appear only 10 and 18 times respectively. Every subsequent decision was shaped by this imbalance.

### Feature Engineering
Log transformation (`np.log1p`) was applied to five right-skewed features — residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, and sulphates — to compress extreme outliers and give the model a cleaner signal. Human-readable tier labels were derived from the quality scores for the application layer.

### Train-Test Split
An **80/20 stratified split** (`stratify=y`) was used to preserve class proportions across both sets. Without stratification on this dataset, the test set can end up with zero examples of the rarest classes, making accuracy metrics meaningless.

The `StandardScaler` was **fit on training data only**, then applied to both sets. Fitting the scaler on the full dataset before splitting is data leakage — the test set's statistics influence the preprocessing. This is a small line of code with a large impact on whether you can trust your evaluation.

### Modelling
Two models were compared:

- **Random Forest (300 trees, `class_weight='balanced'`)** — 59.8% accuracy, weighted F1 of 0.58
- **Gradient Boosting (200 estimators)** — 50.5% accuracy

Random Forest was selected. On a 6-class problem where random guessing achieves ~17%, 59.8% weighted accuracy reflects genuine learning. Gradient Boosting's sequential correction mechanism tends to overfit on small, imbalanced datasets like this one — Random Forest's ensemble averaging generalised better.

`class_weight='balanced'` compensates for the minority classes by upweighting rare examples during training. It does not solve the small-sample problem but partially offsets it.

### Explainability — SHAP
`shap.TreeExplainer` was applied to the trained Random Forest to compute SHAP values for every prediction. The global importance ranking (averaged across the test set) confirmed what EDA suggested:

**alcohol → sulphates → volatile acidity → total sulfur dioxide → chlorides**

The fact that model-learned rankings match established wine chemistry is evidence the model is learning real signal, not noise.

More importantly, SHAP is computed **per prediction** in the live app. Every wine gets its own explanation chart showing exactly which features drove that specific score — not just the average story across the dataset.

---

## Key Findings

Alcohol is the single most important chemical signal for quality. Volatile acidity is the most damaging. Sulphates at controlled levels are a positive signal. The model performs well on wines in the 5–7 range, which is where most real-world red wines sit, and is honest about its limitations on the extremely rare scores at either end of the scale.

---

## Run Locally

```bash
git clone https://github.com/Tee-ai63/Sommelier-AI-Wine-Intelligence-Platform.git
cd Sommelier-AI-Wine-Intelligence-Platform
pip install -r requirements.txt
streamlit run app.py
```

All model artifacts are included — no retraining required.

---

## Tech Stack

**Data:** pandas, numpy  
**Modelling:** scikit-learn (RandomForestClassifier, StandardScaler, train_test_split)  
**Explainability:** SHAP (TreeExplainer)  
**Visualisation:** matplotlib, seaborn  
**Application:** Streamlit  
**Deployment:** Streamlit Community Cloud  
**Language:** Python 3.10+

---

## Dataset

[UCI Wine Quality Dataset](https://archive.ics.uci.edu/dataset/186/wine+quality) — P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis, 2009.  
Red wine variant. 1,599 original samples · 1,143 after deduplication · 11 physicochemical features · Quality scores 3–8.

---

*Built by Tess Kamau · [Medium](https://medium.com/@wanjirutee4/i-taught-a-machine-to-think-like-a-sommelier-heres-everything-i-learned-8f862b75434f) · [GitHub](https://github.com/Tee-ai63)*
