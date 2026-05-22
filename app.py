import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shap
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Sommelier AI",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS  — dark wine aesthetic
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;1,300;1,400&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0f0a0a;
    color: #f0e8e0;
}
.stApp { background-color: #0f0a0a; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a0a0a 0%, #120808 100%);
    border-right: 1px solid #3d1515;
}
[data-testid="stSidebar"] .stMarkdown h2 {
    font-family: 'Cormorant Garamond', serif;
    color: #c9846a;
    font-size: 1.1rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
}

/* Main heading */
h1 {
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 3.2rem !important;
    font-weight: 300 !important;
    color: #f0e8e0 !important;
    letter-spacing: 0.05em;
    line-height: 1.1 !important;
}
h2, h3 {
    font-family: 'Cormorant Garamond', serif !important;
    font-weight: 400 !important;
    color: #e8d5c4 !important;
}
h2 { font-size: 1.8rem !important; }
h3 { font-size: 1.3rem !important; color: #c9846a !important; }

/* Metric cards */
[data-testid="stMetric"] {
    background: #1a0f0f;
    border: 1px solid #3d1515;
    border-radius: 8px;
    padding: 16px 20px;
}
[data-testid="stMetricLabel"] { color: #a08070 !important; font-size: 0.75rem !important; letter-spacing: 0.1em; text-transform: uppercase; }
[data-testid="stMetricValue"] { color: #e8d5c4 !important; font-family: 'Cormorant Garamond', serif !important; font-size: 2rem !important; }

/* Sliders */
[data-testid="stSlider"] > div > div { background-color: #3d1515 !important; }
.stSlider [data-baseweb="slider"] div[role="slider"] { background-color: #c9846a !important; border-color: #c9846a !important; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #8B0000, #c9846a);
    color: #f0e8e0;
    border: none;
    border-radius: 4px;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.85rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 12px 32px;
    width: 100%;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }

/* Divider */
hr { border-color: #3d1515 !important; }

/* Score badge */
.score-badge {
    display: inline-block;
    background: linear-gradient(135deg, #8B0000 0%, #c9846a 100%);
    color: #f9f0e8;
    font-family: 'Cormorant Garamond', serif;
    font-size: 5rem;
    font-weight: 300;
    width: 120px;
    height: 120px;
    line-height: 120px;
    text-align: center;
    border-radius: 50%;
    margin: 0 auto;
}
.tier-label {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.6rem;
    font-style: italic;
    color: #c9846a;
    text-align: center;
    margin-top: 8px;
}
.result-card {
    background: #1a0f0f;
    border: 1px solid #3d1515;
    border-radius: 10px;
    padding: 28px;
    text-align: center;
}
.insight-box {
    background: #1a0f0f;
    border-left: 3px solid #c9846a;
    border-radius: 0 8px 8px 0;
    padding: 16px 20px;
    margin: 10px 0;
    font-size: 0.9rem;
    color: #d0bfb0;
}
.insight-positive { border-left-color: #5a9e6f; }
.insight-negative { border-left-color: #c05050; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD ARTIFACTS
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    model  = pickle.load(open('sommelier_model.pkl',  'rb'))
    scaler = pickle.load(open('sommelier_scaler.pkl', 'rb'))
    with open('model_metrics.json') as f:
        metrics = json.load(f)
    return model, scaler, metrics

@st.cache_data
def load_dataset():
    df = pd.read_csv('WineQT.csv').drop(columns=['Id']).drop_duplicates()
    skewed = ['residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','sulphates']
    for col in skewed:
        df[col] = np.log1p(df[col])
    return df

model, scaler, metrics = load_model()
df_ref = load_dataset()
features = metrics['features']

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
TIER_CONFIG = {
    'Poor':      {'emoji': '😞', 'color': '#c05050', 'range': '3–4', 'desc': 'Significant flaws detected. High volatile acidity or imbalanced chemistry.'},
    'Average':   {'emoji': '😐', 'color': '#b07030', 'range': '5',   'desc': 'Drinkable but unremarkable. Typical everyday wine profile.'},
    'Good':      {'emoji': '😊', 'color': '#5a9e6f', 'range': '6',   'desc': 'Balanced and pleasant. Would pair well at a dinner table.'},
    'Excellent': {'emoji': '🌟', 'color': '#c9846a', 'range': '7–8', 'desc': 'Exceptional chemistry. High alcohol, well-balanced acidity and sulphates.'},
}

def quality_tier(score):
    if score <= 4: return 'Poor'
    elif score == 5: return 'Average'
    elif score == 6: return 'Good'
    else: return 'Excellent'

def percentile_rank(value, col_data):
    return round((col_data < value).mean() * 100, 1)

def preprocess_input(raw_inputs):
    """Apply the same log transform we applied during training."""
    inp = raw_inputs.copy()
    for col in ['residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','sulphates']:
        inp[col] = np.log1p(inp[col])
    return inp

# ─────────────────────────────────────────────
# SIDEBAR — INPUT SLIDERS
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🍷 Wine Parameters")
    st.markdown("<p style='color:#7a6060;font-size:0.8rem;margin-top:-10px;'>Adjust the sliders to match your wine's chemistry profile.</p>", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("**Acidity**")
    fixed_acidity    = st.slider("Fixed Acidity (g/dm³)",     4.0, 16.0, 8.3,  0.1)
    volatile_acidity = st.slider("Volatile Acidity (g/dm³)",  0.1,  1.6, 0.53, 0.01)
    citric_acid      = st.slider("Citric Acid (g/dm³)",       0.0,  1.0, 0.27, 0.01)
    pH_val           = st.slider("pH",                        2.7,  4.1, 3.31, 0.01)

    st.markdown("**Sugars & Salts**")
    residual_sugar   = st.slider("Residual Sugar (g/dm³)",    1.0, 16.0, 2.6,  0.1)
    chlorides        = st.slider("Chlorides (g/dm³)",         0.01, 0.48, 0.087, 0.001)

    st.markdown("**Sulfur Dioxide**")
    free_so2         = st.slider("Free SO₂ (mg/dm³)",         1.0, 70.0, 15.0, 0.5)
    total_so2        = st.slider("Total SO₂ (mg/dm³)",        5.0, 290.0, 46.0, 1.0)

    st.markdown("**Body & Finish**")
    density          = st.slider("Density (g/cm³)",           0.990, 1.004, 0.9967, 0.0001, format="%.4f")
    sulphates        = st.slider("Sulphates (g/dm³)",         0.3, 2.0, 0.66, 0.01)
    alcohol          = st.slider("Alcohol (% vol)",           8.0, 15.0, 10.4, 0.1)

    st.markdown("---")
    predict_btn = st.button("✦ Analyse This Wine")

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
col_logo, col_title = st.columns([1, 8])
with col_title:
    st.markdown("# Sommelier AI")
    st.markdown("<p style='color:#7a6060;font-size:1rem;margin-top:-12px;font-style:italic;'>Machine learning meets the art of wine. Enter a wine's chemical profile — get a quality prediction with full explainability.</p>", unsafe_allow_html=True)

st.markdown("---")

# ─────────────────────────────────────────────
# MODEL STATS BAR
# ─────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Model Accuracy",  f"{metrics['accuracy']*100:.1f}%")
c2.metric("Training Samples", f"{metrics['n_train']:,}")
c3.metric("Test Samples",     f"{metrics['n_test']:,}")
c4.metric("Features Used",    len(features))

st.markdown("---")

# ─────────────────────────────────────────────
# MAIN PREDICTION PANEL
# ─────────────────────────────────────────────
if predict_btn:
    # Build raw input dict (pre-transform)
    raw = {
        'fixed acidity':       fixed_acidity,
        'volatile acidity':    volatile_acidity,
        'citric acid':         citric_acid,
        'residual sugar':      residual_sugar,
        'chlorides':           chlorides,
        'free sulfur dioxide': free_so2,
        'total sulfur dioxide':total_so2,
        'density':             density,
        'pH':                  pH_val,
        'sulphates':           sulphates,
        'alcohol':             alcohol,
    }

    # Pre-process (log transforms) then scale
    processed = preprocess_input(raw)
    input_df  = pd.DataFrame([processed])[features]
    input_sc  = scaler.transform(input_df)

    # Predict
    score      = int(model.predict(input_sc)[0])
    proba      = model.predict_proba(input_sc)[0]
    confidence = round(float(proba.max()) * 100, 1)
    tier       = quality_tier(score)
    tier_cfg   = TIER_CONFIG[tier]

    # ── RESULT SECTION ──
    st.markdown("## Prediction Results")

    res_col, shap_col = st.columns([1, 2])

    with res_col:
        st.markdown(f"""
        <div class="result-card">
            <div class="score-badge">{score}</div>
            <div class="tier-label">{tier_cfg['emoji']} {tier}</div>
            <p style='color:#7a6060;font-size:0.8rem;margin-top:12px;'>Score range: {tier_cfg['range']} / 10</p>
            <p style='color:#d0bfb0;font-size:0.88rem;margin-top:8px;'>{tier_cfg['desc']}</p>
            <hr style='border-color:#3d1515;margin:16px 0;'>
            <p style='color:#a08070;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.1em;'>Model Confidence</p>
            <p style='font-family:"Cormorant Garamond",serif;font-size:2rem;color:#c9846a;margin:0;'>{confidence}%</p>
        </div>
        """, unsafe_allow_html=True)

        # Probability bar
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Score probability distribution**")
        classes = model.classes_
        prob_df = pd.DataFrame({'Score': classes, 'Probability': proba})
        fig_p, ax_p = plt.subplots(figsize=(4, 2.5))
        fig_p.patch.set_facecolor('#1a0f0f')
        ax_p.set_facecolor('#1a0f0f')
        bar_colors = ['#c9846a' if c == score else '#3d1515' for c in classes]
        ax_p.bar([str(c) for c in classes], proba * 100, color=bar_colors, width=0.6)
        ax_p.set_xlabel('Quality Score', color='#7a6060', fontsize=9)
        ax_p.set_ylabel('Probability (%)', color='#7a6060', fontsize=9)
        ax_p.tick_params(colors='#7a6060', labelsize=8)
        for spine in ax_p.spines.values(): spine.set_color('#3d1515')
        plt.tight_layout()
        st.pyplot(fig_p)
        plt.close()

    with shap_col:
        # ── SHAP PER-PREDICTION ──
        st.markdown("### Why this score?")
        st.markdown("<p style='color:#7a6060;font-size:0.85rem;'>SHAP values show how each chemical property pushed the score <span style='color:#5a9e6f;'>up ↑</span> or <span style='color:#c05050;'>down ↓</span> for this specific wine.</p>", unsafe_allow_html=True)

        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_sc)
        # shap_values shape: (1, 11, 6) — pick the predicted class index
        class_idx   = list(model.classes_).index(score)
        sv          = shap_values[0, :, class_idx]

        shap_df = pd.DataFrame({
            'Feature': features,
            'SHAP':    sv
        }).sort_values('SHAP')

        fig_s, ax_s = plt.subplots(figsize=(6.5, 5))
        fig_s.patch.set_facecolor('#1a0f0f')
        ax_s.set_facecolor('#1a0f0f')
        colors_shap = ['#5a9e6f' if v > 0 else '#c05050' for v in shap_df['SHAP']]
        ax_s.barh(shap_df['Feature'], shap_df['SHAP'], color=colors_shap, height=0.6)
        ax_s.axvline(0, color='#7a6060', linewidth=0.8, linestyle='--')
        ax_s.set_xlabel('SHAP value (impact on predicted score)', color='#a08070', fontsize=9)
        ax_s.tick_params(colors='#d0bfb0', labelsize=9)
        for spine in ax_s.spines.values(): spine.set_color('#3d1515')
        ax_s.set_title(f'Feature contributions for this wine (predicted score: {score})',
                       color='#e8d5c4', fontsize=10, pad=10)
        green_p = mpatches.Patch(color='#5a9e6f', label='Pushes score up')
        red_p   = mpatches.Patch(color='#c05050', label='Pushes score down')
        ax_s.legend(handles=[green_p, red_p], facecolor='#1a0f0f',
                    edgecolor='#3d1515', labelcolor='#d0bfb0', fontsize=8)
        plt.tight_layout()
        st.pyplot(fig_s)
        plt.close()

        # Human-readable top insights
        top_pos = shap_df[shap_df['SHAP'] > 0].tail(2)
        top_neg = shap_df[shap_df['SHAP'] < 0].head(2)
        st.markdown("**Key insights for this wine:**")
        for _, row in top_pos.iloc[::-1].iterrows():
            st.markdown(f"<div class='insight-box insight-positive'>✦ <strong>{row['Feature']}</strong> is contributing positively — it is helping this wine's score.</div>", unsafe_allow_html=True)
        for _, row in top_neg.iterrows():
            st.markdown(f"<div class='insight-box insight-negative'>✦ <strong>{row['Feature']}</strong> is pulling the score down — consider this a flag.</div>", unsafe_allow_html=True)

    st.markdown("---")

    # ── PERCENTILE COMPARISON ──
    st.markdown("## How does this wine compare?")
    st.markdown("<p style='color:#7a6060;font-size:0.85rem;'>Percentile ranking against 1,143 red wines in the reference dataset. The higher the percentile, the rarer and better that value is.</p>", unsafe_allow_html=True)

    highlight_features = ['alcohol', 'volatile acidity', 'sulphates', 'citric acid', 'pH']
    comp_data = []
    for feat in highlight_features:
        col_val = processed[feat]
        pct     = percentile_rank(col_val, df_ref[feat])
        # Higher alcohol/sulphates/citric acid = better. Lower volatile acidity = better.
        if feat == 'volatile acidity':
            good = pct < 40
            direction = f"lower than {pct}% of wines (lower is better for this)"
        else:
            good = pct > 55
            direction = f"higher than {pct}% of wines"
        comp_data.append({'feat': feat, 'pct': pct, 'good': good, 'direction': direction})

    cols = st.columns(len(highlight_features))
    for i, item in enumerate(comp_data):
        color = "#5a9e6f" if item['good'] else "#c05050"
        cols[i].markdown(f"""
        <div style='background:#1a0f0f;border:1px solid #3d1515;border-radius:8px;padding:14px;text-align:center;'>
            <p style='color:#a08070;font-size:0.7rem;text-transform:uppercase;letter-spacing:0.1em;margin:0;'>{item['feat']}</p>
            <p style='font-family:"Cormorant Garamond",serif;font-size:2.4rem;color:{color};margin:4px 0;'>{item['pct']}<span style='font-size:1rem;'>%ile</span></p>
            <p style='color:#7a6060;font-size:0.72rem;margin:0;'>{item['direction']}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── MODEL TRANSPARENCY ──
    with st.expander("📊 Model Transparency — How this works"):
        st.markdown("""
        **Model:** Random Forest Classifier (300 trees, class_weight='balanced')

        **Training data:** 1,143 red wines from the UCI Wine Quality dataset (80/20 stratified split)

        **Preprocessing:** Log transformation applied to right-skewed features (residual sugar, chlorides, sulfur dioxide, sulphates). StandardScaler applied to all features.

        **Accuracy:** 59.8% on held-out test set. On a 6-class problem where random guessing = ~17%, this represents strong performance.

        **Known limitation:** The model has very few training examples for quality scores 3 and 8 (6 and 16 samples respectively). Predictions for wines that are truly exceptional or truly poor should be interpreted with caution.

        **SHAP explainability:** Each prediction is explained using TreeSHAP — showing exactly how much each feature pushed the prediction up or down for that specific wine.
        """)

        st.markdown("**Global feature importance (SHAP, averaged across all test predictions):**")
        shap_imp = metrics['shap_importance']
        fig_g, ax_g = plt.subplots(figsize=(7, 3.5))
        fig_g.patch.set_facecolor('#1a0f0f')
        ax_g.set_facecolor('#1a0f0f')
        items = sorted(shap_imp.items(), key=lambda x: x[1])
        ax_g.barh([i[0] for i in items], [i[1] for i in items],
                  color='#8B0000', height=0.6)
        ax_g.set_xlabel('Mean |SHAP Value|', color='#a08070', fontsize=9)
        ax_g.tick_params(colors='#d0bfb0', labelsize=9)
        for spine in ax_g.spines.values(): spine.set_color('#3d1515')
        plt.tight_layout()
        st.pyplot(fig_g)
        plt.close()

else:
    # ── LANDING STATE (no prediction yet) ──
    st.markdown("## How to use Sommelier AI")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("""
        <div style='background:#1a0f0f;border:1px solid #3d1515;border-radius:10px;padding:24px;'>
            <p style='font-family:"Cormorant Garamond",serif;font-size:1.4rem;color:#c9846a;'>01 — Input</p>
            <p style='color:#d0bfb0;font-size:0.88rem;'>Use the sliders on the left to set a wine's chemical properties. Default values are set to the dataset average.</p>
        </div>
        """, unsafe_allow_html=True)
    with col_b:
        st.markdown("""
        <div style='background:#1a0f0f;border:1px solid #3d1515;border-radius:10px;padding:24px;'>
            <p style='font-family:"Cormorant Garamond",serif;font-size:1.4rem;color:#c9846a;'>02 — Predict</p>
            <p style='color:#d0bfb0;font-size:0.88rem;'>Hit "Analyse This Wine" to run the Random Forest model. It predicts a quality score from 3 to 8.</p>
        </div>
        """, unsafe_allow_html=True)
    with col_c:
        st.markdown("""
        <div style='background:#1a0f0f;border:1px solid #3d1515;border-radius:10px;padding:24px;'>
            <p style='font-family:"Cormorant Garamond",serif;font-size:1.4rem;color:#c9846a;'>03 — Explain</p>
            <p style='color:#d0bfb0;font-size:0.88rem;'>A SHAP chart breaks down which chemical properties drove the score up or down — not just what, but why.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background:#1a0f0f;border-left:3px solid #c9846a;border-radius:0 8px 8px 0;padding:20px 24px;'>
        <p style='font-family:"Cormorant Garamond",serif;font-size:1.1rem;color:#c9846a;margin:0 0 8px;'>What makes this different from a basic prediction app?</p>
        <p style='color:#d0bfb0;font-size:0.88rem;margin:0;'>Most wine quality apps stop at the number. Sommelier AI goes further — it uses SHAP (SHapley Additive exPlanations) to show you the exact chemical reason behind every prediction, and ranks your wine against 1,143 reference wines in the dataset. It also exposes the model's known limitations honestly, which is what professional ML work looks like.</p>
    </div>
    """, unsafe_allow_html=True)

# ── FOOTER ──
st.markdown("<br><br>")
st.markdown("<p style='text-align:center;color:#3d2020;font-size:0.8rem;'>Sommelier AI · Built by Tess Kamau · Random Forest + SHAP · UCI Wine Quality Dataset</p>", unsafe_allow_html=True)