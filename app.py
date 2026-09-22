"""
Diabetes Risk Prediction — Explainable AI demo (thesis defense)
---------------------------------------------------------------
Final model : Soft Voting ensemble (XGBoost + Logistic Regression + CatBoost)
Train/valid : BRFSS 2017     |    External test : Cambodia STEPS
Threshold   : 0.43           |    Features : 11   |   Target : Diabetes_binary

This app supports early diabetes RISK SCREENING and explanation.
It does NOT provide a medical diagnosis.

Run:  streamlit run app.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import os

# matplotlib + SHAP are optional — the app must not crash if missing.
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MPL_AVAILABLE = True
except Exception:
    MPL_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
APP_DIR    = Path(__file__).parent
MODELS_DIR = APP_DIR / "models"

EXPECTED_FEATURES = [
    'Sex', 'Age', 'Education', 'HighBP', 'HighChol', 'BMI',
    'Smoker', 'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump'
]
DEFAULT_THRESHOLD = 0.43

FEATURE_LABELS = {
    'Sex': 'Sex', 'Age': 'Age group', 'Education': 'Education level',
    'HighBP': 'High blood pressure', 'HighChol': 'High cholesterol', 'BMI': 'BMI',
    'Smoker': 'Smoking history', 'PhysActivity': 'Physical activity',
    'Fruits': 'Fruit consumption', 'Veggies': 'Vegetable consumption',
    'HvyAlcoholConsump': 'Heavy alcohol consumption',
}

EDUCATION_OPTIONS = {
    'No formal schooling':              1,
    'Elementary or primary school':     2,
    'Some high school':                 3,
    'High school graduate':             4,
    'Some college or technical school': 5,
    'College graduate or above':        6,
}

RISK_COLORS = {'Low': '#2E7D32', 'Moderate': '#F39C12', 'High': '#C0392B'}
RISK_BG     = {'Low': '#E8F5E9', 'Moderate': '#FFF4E5', 'High': '#FDEDEC'}

DISCLAIMER = (
    "This demo is for academic research and early screening support only. "
    "It is not a medical diagnosis. Please consult a qualified healthcare "
    "professional for clinical advice."
)


# ──────────────────────────────────────────────────────────────────────────────
# Core functions
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_artifacts():
    """Load model, scaler, threshold, feature names, and optional SHAP proxy."""
    required = {
        'model':         MODELS_DIR / 'best_model_2017.pkl',
        'scaler':        MODELS_DIR / 'scaler_2017.pkl',
        'threshold':     MODELS_DIR / 'threshold_2017.pkl',
        'feature_names': MODELS_DIR / 'feature_names_2017.pkl',
    }
    missing = [p.name for p in required.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing model file(s) in 'models/': " + ", ".join(missing))

    art = {
        'model':         joblib.load(required['model']),
        'scaler':        joblib.load(required['scaler']),
        'threshold':     float(joblib.load(required['threshold'])),
        'feature_names': list(joblib.load(required['feature_names'])),
    }
    proxy_path = MODELS_DIR / 'xgboost_proxy_2017.pkl'
    art['proxy'] = joblib.load(proxy_path) if proxy_path.exists() else None
    return art


def steps_age_to_brfss_group(age):
    """Map a real age in years to a BRFSS age group (1–13)."""
    bins   = [0, 24, 29, 34, 39, 44, 49, 54, 59, 64, 69, 74, 79, 999]
    groups = list(range(1, 14))
    for i, (lo, hi) in enumerate(zip(bins, bins[1:])):
        if lo < age <= hi:
            return groups[i]
    return 13


def calculate_bmi(weight_kg, height_cm):
    """BMI = weight(kg) / height(m)^2, rounded to 1 decimal place."""
    height_m = height_cm / 100.0
    if height_m <= 0:
        return None
    return round(weight_kg / (height_m ** 2), 1)


def predict_risk(input_df, scaler, model, threshold, feature_names):
    """Scale, predict probability via predict_proba, apply threshold."""
    input_df = input_df[feature_names]              # enforce model feature order
    X_scaled = scaler.transform(input_df)
    prob  = float(model.predict_proba(X_scaled)[:, 1][0])
    label = "Diabetes Risk" if prob >= threshold else "No Diabetes Risk"
    return prob, label, X_scaled


def assign_risk_tier(prob):
    """Map probability (0–1) to a demo communication tier."""
    pct = prob * 100
    if pct < 30:
        return "Low"
    elif pct < 70:
        return "Moderate"
    return "High"


def generate_recommendations(raw):
    """Rule-based lifestyle recommendations from the user's inputs."""
    recs = []
    if raw['BMI'] and raw['BMI'] >= 25:
        recs.append("Your BMI is in the overweight range — focus on weight management, "
                    "a balanced diet, and regular physical activity.")
    if raw['HighBP'] == 1:
        recs.append("Monitor your blood pressure regularly and have a professional check-up.")
    if raw['HighChol'] == 1:
        recs.append("Monitor your cholesterol and reduce unhealthy (saturated/trans) fats.")
    if raw['PhysActivity'] == 0:
        recs.append("Add regular physical activity — aim for ~150 minutes of moderate exercise per week.")
    if raw['Fruits'] == 0:
        recs.append("Add fruit to your daily diet.")
    if raw['Veggies'] == 0:
        recs.append("Add more vegetables to your daily diet.")
    if raw['Smoker'] == 1:
        recs.append("Consider reducing or quitting smoking.")
    if raw['HvyAlcoholConsump'] == 1:
        recs.append("Consider reducing alcohol consumption.")
    recs.append("Consult a qualified healthcare professional for personalised advice.")
    return recs


def explain_prediction(X_scaled, proxy, feature_names, raw):
    """SHAP proxy explanation if available; otherwise a rule-based fallback."""
    if proxy is not None and SHAP_AVAILABLE:
        try:
            explainer   = shap.TreeExplainer(proxy)
            shap_values = explainer(X_scaled)
            vals = shap_values.values
            if vals.ndim == 3:
                vals = vals[:, :, 1]
            contrib = vals[0]
            order = np.argsort(np.abs(contrib))[::-1][:5]
            items = [
                {'feature': FEATURE_LABELS.get(feature_names[i], feature_names[i]),
                 'value': float(contrib[i]),
                 'direction': 'increases' if contrib[i] > 0 else 'reduces'}
                for i in order
            ]
            return {'mode': 'shap', 'items': items}
        except Exception:
            pass

    reasons = []
    if raw['BMI'] and raw['BMI'] >= 25:
        reasons.append(f"BMI is {raw['BMI']} (overweight range)")
    if raw['HighBP'] == 1:
        reasons.append("history of high blood pressure")
    if raw['HighChol'] == 1:
        reasons.append("history of high cholesterol")
    if raw['AgeGroup'] >= 8:
        reasons.append("older age group")
    if raw['PhysActivity'] == 0:
        reasons.append("no regular physical activity")
    if raw['Smoker'] == 1:
        reasons.append("smoking history")
    if raw['Fruits'] == 0 or raw['Veggies'] == 0:
        reasons.append("limited fruit / vegetable intake")
    if not reasons:
        reasons.append("no major high-risk inputs were flagged")
    return {'mode': 'rules', 'items': reasons}


# ──────────────────────────────────────────────────────────────────────────────
# UI: Single-patient input form (main page, columns)
# ──────────────────────────────────────────────────────────────────────────────
def collect_user_input():
    """Render the main-page form and return (input_df, raw, submitted, valid)."""
    st.subheader("Patient Information")

    with st.form("patient_form"):
        c1, c2, c3 = st.columns(3)

        # Column 1 — Demographic
        with c1:
            st.markdown("**Demographic**")
            sex = 1 if st.radio("Sex", ["Female", "Male"], horizontal=True) == "Male" else 0
            age_years = st.number_input("Age (years)", min_value=1, max_value=120, value=45, step=1)
            edu_label = st.selectbox("Education level", list(EDUCATION_OPTIONS.keys()), index=3)
            education = EDUCATION_OPTIONS[edu_label]

        # Column 2 — Body measurement
        with c2:
            st.markdown("**Body measurement**")
            height_cm = st.number_input("Height (cm)", min_value=80.0, max_value=250.0, value=165.0, step=0.5)
            weight_kg = st.number_input("Weight (kg)", min_value=20.0, max_value=300.0, value=70.0, step=0.5)
            bmi = calculate_bmi(weight_kg, height_cm)
            st.metric("Auto-calculated BMI", f"{bmi}" if bmi else "—")

        # Column 3 — Health conditions
        with c3:
            st.markdown("**Health conditions**")
            highbp = 1 if st.radio("High blood pressure?", ["No", "Yes"], horizontal=True) == "Yes" else 0
            highchol = 1 if st.radio("High cholesterol?", ["No", "Yes"], horizontal=True) == "Yes" else 0

        # Lifestyle (section below, spread across columns)
        st.markdown("**Lifestyle**")
        l1, l2, l3, l4, l5 = st.columns(5)
        with l1:
            smoker = 1 if st.radio("Smoking history", ["No", "Yes"], horizontal=True) == "Yes" else 0
        with l2:
            phys = 1 if st.radio("Physical activity", ["Yes", "No"], horizontal=True) == "Yes" else 0
        with l3:
            fruits = 1 if st.radio("Eats fruit", ["Yes", "No"], horizontal=True) == "Yes" else 0
        with l4:
            veggies = 1 if st.radio("Eats vegetables", ["Yes", "No"], horizontal=True) == "Yes" else 0
        with l5:
            alcohol = 1 if st.radio("Heavy alcohol", ["No", "Yes"], horizontal=True) == "Yes" else 0

        st.markdown("")
        submitted = st.form_submit_button("Predict Diabetes Risk", type="primary", use_container_width=True)

    # Validation
    valid = True
    if bmi is None or height_cm <= 0 or weight_kg <= 0:
        st.warning("Please enter valid height and weight values.")
        valid = False

    age_group = steps_age_to_brfss_group(age_years)
    raw = {
        'Sex': sex, 'AgeYears': age_years, 'AgeGroup': age_group, 'Education': education,
        'HighBP': highbp, 'HighChol': highchol, 'BMI': bmi if bmi else 0.0,
        'Smoker': smoker, 'PhysActivity': phys, 'Fruits': fruits, 'Veggies': veggies,
        'HvyAlcoholConsump': alcohol, 'WeightKg': weight_kg, 'HeightCm': height_cm,
    }
    row = {
        'Sex': sex, 'Age': age_group, 'Education': education,
        'HighBP': highbp, 'HighChol': highchol, 'BMI': bmi if bmi else 0.0,
        'Smoker': smoker, 'PhysActivity': phys, 'Fruits': fruits,
        'Veggies': veggies, 'HvyAlcoholConsump': alcohol,
    }
    input_df = pd.DataFrame([row])[EXPECTED_FEATURES]
    return input_df, raw, submitted, valid


def display_result(prob, label, threshold, tier):
    """Render result cards: prediction, probability, threshold, risk tier."""
    pct, thr_pct = prob * 100, threshold * 100
    color, bg = RISK_COLORS[tier], RISK_BG[tier]
    is_risk = label == "Diabetes Risk"
    label_col = RISK_COLORS['High'] if is_risk else RISK_COLORS['Low']

    st.markdown("### Prediction result")
    rc1, rc2 = st.columns([1.3, 1])

    with rc1:
        st.markdown(
            f"""
            <div style="background:{bg};border-left:8px solid {color};border-radius:10px;
                        padding:18px 22px;">
                <div style="font-size:14px;color:#555;">Final prediction</div>
                <div style="font-size:28px;font-weight:800;color:{label_col};margin:2px 0 10px 0;">
                    {label}
                </div>
                <div style="font-size:15px;color:#333;">
                    Model-predicted diabetes risk:
                    <b style="font-size:22px;color:{color};">{pct:.0f}%</b>
                </div>
                <div style="font-size:13px;color:#666;margin-top:4px;">
                    Decision threshold: <b>{threshold:.2f}</b> &nbsp;→&nbsp;
                    {pct:.0f}% {"≥" if is_risk else "<"} {thr_pct:.0f}% ⇒ <b>{label}</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(min(max(prob, 0.0), 1.0))

    with rc2:
        st.markdown(
            f"""
            <div style="background:{bg};border-radius:10px;padding:18px 20px;text-align:center;">
                <div style="font-size:14px;color:#555;">Risk tier (demo)</div>
                <div style="font-size:30px;font-weight:800;color:{color};">{tier}</div>
                <div style="font-size:12px;color:#777;margin-top:6px;">
                    Low &lt;30% · Moderate 30–70% · High ≥70%
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.caption(
        "Risk tiers are **model-based communication categories** and are **not clinically "
        "validated diagnostic cut-offs**."
    )


# ──────────────────────────────────────────────────────────────────────────────
# Tabs
# ──────────────────────────────────────────────────────────────────────────────
def single_patient_tab(art):
    model, scaler         = art['model'], art['scaler']
    threshold             = art['threshold'] if art['threshold'] else DEFAULT_THRESHOLD
    feature_names, proxy  = art['feature_names'], art['proxy']

    st.info("Fill in the patient information below and click **Predict Diabetes Risk**.")

    input_df, raw, submitted, valid = collect_user_input()

    if not submitted:
        return
    if not valid:
        st.stop()

    st.caption(
        f"Computed BMI: **{raw['BMI']}** (from {raw['WeightKg']} kg, {raw['HeightCm']} cm) · "
        f"Age {raw['AgeYears']} → BRFSS age group **{raw['AgeGroup']}**"
    )

    prob, label, X_scaled = predict_risk(input_df, scaler, model, threshold, feature_names)
    tier = assign_risk_tier(prob)

    display_result(prob, label, threshold, tier)

    # Why this prediction?
    st.markdown("### Why this prediction?")
    explanation = explain_prediction(X_scaled, proxy, feature_names, raw)
    if explanation['mode'] == 'shap' and MPL_AVAILABLE:
        st.markdown(
            "The final model is a **Soft Voting ensemble**, so SHAP cannot fully explain the "
            "whole ensemble. The chart below uses **XGBoost as a SHAP proxy** — the strongest "
            "individual tree-based model and one member of the ensemble — to indicate which "
            "inputs pushed this prediction up or down."
        )
        items  = explanation['items']
        labels = [it['feature'] for it in items][::-1]
        values = [it['value'] for it in items][::-1]
        colors = ['#C0392B' if v > 0 else '#2E7D32' for v in values]
        fig, ax = plt.subplots(figsize=(7, 3.4))
        ax.barh(labels, values, color=colors)
        ax.axvline(0, color='black', lw=0.8)
        ax.set_xlabel("SHAP value (proxy) — right = increases risk, left = reduces risk")
        ax.set_title("Top contributing features (XGBoost proxy)")
        for sp in ax.spines.values():
            sp.set_edgecolor('#DDD')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.caption("Red = pushes risk up, green = pushes risk down. Proxy view — indicative, "
                   "not exact for the full ensemble.")
    else:
        st.markdown("A SHAP proxy was not available, so here is a simple **rule-based explanation** "
                    "from your higher-risk inputs:")
        for r in explanation['items']:
            st.markdown(f"- {r}")

    # Recommendations
    st.markdown("### Actionable recommendations")
    for rec in generate_recommendations(raw):
        st.markdown(f"- {rec}")

    # Disclaimer
    st.divider()
    st.warning(f"⚕️ **Medical disclaimer** — {DISCLAIMER}")


def batch_tab(art):
    model, scaler = art['model'], art['scaler']
    threshold     = art['threshold'] if art['threshold'] else DEFAULT_THRESHOLD
    features      = art['feature_names']

    st.subheader("Batch CSV Upload")
    st.markdown(
        "Upload a CSV with **exactly these 11 columns** (already encoded as the model expects — "
        "Age as BRFSS group 1–13, binaries as 0/1):"
    )
    st.code(", ".join(features), language=None)

    # Template download
    template = pd.DataFrame(columns=features)
    st.download_button(
        "Download empty CSV template",
        data=template.to_csv(index=False).encode(),
        file_name="diabetes_input_template.csv",
        mime="text/csv",
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if not uploaded:
        return

    df_up = pd.read_csv(uploaded)
    missing = [f for f in features if f not in df_up.columns]
    if missing:
        st.error(f"CSV is missing required columns: {missing}")
        return

    st.write(f"Loaded **{len(df_up)}** rows. Preview:")
    st.dataframe(df_up.head(5), use_container_width=True)

    if st.button("Run Batch Prediction", type="primary"):
        X = scaler.transform(df_up[features])
        proba = model.predict_proba(X)[:, 1]
        out = df_up.copy()
        out['Model_Probability'] = proba.round(4)
        out['Prediction']        = np.where(proba >= threshold, "Diabetes Risk", "No Diabetes Risk")
        out['Risk_Tier']         = [assign_risk_tier(p) for p in proba]

        st.success(f"Predictions complete for {len(out)} rows (threshold = {threshold:.2f}).")

        dist = out['Risk_Tier'].value_counts().reindex(['Low', 'Moderate', 'High']).fillna(0).astype(int)
        ca, cb = st.columns([1, 2])
        with ca:
            st.markdown("**Risk tier distribution**")
            for r, cnt in dist.items():
                st.markdown(
                    f"<span style='color:{RISK_COLORS[r]};font-weight:700;'>{r}</span>: "
                    f"{cnt} ({cnt/len(out)*100:.1f}%)", unsafe_allow_html=True)
        with cb:
            if MPL_AVAILABLE:
                fig, ax = plt.subplots(figsize=(5, 2.6))
                bars = ax.bar(dist.index, dist.values, color=[RISK_COLORS[r] for r in dist.index])
                ax.bar_label(bars, fmt='%d')
                ax.set_ylabel("Count"); ax.set_title("Batch Risk Tier Distribution")
                plt.tight_layout(); st.pyplot(fig); plt.close(fig)

        st.dataframe(
            out[['Model_Probability', 'Prediction', 'Risk_Tier'] + features].head(25),
            use_container_width=True,
        )
        st.download_button(
            "Download full results CSV",
            data=out.to_csv(index=False).encode(),
            file_name="batch_predictions.csv",
            mime="text/csv",
        )
        st.caption(f"⚕️ {DISCLAIMER}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="Diabetes Risk Prediction", layout="centered")

    # Header
    st.markdown(
        "<h1 style='margin-bottom:0;'>Diabetes Risk Prediction</h1>"
        "<p style='font-size:16px;color:#444;margin:4px 0 2px 0;'>"
        "Explainable AI for Diabetes Risk Prediction and Actionable Insights</p>"
        "<p style='font-size:13px;color:#888;margin-top:0;'>"
        "For academic research and early screening support only, not medical diagnosis.</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    # Load artifacts first so the sidebar can show the real threshold
    try:
        art = load_artifacts()
    except FileNotFoundError as e:
        st.error(
            f"⚠️ {e}\n\nPlace these files in the **models/** folder:\n"
            "- best_model_2017.pkl\n- scaler_2017.pkl\n- threshold_2017.pkl\n- feature_names_2017.pkl"
        )
        st.stop()
    except Exception as e:
        st.error(f"⚠️ Failed to load model artifacts: {e}")
        st.stop()

    thr_display = art['threshold'] if art['threshold'] else DEFAULT_THRESHOLD

    # Sidebar = project information only
    with st.sidebar:
        st.markdown("### Project information")
        st.markdown(
            "- **Model:** Soft Voting Ensemble\n"
            "  - XGBoost · Logistic Regression · CatBoost\n"
            "- **Training data:** BRFSS 2017\n"
            "- **Test data:** Cambodia STEPS\n"
            f"- **Threshold:** {thr_display:.2f}\n"
            "- **Features:** 11"
        )
        st.divider()
        st.caption(f"⚕️ {DISCLAIMER}")

    # Feature-order validation
    if list(art['feature_names']) != EXPECTED_FEATURES:
        st.error(
            "⚠️ Feature order mismatch.\n\n"
            f"Expected: {EXPECTED_FEATURES}\n\nLoaded: {list(art['feature_names'])}"
        )
        st.stop()

    # Tabs
    tab1, tab2 = st.tabs(["Single Patient Prediction", "Batch CSV Upload"])
    with tab1:
        single_patient_tab(art)
    with tab2:
        batch_tab(art)


if __name__ == "__main__":
    main()
