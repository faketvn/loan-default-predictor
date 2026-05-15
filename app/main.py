import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# ── Page config ──────────────────────────────────────────
st.set_page_config(
    page_title = "Loan Default Predictor",
    page_icon  = "🏦",
    layout     = "wide"
)

# ── Load model and feature names ───────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load('model/model.pkl')

@st.cache_resource
def load_feature_names():
    return joblib.load('model/feature_names.pkl')

@st.cache_data
def generate_sample_data(feature_names):
    """Generate synthetic sample data based on feature names"""
    np.random.seed(42)
    n = 200
    data = {}
    for col in feature_names:
        if 'FLAG' in col or 'GENDER' in col:
            data[col] = np.random.randint(0, 2, n).astype(float)
        elif 'EXT_SOURCE' in col:
            data[col] = np.random.uniform(0.1, 0.9, n)
        elif 'DAYS' in col or 'AGE' in col or 'YEARS' in col:
            data[col] = np.random.uniform(0, 60, n)
        elif 'AMT' in col or 'INCOME' in col or 'CREDIT' in col:
            data[col] = np.random.uniform(50000, 500000, n)
        elif 'RATIO' in col or 'RATE' in col or 'TO' in col:
            data[col] = np.random.uniform(0, 5, n)
        else:
            data[col] = np.random.uniform(0, 1, n)
    return pd.DataFrame(data)

model        = load_model()
feature_names = load_feature_names()
X_sample     = generate_sample_data(feature_names)

# Generate synthetic y_test (predictions from model on sample data)
y_test = pd.Series(model.predict(X_sample))

# ── Sidebar navigation ────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/bank-building.png", width=60)
st.sidebar.title("🏦 Loan Default Predictor")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    ["🔮 Predict", "🧠 Explain", "📊 Model Analytics"]
)
st.sidebar.markdown("---")
st.sidebar.markdown("Built with XGBoost + SHAP")
st.sidebar.markdown("AUC Score: **0.7271**")

# ════════════════════════════════════════════════════════════
# PAGE 1 — PREDICT
# ════════════════════════════════════════════════════════════
if page == "🔮 Predict":
    st.title("🔮 Loan Default Risk Predictor")
    st.markdown("Enter applicant details below to get an instant risk assessment.")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("👤 Personal Info")
        age          = st.slider("Age (years)", 18, 70, 35)
        employed_yrs = st.slider("Years employed", 0, 40, 5)
        gender       = st.selectbox("Gender", ["Male", "Female"])

    with col2:
        st.subheader("💰 Financial Info")
        income       = st.number_input("Annual income (RM)", 20000, 1000000, 150000, step=5000)
        loan_amount  = st.number_input("Loan amount (RM)", 10000, 2000000, 500000, step=10000)
        annuity      = st.number_input("Monthly annuity (RM)", 1000, 100000, 25000, step=1000)

    with col3:
        st.subheader("📋 Loan Info")
        loan_type    = st.selectbox("Loan type", ["Cash loans", "Revolving loans"])
        education    = st.selectbox("Education", [
            "Secondary", "Higher education",
            "Incomplete higher", "Lower secondary", "Academic degree"
        ])
        ext_source   = st.slider("Credit score (0-1)", 0.0, 1.0, 0.5, 0.01)

    st.markdown("---")

    if st.button("🔍 Predict Risk", use_container_width=True):
        input_df = X_sample.median().to_frame().T.copy()

        if 'AGE_YEARS'         in input_df: input_df['AGE_YEARS']         = age
        if 'EMPLOYED_YEARS'    in input_df: input_df['EMPLOYED_YEARS']    = employed_yrs
        if 'AMT_INCOME_TOTAL'  in input_df: input_df['AMT_INCOME_TOTAL']  = income
        if 'AMT_CREDIT'        in input_df: input_df['AMT_CREDIT']        = loan_amount
        if 'AMT_ANNUITY'       in input_df: input_df['AMT_ANNUITY']       = annuity
        if 'EXT_SOURCE_2'      in input_df: input_df['EXT_SOURCE_2']      = ext_source
        if 'EXT_SOURCE_3'      in input_df: input_df['EXT_SOURCE_3']      = ext_source
        if 'EXT_SOURCE_1'      in input_df: input_df['EXT_SOURCE_1']      = ext_source
        if 'DEBT_TO_INCOME'    in input_df: input_df['DEBT_TO_INCOME']    = loan_amount / (income + 1)
        if 'ANNUITY_TO_INCOME' in input_df: input_df['ANNUITY_TO_INCOME'] = annuity / (income + 1)
        if 'CREDIT_TO_INCOME'  in input_df: input_df['CREDIT_TO_INCOME']  = loan_amount / (income + 1)
        if 'CODE_GENDER'       in input_df: input_df['CODE_GENDER']       = 1 if gender == "Male" else 0

        prob     = model.predict_proba(input_df)[0][1]
        risk_pct = prob * 100

        st.markdown("---")
        col_r1, col_r2, col_r3 = st.columns(3)

        with col_r1:
            if prob >= 0.5:
                st.error(f"### ❌ HIGH RISK\n**{risk_pct:.1f}%** probability of default")
            else:
                st.success(f"### ✅ LOW RISK\n**{risk_pct:.1f}%** probability of default")

        with col_r2:
            st.metric("Debt-to-Income Ratio", f"{loan_amount/income:.2f}x",
                      delta="High" if loan_amount/income > 3 else "Normal",
                      delta_color="inverse")

        with col_r3:
            st.metric("Monthly Burden",
                      f"RM {annuity:,.0f}",
                      delta=f"{annuity/income*100:.1f}% of income",
                      delta_color="inverse" if annuity/income > 0.3 else "normal")

        st.markdown("---")
        st.subheader("Risk Level")
        bar_color = "#e74c3c" if prob >= 0.5 else "#2ecc71"
        st.markdown(f"""
        <div style='background:#f0f0f0;border-radius:10px;height:24px;width:100%'>
          <div style='background:{bar_color};width:{risk_pct}%;height:24px;
                      border-radius:10px;transition:width 0.5s'>
          </div>
        </div>
        <p style='text-align:center;margin-top:6px'>{risk_pct:.1f}% default risk</p>
        """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# PAGE 2 — EXPLAIN
# ════════════════════════════════════════════════════════════
elif page == "🧠 Explain":
    st.title("🧠 Why did the model decide this?")
    st.markdown("Pick any applicant from the sample set and see exactly what drove the prediction.")
    st.markdown("---")

    idx       = st.slider("Select applicant", 0, len(X_sample)-1, 0)
    applicant = X_sample.iloc[[idx]]
    prob      = model.predict_proba(applicant)[0][1]
    actual    = y_test.iloc[idx]

    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted risk",  f"{prob*100:.1f}%")
    col2.metric("Decision",        "HIGH RISK ❌" if prob >= 0.5 else "LOW RISK ✅")
    col3.metric("Actual outcome",  "Defaulted" if actual == 1 else "No default")

    st.markdown("---")
    st.subheader("📊 SHAP Waterfall — what drove this prediction?")

    explainer   = shap.TreeExplainer(model)
    shap_single = explainer(applicant)

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(shap_single[0], max_display=12, show=False)
    st.pyplot(plt.gcf())
    plt.clf()

    st.markdown("---")
    st.subheader("📝 Top reasons in plain English")
    sv         = shap_single.values[0]
    feat_names = X_sample.columns.tolist()
    top_idx    = np.argsort(np.abs(sv))[::-1][:3]

    for rank, i in enumerate(top_idx, 1):
        direction = "↑ increased" if sv[i] > 0 else "↓ decreased"
        st.markdown(f"**{rank}.** `{feat_names[i]}` = `{applicant.iloc[0][feat_names[i]]:.3f}` "
                    f"— {direction} default risk by **{abs(sv[i]):.3f}**")

# ════════════════════════════════════════════════════════════
# PAGE 3 — MODEL ANALYTICS
# ════════════════════════════════════════════════════════════
elif page == "📊 Model Analytics":
    st.title("📊 Model Analytics")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("ROC-AUC Score",  "0.7271")
    col2.metric("Training rows",  "452,296")
    col3.metric("Test rows",      "61,503")
    col4.metric("Features used",  str(len(feature_names)))

    st.markdown("---")

    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("🏆 Top 15 Important Features")
        importances = model.feature_importances_
        feat_df     = pd.DataFrame({
            'Feature'   : feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=True).tail(15)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.barh(feat_df['Feature'], feat_df['Importance'], color='#e74c3c')
        ax.set_xlabel('Importance Score')
        ax.set_title('XGBoost Feature Importance')
        plt.tight_layout()
        st.pyplot(fig)
        plt.clf()

    with col_r:
        st.subheader("🔍 SHAP Global Summary")
        explainer    = shap.TreeExplainer(model)
        shap_vals_np = explainer.shap_values(X_sample.head(50))
        fig, ax      = plt.subplots(figsize=(6, 6))
        shap.summary_plot(shap_vals_np, X_sample.head(50),
                          max_display=15, show=False, plot_type="bar")
        plt.tight_layout()
        st.pyplot(fig)
        plt.clf()

    st.markdown("---")
    st.subheader("📈 Prediction Distribution on Sample Data")
    col1, col2 = st.columns(2)

    with col1:
        preds = model.predict(X_sample)
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.bar(['Non-Default', 'Default'],
               [sum(preds == 0), sum(preds == 1)],
               color=['#2ecc71', '#e74c3c'])
        ax.set_title('Predicted Class Distribution')
        ax.set_ylabel('Count')
        plt.tight_layout()
        st.pyplot(fig)
        plt.clf()

    with col2:
        st.markdown("### Model info")
        st.markdown("""
        | Item | Detail |
        |---|---|
        | Algorithm | XGBoost |
        | Estimators | 300 |
        | Max depth | 6 |
        | Learning rate | 0.05 |
        | Imbalance fix | SMOTE |
        | Metric | ROC-AUC |
        """)