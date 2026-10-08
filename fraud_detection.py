import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------------------------------------------------------
# Page Setup & Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Sentinel AI | Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Trained Model Pipeline
@st.cache_resource
def load_pipeline():
    return joblib.load("fraud_detection_pipeline.pkl")

try:
    model = load_pipeline()
    model_loaded = True
except Exception as e:
    model_loaded = False
    model_error = str(e)

# Custom Styling (Responsive & Mobile-First)
st.markdown("""
<style>
    /* Global & Container Spacing */
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Typography Fluid Sizing */
    .main-header {
        font-size: clamp(1.5rem, 4vw, 2.3rem);
        font-weight: 800;
        letter-spacing: -0.02em;
        margin-bottom: 0.2rem;
        background: linear-gradient(90deg, #1E3A8A, #2563EB, #3B82F6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        color: #64748B;
        font-size: clamp(0.85rem, 2vw, 1.05rem);
        margin-bottom: 1.2rem;
    }
    
    /* Risk Outcome Box Styling */
    .risk-box-safe {
        background-color: rgba(16, 185, 129, 0.1);
        border: 1px solid #10B981;
        border-radius: 12px;
        padding: 1.25rem;
        color: #065F46;
        margin-top: 1rem;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.08);
    }
    .risk-box-danger {
        background-color: rgba(239, 68, 68, 0.1);
        border: 1px solid #EF4444;
        border-radius: 12px;
        padding: 1.25rem;
        color: #991B1B;
        margin-top: 1rem;
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.08);
    }
    
    /* Tab header responsive scrolling for mobile */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        overflow-x: auto;
        white-space: nowrap;
        padding-bottom: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 14px;
        font-size: clamp(0.85rem, 2vw, 1rem);
        border-radius: 8px;
    }

    /* Mobile Responsive Breakpoint Adjustments */
    @media (max-width: 768px) {
        .main .block-container {
            padding-top: 1rem !important;
            padding-left: 0.75rem !important;
            padding-right: 0.75rem !important;
        }
        div[data-testid="column"] {
            width: 100% !important;
            flex: 1 1 100% !important;
            min-width: 100% !important;
            margin-bottom: 0.5rem;
        }
        .stButton > button {
            width: 100% !important;
            font-size: 1rem !important;
            padding: 0.65rem !important;
        }
        .risk-box-safe, .risk-box-danger {
            padding: 1rem !important;
        }
        .risk-box-safe h3, .risk-box-danger h3 {
            font-size: 1.2rem !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Sidebar: System Status & Model Performance Summary
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/shield-gradient.png", width=64)
    st.title("Sentinel AI Engine")
    st.caption("Real-Time Financial Transaction Classifier")
    
    st.divider()
    
    st.subheader("📊 Model Diagnostics")
    st.markdown("**Algorithm:** Logistic Regression (Balanced)")
    st.markdown("**Dataset Size:** 6,362,620 transactions")
    
    col_sb1, col_sb2 = st.columns(2)
    col_sb1.metric("ROC-AUC", "98.86%")
    col_sb2.metric("Recall", "93.71%")
    
    col_sb3, col_sb4 = st.columns(2)
    col_sb3.metric("Accuracy", "94.55%")
    col_sb4.metric("Precision", "2.18%")

    st.divider()
    
    st.subheader("💡 Quick Preset Scenarios")
    preset = st.radio(
        "Load Preset Data:",
        ["Custom Input", "Legitimate Payment", "Suspicious Drain (Fraud)", "High-Value Transfer (Fraud)"],
        index=0
    )
    
    st.info("ℹ️ Note: High Recall (~93.7%) prioritizes catching fraud over precision to minimize false negatives.")

# Default preset values
default_type = "PAYMENT"
default_amount = 1000.00
default_old_orig = 10000.00
default_new_orig = 9000.00
default_old_dest = 0.00
default_new_dest = 0.00

if preset == "Legitimate Payment":
    default_type = "PAYMENT"
    default_amount = 1864.28
    default_old_orig = 21249.00
    default_new_orig = 19384.72
    default_old_dest = 0.00
    default_new_dest = 0.00
elif preset == "Suspicious Drain (Fraud)":
    default_type = "CASH_OUT"
    default_amount = 181.00
    default_old_orig = 181.00
    default_new_orig = 0.00
    default_old_dest = 21182.00
    default_new_dest = 0.00
elif preset == "High-Value Transfer (Fraud)":
    default_type = "TRANSFER"
    default_amount = 2806000.00
    default_old_orig = 2806000.00
    default_new_orig = 0.00
    default_old_dest = 0.00
    default_new_dest = 0.00

# -----------------------------------------------------------------------------
# Main Application Content
# -----------------------------------------------------------------------------
st.markdown("<div class='main-header'>🛡️ Sentinel AI — Fraud Detection</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Analyze transactions for financial fraud risks using machine learning pipeline</div>", unsafe_allow_html=True)

if not model_loaded:
    st.error(f"❌ Failed to load model artifact `fraud_detection_pipeline.pkl`: {model_error}")
    st.stop()

tab1, tab2, tab3 = st.tabs(["⚡ Live Transaction Inspector", "📊 Model Evaluation Metrics", "📁 Batch CSV Scoring"])

# TAB 1: Live Inspector
with tab1:
    st.markdown("### 📝 Transaction Parameters")
    
    with st.container():
        col1, col2 = st.columns(2, gap="large")
        
        with col1:
            st.markdown("##### 💳 Sender Account Details")
            transaction_type = st.selectbox(
                "Transaction Type",
                ["PAYMENT", "TRANSFER", "CASH_OUT", "DEPOSIT", "DEBIT"],
                index=["PAYMENT", "TRANSFER", "CASH_OUT", "DEPOSIT", "DEBIT"].index(default_type) if default_type in ["PAYMENT", "TRANSFER", "CASH_OUT", "DEPOSIT", "DEBIT"] else 0,
                help="Category of the banking transaction"
            )
            amount = st.number_input(
                "Transaction Amount ($)",
                min_value=0.0,
                value=default_amount,
                step=100.0,
                format="%.2f"
            )
            oldbalanceOrg = st.number_input(
                "Sender Balance (Before Transaction)",
                min_value=0.0,
                value=default_old_orig,
                step=500.0,
                format="%.2f"
            )
            newbalanceOrig = st.number_input(
                "Sender Balance (After Transaction)",
                min_value=0.0,
                value=default_new_orig,
                step=500.0,
                format="%.2f"
            )
            
            # Live Validation / Info
            expected_orig_bal = max(0.0, oldbalanceOrg - amount) if transaction_type in ["PAYMENT", "TRANSFER", "CASH_OUT"] else oldbalanceOrg + amount
            orig_diff = abs(newbalanceOrig - expected_orig_bal)
            if orig_diff > 0.01:
                st.warning(f"⚠️ Balance anomaly: Expected new sender balance is ~${expected_orig_bal:,.2f} (Mismatch: ${orig_diff:,.2f})")

        with col2:
            st.markdown("##### 🏦 Receiver Account Details")
            oldbalanceDest = st.number_input(
                "Receiver Balance (Before Transaction)",
                min_value=0.0,
                value=default_old_dest,
                step=500.0,
                format="%.2f"
            )
            newbalanceDest = st.number_input(
                "Receiver Balance (After Transaction)",
                min_value=0.0,
                value=default_new_dest,
                step=500.0,
                format="%.2f"
            )
            
            expected_dest_bal = oldbalanceDest + amount if transaction_type in ["TRANSFER", "CASH_OUT", "PAYMENT"] else oldbalanceDest
            dest_diff = abs(newbalanceDest - expected_dest_bal)
            if dest_diff > 0.01:
                st.caption(f"ℹ️ Receiver Balance Δ mismatch: ${dest_diff:,.2f}")

    st.markdown("---")
    
    predict_btn = st.button("🔍 Run Fraud Risk Analysis", type="primary", use_container_width=True)

    if predict_btn:
        input_data = pd.DataFrame([{
            "type": transaction_type,
            "amount": amount,
            "oldbalanceOrg": oldbalanceOrg,
            "newbalanceOrig": newbalanceOrig,
            "oldbalanceDest": oldbalanceDest,
            "newbalanceDest": newbalanceDest
        }])

        prediction = model.predict(input_data)[0]
        
        # Calculate Risk Probabilities
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)[0]
            non_fraud_prob = probabilities[0] * 100
            fraud_prob = probabilities[1] * 100
        else:
            fraud_prob = 100.0 if prediction == 1 else 0.0
            non_fraud_prob = 100.0 - fraud_prob

        st.subheader("🎯 Risk Assessment Output")
        
        res_col1, res_col2 = st.columns([1, 2])
        
        with res_col1:
            st.metric("Fraud Risk Score", f"{fraud_prob:.1f}%")
            st.progress(fraud_prob / 100.0)
            
        with res_col2:
            if prediction == 1:
                st.markdown(f"""
                <div class="risk-box-danger">
                    <h3>⚠️ HIGH FRAUD RISK DETECTED</h3>
                    <p><b>Prediction Outcome:</b> Class 1 (Fraudulent Transaction)</p>
                    <p><b>Probability Score:</b> {fraud_prob:.2f}%</p>
                    <p><b>Recommended Action:</b> Hold transaction for secondary manual verification & security review.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="risk-box-safe">
                    <h3>✅ TRANSACTION APPEARS LEGITIMATE</h3>
                    <p><b>Prediction Outcome:</b> Class 0 (Normal Transaction)</p>
                    <p><b>Safety Probability:</b> {non_fraud_prob:.2f}%</p>
                    <p><b>Recommended Action:</b> Approve transaction for normal processing.</p>
                </div>
                """, unsafe_allow_html=True)

# TAB 2: Model Performance & Insights
with tab2:
    st.markdown("### 📈 Evaluation Metrics Summary")
    st.markdown("The underlying pipeline is trained on **6.36M transaction records** using **Logistic Regression** with class balancing to tackle extreme target imbalance.")
    
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    m_col1.metric("ROC-AUC Score", "98.86%", help="Area under the ROC curve on unseen test split")
    m_col2.metric("Fraud Recall", "93.71%", help="Proportion of actual frauds correctly flagged")
    m_col3.metric("Model Accuracy", "94.55%", help="Overall classification accuracy")
    m_col4.metric("Fraud Precision", "2.18%", help="Low precision due to heavy class weight tuning for high recall")
    
    st.markdown("---")
    st.markdown("#### 🧪 Confusion Matrix Details (Test Set = 1,908,786 rows)")
    
    cm_col1, cm_col2 = st.columns(2)
    with cm_col1:
        st.markdown("""
        | Outcome | Count | Description |
        |---|---|---|
        | **True Negatives (TN)** | 1,802,509 | Correctly identified normal transactions |
        | **False Positives (FP)** | 103,813 | Normal transactions flagged for review |
        | **False Negatives (FN)** | 155 | Missed fraudulent transactions |
        | **True Positives (TP)** | 2,309 | Correctly caught fraudulent transactions |
        """)
    with cm_col2:
        st.info("""
        💡 **Trade-off Analysis:**
        In financial fraud detection, **False Negatives** (missing real fraud) are exponentially more expensive than **False Positives** (flagging safe transactions for quick review). 
        The model is tuned to guarantee **~93.7% Recall** to catch fraudulent operations immediately.
        """)

# TAB 3: Batch Scoring
with tab3:
    st.markdown("### 📁 Upload Batch Transaction File (CSV)")
    uploaded_file = st.file_uploader("Upload CSV containing transaction features (`type`, `amount`, `oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest`)", type=["csv"])
    
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.write("##### 📄 Preview Uploaded Data (First 5 rows):")
            st.dataframe(batch_df.head(), use_container_width=True)
            
            required_cols = {"type", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"}
            if required_cols.issubset(set(batch_df.columns)):
                if st.button("🚀 Score Batch Transactions"):
                    with st.spinner("Processing batch predictions..."):
                        preds = model.predict(batch_df[list(required_cols)])
                        probas = model.predict_proba(batch_df[list(required_cols)])[:, 1] if hasattr(model, "predict_proba") else [0]*len(preds)
                        
                        batch_df["Fraud_Prediction"] = preds
                        batch_df["Fraud_Risk_Score_%"] = (probas * 100).round(2)
                        
                        fraud_count = int((preds == 1).sum())
                        st.success(f"Successfully processed {len(batch_df):,} transactions. Detected {fraud_count:,} high-risk transactions.")
                        
                        st.dataframe(batch_df, use_container_width=True)
                        
                        csv_data = batch_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Scored CSV Results",
                            data=csv_data,
                            file_name="scored_fraud_predictions.csv",
                            mime="text/csv"
                        )
            else:
                st.error(f"Missing required columns in CSV! Required: `{list(required_cols)}`")
        except Exception as batch_e:
            st.error(f"Error processing CSV: {batch_e}")