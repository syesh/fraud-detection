import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------------------------------------------------------
# 1. Page Configuration & Theme
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Sentinel AI | Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# 2. Pipeline Loading with Caching
# -----------------------------------------------------------------------------
@st.cache_resource
def load_pipeline():
    return joblib.load("fraud_detection_pipeline.pkl")

try:
    model = load_pipeline()
    model_loaded = True
except Exception as e:
    model_loaded = False
    model_error = str(e)

# -----------------------------------------------------------------------------
# 3. Custom Modern CSS
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Global Container */
    .main .block-container {
        padding-top: 1.25rem;
        padding-bottom: 2.5rem;
        max-width: 1250px;
    }
    
    /* Hero Banner */
    .hero-container {
        background: linear-gradient(135deg, rgba(30, 58, 138, 0.08) 0%, rgba(37, 99, 235, 0.04) 100%);
        border: 1px solid rgba(59, 130, 246, 0.2);
        border-radius: 16px;
        padding: 1.5rem 1.75rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.03);
    }
    .hero-title {
        font-size: clamp(1.6rem, 3.5vw, 2.3rem);
        font-weight: 800;
        letter-spacing: -0.025em;
        margin: 0;
        background: linear-gradient(90deg, #1E3A8A 0%, #2563EB 50%, #0284C7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-subtitle {
        color: #475569;
        font-size: clamp(0.9rem, 1.8vw, 1.05rem);
        margin-top: 0.35rem;
        font-weight: 400;
    }
    .badge-chip {
        display: inline-block;
        padding: 0.2rem 0.65rem;
        font-size: 0.75rem;
        font-weight: 600;
        border-radius: 9999px;
        background-color: rgba(37, 99, 235, 0.12);
        color: #1D4ED8;
        border: 1px solid rgba(37, 99, 235, 0.25);
        margin-top: 0.5rem;
    }
    
    /* Card Container */
    .feature-card {
        background: rgba(255, 255, 255, 0.6);
        border: 1px solid rgba(226, 232, 240, 0.9);
        border-radius: 14px;
        padding: 1.25rem 1.4rem;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.04);
        margin-bottom: 1rem;
    }
    
    /* Result Banners */
    .risk-banner-danger {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.12) 0%, rgba(220, 38, 38, 0.05) 100%);
        border: 1.5px solid #EF4444;
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        color: #7F1D1D;
        margin-top: 1rem;
        box-shadow: 0 8px 24px rgba(239, 68, 68, 0.12);
    }
    .risk-banner-safe {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.12) 0%, rgba(5, 150, 105, 0.05) 100%);
        border: 1.5px solid #10B981;
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        color: #064E3B;
        margin-top: 1rem;
        box-shadow: 0 8px 24px rgba(16, 185, 129, 0.12);
    }
    .risk-banner-warning {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.12) 0%, rgba(217, 119, 6, 0.05) 100%);
        border: 1.5px solid #F59E0B;
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        color: #78350F;
        margin-top: 1rem;
        box-shadow: 0 8px 24px rgba(245, 158, 11, 0.12);
    }
    
    /* Confusion Matrix Card */
    .cm-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 6px;
        margin-top: 0.5rem;
    }
    .cm-cell {
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .cm-tn {
        background-color: rgba(16, 185, 129, 0.15);
        border: 1px solid rgba(16, 185, 129, 0.4);
    }
    .cm-tp {
        background-color: rgba(59, 130, 246, 0.15);
        border: 1px solid rgba(59, 130, 246, 0.4);
    }
    .cm-fp {
        background-color: rgba(245, 158, 11, 0.15);
        border: 1px solid rgba(245, 158, 11, 0.4);
    }
    .cm-fn {
        background-color: rgba(239, 68, 68, 0.15);
        border: 1px solid rgba(239, 68, 68, 0.4);
    }
    
    /* Responsive Buttons & Columns */
    @media (max-width: 768px) {
        .main .block-container {
            padding: 0.75rem !important;
        }
        div[data-testid="column"] {
            width: 100% !important;
            flex: 1 1 100% !important;
            min-width: 100% !important;
        }
        .hero-container {
            padding: 1rem !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 4. State Management for Presets
# -----------------------------------------------------------------------------
if "txn_type" not in st.session_state:
    st.session_state.txn_type = "PAYMENT"
if "txn_amount" not in st.session_state:
    st.session_state.txn_amount = 1000.00
if "old_orig" not in st.session_state:
    st.session_state.old_orig = 10000.00
if "new_orig" not in st.session_state:
    st.session_state.new_orig = 9000.00
if "old_dest" not in st.session_state:
    st.session_state.old_dest = 0.00
if "new_dest" not in st.session_state:
    st.session_state.new_dest = 0.00

def set_preset(t, a, oo, no, od, nd):
    st.session_state.txn_type = t
    st.session_state.txn_amount = float(a)
    st.session_state.old_orig = float(oo)
    st.session_state.new_orig = float(no)
    st.session_state.old_dest = float(od)
    st.session_state.new_dest = float(nd)

# -----------------------------------------------------------------------------
# 5. Sidebar Controls & Diagnostics
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 0.5rem;">
            <span style="font-size: 2.2rem;">🛡️</span>
            <div>
                <h2 style="margin: 0; font-size: 1.35rem; font-weight: 700; line-height: 1.2;">Sentinel AI</h2>
                <span style="font-size: 0.78rem; color: #64748B;">Fraud Detection Engine</span>
            </div>
        </div>
        <div style="margin-bottom: 1.2rem;">
            <span style="background: rgba(16,185,129,0.15); color: #065F46; padding: 2px 8px; border-radius: 9999px; font-size: 0.72rem; font-weight: 600; border: 1px solid rgba(16,185,129,0.3);">
                🟢 Pipeline v1.0 Active
            </span>
        </div>
    """, unsafe_allow_html=True)

    st.subheader("⚙️ Decision Threshold")
    threshold = st.slider(
        "Fraud Probability Cutoff (%)",
        min_value=10,
        max_value=90,
        value=50,
        step=5,
        help="Transactions with risk probability >= this threshold are flagged as fraud."
    )
    st.caption(f"Currently flagging transactions at **≥ {threshold}%** risk probability.")
    
    st.divider()

    st.subheader("📋 Quick Presets")
    preset_choice = st.selectbox(
        "Load Predefined Scenarios:",
        ["Select a preset...", "🟢 Legitimate Small Payment", "🔴 Account Drain (Fraud)", "🚨 High-Value Outlier Transfer (Fraud)"],
        index=0
    )
    if preset_choice == "🟢 Legitimate Small Payment":
        set_preset("PAYMENT", 1864.28, 21249.00, 19384.72, 0.00, 0.00)
    elif preset_choice == "🔴 Account Drain (Fraud)":
        set_preset("CASH_OUT", 181.00, 181.00, 0.00, 21182.00, 0.00)
    elif preset_choice == "🚨 High-Value Outlier Transfer (Fraud)":
        set_preset("TRANSFER", 2806000.00, 2806000.00, 0.00, 0.00, 0.00)

    st.divider()

    st.subheader("📊 Engine Benchmarks")
    st.caption("Evaluated on 1,908,786 unseen test records:")
    c1, c2 = st.columns(2)
    c1.metric("ROC-AUC", "98.86%")
    c2.metric("Recall", "93.71%")
    c3, c4 = st.columns(2)
    c3.metric("Accuracy", "94.55%")
    c4.metric("Dataset", "6.36M")

    st.divider()
    st.markdown("<span style='font-size:0.75rem; color:#94A3B8;'>Sentinel AI • Secure Financial Pipeline</span>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 6. Main Hero Banner
# -----------------------------------------------------------------------------
st.markdown("""
<div class="hero-container">
    <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 10px;">
        <div>
            <h1 class="hero-title">🛡️ Sentinel AI — Fraud Detection</h1>
            <p class="hero-subtitle">Real-time machine learning inference for identifying suspicious financial transactions.</p>
        </div>
        <div style="text-align: right;">
            <span class="badge-chip">🚀 High-Recall Class-Balanced Model</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.error(f"❌ Failed to load model artifact `fraud_detection_pipeline.pkl`: {model_error}")
    st.stop()

# -----------------------------------------------------------------------------
# 7. Main Navigation Tabs
# -----------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "⚡ Live Transaction Inspector", 
    "📈 Model Diagnostics & Explainability", 
    "📁 Batch CSV Scoring Engine"
])

# =============================================================================
# TAB 1: Live Transaction Inspector
# =============================================================================
with tab1:
    # Preset quick-action pills
    st.markdown("##### ⚡ Quick Scenario Loader")
    col_p1, col_p2, col_p3, col_p4 = st.columns(4)
    with col_p1:
        if st.button("🟢 Safe Payment ($1.8k)", use_container_width=True):
            set_preset("PAYMENT", 1864.28, 21249.00, 19384.72, 0.00, 0.00)
            st.rerun()
    with col_p2:
        if st.button("🔴 Account Drain ($181)", use_container_width=True):
            set_preset("CASH_OUT", 181.00, 181.00, 0.00, 21182.00, 0.00)
            st.rerun()
    with col_p3:
        if st.button("🚨 Outlier Transfer ($2.8M)", use_container_width=True):
            set_preset("TRANSFER", 2806000.00, 2806000.00, 0.00, 0.00, 0.00)
            st.rerun()
    with col_p4:
        if st.button("🔄 Reset to Default", use_container_width=True):
            set_preset("PAYMENT", 1000.00, 10000.00, 9000.00, 0.00, 0.00)
            st.rerun()

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
    
    # Input Form Cards
    col_sender, col_receiver = st.columns(2, gap="large")
    
    with col_sender:
        st.markdown("""
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 0.6rem;">
            <span style="font-size: 1.25rem;">💳</span>
            <span style="font-size: 1.1rem; font-weight: 700; color: #1E293B;">Sender (Origin) Account</span>
        </div>
        """, unsafe_allow_html=True)
        
        type_options = ["PAYMENT", "TRANSFER", "CASH_OUT", "DEPOSIT", "DEBIT"]
        default_idx = type_options.index(st.session_state.txn_type) if st.session_state.txn_type in type_options else 0
        
        txn_type = st.selectbox(
            "Transaction Type",
            type_options,
            index=default_idx,
            help="High-risk types identified in training: TRANSFER and CASH_OUT"
        )
        
        amount = st.number_input(
            "Transaction Amount ($)",
            min_value=0.01,
            value=st.session_state.txn_amount,
            step=100.0,
            format="%.2f"
        )
        
        oldbalanceOrg = st.number_input(
            "Sender Initial Balance ($)",
            min_value=0.0,
            value=st.session_state.old_orig,
            step=500.0,
            format="%.2f",
            help="Balance prior to this transaction"
        )
        
        newbalanceOrig = st.number_input(
            "Sender Balance Post-Transaction ($)",
            min_value=0.0,
            value=st.session_state.new_orig,
            step=500.0,
            format="%.2f",
            help="Balance after transaction execution"
        )
        
        # Real-time Sender Balance Delta Validation
        expected_orig_bal = max(0.0, oldbalanceOrg - amount) if txn_type in ["PAYMENT", "TRANSFER", "CASH_OUT"] else oldbalanceOrg + amount
        orig_diff = abs(newbalanceOrig - expected_orig_bal)
        if orig_diff > 0.01:
            st.warning(f"⚠️ **Balance Mismatch:** Expected post-balance is **${expected_orig_bal:,.2f}** (Difference: ${orig_diff:,.2f})")
        else:
            st.caption("✅ Sender balance calculation aligns with transaction amount.")

    with col_receiver:
        st.markdown("""
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 0.6rem;">
            <span style="font-size: 1.25rem;">🏦</span>
            <span style="font-size: 1.1rem; font-weight: 700; color: #1E293B;">Receiver (Destination) Account</span>
        </div>
        """, unsafe_allow_html=True)
        
        oldbalanceDest = st.number_input(
            "Receiver Initial Balance ($)",
            min_value=0.0,
            value=st.session_state.old_dest,
            step=500.0,
            format="%.2f",
            help="Recipient balance before transaction"
        )
        
        newbalanceDest = st.number_input(
            "Receiver Balance Post-Transaction ($)",
            min_value=0.0,
            value=st.session_state.new_dest,
            step=500.0,
            format="%.2f",
            help="Recipient balance after transaction"
        )
        
        # Real-time Receiver Delta
        expected_dest_bal = oldbalanceDest + amount if txn_type in ["TRANSFER", "CASH_OUT", "PAYMENT"] else oldbalanceDest
        dest_diff = abs(newbalanceDest - expected_dest_bal)
        if dest_diff > 0.01:
            st.info(f"ℹ️ **Receiver Delta:** Post-balance differs by ${dest_diff:,.2f} from expected credit (common in merchant/clearing accounts).")
        else:
            st.caption("✅ Receiver balance reflects exact credit amount.")

        # Account Drain Flag
        if oldbalanceOrg > 0 and newbalanceOrig == 0.0 and txn_type in ["TRANSFER", "CASH_OUT"]:
            st.error("🚨 **Total Liquidation:** Sender account is being drained to $0.00!")

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    
    run_btn = st.button("🔍 Run Fraud Risk Analysis", type="primary", use_container_width=True)

    if run_btn:
        input_data = pd.DataFrame([{
            "type": txn_type,
            "amount": amount,
            "oldbalanceOrg": oldbalanceOrg,
            "newbalanceOrig": newbalanceOrig,
            "oldbalanceDest": oldbalanceDest,
            "newbalanceDest": newbalanceDest
        }])

        raw_pred = model.predict(input_data)[0]
        
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(input_data)[0]
            non_fraud_prob = probs[0] * 100.0
            fraud_prob = probs[1] * 100.0
        else:
            fraud_prob = 100.0 if raw_pred == 1 else 0.0
            non_fraud_prob = 100.0 - fraud_prob

        # Threshold Comparison
        is_fraud = fraud_prob >= threshold

        st.markdown("<hr style='margin: 1.5rem 0;'>", unsafe_allow_html=True)
        st.subheader("🎯 Risk Assessment & Decision Output")

        # Top Metric Cards
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            st.metric("Fraud Risk Score", f"{fraud_prob:.1f}%")
        with metric_col2:
            st.metric("Legitimacy Score", f"{non_fraud_prob:.1f}%")
        with metric_col3:
            st.metric("Applied Threshold", f"{threshold}%")
        with metric_col4:
            st.metric("Decision", "FLAG FOR REVIEW" if is_fraud else "APPROVE")

        # Progress Visualizer
        st.progress(float(min(max(fraud_prob / 100.0, 0.0), 1.0)))

        # Risk Banner & Recommendations
        if is_fraud:
            st.markdown(f"""
            <div class="risk-banner-danger">
                <h3 style="margin-top: 0; font-size: 1.3rem;">⚠️ HIGH RISK FRAUD DETECTED</h3>
                <p><b>Risk Probability:</b> <span style="font-size: 1.15rem; font-weight: 700;">{fraud_prob:.2f}%</span> (exceeds operational threshold of {threshold}%)</p>
                <p><b>Automated Recommendation:</b> Place transaction on immediate hold and trigger secondary verification protocol.</p>
                <div style="margin-top: 0.8rem; padding-top: 0.8rem; border-top: 1px dashed rgba(239, 68, 68, 0.4);">
                    <b>⚡ Action Checklist:</b>
                    <ul style="margin-bottom: 0; padding-left: 1.2rem;">
                        <li>Suspend immediate disbursement.</li>
                        <li>Dispatch SMS / Biometric 2FA verification challenge to sender.</li>
                        <li>Escalate origin account <code>{txn_type}</code> profile for anti-money laundering (AML) pattern review.</li>
                    </ul>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="risk-banner-safe">
                <h3 style="margin-top: 0; font-size: 1.3rem;">✅ TRANSACTION VERIFIED & APPROVED</h3>
                <p><b>Safety Probability:</b> <span style="font-size: 1.15rem; font-weight: 700;">{non_fraud_prob:.2f}%</span> (Risk score {fraud_prob:.2f}% is below threshold of {threshold}%)</p>
                <p><b>Automated Recommendation:</b> Release funds and clear transaction for regular automated settlement.</p>
                <div style="margin-top: 0.8rem; padding-top: 0.8rem; border-top: 1px dashed rgba(16, 185, 129, 0.4);">
                    <b>⚡ Action Checklist:</b>
                    <ul style="margin-bottom: 0; padding-left: 1.2rem;">
                        <li>Process automated clearing and settlement.</li>
                        <li>Record cryptographic audit log with Risk Score: {fraud_prob:.2f}%.</li>
                    </ul>
                </div>
            </div>
            """, unsafe_allow_html=True)

# =============================================================================
# TAB 2: Model Performance & Explainability
# =============================================================================
with tab2:
    st.markdown("### 📊 Comprehensive Model Diagnostics")
    st.markdown("""
    The detection engine uses a **scikit-learn Pipeline** combining **StandardScaler**, **OneHotEncoder**, and **class-weighted Logistic Regression** trained on **6,362,620 real-world simulated transactions**.
    """)

    # Metric Row
    row1, row2, row3, row4 = st.columns(4)
    row1.metric("ROC-AUC Score", "98.86%", help="Separability metric on stratified test split")
    row2.metric("Fraud Recall", "93.71%", help="Catches ~94 of every 100 fraudulent transactions")
    row3.metric("Overall Accuracy", "94.55%", help="Baseline accuracy across all transaction classes")
    row4.metric("Missed Fraud Rate", "0.008%", help="Only 155 of 1.9M test cases missed")

    st.markdown("<hr style='margin: 1.5rem 0;'>", unsafe_allow_html=True)

    # Confusion Matrix Visual & Domain Explanation
    col_cm, col_tradeoff = st.columns([1.1, 1], gap="large")

    with col_cm:
        st.markdown("#### 🧪 Confusion Matrix (Test Split: 1,908,786 rows)")
        st.markdown("""
        <table class="cm-table">
            <tr>
                <td class="cm-cell cm-tn">
                    <div style="font-size: 0.75rem; color: #065F46; font-weight: 600;">TRUE NEGATIVE (TN)</div>
                    <div style="font-size: 1.25rem; font-weight: 800; color: #064E3B;">1,802,509</div>
                    <div style="font-size: 0.72rem; color: #047857;">Normal Approved</div>
                </td>
                <td class="cm-cell cm-fp">
                    <div style="font-size: 0.75rem; color: #92400E; font-weight: 600;">FALSE POSITIVE (FP)</div>
                    <div style="font-size: 1.25rem; font-weight: 800; color: #78350F;">103,813</div>
                    <div style="font-size: 0.72rem; color: #B45309;">Held for Verification</div>
                </td>
            </tr>
            <tr>
                <td class="cm-cell cm-fn">
                    <div style="font-size: 0.75rem; color: #991B1B; font-weight: 600;">FALSE NEGATIVE (FN)</div>
                    <div style="font-size: 1.25rem; font-weight: 800; color: #7F1D1D;">155</div>
                    <div style="font-size: 0.72rem; color: #B91C1C;">Missed Frauds</div>
                </td>
                <td class="cm-cell cm-tp">
                    <div style="font-size: 0.75rem; color: #1E40AF; font-weight: 600;">TRUE POSITIVE (TP)</div>
                    <div style="font-size: 1.25rem; font-weight: 800; color: #1E3A8A;">2,309</div>
                    <div style="font-size: 0.72rem; color: #2563EB;">Caught Frauds</div>
                </td>
            </tr>
        </table>
        """, unsafe_allow_html=True)

    with col_tradeoff:
        st.markdown("#### 💡 Asymmetric Risk in Fraud Systems")
        st.markdown("""
        In enterprise financial security, the **cost of a False Negative** (unauthorized transfer of hundreds of thousands of dollars) is exponentially higher than the cost of a **False Positive** (a 30-second SMS step-up challenge for a genuine user).
        
        - 🛡️ **Tuned for Safety:** `class_weight='balanced'` directs the optimizer to penalize missed frauds heavily.
        - 📈 **Result:** Over **93.7%** of all fraudulent transactions are intercepted immediately before settlement.
        """)

    st.markdown("<hr style='margin: 1.5rem 0;'>", unsafe_allow_html=True)

    # Feature Importance / Pipeline Drivers
    st.markdown("#### 🔍 Model Feature Drivers & Odds Ratios")
    st.caption("Extracted directly from the serialized Logistic Regression pipeline coefficients:")
    
    coef_col1, coef_col2 = st.columns(2)
    with coef_col1:
        st.markdown("""
        **🚨 Top Fraud Escalators (Positive Weights):**
        - `TRANSFER` transaction type: **+72.38**
        - `CASH_OUT` transaction type: **+70.77**
        - High `oldbalanceOrg` (Large initial balance): **+65.34**
        - High `oldbalanceDest` (Target account balance): **+32.94**
        """)
    with coef_col2:
        st.markdown("""
        **🛡️ Top Legitimacy Anchors (Negative Weights):**
        - `PAYMENT` transaction type: **-89.04**
        - High `newbalanceOrig` (Remaining balance intact): **-63.75**
        - High `newbalanceDest` (Proper credited balance): **-35.73**
        - `DEBIT` transaction type: **-3.79**
        """)

# =============================================================================
# TAB 3: Batch CSV Scoring Engine
# =============================================================================
with tab3:
    st.markdown("### 📁 Batch Transaction File Scoring")
    st.markdown("Upload a CSV file containing transactions or test with our built-in pre-populated dataset.")

    # Template and Demo Bar
    col_t1, col_t2 = st.columns([1, 1])
    
    # Pre-generate sample CSV template
    sample_df = pd.DataFrame([
        {"type": "TRANSFER", "amount": 181.00, "oldbalanceOrg": 181.00, "newbalanceOrig": 0.00, "oldbalanceDest": 0.00, "newbalanceDest": 0.00},
        {"type": "PAYMENT", "amount": 1864.28, "oldbalanceOrg": 21249.00, "newbalanceOrig": 19384.72, "oldbalanceDest": 0.00, "newbalanceDest": 0.00},
        {"type": "CASH_OUT", "amount": 25000.00, "oldbalanceOrg": 25000.00, "newbalanceOrig": 0.00, "oldbalanceDest": 10000.00, "newbalanceDest": 35000.00},
        {"type": "PAYMENT", "amount": 54.12, "oldbalanceOrg": 5400.00, "newbalanceOrig": 5345.88, "oldbalanceDest": 0.00, "newbalanceDest": 0.00},
        {"type": "TRANSFER", "amount": 950000.00, "oldbalanceOrg": 950000.00, "newbalanceOrig": 0.00, "oldbalanceDest": 0.00, "newbalanceDest": 0.00},
        {"type": "DEPOSIT", "amount": 5000.00, "oldbalanceOrg": 2000.00, "newbalanceOrig": 7000.00, "oldbalanceDest": 0.00, "newbalanceDest": 0.00},
        {"type": "CASH_OUT", "amount": 181.00, "oldbalanceOrg": 181.00, "newbalanceOrig": 0.00, "oldbalanceDest": 21182.00, "newbalanceDest": 0.00},
        {"type": "DEBIT", "amount": 120.00, "oldbalanceOrg": 4000.00, "newbalanceOrig": 3880.00, "oldbalanceDest": 0.00, "newbalanceDest": 0.00},
    ])
    
    with col_t1:
        st.download_button(
            label="📥 Download Sample CSV Template",
            data=sample_df.to_csv(index=False).encode('utf-8'),
            file_name="sample_transactions_template.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col_t2:
        use_demo_data = st.checkbox("⚡ Load Built-in Demo Batch (8 Transactions)", value=False)

    uploaded_file = st.file_uploader(
        "Upload CSV file containing: `type`, `amount`, `oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest`",
        type=["csv"]
    )

    df_to_score = None
    if uploaded_file is not None:
        try:
            df_to_score = pd.read_csv(uploaded_file)
        except Exception as read_e:
            st.error(f"Error reading uploaded CSV: {read_e}")
    elif use_demo_data:
        df_to_score = sample_df.copy()

    if df_to_score is not None:
        st.markdown("##### 📄 Data Preview:")
        st.dataframe(df_to_score.head(5), use_container_width=True)

        req_cols = {"type", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"}
        if req_cols.issubset(set(df_to_score.columns)):
            if st.button("🚀 Score Entire Batch", type="primary", use_container_width=True):
                with st.spinner("Executing machine learning pipeline across batch..."):
                    features_df = df_to_score[list(req_cols)]
                    
                    if hasattr(model, "predict_proba"):
                        batch_probs = model.predict_proba(features_df)[:, 1]
                    else:
                        batch_probs = model.predict(features_df)

                    df_to_score["Fraud_Risk_%"] = (batch_probs * 100.0).round(2)
                    df_to_score["Decision"] = np.where(df_to_score["Fraud_Risk_%"] >= threshold, "🚨 FRAUD ALERT", "✅ APPROVED")
                    
                    # Batch Statistics Summary
                    total_count = len(df_to_score)
                    fraud_count = int((df_to_score["Decision"] == "🚨 FRAUD ALERT").sum())
                    clean_count = total_count - fraud_count
                    fraud_vol = df_to_score[df_to_score["Decision"] == "🚨 FRAUD ALERT"]["amount"].sum()
                    total_vol = df_to_score["amount"].sum()

                    st.markdown("<hr style='margin: 1rem 0;'>", unsafe_allow_html=True)
                    st.subheader("📊 Batch Scoring Results Summary")

                    bc1, bc2, bc3, bc4 = st.columns(4)
                    bc1.metric("Total Transactions", f"{total_count:,}")
                    bc2.metric("Flagged Fraud Count", f"{fraud_count:,}", f"{(fraud_count/total_count)*100:.1f}%")
                    bc3.metric("Volume at Risk", f"${fraud_vol:,.2f}")
                    bc4.metric("Total Scored Volume", f"${total_vol:,.2f}")

                    # Filter Views
                    filter_view = st.radio(
                        "Filter Results:",
                        ["Show All Records", "Only Flagged Fraud", "Only Approved"],
                        horizontal=True
                    )
                    
                    if filter_view == "Only Flagged Fraud":
                        display_df = df_to_score[df_to_score["Decision"] == "🚨 FRAUD ALERT"]
                    elif filter_view == "Only Approved":
                        display_df = df_to_score[df_to_score["Decision"] == "✅ APPROVED"]
                    else:
                        display_df = df_to_score

                    st.dataframe(
                        display_df,
                        column_config={
                            "Fraud_Risk_%": st.column_config.ProgressColumn(
                                "Risk Score (%)",
                                min_value=0.0,
                                max_value=100.0,
                                format="%.1f%%"
                            ),
                            "amount": st.column_config.NumberColumn("Amount ($)", format="$%.2f"),
                            "oldbalanceOrg": st.column_config.NumberColumn("Old Bal Orig", format="$%.2f"),
                            "newbalanceOrig": st.column_config.NumberColumn("New Bal Orig", format="$%.2f"),
                            "oldbalanceDest": st.column_config.NumberColumn("Old Bal Dest", format="$%.2f"),
                            "newbalanceDest": st.column_config.NumberColumn("New Bal Dest", format="$%.2f"),
                        },
                        use_container_width=True
                    )

                    export_csv = df_to_score.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Scored Batch Results (CSV)",
                        data=export_csv,
                        file_name="sentinel_ai_batch_scored.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
        else:
            missing = req_cols - set(df_to_score.columns)
            st.error(f"❌ Missing required columns in CSV: `{list(missing)}`")
