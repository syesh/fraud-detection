# 🛡️ Sentinel AI — Financial Fraud Detection Engine

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-Pipeline-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458.svg?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![ROC-AUC](https://img.shields.io/badge/ROC--AUC-98.86%25-success.svg)](#-model-evaluation--diagnostics)
[![Recall](https://img.shields.io/badge/Fraud%20Recall-93.71%25-brightgreen.svg)](#-model-evaluation--diagnostics)

An end-to-end machine learning system and interactive Streamlit web application designed to detect fraudulent financial transactions in real time. Trained on **6.36+ million transaction records**, Sentinel AI prioritizes fraud recall to catch suspicious patterns and balance anomalies before financial loss occurs.

---

## 🌟 Key Features

- **⚡ Live Transaction Inspector:**
  - Real-time scoring for single transactions across major transfer types (`PAYMENT`, `TRANSFER`, `CASH_OUT`, `DEPOSIT`, `DEBIT`).
  - Automated balance delta calculations flagging sender and receiver balance anomalies before model scoring.
  - Granular fraud probability percentage, visual progress risk meter, and clear operational recommendations (*Approve* vs *Hold for Security Review*).
- **💡 Built-in Preset Scenarios:**
  - One-click loading of realistic transaction profiles: *Legitimate Payment*, *Suspicious Drain (Fraud)*, and *High-Value Transfer (Fraud)*.
- **📁 Batch CSV Scoring & Export:**
  - Upload batch CSV datasets for instant multi-row inference.
  - Previews data, computes risk probability scores, flags fraud candidates, and exports scored results via one-click CSV download.
- **📊 Embedded Diagnostics Dashboard:**
  - In-app access to model evaluation metrics, ROC-AUC curves, confusion matrix breakdown, and the domain rationale behind class-weighted recall tuning.
- **📱 Responsive & Modern UI:**
  - Custom fluid CSS styling designed for both desktop workstations and mobile screens.

---

## 🏗️ System Architecture & ML Pipeline

```
Raw Data (6.36M Records)
        │
        ▼
Data Preprocessing & Feature Selection
  ├── Numeric Features: amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest
  │     └─► StandardScaler()
  └── Categorical Features: type (PAYMENT, TRANSFER, CASH_OUT, DEPOSIT, DEBIT)
        └─► OneHotEncoder(drop='first')
        │
        ▼
Scikit-Learn ColumnTransformer
        │
        ▼
Logistic Regression Classifier (class_weight='balanced', max_iter=1000)
        │
        ▼
Serialized Pipeline Artifact (fraud_detection_pipeline.pkl)
        │
        ▼
Streamlit Interactive Dashboard (fraud_detection.py)
  ├── Live Transaction Inspector
  ├── Model Evaluation Metrics & Insights
  └── Batch CSV Scoring Engine
```

### Why Balanced Class Weights?
Financial fraud represents only **~0.13%** of all transactions in the dataset. A standard model optimizing purely for raw accuracy could achieve >99.8% accuracy simply by classifying *every* transaction as non-fraudulent while missing 100% of actual crimes.

Sentinel AI implements **balanced class weighting**, penalizing False Negatives significantly harder. In financial security, the cost of reviewing a flagged legitimate transaction (False Positive) is minimal compared to the catastrophic loss of an undetected fraud operation (False Negative).

---

## 📈 Model Evaluation & Diagnostics

Evaluated on a **30% stratified test split** (**1,908,786 unseen transactions**):

| Metric | Score | Rationale |
|---|---|---|
| **ROC-AUC** | **98.86%** | Exceptional discrimination capability between legitimate and fraudulent transactions |
| **Fraud Recall** | **93.71%** | Flags ~94 out of every 100 fraudulent transactions |
| **Overall Accuracy** | **94.55%** | High baseline accuracy across millions of test records |
| **Fraud Precision** | **2.18%** | Expected trade-off: deliberately sensitive to minimize costly missed frauds |

### Confusion Matrix Breakdown (Test Set: 1,908,786 rows)

| Outcome | Count | Description |
|---|---|---|
| **True Negatives (TN)** | **1,802,509** | Correctly identified legitimate transactions |
| **True Positives (TP)** | **2,309** | Correctly caught fraudulent transactions |
| **False Positives (FP)** | **103,813** | Legitimate transactions held for secondary verification |
| **False Negatives (FN)** | **155** | Undetected fraudulent transactions |

---

## 📁 Repository Structure

```
fraud-detection/
├── AIML Dataset.csv               # Raw financial transaction dataset (~6.36M rows)
├── analysis_model.ipynb           # EDA, preprocessing, model training & evaluation
├── fraud_detection.py             # Sentinel AI Streamlit interactive application
├── fraud_detection_pipeline.pkl   # Serialized end-to-end preprocessing + ML pipeline
├── requirements.txt               # Project dependencies
├── .gitignore                     # Git exclusion rules
└── README.md                      # Project documentation
```

---

## 🚀 Quickstart Guide

### 1. Clone the Repository
```bash
git clone https://github.com/syesh/fraud-detection.git
cd fraud-detection
```

### 2. Create and Activate a Virtual Environment
```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

# Windows
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install Dependencies
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the Streamlit Application
```bash
streamlit run fraud_detection.py
```
Open your browser at `http://localhost:8501` to access the Sentinel AI dashboard.

### 5. Explore the Model Notebook (Optional)
To review data cleaning, exploratory data analysis, and the pipeline training process:
```bash
jupyter notebook analysis_model.ipynb
```

---

## 📄 Batch CSV Schema

To use the **Batch CSV Scoring** feature, upload a CSV file with the following columns:

| Column Name | Data Type | Example Value | Description |
|---|---|---|---|
| `type` | String | `TRANSFER` | Transaction type (`PAYMENT`, `TRANSFER`, `CASH_OUT`, `DEPOSIT`, `DEBIT`) |
| `amount` | Float | `150000.00` | Transaction value in currency units |
| `oldbalanceOrg` | Float | `150000.00` | Initial sender balance before transaction |
| `newbalanceOrig` | Float | `0.00` | Sender balance after transaction |
| `oldbalanceDest` | Float | `0.00` | Initial recipient balance before transaction |
| `newbalanceDest` | Float | `150000.00` | Recipient balance after transaction |

Example CSV snippet:
```csv
type,amount,oldbalanceOrg,newbalanceOrig,oldbalanceDest,newbalanceDest
TRANSFER,150000.00,150000.00,0.00,0.00,150000.00
PAYMENT,1864.28,21249.00,19384.72,0.00,0.00
CASH_OUT,181.00,181.00,0.00,21182.00,0.00
```

---

## 🛠️ Roadmap & Future Enhancements

- [ ] **Ensemble Gradient Boosters:** Train and benchmark XGBoost, LightGBM, and CatBoost models for improved precision without sacrificing recall.
- [ ] **Explainable AI (XAI):** Integrate SHAP (SHapley Additive exPlanations) in the Streamlit inspector to show feature impact per transaction.
- [ ] **REST API Layer:** Expose predictions as a lightweight FastAPI microservice for integration with payment processing backends.
- [ ] **Real-Time Stream Processing:** Implement Kafka / Redis streams for high-throughput streaming fraud ingestion.

---

## 📜 License

This project is open-source under the [MIT License](LICENSE).
