# Fraud Detection

Project: a small fraud detection demo that trains a scikit-learn pipeline from a CSV dataset and exposes a lightweight Streamlit app for scoring transactions.

**Purpose:** Train and evaluate a classifier to flag potentially fraudulent records, and provide a simple UI to inspect predictions.

**Contents**
- `AIML Dataset.csv` — raw data used for EDA and model training.
- `analysis_model.ipynb` — exploratory analysis, preprocessing, model training, and evaluation.
- `fraud_detection.py` — Streamlit app that loads the saved pipeline and serves predictions.
- `fraud_detection_pipeline.pkl` — serialized preprocessing + model pipeline (artifact).
- `requirements.txt` — Python dependencies.
- `.gitignore` — ignored files and folders.

**Quickstart**
1. Create and activate a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
```
2. Install dependencies:
```bash
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```
3. Run the notebook for EDA and training (optional):
```bash
jupyter notebook analysis_model.ipynb
```
4. Run the Streamlit demo:
```bash
streamlit run fraud_detection.py
```
