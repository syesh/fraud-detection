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

**How it works**
- Use the notebook to load `AIML Dataset.csv`, clean and engineer features, and train a scikit-learn pipeline.
- Save the trained pipeline to `fraud_detection_pipeline.pkl`.
- The Streamlit app loads that pipeline and exposes a simple form to score new examples or sample rows from the dataset.

**Development notes & recommendations**
- Avoid committing large binary model artifacts in the repo; use Git LFS or regenerate the model from the notebook.
- Keep `AIML Dataset.csv` in `data/` (update app/notebook paths) if you want to add other datasets.
- Add unit tests that load the pipeline and run a smoke prediction to catch serialization/API changes.

**Contributing**
- Fork, create a branch, and open a PR with a description of changes.

**License**
- Add a license file if you plan to distribute the code.
