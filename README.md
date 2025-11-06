# Customer Churn Project (Data Scientist)
Industry-style Customer Churn Prediction project focused on Data Science responsibilities (no Docker, no FastAPI).

## Structure
```
customer_churn_project/
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── notebooks/
│   └── EDA_customer_churn.ipynb  # your notebook or converted version
├── src/
│   ├── preprocess.py
│   ├── db_utils.py
│   ├── features.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict_function.py
├── models/
│   └── model.pkl
├── mlflow_tracking/
├── requirements.txt
├── environment.yml
├── Makefile
└── README.md
```

## Quick start (summary)
1. Create environment:
```bash
conda env create -f environment.yml
conda activate churn_env
pip install -r requirements.txt
```
or using virtualenv:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Put your raw CSV: `data/raw/customer_churn_raw.csv`

3. Clean:
```bash
python src/preprocess.py --input data/raw/customer_churn_raw.csv --output data/interim/customer_clean.csv
```

4. Upload to PostgreSQL (set env or .env):
```bash
# Example: fill .env with PGHOST, PGPORT, PGUSER, PGPASSWORD, PGDATABASE
python src/db_utils.py --csv data/interim/customer_clean.csv
```

5. Train (MLflow optional):
```bash
mlflow ui --port 5000  # optional
python src/train.py --csv data/interim/customer_clean.csv --out models/model.pkl --experiment churn_experiment
```

6. Evaluate:
```bash
python src/evaluate.py --model models/model.pkl --csv data/interim/customer_clean.csv
```

7. Predict locally:
```bash
python src/predict_function.py
```

## Notes
- Column names will be converted to snake_case and nulls will be imputed during preprocess step.
- DVC is recommended for data versioning (not included in this zip by default).
- You can place your original notebook at `notebooks/EDA_customer_churn.ipynb` (if available it has been copied here).
