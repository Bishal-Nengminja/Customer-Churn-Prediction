# 🧩 Customer Churn Prediction — Data Science & MLOps Project  

**Author:** Bishal Nengminja (Jr. Data Scientist)  
**Goal:** Predict customer churn for a subscription-based business using an end-to-end, reproducible ML pipeline with experiment tracking and database integration.

---

## 🚀 Overview

Customer churn is a key business problem that directly impacts revenue and customer retention.  
This project demonstrates how a **Data Scientist** builds a complete, production-like pipeline — from data ingestion to model evaluation — while applying **MLOps best practices** (versioning, tracking, reproducibility).

✅ Focused on *Data Science role only*  
🚫 Excludes full-stack, Docker, and deployment engineering tasks.

---

## 🧱 Project Structure

```

customer_churn_project/
│
├── data/                   # Raw → Interim → Processed datasets
├── notebooks/              # EDA and analysis
├── src/                    # Scripts for data prep and modeling
├── models/                 # Trained model artifacts
├── mlflow_tracking/        # MLflow experiment logs
├── .gitignore / .dvcignore # Versioning configs
├── requirements.txt        # Dependencies
├── environment.yml         # Conda environment setup
├── Makefile                # Common command shortcuts
└── README.md               # Project documentation

```

---

## ⚙️ Tech Stack

| Component | Tool / Library |
|------------|----------------|
| **Language** | Python 3.10+ |
| **Database** | PostgreSQL |
| **Version Control** | Git |
| **Data Versioning** | DVC |
| **Experiment Tracking** | MLflow + DagsHub |
| **Modeling** | scikit-learn |
| **EDA & Visualization** | pandas, seaborn, matplotlib |
| **Environment** | Conda |
| **Orchestration** | Makefile |

---

## 🧩 Workflow Diagram

```

```
    ┌────────────┐
    │ Raw CSV    │
    └────┬───────┘
         │
         ▼
```

┌────────────────┐
│ Preprocessing  │  ➜ Fill nulls, rename columns, clean data
└────┬───────────┘
│
▼
┌───────────────────┐
│ Upload to Postgres│
└────┬──────────────┘
│
▼
┌────────────────────┐
│  Train (MLflow)    │
│  + Feature Eng.     │
└────┬───────────────┘
│
▼
┌────────────────────┐
│  Evaluate & Log    │
│  + DVC Track Model │
└────────────────────┘

````

---

## 📊 Key Results

| Metric | Score |
|--------|--------|
| Accuracy | **0.91** |
| Precision | 0.89 |
| Recall | 0.90 |
| F1-Score | 0.90 |
| ROC-AUC | 0.94 |

---

## 🧮 Models & Features

- **Algorithms:** Logistic Regression, Random Forest, XGBoost  
- **Feature Engineering:** tenure buckets, encoding, scaling  
- **Selection Criterion:** ROC-AUC and interpretability  
- **Experiment Tracking:** MLflow (local & DagsHub integration)

---

## 🧰 How to Run Locally

```bash
# 1️⃣ Clone project
git clone <your_github_repo_url>
cd customer_churn_project

# 2️⃣ Create environment
conda env create -f environment.yml
conda activate churn_env
pip install -r requirements.txt

# 3️⃣ Configure PostgreSQL credentials
cp .env.example .env
# Edit .env to include PGHOST, PGUSER, PGPASSWORD, PGDATABASE

# 4️⃣ Place raw dataset
mkdir -p data/raw
cp /path/to/your.csv data/raw/customer_churn_raw.csv

# 5️⃣ Preprocess (fill nulls, rename columns)
python src/preprocess.py --input data/raw/customer_churn_raw.csv --output data/interim/customer_clean.csv

# 6️⃣ Upload to PostgreSQL
python src/db_utils.py --csv data/interim/customer_clean.csv --table customer_churn

# 7️⃣ Train (logs to MLflow)
python src/train.py --csv data/interim/customer_clean.csv --out models/model.pkl --experiment churn_experiment

# 8️⃣ Evaluate model
python src/evaluate.py --model models/model.pkl --csv data/interim/customer_clean.csv
````

To visualize experiment runs:

```bash
mlflow ui --port 5000
```

Then open [http://localhost:5000](http://localhost:5000).

---

## 🧬 MLOps Highlights

✅ **Data lineage** with DVC
✅ **Reproducibility** via Conda + Makefile
✅ **Experiment tracking** (MLflow)
✅ **Database integration** (PostgreSQL)
✅ **Code modularization** (for reusability)

---

## 📈 Future Enhancements

* Model explainability (SHAP, LIME)
* Automated retraining pipeline with DVC (`dvc.yaml`)
* Streamlit dashboard for interpretability
* CI/CD for continuous evaluation

---

## 👨‍💻 About the Author

**Bishal Nengminja** — Jr. Data Scientist
Passionate about creating reproducible ML pipelines and applying MLOps in real-world projects.
🔗 [LinkedIn](https://www.linkedin.com/in/bishal-nengminja/) • [GitHub](https://github.com/Bishal-Nengminja)

---
