import argparse
import joblib
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

def evaluate(model_path, csv_path, target="churn"):
    model = joblib.load(model_path)
    df = pd.read_csv(csv_path)
    if df[target].dtype == object:
        df[target] = df[target].map(lambda x: 1 if str(x).lower() in ["yes","1","true","y"] else 0)
    X = df.drop(columns=[target, "customer_id"], errors="ignore")
    y = df[target]
    preds = model.predict(X)
    probs = model.predict_proba(X)[:,1]
    print("Classification report:\n", classification_report(y, preds))
    print("ROC AUC:", roc_auc_score(y, probs))
    print("Confusion matrix:\n", confusion_matrix(y, preds))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--csv", required=True)
    args = parser.parse_args()
    evaluate(args.model, args.csv)
