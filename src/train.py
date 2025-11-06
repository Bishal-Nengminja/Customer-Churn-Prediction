import argparse
import pandas as pd
import mlflow
import joblib
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

from features import build_transformer

def train(csv_path, target="churn", model_out="models/model.pkl", mlflow_experiment="churn_exp"):
    df = pd.read_csv(csv_path)
    if df[target].dtype == object:
        df[target] = df[target].map(lambda x: 1 if str(x).lower() in ["yes","1","true","y"] else 0)

    drop_cols = [c for c in ["customer_id"] if c in df.columns]
    features = [c for c in df.columns if c not in drop_cols + [target]]
    numeric_features = df[features].select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [c for c in features if c not in numeric_features]

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    preprocessor = build_transformer(numeric_features, categorical_features)

    from sklearn.pipeline import Pipeline
    pipe = Pipeline([
        ("pre", preprocessor),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
    ])

    mlflow.set_experiment(mlflow_experiment)
    with mlflow.start_run():
        pipe.fit(X_train, y_train)

        preds = pipe.predict(X_test)
        preds_proba = pipe.predict_proba(X_test)[:,1]

        auc = roc_auc_score(y_test, preds_proba)
        acc = accuracy_score(y_test, preds)

        mlflow.log_param("model", "RandomForest")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("auc", float(auc))
        mlflow.log_metric("accuracy", float(acc))

        os.makedirs(os.path.dirname(model_out), exist_ok=True)
        joblib.dump(pipe, model_out)
        mlflow.log_artifact(model_out, artifact_path="model")

        print(f"AUC: {auc:.4f}  ACC: {acc:.4f}")
        print(f"Saved model to {model_out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--target", default="churn")
    parser.add_argument("--out", default="models/model.pkl")
    parser.add_argument("--experiment", default="churn_experiment")
    args = parser.parse_args()
    train(args.csv, args.target, args.out, args.experiment)
