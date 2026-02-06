# scripts/02_train.py
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

DATA_PATH = Path("data/processed/dataset_v1.csv")
MODEL_PATH = Path("models/rf_v1.joblib")
FEATURES_PATH = Path("models/features_v1.json")
METRICS_PATH = Path("reports/metrics_v1.json")

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing {DATA_PATH}. Run scripts/01_build_dataset.py first.")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    y = df["is_malicious"].astype(int)
    X = df.drop(columns=["is_malicious"], errors="ignore")

    # keep source_file out of features
    if "source_file" in X.columns:
        X = X.drop(columns=["source_file"], errors="ignore")

    feature_columns = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)),
    ])

    print("[*] Training model...")
    model.fit(X_train, y_train)

    print("[*] Evaluating...")
    preds = model.predict(X_test)

    cm = confusion_matrix(y_test, preds)
    report = classification_report(y_test, preds, digits=4, output_dict=True)

    print("Confusion matrix:\n", cm)
    print("\nReport:\n", classification_report(y_test, preds, digits=4))

    joblib.dump(model, MODEL_PATH)
    FEATURES_PATH.write_text(json.dumps({"feature_columns": feature_columns}, indent=2))
    METRICS_PATH.write_text(json.dumps({"confusion_matrix": cm.tolist(), "report": report}, indent=2))

    print(f"[+] Saved model: {MODEL_PATH}")
    print(f"[+] Saved features: {FEATURES_PATH}")
    print(f"[+] Saved metrics: {METRICS_PATH}")

if __name__ == "__main__":
    main()
