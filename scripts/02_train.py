# scripts/02_train.py
"""
Train a baseline machine learning IDS classifier on a processed dataset.
Pipeline goes as follows: 01_build_dataset.py -> 02_train.py -> 03_predict.py

This file exists to split data into training and testing sets, train a classifier model, measure the model and then save metrics and artifacts 
(model file, feature list) for later

"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/dataset_v1.csv", help="Processed dataset CSV path")
    parser.add_argument("--tag", default="v1", help="Suffix tag used for output artifact filenames")

    # Practical knobs
    parser.add_argument("--n_estimators", type=int, default=300, help="RandomForest trees")
    parser.add_argument("--class_weight", default="balanced", help="Use 'balanced' to improve recall")
    parser.add_argument("--seed", type=int, default=42)

    # If full dataset is huge, optionally downsample training data
    parser.add_argument("--train_sample_rows", type=int, default=0,
                        help="If >0, sample this many rows from TRAINING set only")

    # Day split options
    parser.add_argument("--test_day_keyword", default="Friday",
                        help="Rows whose source_file contains this keyword become TEST set")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Missing {data_path}. Run scripts/01_build_dataset.py first.")

    # Output paths (tagged so you don't overwrite artifacts)
    model_path = Path(f"models/rf_{args.tag}.joblib")
    features_path = Path(f"models/features_{args.tag}.json")
    metrics_path = Path(f"reports/metrics_{args.tag}.json")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    features_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[*] Loading dataset: {data_path}")
    df = pd.read_csv(data_path)

    # Basic checks
    needed = {"is_malicious", "source_file"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing columns {missing}. Re-run scripts/01_build_dataset.py.")

    # Day-based split: train = not test day, test = test day
    is_test_day = df["source_file"].astype(str).str.contains(args.test_day_keyword, case=False, na=False)

    train_df = df[~is_test_day].copy()
    test_df = df[is_test_day].copy()

    if len(test_df) == 0:
        raise ValueError(
            f"No rows matched test_day_keyword='{args.test_day_keyword}'. "
            f"Check source_file values in your dataset."
        )

    print("[*] Train rows:", len(train_df))
    print("[*] Test rows:", len(test_df))
    print("[*] Train label distribution:\n", train_df["is_malicious"].value_counts())
    print("[*] Test label distribution:\n", test_df["is_malicious"].value_counts())

    # Optionally sample training set only (helps with huge full datasets)
    if args.train_sample_rows and args.train_sample_rows > 0 and len(train_df) > args.train_sample_rows:
        train_df = train_df.sample(n=args.train_sample_rows, random_state=args.seed)
        print(f"[*] Sampled train_df down to {len(train_df)} rows for training")

    y_train = train_df["is_malicious"].astype(int)
    X_train = train_df.drop(columns=["is_malicious", "source_file"], errors="ignore")

    y_test = test_df["is_malicious"].astype(int)
    X_test = test_df.drop(columns=["is_malicious", "source_file"], errors="ignore")

    feature_columns = list(X_train.columns)

    # Pipeline: impute NaNs -> model
    model = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier(
            n_estimators=args.n_estimators,
            random_state=args.seed,
            n_jobs=-1,
            class_weight=args.class_weight if args.class_weight != "none" else None
        )),
    ])

    print("[*] Training model...")
    model.fit(X_train, y_train)

    print("[*] Evaluating...")
    preds = model.predict(X_test)

    cm = confusion_matrix(y_test, preds)
    report = classification_report(y_test, preds, digits=4, output_dict=True)

    print("Confusion matrix:\n", cm)
    print("\nReport:\n", classification_report(y_test, preds, digits=4))

    # Save artifacts
    joblib.dump(model, model_path)
    features_path.write_text(json.dumps({"feature_columns": feature_columns}, indent=2))
    metrics_path.write_text(json.dumps(
        {
            "data": str(data_path),
            "tag": args.tag,
            "test_day_keyword": args.test_day_keyword,
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "n_features": int(len(feature_columns)),
            "confusion_matrix": cm.tolist(),
            "report": report,
        },
        indent=2
    ))

    print(f"[+] Saved model: {model_path}")
    print(f"[+] Saved features: {features_path}")
    print(f"[+] Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
