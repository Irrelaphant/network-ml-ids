# scripts/04_demo.py
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import joblib

# Paths
DATASET_PATH = Path("data/processed/dataset_v1.csv")
MODEL_PATH = Path("models/rf_v1.joblib")
FEATURES_PATH = Path("models/features_v1.json")

# Import your scripts as modules by running from project root
# We'll just shell out to keep it beginner-simple and avoid import issues.
import subprocess
import sys


def run(cmd: list[str]):
    print("\n$ " + " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"[!] Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description="Demo runner for ML IDS pipeline")
    parser.add_argument("--input", required=True, help="Raw CSV to score (e.g., Friday PortScan CSV)")
    parser.add_argument("--threshold", type=float, default=0.7, help="Alert threshold")
    parser.add_argument("--nrows", type=int, default=200000, help="Rows to score (0=all)")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild processed dataset first")
    parser.add_argument("--retrain", action="store_true", help="Retrain model first")
    parser.add_argument("--output", default="reports/alerts_demo.csv", help="Alerts output CSV path")
    args = parser.parse_args()

    # Step 1: rebuild dataset if requested (or if missing)
    if args.rebuild or not DATASET_PATH.exists():
        print("[*] Building processed dataset...")
        run(["python", "scripts/01_build_dataset.py"])

    # Step 2: retrain if requested (or if missing)
    if args.retrain or not MODEL_PATH.exists() or not FEATURES_PATH.exists():
        print("[*] Training model...")
        run(["python", "scripts/02_train.py"])

    # Step 3: predict
    predict_cmd = [
        "python", "scripts/03_predict.py",
        "--input", args.input,
        "--output", args.output,
        "--threshold", str(args.threshold),
    ]
    if args.nrows and args.nrows > 0:
        predict_cmd += ["--nrows", str(args.nrows)]

    print("[*] Scoring input and generating alerts...")
    run(predict_cmd)

    # Step 4: show summary + top alerts
    out_path = Path(args.output)
    df = pd.read_csv(out_path)

    if "prob_malicious" not in df.columns or "pred_is_malicious" not in df.columns:
        print("[!] Output alerts file missing expected columns.")
        print(df.head())
        return

    total = len(df)
    alerts = int((df["pred_is_malicious"] == 1).sum())
    print("\n========== DEMO SUMMARY ==========")
    print(f"Input: {args.input}")
    print(f"Scored flows: {total}")
    print(f"Alerts (threshold={args.threshold}): {alerts}")

    top = df.sort_values("prob_malicious", ascending=False).head(25)
    print("\nTop 25 suspicious flows:")
    print(top.to_string(index=False))


if __name__ == "__main__":
    main()
