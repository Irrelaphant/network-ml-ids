# scripts/03_predict.py
from __future__ import annotations

from pathlib import Path
import argparse
import json
import joblib
import pandas as pd
import numpy as np

DEFAULT_MODEL_PATH = Path("models/rf_full.joblib")
DEFAULT_FEATURES_PATH = Path("models/features_full.json")

DROP_COLS = ["Flow ID", "Timestamp"]

def preprocess_for_inference(df:pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]: 
    df = df.copy()
    df.columns = df.columns.str.strip()  # Strip whitespace from column names

    # Support both CIC-IDS2017 names and shorter names
    META_CANDIDATES = [
        "Src IP", "Dst IP", "Src Port", "Dst Port",
        "Source IP", "Destination IP", "Source Port", "Destination Port",
        "Protocol", "Timestamp"
    ]

    meta_cols = [c for c in META_CANDIDATES if c in df.columns]
    meta = df[meta_cols].copy() if meta_cols else pd.DataFrame(index=df.index)

    # Normalize column names so SOC output is consistent
    rename_map = {
        "Source IP": "Src IP",
        "Destination IP": "Dst IP",
        "Source Port": "Src Port",
        "Destination Port": "Dst Port",
    }
    meta = meta.rename(columns=rename_map)


    # Drop label if present and drop selected non-feature columns
    df = df.drop(columns=["Label"], errors="ignore")
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")

    # Drop any IP columns to avoid leaking info and for privacy (we keep them in meta for SOC context but not as model features)
    df = df.drop(columns=["Src IP", "Dst IP", "Source IP", "Destination IP"], errors="ignore")

    # Convert to numeric and clean inf
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)

    return df, meta


def align_features(X: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Ensure inference dataframe has exactly the training feature columns, in the right order."""
    X = X.copy()
    for col in feature_columns:
        if col not in X.columns:
            X[col] = np.nan
    # Drop any extra columns not used in training
    X = X[feature_columns]
    return X


def severity_from_prob(prob: float, threshold: float) -> str:
    if prob >= 0.90:
        return "HIGH"
    if prob >= threshold:
        return "MED"
    return "LOW"


def main():
    parser = argparse.ArgumentParser(description="Score a CIC-style flow CSV and output IDS alerts.")
    parser.add_argument("--input", required=True, help="Raw CIC CSV to score")
    parser.add_argument("--output", default="reports/alerts.csv", help="Output alerts CSV")
    parser.add_argument("--threshold", type=float, default=0.3, help="Alert threshold for MED/HIGH")
    parser.add_argument("--nrows", type=int, default=0, help="0 = all rows")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH), help="Path to trained .joblib model")
    parser.add_argument("--features", default=str(DEFAULT_FEATURES_PATH), help="Path to features JSON")
    parser.add_argument("--top", type=int, default=0, help="If >0, keep only top N rows by probability")
    parser.add_argument("--alerts_only", action="store_true", help="Only write rows predicted malicious")
    parser.add_argument("--quiet", action="store_true", help="Suppress detailed terminal output")
    args = parser.parse_args()

    model_path = Path(args.model)
    features_path = Path(args.features)

    if not model_path.exists():
        raise FileNotFoundError(f"Missing model: {model_path}")
    if not features_path.exists():
        raise FileNotFoundError(f"Missing features JSON: {features_path}")

    model = joblib.load(model_path)
    feature_columns = json.loads(features_path.read_text())["feature_columns"]

    # Load input
    nrows = (args.nrows if args.nrows and args.nrows > 0 else None)
    df_raw = pd.read_csv(args.input, nrows=nrows)

    X, meta = preprocess_for_inference(df_raw)
    X = align_features(X, feature_columns)

    # Predict
    prob = model.predict_proba(X)[:, 1]
    pred = (prob >= args.threshold).astype(int)

    out = meta.copy()
    out["prob_malicious"] = prob
    out["pred_is_malicious"] = pred
    out["severity"] = [severity_from_prob(p, args.threshold) for p in prob]

    # Sort by probability (most suspicious first)
    out = out.sort_values("prob_malicious", ascending=False)

    # Optionally filter to alerts only
    if args.alerts_only:
        out = out[out["pred_is_malicious"] == 1]

    # Optionally take top N
    if args.top and args.top > 0:
        out = out.head(args.top)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    if not args.quiet:
        total = len(df_raw)
        alerts = int((pred == 1).sum())
        rate = (alerts / total * 100.0) if total else 0.0
        high = int((out["severity"] == "HIGH").sum()) if len(out) else 0
        med = int((out["severity"] == "MED").sum()) if len(out) else 0

        print("\nML-IDS Scan Summary")
        print("-------------------")
        print(f"Input: {args.input}")
        print(f"Model: {model_path}")
        print(f"Threshold: {args.threshold}")
        print(f"Flows scored: {total:,}")
        print(f"Alerts: {alerts:,} ({rate:.4f}%)")
        print(f"Severity (in written output): HIGH={high:,}  MED={med:,}")

        # Show top 10 alerts (or top 10 scored rows if not alerts_only)
        preview = out.head(10)
        if len(preview):
            print("\nTop results:")
            print(preview.to_string(index=False))
        else:
            print("\nNo rows written (try lowering threshold or disable --alerts_only).")

    print(f"\n[+] Wrote: {out_path}")


if __name__ == "__main__":
    main()