# scripts/03_predict.py
from pathlib import Path
import argparse
import json
import joblib
import pandas as pd
import numpy as np

MODEL_PATH = Path("models/rf_v1.joblib")
FEATURES_PATH = Path("models/features_v1.json")

DROP_COLS = ["Flow ID", "Timestamp"]  # keep IPs/ports in output, but not in features

def preprocess_for_inference(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()

    # Preserve some meta columns for reporting if present
    meta_cols = [c for c in ["Src IP", "Dst IP", "Src Port", "Dst Port", "Protocol"] if c in df.columns]
    meta = df[meta_cols].copy() if meta_cols else pd.DataFrame(index=df.index)

    # Drop label if present and drop selected non-feature columns
    df = df.drop(columns=["Label"], errors="ignore")
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")
    df = df.drop(columns=["Src IP", "Dst IP"], errors="ignore")  # keep in meta only

    # Convert to numeric and clean inf
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)

    return df, meta

def align_features(X: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    X = X.copy()
    for col in feature_columns:
        if col not in X.columns:
            X[col] = np.nan
    X = X[feature_columns]
    return X

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Raw CIC CSV to score")
    parser.add_argument("--output", default="reports/alerts.csv")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--nrows", type=int, default=0)
    args = parser.parse_args()

    if not MODEL_PATH.exists() or not FEATURES_PATH.exists():
        raise FileNotFoundError("Missing model/features. Run scripts/02_train.py first.")

    model = joblib.load(MODEL_PATH)
    feature_columns = json.loads(FEATURES_PATH.read_text())["feature_columns"]

    df_raw = pd.read_csv(args.input, nrows=(args.nrows if args.nrows > 0 else None))
    X, meta = preprocess_for_inference(df_raw)
    X = align_features(X, feature_columns)

    prob = model.predict_proba(X)[:, 1]
    pred = (prob >= args.threshold).astype(int)

    out = meta.copy()
    out["prob_malicious"] = prob
    out["pred_is_malicious"] = pred

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print(f"[+] Wrote: {out_path}")
    print(out["pred_is_malicious"].value_counts())

if __name__ == "__main__":
    main()
