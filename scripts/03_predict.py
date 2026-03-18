"""

This file makes a scored CSV output based on a trained model, with threshold-based severity levels.
Training produces a model artifact, but an IDS-like deliverable that needs an inference tool that
can score new traffic logs and return a ranked list of suspicious logs with context for the SOC to handle.
This script is the scanning tool that would be used in application to generate alerts on new traffic logs.

this script:
1. loads a trained model (.joblib) and its feature schema
2. loadas a input CSV (same format as raw data, but different from processed dataset trained on)
3. preproccesses input data for inference:
    strips column whitespace
    removes non-feature columns
    keeps meta columns for SOCK context, like IPs and ports but doesn't use them to train 
    converts feature columns to metric values
    replaces inf with NaN
4. aligns input features with columns used during training, adding missing columns with NaN and dropping extra columns
5. Runs a prediction to get probability score 
6. Applies a user-defined thresehold to convert probabilities into alerts
7. writes an output CSV with scores, labels (that are predicted as malicious or not), severity levels, and metadata for SOC context

Why thresheholding?
    In a legitimate application of an IDS, probability thresheholds control the volume of alerts
    low threshehold means high sensitivity and a lot of false-positives or noise
    high threshehold means low sensitivity and fewer false positives, but a lot of missed detections
    this makes the model usable as an operational tool rather than a fixed program that will miss detections

limitations:
    uhhh everything here depends on the input CSV lol, all the usable metadata depends on whats in the input csv
    the script can only report on what the model can see, so if the input csv doesn't have good features, or enough features, 
    the model won't perform well


"""



# scripts/03_predict.py


from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys
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

def prob_bar(prob: float, width: int = 20) -> str:
    """
    Return a compact ASCII probability bar for terminal output.
    Example: prob=0.82 → [████████████████    ] 82%
    """
    filled = int(round(prob * width))
    bar = "█" * filled + " " * (width - filled)
    pct = f"{prob * 100:5.1f}%"
    return f"[{bar}] {pct}"
 
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
        total  = len(df_raw)
        alerts = int((pred == 1).sum())
        rate   = (alerts / total * 100.0) if total else 0.0
        high   = int((out["severity"] == "HIGH").sum()) if len(out) else 0
        med    = int((out["severity"] == "MED").sum())  if len(out) else 0
        low    = int((out["severity"] == "LOW").sum())  if len(out) else 0
 
        # NaN diagnostic — warns if the input format may be mismatched
        nan_rate = float(X.isna().mean().mean())
        nan_warn = ""
        if nan_rate > 0.20:
            worst_cols = X.isna().mean().sort_values(ascending=False).head(3)
            col_list = ", ".join(f"{c} ({v:.0%})" for c, v in worst_cols.items())
            nan_warn = f"\n  ⚠  High NaN rate ({nan_rate:.0%}) — possible format mismatch. Top cols: {col_list}"
 
        sep = "─" * 46
 
        print(f"\n{'═' * 46}")
        print(f"  ML-IDS Scan Report")
        print(f"{'═' * 46}")
        print(f"  Input   : {args.input}")
        print(f"  Model   : {model_path}")
        print(f"  Output  : {out_path}")
        print(sep)
        print(f"  Flows scored  : {total:>10,}")
        print(f"  Alerts fired  : {alerts:>10,}  ({rate:.2f}%)")
        print(f"  Threshold     : {args.threshold:>10.2f}")
        print(sep)
        print(f"  {'HIGH':>6}  {'MED':>6}  {'LOW':>6}")
        print(f"  {high:>6,}  {med:>6,}  {low:>6,}")
        if nan_warn:
            print(nan_warn)
        print(sep)
 
        # Top results table with probability bar
        preview = out[out["pred_is_malicious"] == 1].head(10)
        if len(preview) == 0:
            preview = out.head(10)
 
        if len(preview):
            # Build display columns — show whatever meta is available
            display_cols = []
            for col in ["Src IP", "Dst IP", "Src Port", "Dst Port", "Protocol"]:
                if col in preview.columns:
                    display_cols.append(col)
 
            print(f"\n  Top {len(preview)} results (sorted by suspicion):\n")
 
            # Header
            meta_w = 18  # fixed width per meta column
            hdr_meta = "".join(f"  {c:<{meta_w}}" for c in display_cols)
            print(f"  {'SEV':<5}  {'PROB BAR':<30}  {'SCORE':<7}{hdr_meta}")
            print(f"  {'─'*4}  {'─'*28}  {'─'*6}{('  ' + '─'*meta_w) * len(display_cols)}")
 
            for _, row in preview.iterrows():
                prob_val = float(row["prob_malicious"])
                sev      = str(row["severity"])
                bar      = prob_bar(prob_val, width=20)
                score    = f"{prob_val:.4f}"
 
                # Severity coloring (ANSI, degrades gracefully if no tty)
                sev_display = sev
                if hasattr(sys.stdout, "isatty") and sys.stdout.isatty():
                    if sev == "HIGH":
                        sev_display = f"\033[31m{sev}\033[0m"   # red
                    elif sev == "MED":
                        sev_display = f"\033[33m{sev}\033[0m"   # yellow
                    else:
                        sev_display = f"\033[2m{sev}\033[0m"    # dim
 
                meta_str = ""
                for col in display_cols:
                    val = str(row.get(col, "N/A"))
                    meta_str += f"  {val:<{meta_w}}"
 
                print(f"  {sev_display:<5}  {bar:<30}  {score:<7}{meta_str}")
 
        else:
            print("\n  No rows to preview. Try lowering --threshold or removing --alerts_only.")
 
        print(f"\n{'═' * 46}")
 
        print(f"\n[+] Wrote: {out_path}")


if __name__ == "__main__":
    main()