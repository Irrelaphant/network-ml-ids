# scripts/01_build_dataset.py
"""
This file processes the machine-learning-ready dataset from multiple daily flow CSV files into one file to train off of
The raw CIC-style dataset is split across multiple days of network traffic (Mon-Fri).
It includes labels as strings that mark traffic as benign, or different attack types.
Machine learning models are faster to train with a single file, so this script combvines them all into one table with
    1. Consistent column names
    2. numeric features (non-numeric values coerced to NaN)
    3. A binary target column "is_malicious" where 0=BENIGN and 1=anything else

1.This script loads each raw CSV from data/raw, then cleans up column names stripping whitespace
2. Converts the "Label" column into a binary target "is_malicious" (0=BENIGN, 1=anything else)
3. Removes columns not used for model training
4. Converts feature columns to numeric types
5. Replaces inf values with NaN (we will impute these during training)
6. adds a "source_file" column to keep track of which original CSV each row came from 
7. Concatenates all files into one dataset CSV at data/processed/dataset_v1.csv

Design notes:
  This scripts is used only to prepare the dataset 
  Missing values are not imputed here, that is handled during training with a scikit-learn imputer so training and inference behavior should stay consistent
  the source_file is preserved for evaluation purposes, so we can see if the model performs differently on different days of traffic

"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path("data/raw")
DROP_COLS = ["Flow ID", "Src IP", "Dst IP", "Timestamp"]
SEED = 42


def clean_and_label(df: pd.DataFrame) -> pd.DataFrame:
    """Clean a raw CIC dataframe chunk and add binary label."""
    df = df.copy()
    df.columns = df.columns.str.strip()

    if "Label" not in df.columns:
        raise ValueError("No 'Label' column found after stripping column names.")

    # Binary target: 0=BENIGN, 1=anything else
    df["is_malicious"] = (df["Label"].astype(str).str.strip() != "BENIGN").astype(int)

    # Drop leakage/identifier columns and original label
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")
    df = df.drop(columns=["Label"], errors="ignore")

    # Convert everything to numeric; non-numeric becomes NaN
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Replace inf with NaN (we will impute during training)
    df = df.replace([np.inf, -np.inf], np.nan)

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default="data/raw", help="Folder containing per-day CIC CSV files")
    parser.add_argument("--out", default="data/processed/dataset_v1.csv", help="Output processed dataset path")
    parser.add_argument("--sample_per_file", type=int, default=150_000, help="Rows to keep per file (0 = all rows)")
    parser.add_argument("--chunksize", type=int, default=200_000, help="Rows per chunk when reading large CSVs")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Clear old output file if it exists (so reruns don't append forever)
    if out_path.exists():
        out_path.unlink()

    files = sorted(raw_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSVs found in {raw_dir.resolve()}")

    total_rows_written = 0
    first_write = True

    for f in files:
        print(f"[*] Processing {f.name}")

        # Track how many rows we've written for this file (useful if sampling)
        written_this_file = 0

        # Stream the CSV file in chunks to avoid RAM issues
        for chunk in pd.read_csv(f, chunksize=args.chunksize):
            # If sampling is enabled and we've already written enough rows from this file, stop
            if args.sample_per_file and args.sample_per_file > 0 and written_this_file >= args.sample_per_file:
                break

            chunk = clean_and_label(chunk)
            chunk["source_file"] = f.name

            # If sampling is enabled, cap the number of rows we write for this file
            if args.sample_per_file and args.sample_per_file > 0:
                remaining = args.sample_per_file - written_this_file
                if len(chunk) > remaining:
                    # Sample down to remaining rows
                    chunk = chunk.sample(n=remaining, random_state=SEED)

            # Write incrementally (append mode)
            chunk.to_csv(out_path, mode="a", header=first_write, index=False)
            first_write = False

            written_this_file += len(chunk)
            total_rows_written += len(chunk)

        print(f"[+] Wrote {written_this_file:,} rows from {f.name}")

    print(f"[+] Done. Total rows written: {total_rows_written:,}")
    print(f"[+] Saved dataset: {out_path}")


if __name__ == "__main__":
    main()
