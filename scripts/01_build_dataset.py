# Load each day's CSV file from the raw data directory
# Create the "is_malicious" label to indicate if the entry is malicious
# Drop leaky identifiers that may not be useful for the model
# Keep only numeric features for analysis
# Save the combined dataset into a single CSV file

from pathlib import Path
import pandas as pd
import numpy as np

RAW_DIR = Path("data/raw")
OUT_PATH = Path("data/processed/dataset_v1.csv")

SAMPLE_PER_FILE = 150_000 
SEED = 42

DROP_COLS = ["Flow ID", "Src IP", "Dst IP", "Timestamp",]

def clean_and_label(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()  # Create a copy of the DataFrame to avoid modifying the original data
    df.columns = df.columns.str.strip()  # Strip whitespace from column names

    if "Label" not in df.columns:
        raise ValueError("No 'Label' column found after stripping column names.")

    # Create binary target indicating if the entry is malicious (1) or benign (0)
    df["is_malicious"] = (df["Label"].astype(str).str.strip() != "BENIGN").astype(int)

    # Drop leakage/identifier columns and the original label column
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")
    df = df.drop(columns=["Label"], errors="ignore")

    # Convert all columns to numeric types, coercing errors to NaN
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Replace infinite values with NaN for later imputation during training
    df = df.replace([np.inf, -np.inf], np.nan)

    return df

def main():
    # Create the output directory if it doesn't exist
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Get a sorted list of all CSV files in the raw data directory
    files = sorted(RAW_DIR.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSVs found in {RAW_DIR.resolve()}")

    blocks = []
    for f in files:
        # Print the name of the file being loaded
        print(f"[*] Loading {f.name}")
        df = pd.read_csv(f)

        # Sample the data for speed and memory efficiency
        if SAMPLE_PER_FILE and len(df) > SAMPLE_PER_FILE:
            df = df.sample(n=SAMPLE_PER_FILE, random_state=SEED)

        df = clean_and_label(df)
        df["source_file"] = f.name  # helps later for day-based split
        blocks.append(df)

    # Combine all processed DataFrames into a single DataFrame
    combined = pd.concat(blocks, ignore_index=True)

    # Print the shape of the combined DataFrame
    print("[+] Combined shape:", combined.shape)
    # Print the distribution of the 'is_malicious' label
    print("[+] is_malicious distribution:")
    print(combined["is_malicious"].value_counts())

    # Save the combined DataFrame to a CSV file
    combined.to_csv(OUT_PATH, index=False)
    # Print confirmation of the saved dataset
    print(f"[+] Saved dataset: {OUT_PATH}")

if __name__ == "__main__":
    main()