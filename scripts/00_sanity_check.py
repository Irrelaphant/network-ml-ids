#sanity check
#verify raw files exit and can be loaded
from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw")

def main(): 
    #files in raw dir put into glob 
    files = sorted(RAW_DIR.glob("*.csv"))

    #no files in raw dir
    if not files:
        raise FileNotFoundError(f"No CSV files found in {RAW_DIR.resolve()}")
    
    #print status if files found 
    print(f"[+] Found {len(files)} CSV files.")
    print("[*] First file:", files[0].name)

    #create a dataframe from first file, load first 5000 rows, strip column names
    df = pd.read_csv(files[0], nrows=5000)
    df.columns = df.columns.str.strip()

    #print status if dataframe loaded successfully
    print ("[+] loaded first 500 rows")
    print("[*] Columns (First 15): ", list(df.columns[:15]))

    #if label exists in df columns, print the first ten unique labels
    if "Label" in df.columns:
        print("[*] Label counts (top): ")
        print(df["Label"].value_counts().head(10))
    else:
        print("[!] no 'Label' column found in the dataframe.")
        print([c for c in df.columns if "label" in c.lower()])  # Show columns that contain 'label' (case-insensitive)

if __name__ == "__main__":
    main()


