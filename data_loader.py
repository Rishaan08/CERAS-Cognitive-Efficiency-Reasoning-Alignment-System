import os
import pandas as pd
from typing import Dict

if "__file__" in globals():
    candidate_base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
else:
    candidate_base = None

cwd = os.path.abspath(os.getcwd())
if os.path.isdir(os.path.join(cwd, "ceras")) or os.path.exists(os.path.join(cwd, "README.md")):
    BASE_DIR = cwd
elif candidate_base and (os.path.isdir(os.path.join(candidate_base, "ceras")) or os.path.exists(os.path.join(candidate_base, "README.md"))):
    BASE_DIR = candidate_base
else:
    BASE_DIR = candidate_base or cwd

BASE_RAW = os.path.expanduser("/Users/rishaan/Desktop/CERAS-Cognitive-Efficiency-Reasoning-Alignment-System/data/raw")
BASE_PROCESSED = os.path.expanduser("/Users/rishaan/Desktop/CERAS-Cognitive-Efficiency-Reasoning-Alignment-System/data/processed")
OULAD_DIR = os.path.join(BASE_DIR, "data", "anonymisedData")
os.makedirs(BASE_PROCESSED, exist_ok=True)

def _read_table(path: str):
    """Try parquet then CSV; return empty DataFrame on failure."""
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception:
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()

def load_oulad(oulad_dir: str = OULAD_DIR) -> Dict[str, pd.DataFrame]:
    """Load common OULAD CSVs/parquets into a dict of DataFrames."""
    oulad = {}
    if not os.path.isdir(oulad_dir):
        raise FileNotFoundError(f"OULAD directory not found: {oulad_dir}")
    print(f"Loading OULAD datasets from: {oulad_dir}")

    for f in sorted(os.listdir(oulad_dir)):
        fpath = os.path.join(oulad_dir, f)
        name = f.rsplit(".", 1)[0]
        if f.lower().endswith((".csv", ".parquet")):
            try:
                df = _read_table(fpath)
                oulad[name] = df
                print(f"Loaded {name}: {df.shape}")
            except Exception as e:
                print(f"Failed to load {f}: {e}")
    if not oulad:
        print("No OULAD CSV or parquet files found in: ", oulad_dir)
    return oulad

def load_meu(path: str):
    """Load MEU Excel file; if it contains 'Subject' rename to student_id."""
    if path and os.path.exists(path):
        print(f"Loading MEU dataset: {path}")
        try:
            df = pd.read_excel(path)
        except Exception:
            try:
                df = pd.read_csv(path)
            except Exception:
                df = pd.DataFrame()
        if not df.empty:
            # rename common subject field to student_id if present
            if "Subject" in df.columns and "student_id" not in df.columns:
                df = df.rename(columns={"Subject": "student_id"})
            # ensure student_id exists (string)
            if "student_id" in df.columns:
                df["student_id"] = df["student_id"].astype("string").str.strip()
        return df
    print("MEU dataset not found:", path)
    return pd.DataFrame()

def load_reveal_eval(path: str):
    """Load reveal eval CSV and ensure student_id column if available."""
    if path and os.path.exists(path):
        print(f"Loading Reveal-Eval dataset: {path}")
        try:
            df = pd.read_csv(path)
        except Exception:
            try:
                df = pd.read_parquet(path)
            except Exception:
                df = pd.DataFrame()
        if not df.empty:
            if "student_id" in df.columns:
                df["student_id"] = df["student_id"].astype("string").str.strip()
        return df
    print("Reveal-Eval dataset not found:", path)
    return pd.DataFrame()

def save_df(df: pd.DataFrame, name: str, base_processed: str = BASE_PROCESSED):
    os.makedirs(base_processed, exist_ok=True)
    p = os.path.join(base_processed, name if name.endswith(".parquet") else f"{name}.parquet")
    df_out = df.copy()
    for c in df_out.columns:
        if df_out[c].dtype == bool:
            df_out[c] = df_out[c].astype("int8")
    df_out.to_parquet(p, index=False)
    print(f"Saved: {p} ({df_out.shape})")
    return p

if __name__ == "__main__":
    print("ceras.data_loader quick test")
    print("Base Directory: ", BASE_DIR)
    print("Base Raw: ", BASE_RAW)
    print("Base Processed: ", BASE_PROCESSED)
    print("OULAD Directory: ", OULAD_DIR)

    print("Loading OULAD CSV files....")
    try:
        oulad = load_oulad(OULAD_DIR)
        print("OULAD tables loaded: ", list(oulad.keys()))
    except FileNotFoundError as e:
        print(e)
        oulad = {}

    def find_first_file_with_substring(base_dir, substrings):
        for root, _, files in os.walk(base_dir):
            for f in files:
                fname = f.lower()
                for s in substrings:
                    if s.lower() in fname:
                        return os.path.join(root, f)
        return None

    print("\nDiscovering MEU and Reveal files inside:", BASE_RAW)
    meu_path = find_first_file_with_substring(BASE_RAW, ["meu"])
    reveal_path = find_first_file_with_substring(BASE_RAW, ["reveal"])

    if meu_path is None:
        for alt in ["MEU-Mobile KSD 2016.xlsx", "MEU_Mobile KSD 2016.xlsx", "MEU-Mobile_KSD_2016.xlsx"]:
            p = os.path.join(BASE_RAW, alt)
            if os.path.exists(p):
                meu_path = p
                break

    if reveal_path is None:
        for alt in ["reveal_eval.csv", "reveal-eval.csv", "reveal_eval_final.csv"]:
            p = os.path.join(BASE_RAW, alt)
            if os.path.exists(p):
                reveal_path = p
                break

    if meu_path:
        print(f"MEU file found: {meu_path}")
        meu = load_meu(meu_path)
    else:
        print("MEU file not found in", BASE_RAW)
        meu = pd.DataFrame()

    if reveal_path:
        print(f"Reveal-Eval file found: {reveal_path}")
        reveal = load_reveal_eval(reveal_path)
    else:
        print("Reveal-Eval file not found:", BASE_RAW)
        reveal = pd.DataFrame()

    print("MEU Shape: ", meu.shape if not meu.empty else "not found")
    print("Reveal Shape: ", reveal.shape if not reveal.empty else "not found")