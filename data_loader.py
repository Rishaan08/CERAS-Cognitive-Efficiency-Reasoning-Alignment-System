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

def load_pisa_parquet(base_raw=BASE_RAW) -> Dict[str, pd.DataFrame]:
    out = {}
    mapping = {
        "stu_q": ["CY08MSP_STU_QQQ.parquet"],
        "cog": ["CY08MSP_STU_COG.parquet"],
        "tim": ["CY08MSP_STU_TIM.parquet"],
        "sch": ["CY08MSP_SCH_QQQ.parquet"]
    }

    for key, names in mapping.items():
        for name in names:
            p = os.path.join(base_raw, name)
            if os.path.exists(p):
                out[key] = pd.read_parquet(p)
                break
        if key not in out:
            out[key] = pd.DataFrame()
    return out

def load_oulad(oulad_dir: str = OULAD_DIR, use_chunked: bool = False):
    oulad = {}
    if not os.path.isdir(oulad_dir):
        raise FileNotFoundError(
            f"OULAD directory not found: {oulad_dir}\n"
        )
    print(f"Loading OULAD datasets from: {oulad_dir}")

    for f in sorted(os.listdir(oulad_dir)):
        fpath = os.path.join(oulad_dir, f)
        name = f.rsplit(".", 1)[0]
        try:
            if f.lower().endswith(".csv"):
                df = pd.read_csv(fpath)
            elif f.lower().endswith(".parquet"):
                df = pd.read_parquet(fpath)
            else:
                continue

            oulad[name] = df
            print(f"Loaded {name}: {df.shape}")
        except Exception as e:
            print(f"Failed to load {f}: {e}")

    if not oulad:
        print("No OULAD CSV or parquet files found in: ", oulad_dir)
    
    return oulad

def load_meu(path: str):
    if path and os.path.exists(path):
        print(f"Loading MEU dataset: {path}")
        return pd.read_excel(path)
    print("MEU dataset not found:", path)
    return pd.DataFrame()

def load_reveal_eval(path: str):
    if path and os.path.exists(path):
        print(f"Loading Reveal-Eval dataset: {path}")
        return pd.read_csv(path)
    print("Reveal-Eval dataset not found:", path)
    return pd.DataFrame()

def save_df(df: pd.DataFrame, name: str, base_processed: str = BASE_PROCESSED):
    os.makedirs(base_processed, exist_ok = True)
    p = os.path.join(base_processed, name if name.endswith(".parquet") else f"{name}.parquet")
    df.to_parquet(p, index=False)
    print(f"Saved: {p} ({df.shape})")
    return p

if __name__ == "__main__":
    print("ceras.data_loader quick test")
    print("Base Directory: ", BASE_DIR)
    print("Base Raw: ", BASE_RAW)
    print("Base Processed: ", BASE_PROCESSED)
    print("OULAD Directory: ", OULAD_DIR)

    print("Loading PISA Parquet files....")
    pisa = load_pisa_parquet()

    print("Loading OULAD CSV files....")
    try:
        oulad = load_oulad(OULAD_DIR)
        print("OULAD tables loaded: ", list(oulad.keys()))
    except FileNotFoundError as e:
        print(e)

    def find_first_file_with_substring(base_dir, substrings):
        for root, _, files in os.walk(base_dir):
            for f in files:
                fname = f.lower()
                for s in substrings:
                    if s.lower() in fname:
                        return os.path.join(root, f)
        return None

    print("\nDiscovering MEU and Reveal files inside:", BASE_RAW)
    meu_candidates = ["meu", "MEU", "MEU-Mobile", "MEU_Mobile"]
    reveal_candidates = ["reveal", "reveal_eval", "reveal-eval", "reveal_eval.csv"]

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
        print("Reveal-Eval file not found in", BASE_RAW)
        reveal = pd.DataFrame()

    print("MEU Shape: ", meu.shape if not meu.empty else "not found")
    print("Reveal Shape: ", reveal.shape if not reveal.empty else "not found")