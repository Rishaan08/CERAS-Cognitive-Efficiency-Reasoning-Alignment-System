#Imports
from pathlib import Path
import pandas as pd
import numpy as np
import time

#Paths
DATA_DIR = Path("./data/processed")
COG_FILE = DATA_DIR / "pisa_2022_student_cog.parquet"
OUTPUT_FILE = DATA_DIR / "pisa_2022_cog_with_ce.parquet"

#Logger
def log(msg: str):
    print(f"[INFO] {msg}")

#Cognitive signal detection
def detect_cognitive_signal_columns(df: pd.DataFrame, student_id_col: str):
    """
    Detect numeric cognitive signal columns in PISA 2022 COG.
    This is PV-free and item-agnostic.
    """

    exclude_keywords = [
        "id", "cnt", "school", "country",
        "weight", "ver_", "strat", "repwt",
        "finalwt", "senwt"
    ]

    signal_cols = []

    for c in df.columns:
        if c == student_id_col:
            continue

        if any(k in c for k in exclude_keywords):
            continue

        if pd.api.types.is_numeric_dtype(df[c]):
            signal_cols.append(c)

    return signal_cols

#CE computation
def compute_ce(df: pd.DataFrame, student_id_col: str):
    """
    Cognitive Efficiency (CE) computation for PISA 2022.

    CE = normalized density of valid cognitive signals per student
    """

    log("Detecting cognitive signal columns (PISA 2022 safe)...")
    signal_cols = detect_cognitive_signal_columns(df, student_id_col)

    if len(signal_cols) < 50:
        raise ValueError(
            f"Too few cognitive signal columns detected ({len(signal_cols)}). "
            "Check COG file integrity."
        )

    log(f"Detected {len(signal_cols)} cognitive signal columns")

    signal_matrix = df[signal_cols]

    #Valid signal = non-null & finite
    valid_mask = signal_matrix.notna() & np.isfinite(signal_matrix)

    #Raw CE: proportion of valid signals
    ce_raw = valid_mask.sum(axis=1) / len(signal_cols)

    #Normalize to 0–100
    ce_score = 100 * (ce_raw - ce_raw.min()) / (ce_raw.max() - ce_raw.min())

    ce_df = pd.DataFrame({
        student_id_col: df[student_id_col],
        "ce_score": ce_score
    })

    return ce_df

#Main pipeline
def build_pisa_ce():
    start = time.time()
    log("Starting PISA Cognitive Efficiency (CE) builder")

    # Load COG
    log("Loading PISA COG dataset...")
    df = pd.read_parquet(COG_FILE)
    log(f"COG loaded | Shape: {df.shape}")

    #Normalize column names
    df.columns = df.columns.str.strip().str.lower()

    #Drop VER_DAT if present
    if "ver_dat" in df.columns:
        df.drop(columns=["ver_dat"], inplace=True)
        log("Dropped 'ver_dat' column")

    #Detect student identifier
    log("Detecting student identifier...")
    id_candidates = [c for c in df.columns if "stuid" in c]

    if not id_candidates:
        raise ValueError("No student ID column found (expected '*stuid*')")

    student_id_col = id_candidates[0]
    log(f"Using student ID column: '{student_id_col}'")

    #Compute CE
    ce_df = compute_ce(df, student_id_col)

    #Merge CE back to COG
    log("Merging CE score back into COG dataset...")
    df_out = df.merge(ce_df, on=student_id_col, how="left")

    #Save
    log("Saving PISA COG dataset with CE score...")
    df_out.to_parquet(OUTPUT_FILE, index=False)

    #Final summary
    elapsed = time.time() - start
    print("\nFINAL CE DATASET SUMMARY")
    print(f"Rows (students): {df_out.shape[0]:,}")
    print(f"Columns (features): {df_out.shape[1]:,}")
    print(f"CE column: 'ce_score'")
    print(f"Saved to: {OUTPUT_FILE}")
    print(f"Time taken: {elapsed:.2f} seconds")

#Entry point
if __name__ == "__main__":
    build_pisa_ce()