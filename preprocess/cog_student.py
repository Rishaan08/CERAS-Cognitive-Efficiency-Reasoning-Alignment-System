# Imports
from pathlib import Path
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
import pyarrow.parquet as pq

#Paths
DATA_DIR = Path("./data/processed")
INPUT_FILE = DATA_DIR / "pisa_2022_cog_with_ce.parquet"
OUTPUT_FILE = DATA_DIR / "pisa_student.parquet"

#Logger
def log(msg):
    print(f"{msg}")

#Main
def main():
    start = time.time()
    log("Starting PISA student-level feature extraction")

    #Read column metadata only
    log("Reading column metadata only (Parquet schema)...")
    parquet_file = pq.ParquetFile(INPUT_FILE)
    all_cols = parquet_file.schema.names

    log(f"Total columns detected: {len(all_cols)}")

    #Detect student ID
    student_id_candidates = [c for c in all_cols if "stuid" in c.lower()]
    if not student_id_candidates:
        raise ValueError(f"Student ID column not found. Columns sample: {all_cols[:20]}")

    student_id_col = student_id_candidates[0]
    ce_col = "ce_score"

    log(f"Using student ID column: {student_id_col}")

    #Candidate cognitive columns
    cognitive_cols = [
        c for c in all_cols
        if c not in [student_id_col, ce_col]
    ]

    log(f"Candidate cognitive columns: {len(cognitive_cols)}")

    #Load only student_id + CE (cheap)
    log("Loading student ID and CE columns only...")
    base_df = pd.read_parquet(INPUT_FILE, columns=[student_id_col, ce_col])
    students = base_df[student_id_col].values

    #Initialize accumulators
    log("Initializing student accumulators...")

    acc = {}
    for sid in students:
        acc[sid] = {
            "sum": 0.0,
            "sum_sq": 0.0,
            "count": 0,
            "non_null": 0,
            "min": np.inf,
            "max": -np.inf,
            "entropy_counts": {}
        }

    #Process columns in Chunks
    chunk_size = 50
    chunks = [
        cognitive_cols[i:i + chunk_size]
        for i in range(0, len(cognitive_cols), chunk_size)
    ]

    log(f"Processing {len(chunks)} column chunks ({chunk_size} columns each)")

    for cols in tqdm(chunks, desc="Processing cognitive columns", unit="chunk"):

        df_chunk = pd.read_parquet(
            INPUT_FILE,
            columns=[student_id_col] + cols
        )

        sids = df_chunk[student_id_col].values

        for c in cols:
            col_vals = df_chunk[c].values

            for sid, val in zip(sids, col_vals):

                # HARD SAFETY FILTER
                if not isinstance(val, (int, float, np.integer, np.floating)):
                    continue
                if not np.isfinite(val):
                    continue

                a = acc[sid]
                v = float(val)

                a["sum"] += v
                a["sum_sq"] += v * v
                a["count"] += 1
                a["non_null"] += 1
                a["min"] = min(a["min"], v)
                a["max"] = max(a["max"], v)

                key = round(v, 2)
                a["entropy_counts"][key] = a["entropy_counts"].get(key, 0) + 1

        del df_chunk  

    #Compute student-level features
    log("Computing final student-level features...")

    rows = []
    total_items = len(cognitive_cols)

    for sid, a in acc.items():
        if a["count"] == 0:
            rows.append([sid] + [np.nan] * 15)
            continue

        mean = a["sum"] / a["count"]
        variance = max(a["sum_sq"] / a["count"] - mean ** 2, 0)
        std = np.sqrt(variance)
        rms = np.sqrt(a["sum_sq"] / a["count"])
        item_range = a["max"] - a["min"]
        cv = std / mean if mean != 0 else 0.0
        energy = a["sum_sq"]
        attempt_rate = a["non_null"] / a["count"]
        response_density = a["count"] / total_items

        total = sum(a["entropy_counts"].values())
        entropy = -sum(
            (c / total) * np.log(c / total + 1e-9)
            for c in a["entropy_counts"].values()
        )
        normalized_entropy = entropy / np.log(total + 1e-9)
        zscore_spread = std / (item_range + 1e-9)
        mean_min_ratio = (mean - a["min"]) / (mean + 1e-9)

        rows.append([
            sid,
            mean,
            std,
            variance,
            rms,
            a["min"],
            a["max"],
            item_range,
            cv,
            energy,
            attempt_rate,
            response_density,
            entropy,
            normalized_entropy,
            zscore_spread,
            mean_min_ratio
        ])

    feature_df = pd.DataFrame(
        rows,
        columns=[
            student_id_col,
            "item_mean",
            "item_std",
            "item_variance",
            "item_rms",
            "item_min",
            "item_max",
            "item_range",
            "item_cv",
            "item_energy",
            "attempt_rate",
            "response_density",
            "response_entropy",
            "normalized_entropy",
            "zscore_spread",
            "mean_minus_min_ratio"
        ]
    )

    #Attach CE
    ce_map = base_df.set_index(student_id_col)[ce_col]
    feature_df[ce_col] = feature_df[student_id_col].map(ce_map)

    #Save
    log("Saving student-level dataset...")
    feature_df.to_parquet(OUTPUT_FILE, index=False)

    print("\nFinal Student-level Dataset")
    print(f"Rows (students): {feature_df.shape[0]:,}")
    print(f"Columns (features): {feature_df.shape[1]:,}")
    print(f"Saved to: {OUTPUT_FILE}")
    print(f"Time taken: {time.time() - start:.2f}s")

#Entry Point
if __name__ == "__main__":
    main()