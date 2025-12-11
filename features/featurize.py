import os
import json
from typing import Dict, List
import numpy as np
import pandas as pd

#Import helper utilities from top-level data_loader where available
try:
    from data_loader import BASE_PROCESSED, save_df
except Exception:
    BASE_PROCESSED = os.path.join("data", "processed")
    def save_df(df, name, base_processed=BASE_PROCESSED):
        os.makedirs(base_processed, exist_ok=True)
        path = os.path.join(base_processed, f"{name}.parquet")
        df.to_parquet(path, index=False)
        print(f"Saved: {path} ({df.shape})")
        return path

#Try importing the feature_quality utilities if present
try:
    from features.feature_quality import evaluate_feature_quality, save_feature_quality_report
except Exception:
    def evaluate_feature_quality(df: pd.DataFrame) -> Dict:
        return {"rows": int(df.shape[0]), "cols": int(df.shape[1])}
    def save_feature_quality_report(metrics: Dict, processed_dir: str, filename: str = "feature_quality.json"):
        p = os.path.join(processed_dir, filename)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved feature quality: {p}")

# Helpers
def _to_str_id_series(ser: pd.Series) -> pd.Series:
    """Robustly coerce to a pandas string id column."""
    if ser is None:
        return pd.Series(dtype="string")
    s = ser.copy()
    try:
        if s.dtype.kind == "f":
            nonnull = s.dropna()
            if len(nonnull) > 0:
                frac = (nonnull % 1.0).abs()
                if (frac <= 1e-9).all():
                    s = s.astype("Int64").astype("string")
                else:
                    s = s.astype("string")
            else:
                s = s.astype("string")
        else:
            s = s.astype("string")
    except Exception:
        s = s.astype("string")
    s = s.str.strip().replace("", pd.NA)
    return s

def _sanitize_column(series: pd.Series) -> pd.Series:
    """Make column parquet-friendly: bool->int8, object->string, numeric->float32."""
    if series.dtype == bool:
        return series.astype("int8")
    if series.dtype == object:
        return series.astype("string")
    if series.dtype.kind in "iu":
        try:
            return pd.to_numeric(series, errors="coerce").astype("float32")
        except Exception:
            return series.astype("string")
    if series.dtype.kind in "f":
        return pd.to_numeric(series, errors="coerce").astype("float32")
    return series.astype("string")

def _maybe_load(paths: List[str], processed_dir: str):
    for p in paths:
        full = os.path.join(processed_dir, p)
        if os.path.exists(full):
            try:
                if full.lower().endswith(".parquet"):
                    return pd.read_parquet(full)
                else:
                    return pd.read_csv(full)
            except Exception as e:
                print(f"Could not load {full}: {e}")
    return pd.DataFrame()

#Load processed inputs 
def load_processed_inputs(processed_dir: str = BASE_PROCESSED) -> Dict[str, pd.DataFrame]:
    out = {}
    out["canonical"] = _maybe_load(["canonical_merged_normalized.parquet", "canonical_merged.parquet"], processed_dir)
    out["oulad"] = _maybe_load(["oulad_behavior.parquet", "oulad_behavior.csv"], processed_dir)
    out["meu"] = _maybe_load(["meu_raw.parquet", "meu_raw.csv"], processed_dir)
    out["reveal"] = _maybe_load(["reveal_raw.parquet", "reveal_raw.csv"], processed_dir)
    return out

#Per-source feature builders
def build_oulad_features(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    if "student_id" not in df.columns:
        candidates = [c for c in df.columns if "student" in c.lower() or "id_student" in c.lower()]
        if candidates:
            df = df.rename(columns={candidates[0]: "student_id"})
        else:
            df["student_id"] = pd.NA
    df["student_id"] = _to_str_id_series(df["student_id"])

    # Keep numeric columns
    num_cols = [c for c in df.columns if df[c].dtype.kind in "biufc" and c != "student_id"]
    out = df[["student_id"] + num_cols].copy()

    #Fill numeric NA with median per-column
    medians = out.median(numeric_only=True)
    out = out.fillna(medians)

    #Compute cognitive_efficiency: avg_assessment_score / (active_days + 1)
    if "oulad_avg_assessment_score" in out.columns:
        denom = None
        if "oulad_active_days" in out.columns:
            denom = out["oulad_active_days"]
        elif "oulad_avg_clicks" in out.columns:
            denom = out["oulad_avg_clicks"]
        if denom is not None:
            out["cognitive_efficiency"] = out["oulad_avg_assessment_score"] / (denom.fillna(0) + 1)
        else:
            out["cognitive_efficiency"] = out["oulad_avg_assessment_score"]

    out["dataset"] = "oulad"
    return out

def build_meu_features(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    low = {c.lower(): c for c in df.columns}
    if "subject" in low and "student_id" not in df.columns:
        df = df.rename(columns={low["subject"]: "student_id"})
    if "student_id" not in df.columns:
        df["student_id"] = pd.NA
    df["student_id"] = _to_str_id_series(df["student_id"])
    num_cols = [c for c in df.columns if df[c].dtype.kind in "biufc" and c != "student_id"]
    out = df[["student_id"] + num_cols].copy()
    medians = out.median(numeric_only=True)
    out = out.fillna(medians)
    out["dataset"] = "meu"
    return out

def build_reveal_features(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    low = {c.lower(): c for c in df.columns}
    if "answer_model" in low and "student_id" not in df.columns:
        df = df.rename(columns={low["answer_model"]: "student_id"})
    if "student_id" not in df.columns:
        for cand in ("student", "subject"):
            if cand in low:
                df = df.rename(columns={low[cand]: "student_id"})
                break
    if "student_id" not in df.columns:
        df["student_id"] = pd.NA
    df["student_id"] = _to_str_id_series(df["student_id"])
    num_cols = [c for c in df.columns if df[c].dtype.kind in "biufc" and c != "student_id"]
    out = df[["student_id"] + num_cols].copy()
    medians = out.median(numeric_only=True)
    out = out.fillna(medians)
    out["dataset"] = "reveal"
    return out

#Grouping
def group_features(df: pd.DataFrame) -> Dict[str, List[str]]:
    groups = {
        "behavioral": [c for c in df.columns if "click" in c.lower() or "vle" in c.lower() or "active_days" in c.lower()],
        "cognitive": [c for c in df.columns if "cognitive_efficiency" in c.lower() or "score" in c.lower() or "accuracy" in c.lower()],
        "temporal": [c for c in df.columns if "time" in c.lower() or "duration" in c.lower()],
        "keystroke": [c for c in df.columns if "hold" in c.lower() or "typing" in c.lower()],
        "self_report": [c for c in df.columns if "effort" in c.lower() or "confid" in c.lower() or "anx" in c.lower()],
    }
    return {k: v for k, v in groups.items() if v}

#Orchestration
def build_features_pipeline(processed_dir: str = BASE_PROCESSED) -> pd.DataFrame:
    print("Loading processed inputs from:", processed_dir)
    inputs = load_processed_inputs(processed_dir)

    canonical = inputs.get("canonical", pd.DataFrame())
    oulad_df = inputs.get("oulad", pd.DataFrame())
    meu_df = inputs.get("meu", pd.DataFrame())
    reveal_df = inputs.get("reveal", pd.DataFrame())

    for name, df in [("canonical", canonical), ("oulad", oulad_df), ("meu", meu_df), ("reveal", reveal_df)]:
        if df is not None and not df.empty:
            print(f"Loaded {name}: {df.shape}")

    oulad_feats = build_oulad_features(oulad_df)
    meu_feats = build_meu_features(meu_df)
    reveal_feats = build_reveal_features(reveal_df)

    all_frames = [df for df in [oulad_feats, meu_feats, reveal_feats] if df is not None and not df.empty]
    if not all_frames:
        raise FileNotFoundError(f"No processed input data found in {processed_dir} (oulad/meu/reveal)")

    all_features = pd.concat(all_frames, ignore_index=True, sort=False)

    #try to map canonical CE into all_features
    if canonical is not None and not canonical.empty and "cognitive_efficiency" in canonical.columns:
        canon_ce = canonical[["student_id", "cognitive_efficiency"]].copy()
        canon_ce["student_id"] = _to_str_id_series(canon_ce["student_id"])
        all_features = all_features.merge(canon_ce, on="student_id", how="left", suffixes=("", "_canon"))
        if "cognitive_efficiency" in all_features.columns and "cognitive_efficiency_canon" in all_features.columns:
            all_features["cognitive_efficiency"] = all_features["cognitive_efficiency"].fillna(all_features["cognitive_efficiency_canon"])
            all_features = all_features.drop(columns=[c for c in all_features.columns if c.endswith("_canon")])
        elif "cognitive_efficiency_canon" in all_features.columns:
            all_features = all_features.rename(columns={"cognitive_efficiency_canon": "cognitive_efficiency"})

    #sanitize all columns
    for col in all_features.columns:
        all_features[col] = _sanitize_column(all_features[col])

    #ensure student_id type
    if "student_id" in all_features.columns:
        all_features["student_id"] = all_features["student_id"].astype("string")
    else:
        all_features.insert(0, "student_id", pd.Series([pd.NA] * len(all_features), dtype="string"))

    save_df(all_features, "features_final", base_processed=processed_dir)

    #groups & quality
    groups = group_features(all_features)
    os.makedirs(processed_dir, exist_ok=True)
    with open(os.path.join(processed_dir, "feature_groups.json"), "w", encoding="utf-8") as f:
        json.dump(groups, f, indent=2)

    metrics = evaluate_feature_quality(all_features)
    save_feature_quality_report(metrics, processed_dir, filename="feature_quality.json")

    print("Features pipeline complete. Final shape:", all_features.shape)
    return all_features

if __name__ == "__main__":
    print("Running features pipeline...")
    df = build_features_pipeline()
    print("Done. Saved engineered features to", BASE_PROCESSED)