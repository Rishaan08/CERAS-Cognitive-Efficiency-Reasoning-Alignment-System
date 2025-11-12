import os
import json
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd

#Import helper utilities
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

#Import feature quality functions
try:
    from features.feature_quality import evaluate_feature_quality, save_feature_quality_report
except Exception:
    from feature_quality import evaluate_feature_quality, save_feature_quality_report

#Helper functions
def _safe_div(a, b, eps: float = 1e-9):
    return np.divide(a, (b + eps))

def _to_str_id(df: pd.DataFrame) -> pd.DataFrame:
    if "student_id" not in df.columns:
        id_cols = [c for c in df.columns if "student" in c.lower() or "id" in c.lower()]
        if id_cols:
            df = df.rename(columns={id_cols[0]: "student_id"})
        else:
            df["student_id"] = np.arange(len(df)).astype(str)
    df["student_id"] = df["student_id"].astype("string").str.strip()
    return df

def _fill_numeric_na_with_median(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in df.columns:
        if df[c].dtype.kind in "biufc":
            med = df[c].median(skipna=True)
            df.loc[:, c] = df[c].fillna(med)
    return df

#Data loading
def load_processed_inputs(processed_dir: str = BASE_PROCESSED) -> Dict[str, pd.DataFrame]:
    out = {}
    def maybe_load(fnames):
        for fn in fnames:
            path = os.path.join(processed_dir, fn)
            if os.path.exists(path):
                try:
                    return pd.read_parquet(path)
                except Exception:
                    try:
                        return pd.read_csv(path)
                    except Exception:
                        pass
        return pd.DataFrame()

    out["canonical"] = maybe_load(["canonical_merged_normalized.parquet", "canonical_merged.parquet"])
    out["oulad"] = maybe_load(["oulad_behavior.parquet"])
    out["pisa"] = maybe_load(["pisa_enriched.parquet", "pisa_features.parquet"])
    out["meu"] = maybe_load(["meu_raw.parquet"])
    out["reveal"] = maybe_load(["reveal_raw.parquet"])
    return out

#Base feature extraction
def extract_base_features(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = _to_str_id(df)
    num_cols = [c for c in df.columns if df[c].dtype.kind in "biufc"]
    extra_cols = [c for c in df.columns if any(x in c.lower() for x in ["effort", "confid", "anx", "score"])]
    keep = sorted(set(num_cols + extra_cols + ["student_id"]))
    df = df[keep]
    df = _fill_numeric_na_with_median(df)
    return df

#Derived features
def compute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    df = _to_str_id(df)

    #Engagement
    click_cols = [c for c in df.columns if "click" in c.lower()]
    day_cols = [c for c in df.columns if "day" in c.lower()]
    if click_cols and day_cols:
        df["engagement_intensity"] = _safe_div(df[click_cols[0]], df[day_cols[0]] + 1)

    #Cognitive efficiency
    acc_cols = [c for c in df.columns if "accuracy" in c.lower()]
    rt_cols = [c for c in df.columns if "response" in c.lower() or "time" in c.lower()]
    if acc_cols and rt_cols:
        df["cognitive_efficiency"] = _safe_div(df[acc_cols[0]], df[rt_cols[0]] + 1)

    #Cross-domain performance
    score_cols = [c for c in df.columns if "score" in c.lower()]
    if acc_cols and score_cols:
        df["cross_domain_performance"] = (df[acc_cols[0]] + df[score_cols[0]]) / 2

    #Effort/confidence ratio & stress
    eff_cols = [c for c in df.columns if "effort" in c.lower()]
    conf_cols = [c for c in df.columns if "confid" in c.lower()]
    anx_cols = [c for c in df.columns if "anx" in c.lower()]
    if eff_cols and conf_cols:
        df["effort_confidence_ratio"] = _safe_div(df[eff_cols[0]], df[conf_cols[0]] + 1)
    if anx_cols:
        df["stress_index"] = df[anx_cols[0]]

    #Typing consistency (MEU)
    hold_cols = [c for c in df.columns if "hold" in c.lower()]
    if hold_cols:
        df["typing_consistency"] = _safe_div(df[hold_cols].std(axis=1), df[hold_cols].mean(axis=1) + 1e-9)
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    return df

#Grouping
def group_features(df: pd.DataFrame) -> Dict[str, List[str]]:
    groups = {
        "behavioral": [c for c in df.columns if "click" in c.lower() or "engage" in c.lower()],
        "cognitive": [c for c in df.columns if "accuracy" in c.lower() or "score" in c.lower()],
        "temporal": [c for c in df.columns if "time" in c.lower() or "duration" in c.lower()],
        "keystroke": [c for c in df.columns if "hold" in c.lower() or "typing" in c.lower()],
        "self_report": [c for c in df.columns if "effort" in c.lower() or "confid" in c.lower() or "anx" in c.lower()],
        "derived": [c for c in df.columns if "efficiency" in c.lower() or "ratio" in c.lower() or "cross_domain" in c.lower()],
    }
    return {k: v for k, v in groups.items() if v}

def save_feature_groups(groups: Dict[str, List[str]], processed_dir: str = BASE_PROCESSED):
    os.makedirs(processed_dir, exist_ok=True)
    path = os.path.join(processed_dir, "feature_groups.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(groups, f, indent=2)
    print(f"Saved feature groups: {path}")
    return path

#Orchestration
def build_features_pipeline(processed_dir: str = BASE_PROCESSED) -> pd.DataFrame:
    print("Loading processed inputs from:", processed_dir)
    inputs = load_processed_inputs(processed_dir)

    canonical = inputs.get("canonical")
    if canonical is None or canonical.empty:
        pieces = [v for v in inputs.values() if v is not None and not v.empty]
        if pieces:
            canonical = pieces[0]
            print("Warning: canonical dataset missing — using fallback source.")
        else:
            raise FileNotFoundError(f"No processed input data found in {processed_dir}")

    for name, df in inputs.items():
        if df is not None and not df.empty:
            print(f"Loaded {name}: {df.shape}")
    base = extract_base_features(canonical)
    print("Base features:", base.shape)
    derived = compute_derived_features(base)
    print("Derived features:", derived.shape)
    derived = derived.copy()
    for col in derived.columns:
        if derived[col].dtype == object:
            unique_vals = derived[col].dropna().unique()
            if set(unique_vals).issubset({True, False, 0, 1}):
                derived.loc[:, col] = derived[col].astype("int8")
            else:
                derived.loc[:, col] = derived[col].astype("string")
        elif derived[col].dtype == bool:
            derived.loc[:, col] = derived[col].astype("int8")
    save_df(derived, "features_final", base_processed=processed_dir)
    groups = group_features(derived)
    save_feature_groups(groups, processed_dir)

    try:
        metrics = evaluate_feature_quality(derived)
        save_feature_quality_report(metrics, processed_dir, filename="feature_quality.json")
    except Exception as e:
        print(f"Feature quality evaluation skipped: {e}")

    print(f"Features pipeline complete. Final shape: {derived.shape}")
    return derived

if __name__ == "__main__":
    print("Running features pipeline...")
    df = build_features_pipeline()
    print("Done! Saved engineered features to data/processed/")