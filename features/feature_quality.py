import os
import json
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd

DEFAULT_PROCESSED = os.path.join("data", "processed")

def evaluate_feature_quality(df: pd.DataFrame) -> Dict[str, Dict]:
    if df is None or df.empty:
        return {}
    
    report = {}
    nrows = len(df)
    for c in df.columns:
        ser = df[c]
        entry = {
            "dtype": str(ser.dtype),
            "n_missing": int(ser.isna().sum()),
            "pct_missing": float(ser.isna().sum() / max(1, nrows)),
            "n_unique": int(ser.nunique(dropna=True))
        }

        if ser.dtype.kind in "biufc":
            entry.update({
                "mean": float(ser.mean(skipna=True)) if not ser.dropna().empty else None,
                "std": float(ser.std(skipna=True)) if not ser.dropna().empty else None,
                "var": float(ser.var(skipna=True)) if not ser.dropna().empty else None,
                "min": float(ser.min(skipna=True)) if not ser.dropna().empty else None,
                "max": float(ser.max(skipna=True)) if not ser.dropna().empty else None
            })
        else:
            try:
                entry["sample_values"] = ser.dropna().astype(str).unique().tolist()[:10]
            except Exception:
                entry["sample_values"] = []
        report[c] = entry
    return report

def feature_correlation_pairs(df: pd.DataFrame, threshold: float = 0.9) -> List[Tuple[str, str, float]]:
    if df is None or df.empty:
        return []
    numeric = df.select_dtypes(include=[np.number]).copy()
    if numeric.shape[1] < 2:
        return []
    
    corr = numeric.corr().abs()
    pairs = []
    cols = corr.columns
    for i, a in enumerate(cols):
        for j in range(i + 1, len(cols)):
            b = cols[j]
            val = corr.at[a, b]
            if not np.isnan(val) and val >- threshold:
                pairs.append((a, b, float(val)))
    pairs = sorted(pairs, key=lambda x: -x[2])
    return pairs

def redundancy_prune_suggestion(df: pd.DataFrame, corr_threshold: float = 0.9) -> List[Dict]:
    suggestions = []
    pairs = feature_correlation_pairs(df, threshold=corr_threshold)
    quality = evaluate_feature_quality(df)
    for a, b, corr in pairs:
        qa = quality.get(a, {})
        qb = quality.get(b, {})
        var_a = qa.get("var") or 0.0
        var_b = qb.get("var") or 0.0
        miss_a = qa.get("pct_missing") or 1.0
        miss_b = qb.get("pct_missing") or 1.0
        drop = a if (miss_a > miss_b or var_a < var_b) else b
        keep = b if drop == a else a
        suggestions.append({
            "pairs": (a, b),
            "correlation": corr,
            "suggested_drop": drop,
            "suggested_keep": keep,
            "reason": f"drop {drop} beacuse higher missingness / lower variance"
        })

    return suggestions

def save_feature_quality_report(metrics: Dict, out_dir: str = DEFAULT_PROCESSED, filename: str = "feature_quality.json") ->str:
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, filename)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"Saved feature quality report: {p}")
    return p

if __name__ == "__main__":
    p = os.path.join(DEFAULT_PROCESSED, "features_final.parquet")
    if os.path.exists(p):
        df = pd.read_parquet(p)
        metrics = evaluate_feature_quality(df)
        save_feature_quality_report(metrics, DEFAULT_PROCESSED)
        print("Top correlated pairs:", feature_correlation_pairs(df)[:10])
    else:
        print("No features_final.parquet found in", DEFAULT_PROCESSED)