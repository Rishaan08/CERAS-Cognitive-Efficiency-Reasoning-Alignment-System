import pandas as pd
import json
import os
from typing import Dict, Any

def _to_py(x: Any):
    """Convert numpy / pandas scalar to plain python for JSON serialization."""
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass
    try:
        import numpy as np
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return float(x)
        if isinstance(x, (np.bool_,)):
            return bool(x)
        if isinstance(x, (np.ndarray,)):
            return x.tolist()
    except Exception:
        pass
    try:
        if hasattr(x, "item"):
            return x.item()
    except Exception:
        pass
    return x if isinstance(x, (str, int, float, bool, type(None))) else str(x)

def audit_dataframe(df: pd.DataFrame, name: str, out_dir: str = "data/processed") -> str:
    report: Dict[str, Any] = {}
    report["name"] = name
    report["rows"] = int(len(df))
    report["cols"] = int(len(df.columns))
    report["columns"] = {}

    SAMPLE_THRESHOLD = 200_000
    SAMPLE_COUNT = 10_000

    for c in df.columns:
        try:
            ser = df[c]
            n_missing = int(ser.isna().sum())
            pct_missing = float(n_missing / max(1, len(ser)))
            n_unique = int(ser.nunique(dropna=True))

            if len(ser) > SAMPLE_THRESHOLD:
                sample = ser.dropna()
                sample = sample.sample(min(SAMPLE_COUNT, len(sample)), random_state=42).astype(str)
                vs = sample.value_counts().head(5)
            else:
                vs = ser.dropna().astype(str).value_counts().head(5)

            top_vals = {str(k): int(v) for k, v in vs.items()}

            report["columns"][c] = {
                "dtype": str(ser.dtype),
                "n_missing": n_missing,
                "pct_missing": pct_missing,
                "n_unique": n_unique,
                "top_values": top_vals
            }
        except Exception as e:
            report["columns"][c] = {
                "dtype": "error",
                "n_missing": None,
                "pct_missing": None,
                "n_unique": None,
                "top_values": {},
                "error": str(e)
            }

    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, f"audit_{name}.json")

    def _sanitize(obj):
        if isinstance(obj, dict):
            return {str(k): _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize(x) for x in obj]
        return _to_py(obj)

    with open(p, "w", encoding="utf-8") as f:
        json.dump(_sanitize(report), f, indent=2, ensure_ascii=False)
    return p

def bulk_audit(dfs: Dict[str, pd.DataFrame], out_dir: str = "data/processed") -> Dict[str, str]:
    results: Dict[str, str] = {}
    for name, df in dfs.items():
        try:
            path = audit_dataframe(df, name, out_dir=out_dir)
            results[name] = path
        except Exception as e:
            results[name] = f"error: {e}"
    return results