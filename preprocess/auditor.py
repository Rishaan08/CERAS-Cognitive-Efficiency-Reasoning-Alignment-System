# preprocess/auditor.py
import pandas as pd
import json
import os
from typing import Dict

def audit_dataframe(df: pd.DataFrame, name: str, out_dir: str = "data/processed") -> str:
    report = {}
    report["name"] = name
    report["rows"] = int(len(df))
    report["cols"] = int(len(df.columns))
    report["columns"] = {}

    for c in df.columns:
        ser = df[c]
        top_vals = ser.dropna().astype(str).value_counts().head(5).to_dict()
        report["columns"][c] = {
            "dtype": str(ser.dtype),
            "n_missing": int(ser.isna().sum()),
            "pct_missing": float(ser.isna().sum() / max(1, len(ser))),
            "n_unique": int(ser.nunique(dropna=True)),
            "top_values": top_vals
        }

    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, f"audit_{name}.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return p

def bulk_audit(dfs: Dict[str, pd.DataFrame], out_dir: str = "data/processed") -> Dict[str, str]:
    results = {}
    for name, df in dfs.items():
        try:
            path = audit_dataframe(df, name, out_dir=out_dir)
            results[name] = path
        except Exception as e:
            results[name] = f"error: {e}"
    return results