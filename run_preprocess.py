import os
import pandas as pd
import numpy as np
from data_loader import load_oulad, save_df, BASE_RAW, BASE_PROCESSED, OULAD_DIR
from preprocess.signal_fusion import aggregate_oulad_behavior, normalize_numeric
from preprocess.auditor import bulk_audit

#Helpers
def _to_str_id_series(s: pd.Series) -> pd.Series:
    s = s.copy()
    try:
        s = s.astype("string").str.strip()
    except Exception:
        s = s.fillna("").astype("string").str.strip()
    s = s.replace("", pd.NA).astype("string")
    return s

def _find_student_col(df: pd.DataFrame, prefer=None) -> str | None:
    if df is None or df.empty:
        return None
    cols = list(df.columns)
    prefer = prefer or ["CNTSTUID", "CNTSTU", "CNT_STU", "CNTSTU_ID", "CNTSTUD", "STUID", "STUDENT", "CNT"]
    for p in prefer:
        for c in cols:
            if p.lower() in c.lower():
                return c
    for c in cols:
        lc = c.lower()
        if "cntstuid" in lc or "cntst" in lc or "stuid" in lc or ("student" in lc and "id" in lc):
            return c
    return cols[0] if cols else None

def _safe_numeric(series):
    return pd.to_numeric(series, errors="coerce")

#Main preprocess runner (OULAD / MEU / Reveal only)
def run_preprocess(oulad_dir: str = OULAD_DIR, base_raw: str = BASE_RAW, base_processed: str = BASE_PROCESSED):
    print("Running preprocess")
    print("Raw:", base_raw)
    print("OULAD:", oulad_dir)
    print("Processed:", base_processed)

    #Load raw sources
    oulad = {}
    if os.path.isdir(oulad_dir):
        try:
            oulad = load_oulad(oulad_dir)
        except Exception as e:
            print("Failed to load OULAD:", e)
            oulad = {}

    #discover MEU and reveal files in base_raw
    meu_candidate = None
    reveal_candidate = None
    for root, _, files in os.walk(base_raw):
        for f in files:
            lf = f.lower()
            if "meu" in lf and meu_candidate is None:
                meu_candidate = os.path.join(root, f)
            if "reveal" in lf and reveal_candidate is None:
                reveal_candidate = os.path.join(root, f)

    meu = pd.read_excel(meu_candidate) if meu_candidate else pd.DataFrame()
    reveal = pd.read_csv(reveal_candidate) if reveal_candidate else pd.DataFrame()

    #Signal fusion
    print("Aggregating OULAD behavior...")
    oulad_behavior = aggregate_oulad_behavior(oulad)
    if not oulad_behavior.empty:
        save_df(oulad_behavior, "oulad_behavior")

    #MEU / Reveal feature save
    if not meu.empty:
        if "Subject" in meu.columns and "student_id" not in meu.columns:
            meu = meu.rename(columns={"Subject": "student_id"})
        if "student_id" in meu.columns:
            meu["student_id"] = _to_str_id_series(meu["student_id"])
        save_df(meu, "meu_raw")
    if not reveal.empty:
        if "answer_model" in reveal.columns and "student_id" not in reveal.columns:
            reveal = reveal.rename(columns={"answer_model": "student_id"})
        if "student_id" in reveal.columns:
            reveal["student_id"] = _to_str_id_series(reveal["student_id"])
        save_df(reveal, "reveal_raw")

    #Build canonical merged table
    print("Building canonical merged table (dtype-safe merge)...")
    dfs = []
    names = []

    #include available sources (oulad_behavior, meu, reveal)
    if not oulad_behavior.empty:
        ob = oulad_behavior.copy()
        if "student_id" not in ob.columns and "id_student" in ob.columns:
            ob = ob.rename(columns={"id_student": "student_id"})
        dfs.append(ob)
        names.append("oulad_behavior")

    if not meu.empty:
        m = meu.copy()
        if "student_id" not in m.columns:
            possible = [c for c in m.columns if "subject" in c.lower() or "student" in c.lower()]
            if possible:
                m = m.rename(columns={possible[0]: "student_id"})
        dfs.append(m)
        names.append("meu")

    if not reveal.empty:
        r = reveal.copy()
        if "student_id" not in r.columns:
            possible = [c for c in r.columns if "student" in c.lower() or "answer_model" in c.lower() or "model" in c.lower()]
            if possible:
                r = r.rename(columns={possible[0]: "student_id"})
        dfs.append(r)
        names.append("reveal")

    #Normalize key dtype to string for all dfs and report
    normalized_dfs = []
    for df_, nm in zip(dfs, names):
        df = df_.copy()
        if "student_id" not in df.columns:
            print(f"{nm} has no student_id column; adding empty student_id")
            df["student_id"] = pd.NA
        df["student_id"] = _to_str_id_series(df["student_id"])
        print(f"  * {nm}: rows={len(df)}, student_id dtype={df['student_id'].dtype}, n_unique_ids={df['student_id'].nunique()}")
        normalized_dfs.append(df)

    if normalized_dfs:
        merged = normalized_dfs[0]
        for d, nm in zip(normalized_dfs[1:], names[1:]):
            merged = merged.merge(d, on="student_id", how="outer", suffixes=("", f"_{nm}"))
            print(f"  -> merged with {nm}, shape now {merged.shape}")
        merged["student_id"] = merged["student_id"].replace("", pd.NA).astype("string")
        merged = merged.drop_duplicates(subset=["student_id"])
        save_df(merged, "canonical_merged")
    else:
        merged = pd.DataFrame()
        save_df(merged, "canonical_merged")

    #Audits
    print("Running audits...")
    audit_inputs = {
        "oulad_behavior": oulad_behavior,
        "meu_raw": meu,
        "reveal_raw": reveal,
        "canonical_merged": merged
    }
    audits = bulk_audit(audit_inputs, out_dir=base_processed)
    print("Audit outputs:", audits)

    #Basic normalization of numeric columns in merged (safe)
    if not merged.empty:
        norm = normalize_numeric(merged)
        save_df(norm, "canonical_merged_normalized")

    print("Preprocess complete")

if __name__ == "__main__":
    run_preprocess()