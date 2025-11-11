import os
from data_loader import load_pisa_parquet, load_oulad, save_df, BASE_RAW, BASE_PROCESSED, OULAD_DIR
from preprocess.context_enrichment import enrich_with_school, add_context_tags
from preprocess.signal_fusion import aggregate_oulad_behavior, normalize_numeric
from preprocess.auditor import bulk_audit
import pandas as pd

def run_preprocess(oulad_dir: str = OULAD_DIR, base_raw: str = BASE_RAW, base_processed: str = BASE_PROCESSED):
    print("Running preprocess")
    print("Raw:", base_raw)
    print("OULAD:", oulad_dir)
    print("Processed:", base_processed)

    #Load raw sources
    pisa = load_pisa_parquet(base_raw)
    oulad = load_oulad(oulad_dir) if os.path.isdir(oulad_dir) else {}
    meu_candidate = None
    reveal_candidate = None
    for root, _, files in os.walk(base_raw):
        for f in files:
            lf = f.lower()
            if "meu" in lf:
                meu_candidate = os.path.join(root, f)
            if "reveal" in lf:
                reveal_candidate = os.path.join(root, f)

    meu = pd.read_excel(meu_candidate) if meu_candidate else pd.DataFrame()
    reveal = pd.read_csv(reveal_candidate) if reveal_candidate else pd.DataFrame()

    #Signal fusion
    print("Aggregating OULAD behavior...")
    oulad_behavior = aggregate_oulad_behavior(oulad)
    if not oulad_behavior.empty:
        save_df(oulad_behavior, "oulad_behavior")

    #PISA feature extraction
    print("Building minimal PISA student features...")
    stu_q = pisa.get("stu_q", pd.DataFrame())
    cog = pisa.get("cog", pd.DataFrame())
    tim = pisa.get("tim", pd.DataFrame())

    #Student id detection
    if not stu_q.empty:
        sid_col = next((c for c in stu_q.columns if "stid" in c.lower() or "student" in c.lower()), stu_q.columns[0])
        pisa_stu = stu_q.rename(columns={sid_col: "student_id"})
    else:
        pisa_stu = pd.DataFrame()

    #Cognitive aggregates
    pisa_cog_features = pd.DataFrame()
    if not cog.empty:
        cog_sid = next((c for c in cog.columns if "stid" in c.lower() or "student" in c.lower()), None)
        if cog_sid:
            score_col = next((c for c in cog.columns if "score" in c.lower() or "correct" in c.lower()), None)
            time_col = next((c for c in cog.columns if "time" in c.lower() or "rt" in c.lower()), None)
            tmp = cog.rename(columns={cog_sid: "student_id"})
            agg_map = {}
            if score_col:
                agg_map[score_col] = "mean"
            if time_col:
                agg_map[time_col] = "mean"
            if agg_map:
                pisa_cog_features = tmp.groupby("student_id").agg(agg_map).reset_index()
                pisa_cog_features = pisa_cog_features.rename(columns={score_col: "pisa_avg_accuracy", time_col: "pisa_avg_response_time"})
    #Timing aggregates
    pisa_timing = pd.DataFrame()
    if not tim.empty:
        tim_sid = next((c for c in tim.columns if "stid" in c.lower() or "student" in c.lower()), None)
        dt_cols = [c for c in tim.columns if "start" in c.lower() or "end" in c.lower() or "time" in c.lower() or "date" in c.lower()]
        if tim_sid and len(dt_cols) >= 2:
            ttmp = tim.rename(columns={tim_sid: "student_id"})
            try:
                ttmp["start"] = pd.to_datetime(ttmp[dt_cols[0]], errors="coerce")
                ttmp["end"] = pd.to_datetime(ttmp[dt_cols[1]], errors="coerce")
                ttmp["duration_s"] = (ttmp["end"] - ttmp["start"]).dt.total_seconds()
                pisa_timing = ttmp.groupby("student_id")["duration_s"].mean().reset_index().rename(columns={"duration_s": "pisa_avg_duration_s"})
            except Exception:
                pisa_timing = pd.DataFrame()

    #Merge pisa features
    pisa_features = pisa_stu[["student_id"]].copy() if not pisa_stu.empty else pd.DataFrame()
    if not pisa_cog_features.empty:
        pisa_features = pisa_features.merge(pisa_cog_features, on="student_id", how="left") if not pisa_features.empty else pisa_cog_features
    if not pisa_timing.empty:
        pisa_features = pisa_features.merge(pisa_timing, on="student_id", how="left") if not pisa_features.empty else pisa_timing

    if not pisa_features.empty:
        save_df(pisa_features, "pisa_features")

    #MEU / Reveal feature save
    if not meu.empty:
        save_df(meu, "meu_raw")
    if not reveal.empty:
        save_df(reveal, "reveal_raw")

    #Context enrichment
    sch = pisa.get("sch", pd.DataFrame())
    if not pisa_features.empty:
        enriched = enrich_with_school(pisa_features, sch)
        enriched = add_context_tags(enriched)
        save_df(enriched, "pisa_enriched")
    else:
        enriched = pd.DataFrame()

    #Build canonical merge
    print("Building canonical merged table (dtype-safe merge)...")
    dfs = []
    names = []

    if not enriched.empty:
        dfs.append(enriched.copy())
        names.append("pisa_enriched")
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

    # Normalize key dtype to string for all dfs and drop entirely-empty student_id columns
    normalized_dfs = []
    for df, nm in zip(dfs, names):
        if "student_id" not in df.columns:
            print(f"{nm} has no student_id column; adding empty student_id")
            df["student_id"] = pd.NA
        df["student_id"] = df["student_id"].astype("string").fillna("")
        df["student_id"] = df["student_id"].str.strip()
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

    #Audits
    print("Running audits...")
    audit_inputs = {
        "pisa_features": pisa_features,
        "oulad_behavior": oulad_behavior,
        "pisa_enriched": enriched,
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