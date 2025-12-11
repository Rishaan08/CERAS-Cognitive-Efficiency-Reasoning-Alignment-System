import os
import pandas as pd
import numpy as np
from typing import Dict, Optional

def _safe_numeric(s):
    return pd.to_numeric(s, errors="coerce")

def _winsorize_series(s: pd.Series, lower_q=0.01, upper_q=0.99):
    if s.dropna().empty:
        return s
    lo = s.quantile(lower_q)
    hi = s.quantile(upper_q)
    return s.clip(lower=lo, upper=hi)

def _robust_minmax(s: pd.Series, low_q=0.01, high_q=0.99):
    """Winsorize at quantiles then min-max scale to 0..1 on remaining range"""
    if s.dropna().empty:
        return s
    s_w = _winsorize_series(s, lower_q=low_q, upper_q=high_q)
    lo = s_w.quantile(0.00)
    hi = s_w.quantile(1.00)
    if pd.isna(lo) or pd.isna(hi) or lo == hi:
        return pd.Series(np.zeros(len(s_w)), index=s_w.index)
    scaled = (s_w - lo) / (hi - lo)
    return scaled.fillna(0.0)

def _median_impute(df: pd.DataFrame, cols):
    for c in cols:
        med = df[c].median(skipna=True)
        df[c] = df[c].fillna(med)
    return df

def aggregate_oulad_behavior(oulad: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Aggregate OULAD behavior and compute a cognitive_efficiency proxy from:
      - oulad_avg_assessment_score  (score)
      - oulad_avg_clicks            (avg clicks per row/user)
      - oulad_active_days           (distinct days active)
      - optional: registration duration in days (if registration columns exist)
    Result columns include numeric aggregates plus:
      - score_norm, clicks_norm, days_norm, reg_norm (when available)
      - effort_norm (sum of normalized effort signals)
      - cognitive_efficiency (score_norm / (effort_norm + 1))
    """
    student_info = oulad.get("studentInfo", pd.DataFrame())
    stu_reg = oulad.get("studentRegistration", pd.DataFrame())
    stu_assess = oulad.get("studentAssessment", pd.DataFrame())
    stu_vle = oulad.get("studentVle", pd.DataFrame())
    assessments = oulad.get("assessments", pd.DataFrame())

    #Base student list
    if not student_info.empty:
        base = student_info.copy()
        base_id_col = next((c for c in base.columns if "id_student" in c.lower() or c.lower() == "id_student" or "student" in c.lower()), base.columns[0])
        base = base.rename(columns={base_id_col: "student_id"})
        base = base[["student_id"]].drop_duplicates()
    elif not stu_reg.empty:
        base = stu_reg.copy()
        idcol = next((c for c in base.columns if "student" in c.lower() or "id" in c.lower()), base.columns[0])
        base = base.rename(columns={idcol: "student_id"})
        base = base[["student_id"]].drop_duplicates()
    else:
        ids = []
        for df in (stu_assess, stu_vle):
            if isinstance(df, pd.DataFrame) and not df.empty:
                idcol = next((c for c in df.columns if "id_student" in c.lower() or "student" in c.lower()), None)
                if idcol:
                    ids.extend(df[idcol].astype(str).dropna().unique().tolist())
        base = pd.DataFrame({"student_id": pd.Series(list(set(ids)))})
        if base.empty:
            return pd.DataFrame()

    #Aggregate studentAssessment
    if isinstance(stu_assess, pd.DataFrame) and not stu_assess.empty:
        idcol = next((c for c in stu_assess.columns if "id_student" in c.lower() or "student" in c.lower()), None)
        scorecol = next((c for c in stu_assess.columns if "score" in c.lower() or "mark" in c.lower() or "grade" in c.lower()), None)
        if idcol:
            if scorecol:
                agg_assess = stu_assess.groupby(idcol).agg(
                    oulad_avg_assessment_score=(scorecol, "mean"),
                    oulad_n_assessments=(idcol, "count")
                ).reset_index().rename(columns={idcol: "student_id"})
            else:
                agg_assess = stu_assess.groupby(idcol).agg(
                    oulad_avg_assessment_score=(idcol, "count"),
                    oulad_n_assessments=(idcol, "count")
                ).reset_index().rename(columns={idcol: "student_id"})
        else:
            agg_assess = pd.DataFrame(columns=["student_id", "oulad_avg_assessment_score", "oulad_n_assessments"])
    else:
        agg_assess = pd.DataFrame(columns=["student_id", "oulad_avg_assessment_score", "oulad_n_assessments"])

    #Aggregate studentVle 
    agg_vle = pd.DataFrame(columns=["student_id"])
    if isinstance(stu_vle, pd.DataFrame) and not stu_vle.empty:
        idcol = next((c for c in stu_vle.columns if "id_student" in c.lower() or "student" in c.lower()), None)
        clickcol = next((c for c in stu_vle.columns if "click" in c.lower() or "sum_click" in c.lower() or "num_click" in c.lower()), None)
        datecol = next((c for c in stu_vle.columns if "date" in c.lower() or "day" in c.lower()), None)

        if idcol:
            groups = []
            if clickcol:
                gclick = stu_vle.groupby(idcol)[clickcol].mean().reset_index().rename(columns={clickcol: "oulad_avg_clicks"})
                groups.append(gclick)
            if datecol:
                gd = stu_vle.groupby(idcol)[datecol].nunique().reset_index().rename(columns={datecol: "oulad_active_days"})
                groups.append(gd)

            if groups:
                agg_vle = groups[0]
                for g in groups[1:]:
                    agg_vle = agg_vle.merge(g, on=idcol, how="outer")
                agg_vle = agg_vle.rename(columns={idcol: "student_id"})
            else:
                agg_vle = pd.DataFrame(columns=["student_id"])
        else:
            agg_vle = pd.DataFrame(columns=["student_id"])
    else:
        agg_vle = pd.DataFrame(columns=["student_id"])

    reg_df = pd.DataFrame(columns=["student_id", "registration_duration_days"])
    if isinstance(stu_reg, pd.DataFrame) and not stu_reg.empty:
        idcol = next((c for c in stu_reg.columns if "student" in c.lower() or "id" in c.lower()), None)
        date_start = next((c for c in stu_reg.columns if "reg" in c.lower() and "date" in c.lower()) , None)
        date_end = next((c for c in stu_reg.columns if ("end" in c.lower() or "unreg" in c.lower() or "last" in c.lower()) and "date" in c.lower()), None)

        if idcol:
            if date_start and date_end:
                tmp = stu_reg[[idcol, date_start, date_end]].copy()
                tmp = tmp.rename(columns={idcol: "student_id"})
                tmp["__start"] = pd.to_datetime(tmp[date_start], errors="coerce")
                tmp["__end"] = pd.to_datetime(tmp[date_end], errors="coerce")
                tmp["registration_duration_days"] = (tmp["__end"] - tmp["__start"]).dt.total_seconds() / 86400.0
                reg_df = tmp[["student_id", "registration_duration_days"]].dropna(subset=["registration_duration_days"])
            else:
                #fallback
                reg_df = pd.DataFrame(columns=["student_id", "registration_duration_days"])

    #Merge base + aggregates
    merged = base.merge(agg_assess, on="student_id", how="left") \
                 .merge(agg_vle, on="student_id", how="left") \
                 .merge(reg_df, on="student_id", how="left")

    #Ensure student_id string dtype and strip
    merged["student_id"] = merged["student_id"].astype("string").str.strip()

    #Numeric columns of interest
    for col in ["oulad_avg_assessment_score", "oulad_avg_clicks", "oulad_active_days", "registration_duration_days"]:
        if col not in merged.columns:
            merged[col] = pd.NA

    #Convert to numeric where sensible
    merged["oulad_avg_assessment_score"] = _safe_numeric(merged["oulad_avg_assessment_score"])
    merged["oulad_avg_clicks"] = _safe_numeric(merged["oulad_avg_clicks"])
    merged["oulad_active_days"] = _safe_numeric(merged["oulad_active_days"])
    merged["registration_duration_days"] = _safe_numeric(merged["registration_duration_days"])

    #Median impute numeric columns
    numeric_cols = ["oulad_avg_assessment_score", "oulad_avg_clicks", "oulad_active_days", "registration_duration_days"]
    merged = _median_impute(merged, numeric_cols)
    merged["score_w"] = _winsorize_series(merged["oulad_avg_assessment_score"])
    merged["score_norm"] = _robust_minmax(merged["score_w"])
    merged["clicks_log"] = np.log1p(merged["oulad_avg_clicks"].astype(float))
    merged["clicks_norm"] = _robust_minmax(_winsorize_series(merged["clicks_log"]))
    merged["days_norm"] = _robust_minmax(_winsorize_series(merged["oulad_active_days"]))

    #registration duration (if present and not all zeros)
    if merged["registration_duration_days"].notna().any() and merged["registration_duration_days"].sum() != 0:
        merged["reg_norm"] = _robust_minmax(_winsorize_series(merged["registration_duration_days"]))
        merged["has_reg_norm"] = True
    else:
        merged["reg_norm"] = 0.0
        merged["has_reg_norm"] = False
    merged["effort_norm"] = merged["clicks_norm"].fillna(0.0) + merged["days_norm"].fillna(0.0)
    if merged["has_reg_norm"].any():
        merged["effort_norm"] = merged["effort_norm"] + merged["reg_norm"].fillna(0.0)

    #Final CE: score_norm / (effort_norm + 1)
    merged["cognitive_efficiency"] = merged["score_norm"].astype(float) / (merged["effort_norm"].astype(float) + 1.0)

    #Additional: keep raw and normalized columns, cast numeric dtypes for parquet friendliness
    to_float32 = ["oulad_avg_assessment_score", "oulad_avg_clicks", "oulad_active_days",
                  "registration_duration_days", "score_norm", "clicks_norm", "days_norm", "reg_norm",
                  "effort_norm", "cognitive_efficiency"]
    for c in to_float32:
        if c in merged.columns:
            merged[c] = pd.to_numeric(merged[c], errors="coerce").astype("float32")

    #Replace inf/nans in effort_norm/ce with 0 / NaN appropriately
    merged["effort_norm"] = merged["effort_norm"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    merged["cognitive_efficiency"] = merged["cognitive_efficiency"].replace([np.inf, -np.inf], np.nan)

    #Return merged aggregates
    return merged

def normalize_numeric(df: pd.DataFrame, cols: Optional[list] = None) -> pd.DataFrame:
    out = df.copy()
    if cols is None:
        cols = [c for c in out.columns if out[c].dtype.kind in "biufc" and c != "student_id"]
    for c in cols:
        mean = out[c].mean()
        std = out[c].std(ddof=0) or 1.0
        out[c] = (out[c] - mean) / std
    return out