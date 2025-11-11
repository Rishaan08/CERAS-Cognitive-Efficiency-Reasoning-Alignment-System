import pandas as pd
import numpy as np
from typing import Dict, Optional

def aggregate_oulad_behavior(oulad: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    student_info = oulad.get("studentInfo", pd.DataFrame())
    stu_reg = oulad.get("studentRegistration", pd.DataFrame())
    stu_assess = oulad.get("studentAssessment", pd.DataFrame())
    stu_vle = oulad.get("studentVle", pd.DataFrame())
    assessments = oulad.get("assessments", pd.DataFrame())

    if not student_info.empty:
        base = student_info.copy()
        base_id_col = next((c for c in base.columns if "id_student" in c.lower() or c.lower()=="id_student" or "student" in c.lower()), base.columns[0])
        base = base.rename(columns={base_id_col: "student_id"})
    elif not stu_reg.empty:
        base = stu_reg.copy()
        base = base.rename(columns={next(iter(stu_reg.columns)): "student_id"})
    else:
        ids = []
        for df in (stu_assess, stu_vle):
            if not df.empty:
                idcol = next((c for c in df.columns if "id_student" in c.lower() or "student" in c.lower()), None)
                if idcol:
                    ids.extend(df[idcol].unique().tolist())
        base = pd.DataFrame({"student_id": pd.Series(list(set(ids)))})
        if base.empty:
            return pd.DataFrame()

    if not stu_assess.empty:
        idcol = next((c for c in stu_assess.columns if "id_student" in c.lower() or "student" in c.lower()), None)
        scorecol = next((c for c in stu_assess.columns if "score" in c.lower()), None)
        agg_assess = stu_assess.groupby(idcol).agg(
            oulad_avg_assessment_score=(scorecol, "mean") if scorecol else (stu_assess.columns[-1], "nunique"),
            oulad_n_assessments=(idcol, "count")
        ).reset_index().rename(columns={idcol: "student_id"})
    else:
        agg_assess = pd.DataFrame(columns=["student_id", "oulad_avg_assessment_score", "oulad_n_assessments"])

    if not stu_vle.empty:
        idcol = next((c for c in stu_vle.columns if "id_student" in c.lower() or "student" in c.lower()), None)
        clickcol = next((c for c in stu_vle.columns if "click" in c.lower() or "sum_click" in c.lower()), None)
        datecol = next((c for c in stu_vle.columns if "date" in c.lower()), None)

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

    merged = base.merge(agg_assess, on="student_id", how="left") \
                 .merge(agg_vle, on="student_id", how="left")

    num_cols = [c for c in merged.columns if merged[c].dtype.kind in "biufc"]
    if num_cols:
        merged[num_cols] = merged[num_cols].fillna(0)

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