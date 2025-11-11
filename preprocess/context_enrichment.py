import pandas as pd
from typing import Optional

def enrich_with_school(stu_df: pd.DataFrame, sch_df: Optional[pd.DataFrame],
                       stu_school_key: str = "SCHID", school_key: str = "SCHID") -> pd.DataFrame:
    if sch_df is None or sch_df.empty:
        return stu_df.copy()

    stu = stu_df.copy()
    if stu_school_key not in stu.columns:
        cand = [c for c in stu.columns if "sch" in c.lower()]
        if cand:
            stu_school_key = cand[0]
        else:
            return stu

    if school_key not in sch_df.columns:
        cand = [c for c in sch_df.columns if "sch" in c.lower()]
        if cand:
            school_key = cand[0]
        else:
            return stu

    sch = sch_df.copy()
    sch_cols = [c for c in sch.columns if c.lower().startswith("sc")]
    if not sch_cols:
        sch_cols = [c for c in sch.columns if c != school_key][:5]
    selected = [school_key] + sch_cols
    sch_small = sch[selected].drop_duplicates(subset=[school_key])

    merged = stu.merge(sch_small, left_on=stu_school_key, right_on=school_key, how="left", suffixes=("", "_school"))
    return merged

def add_context_tags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["has_scores"] = out.filter(like="score").notna().any(axis=1)
    out["has_time"] = out.filter(like="time").notna().any(axis=1)
    out["has_behavior"] = out.filter(regex="click|vle|event|log", axis=1).notna().any(axis=1)
    return out