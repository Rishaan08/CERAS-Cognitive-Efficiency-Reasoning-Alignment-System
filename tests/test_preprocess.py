import numpy as np
import pandas as pd
import importlib
import os
import json

def test_audit_dataframe_basic():
    # check dataframe audit runs and creates report
    from preprocess.auditor import audit_dataframe

    df = pd.DataFrame({
        "a": [1, 2, np.nan],
        "b": [0.5, 0.6, 0.7]
    })

    path = audit_dataframe(df, name="test_df", out_dir="data/processed")

    os.path.exists(path)

    with open(path, "r") as f:
        report = json.load(f)

    assert report["rows"] == 3
    assert report["cols"] == 2
    assert "a" in report["columns"]
    assert report["columns"]["a"]["n_missing"] == 1

def test_bulk_audit_basic():
    #check bulk audit runs on multiple dfs
    from preprocess.auditor import bulk_audit

    dfs = {
        "df1": pd.DataFrame({"x": [1, 2, 3]}),
        "df2": pd.DataFrame({"y": [np.nan, 4, 5]})
    }

    results = bulk_audit(dfs, out_dir="data/processed")

    assert isinstance(results, dict)
    assert "df1" in results
    assert "df2" in results

def test_aggregate_oulad_behavior_basic():
    # check oulad aggregation runs on small input
    from preprocess.signal_fusion import aggregate_oulad_behavior

    oulad = {
        "studentInfo": pd.DataFrame({
            "id_student": ["1", "2"]
        }),
        "studentAssessment": pd.DataFrame({
            "id_student": ["1", "1", "2"],
            "score": [70, 80, 60]
        }),
        "studentVle": pd.DataFrame({
            "id_student": ["1", "2", "2"],
            "sum_click": [10, 5, 7],
            "date": ["2020-01-01", "2020-01-01", "2020-01-02"]
        })
    }

    out = aggregate_oulad_behavior(oulad)

    assert isinstance(out, pd.DataFrame)
    assert "student_id" in out.columns
    assert "cognitive_efficiency" in out.columns
    assert len(out) > 0

def test_normalize_numeric_basic():
    #check numeric normalization
    from preprocess.signal_fusion import normalize_numeric

    df = pd.DataFrame({
        "student_id": ["1", "2", "3"],
        "x": [1.0, 2.0, 3.0],
        "y": [2.0, 4.0, 6.0]
    })

    out = normalize_numeric(df)

    assert "x" in out.columns
    assert abs(out["x"].mean()) < 1e-6

def test_import_ce_builder():
    # check ce_builder import
    importlib.import_module("preprocess.ce_builder")

def test_import_cog_student():
    # check cog_student import
    importlib.import_module("preprocess.cog_student")

def test_import_questionnaire_student():
    # check questionnaire_student import
    importlib.import_module("preprocess.questionnaire_student")