import os
import json
import pandas as pd
import numpy as np
import pytest

try:
    from features.feature_quality import (
        evaluate_feature_quality,
        feature_correlation_pairs,
        redundancy_prune_suggestions,
        save_feature_quality_report,
    )
except Exception:
    from features.feature_quality import (
        evaluate_feature_quality,
        feature_correlation_pairs,
        redundancy_prune_suggestion,
        save_feature_quality_report,
    )

PROCESSED = os.path.join("data", "processed")
FEATURES_PATH = os.path.join(PROCESSED, "features_final.parquet")
REPORT_PATH = os.path.join(PROCESSED, "feature_quality_test.json")


def make_synthetic_features():
    rng = np.random.RandomState(0)
    n = 200
    df = pd.DataFrame({
        "student_id": [f"s{i:03d}" for i in range(n)],
        "pisa_accuracy": rng.rand(n) * 1.0,
        "pisa_time": rng.rand(n) * 60 + 1,
        "oulad_avg_clicks": rng.poisson(5, size=n).astype(float),
        "oulad_active_days": rng.randint(1, 30, size=n).astype(float),
        "constant_col": np.ones(n),
        "sparse_col": np.concatenate([np.repeat(np.nan, n - 5), rng.rand(5)])
    })
    df["pisa_accuracy2"] = df["pisa_accuracy"] * 0.98 + rng.rand(n) * 0.02
    return df


@pytest.fixture(scope="module")
def features_df():
    if os.path.exists(FEATURES_PATH):
        df = pd.read_parquet(FEATURES_PATH)
        if not any(df.select_dtypes(include=[np.number]).columns):
            df["__synthetic_num"] = np.arange(len(df))
        return df
    else:
        return make_synthetic_features()

def test_evaluate_feature_quality_creates_metrics(features_df):
    metrics = evaluate_feature_quality(features_df)
    assert isinstance(metrics, dict)
    assert len(metrics) >= 1
    nums = [c for c in features_df.columns if features_df[c].dtype.kind in "biufc"]
    assert nums, "no numeric column found in test dataframe"
    col = nums[0]
    assert col in metrics
    entry = metrics[col]
    assert "n_missing" in entry and "pct_missing" in entry
    assert "mean" in entry and "std" in entry


def test_correlation_pairs_and_redundancy(features_df):
    pairs = feature_correlation_pairs(features_df, threshold=0.9)
    assert isinstance(pairs, list)
    if "pisa_accuracy2" in features_df.columns:
        assert any("pisa_accuracy" in (a + b) for a, b, _ in pairs) or len(pairs) > 0
    suggestions = redundancy_prune_suggestions(features_df, corr_threshold=0.9)
    assert isinstance(suggestions, list)
    if pairs:
        assert suggestions
        s0 = suggestions[0]
        assert "pair" in s0 and "suggested_drop" in s0 and "suggested_keep" in s0

def test_save_feature_quality_report_writes_file(features_df, tmp_path):
    metrics = evaluate_feature_quality(features_df)
    out_dir = tmp_path.as_posix()
    out_path = save_feature_quality_report(metrics, out_dir=out_dir, filename="feature_quality_test.json")
    assert os.path.exists(out_path)
    with open(out_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert isinstance(loaded, dict)