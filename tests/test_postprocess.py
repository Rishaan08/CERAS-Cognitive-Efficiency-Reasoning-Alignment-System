import numpy as np
import pandas as pd
import os
import tempfile
import importlib

#Calibration Test
def test_calibrator_fit_evaluate_and_save_load():
    from postprocess.calibrator import CECalibrator

    y_true = np.array([0.2, 0.4, 0.6, 0.8])
    y_pred = np.array([0.25, 0.35, 0.55, 0.75])

    calibrator = CECalibrator(min_ce=0.0, max_ce=1.0)

    calibrator.fit(y_true, y_pred)

    metrics = calibrator.evaluate(y_true, y_pred)

    assert isinstance(metrics, dict)
    assert "brier_before" in metrics
    assert "brier_after" in metrics
    assert "improvement" in metrics

    assert metrics["brier_after"] <= metrics["brier_before"]

    #Save & Load
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "calibrator.joblib")
        calibrator.save(path)

        new_calibrator = CECalibrator()
        new_calibrator.load(path)

        y_cal = new_calibrator.predict(y_pred)
        assert len(y_cal) == len(y_pred)

# Fusion Insight Test
def test_fusion_produces_diagnostics_and_confidence():
    from ceras.fusion import CERASFusion

    fusion = CERASFusion()

    student_ids = [1, 2, 3]

    cepm = np.array([0.3, 0.6, 0.85])   
    cnn  = np.array([0.4, 0.65, 0.9])   
    anfis = np.array([0.2, 0.6, 0.9])   

    df = fusion.fuse(
        student_ids=student_ids,
        cepm_scores=cepm,
        cnn_scores=cnn,
        anfis_scores=anfis
    )

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3

    required_cols = {
        "student_id",
        "fused_ce_score",
        "confidence",
        "diagnostics"
    }

    for c in required_cols:
        assert c in df.columns

    # Diagnostics must explain what's wrong
    for d in df["diagnostics"]:
        assert isinstance(d, dict)
        assert {
            "concept_gap",
            "effort_gap",
            "strategy_gap",
            "high_disagreement"
        }.issubset(d.keys())

#Monitoring Test
def test_model_monitor_basic_run():
    from postprocess.model_monitor import CERASMonitor

    monitor = CERASMonitor(
        cnn_mean_ref=0.6,
        cnn_std_ref=0.1,
        calib_error_ref=0.05,
        readiness_mean_ref=0.65,
        at_risk_ratio_ref=0.2,
        tolerance=0.2
    )

    cnn_scores = np.array([0.55, 0.6, 0.7, 0.65])
    calib_error_current = 0.04
    readiness_scores = np.array([0.5, 0.6, 0.8, 0.7])
    readiness_labels = ["Needs Support", "Ready", "Highly Ready", "Ready"]

    report = monitor.monitor(
        cnn_scores=cnn_scores,
        calib_error_current=calib_error_current,
        readiness_scores=readiness_scores,
        readiness_labels=readiness_labels
    )

    assert isinstance(report, dict)

    expected_keys = {
        "cnn_mean_current",
        "cnn_std_current",
        "cnn_drift",
        "calibration_error_current",
        "calibration_drift",
        "readiness_mean_current",
        "at_risk_ratio_current",
        "readiness_drift"
    }

    for k in expected_keys:
        assert k in report
        assert report[k] is not None