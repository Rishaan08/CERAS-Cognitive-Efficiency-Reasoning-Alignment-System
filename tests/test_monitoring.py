import numpy as np
import os
import tempfile

#Model Monitor Test
def test_ceras_monitor_basic():
    from postprocess.model_monitor import CERASMonitor

    monitor = CERASMonitor(
        cnn_mean_ref=0.6,
        cnn_std_ref=0.1,
        calib_error_ref=0.05,
        readiness_mean_ref=0.65,
        at_risk_ratio_ref=0.2,
        tolerance=0.2
    )

    report = monitor.monitor(
        cnn_scores=np.array([0.55, 0.6, 0.7]),
        calib_error_current=0.04,
        readiness_scores=np.array([0.5, 0.6, 0.8]),
        readiness_labels=["Needs Support", "Ready", "Highly Ready"]
    )

    assert isinstance(report, dict)

#Alerts Test
def test_alert_generation():
    from monitoring.alerts import CERASAlerts

    alerts_engine = CERASAlerts()

    monitor_report = {
        "cnn_drift": True,
        "calibration_drift": False,
        "readiness_drift": True
    }

    alerts = alerts_engine.generate(monitor_report)

    assert isinstance(alerts, list)
    assert len(alerts) >= 1

    for a in alerts:
        assert "type" in a
        assert "severity" in a
        assert "message" in a

#Monitoring Report Generation Test
def test_monitoring_report_generation():
    from monitoring.reports import CERASReport

    report_engine = CERASReport()

    monitor_report = {
        "cnn_drift": False,
        "calibration_drift": False,
        "readiness_drift": False,
        "cnn_mean_current": 0.6,
        "cnn_std_current": 0.1,
        "calibration_error_current": 0.04,
        "readiness_mean_current": 0.65,
        "at_risk_ratio_current": 0.2
    }

    alerts = [{
        "type": "SYSTEM_HEALTH",
        "severity": "Info",
        "message": "CERAS system operating within normal parameters."
    }]

    final_report = report_engine.generate(monitor_report, alerts)

    assert isinstance(final_report, dict)
    assert "timestamp" in final_report
    assert "health_summary" in final_report
    assert "metrics" in final_report
    assert "alerts" in final_report