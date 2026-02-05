from datetime import datetime

class CERASReport:
    """
    Generates structured monitoring reports for CERAS.
    """

    def generate(self, monitor_report, alerts):
        """
        Parameters
        ----------
        monitor_report : dict
            Output from CERASMonitor.monitor()
        alerts : list
            Output from CERASAlerts.generate()

        Returns
        -------
        report : dict
            System health report
        """

        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "system": "CERAS",
            "health_summary": {
                "cnn_drift": monitor_report.get("cnn_drift"),
                "calibration_drift": monitor_report.get("calibration_drift"),
                "readiness_drift": monitor_report.get("readiness_drift"),
            },
            "metrics": {
                "cnn_mean": monitor_report.get("cnn_mean_current"),
                "cnn_std": monitor_report.get("cnn_std_current"),
                "calibration_error": monitor_report.get("calibration_error_current"),
                "readiness_mean": monitor_report.get("readiness_mean_current"),
                "at_risk_ratio": monitor_report.get("at_risk_ratio_current"),
            },
            "alerts": alerts
        }

        return report