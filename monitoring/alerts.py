class CERASAlerts:
    """
    Generates alert signals from CERAS monitoring outputs.
    Alerts are rule-based and human-interpretable.
    """

    def generate(self, monitor_report):
        """
        Parameters
        ----------
        monitor_report : dict
            Output from CERASMonitor.monitor()

        Returns
        -------
        alerts : list of dict
            List of active alerts
        """

        alerts = []

        if monitor_report.get("cnn_drift"):
            alerts.append({
                "type": "BEHAVIOR_DRIFT",
                "severity": "High",
                "message": "CNN behavioral patterns have drifted beyond acceptable range."
            })

        if monitor_report.get("calibration_drift"):
            alerts.append({
                "type": "CALIBARATION_DRIFT",
                "severity": "Critical",
                "message": "Calibration error has degraded. MOdel confidence may be unreliable."
            })

        if monitor_report.get("readiness_drift"):
            alerts.append({
                "type": "READINESS_SHIFT",
                "severity": "Medium",
                "message": "Learning readiness distribution has shifted significantly."
            })

        if not alerts:
            alerts.append({
                "type": "SYSTEM_HEALTH",
                "severity": "Info",
                "message": "CERAS system operating within normal parameters."
            })

        return alerts
