import numpy as np
import pandas as pd

class CERASMonitor:
    """
    Monitors postprocess health of CERAS:
    - CNN behavioral drift
    - Calibration stability
    - Learning readiness distribution
    """

    def __init__(
        self,
        cnn_mean_ref,
        cnn_std_ref,
        calib_error_ref,
        readiness_mean_ref,
        at_risk_ratio_ref,
        tolerance=0.15
    ):
        
        """
        Parameters
        ----------
        *_ref : baseline reference values (from deployment time)
        tolerance : relative drift tolerance (15% default)
        """

        self.cnn_mean_ref = cnn_mean_ref
        self.cnn_std_ref = cnn_std_ref
        self.calib_error_ref = calib_error_ref
        self.readiness_mean_ref = readiness_mean_ref
        self.at_risk_ratio_ref = at_risk_ratio_ref
        self.tolerance = tolerance

    @staticmethod
    def _relative_change(current, reference):
        return (current - reference) / (reference + 1e-8)
    
    def monitor(
        self,
        cnn_scores,
        calib_error_current,
        readiness_scores,
        readiness_labels
    ):
        """
        Run monitoring checks on a new batch.
        """

        cnn_scores = np.asarray(cnn_scores)
        readiness_scores = np.asarray(readiness_scores)

        #CNN behavior statistics
        cnn_mean = cnn_scores.mean()
        cnn_std = cnn_scores.std()

        cnn_mean_shift = self._relative_change(cnn_mean, self.cnn_mean_ref)
        cnn_std_shift = self._relative_change(cnn_std, self.cnn_std_ref)
        cnn_drift = (
            abs(cnn_mean_shift) > self.tolerance or
            abs(cnn_std_shift) > self.tolerance
        )

        #Calibration drift
        calib_shift = self._relative_change(calib_error_current, self.calib_error_ref)
        calibration_drift = abs(calib_shift) > self.tolerance

        #Learning readiness distribution
        readiness_mean = readiness_scores.mean()
        readiness_shift = self._relative_change(readiness_mean, self.readiness_mean_ref)

        at_risk_ratio = np.mean(np.array(readiness_labels)=="At Risk")
        at_risk_ratio_shift = self._relative_change(at_risk_ratio, self.at_risk_ratio_ref)
        
        readiness_drift = (
            abs(readiness_shift) > self.tolerance or
            abs(at_risk_ratio_shift) > self.tolerance
        )

        #Final Monitoring Report
        return {
            "cnn_mean_current": cnn_mean,
            "cnn_std_current": cnn_std,
            "cnn_drift": cnn_drift,
            "calibration_error_current": calib_error_current,
            "calibration_drift": calibration_drift,
            "readiness_mean_current": readiness_mean,
            "at_risk_ratio_current": at_risk_ratio,
            "readiness_drift": readiness_drift
        }