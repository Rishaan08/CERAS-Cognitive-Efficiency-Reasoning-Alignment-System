import numpy as np
import joblib
from sklearn.isotonic import IsotonicRegression


class CECalibrator:
    """
    Calibration module for continuous Cognitive Efficiency (CE) regression.
    Uses Isotonic Regression and evaluates calibration via regression-Brier (MSE).
    """

    def __init__(self, min_ce=0.0, max_ce=1.0):
        self.calibrator = None
        self.min_ce = min_ce
        self.max_ce = max_ce

    def _normalize_ce(self, y):
        """
        Normalize CE scores to [0, 1].
        """
        y = np.asarray(y)
        return np.clip(
            (y - self.min_ce) / (self.max_ce - self.min_ce),
            0.0,
            1.0
        )

    @staticmethod
    def _regression_brier(y_true, y_pred):
        """
        Continuous Brier score (MSE on normalized scale).
        """
        return np.mean((y_true - y_pred) ** 2)

    def fit(self, y_true, y_pred):
        """
        Fit isotonic calibrator on validation data.
        """
        y_true_norm = self._normalize_ce(y_true)
        y_pred_norm = self._normalize_ce(y_pred)

        self.calibrator = IsotonicRegression(out_of_bounds="clip")
        self.calibrator.fit(y_pred_norm, y_true_norm)

    def predict(self, y_pred):
        """
        Apply calibration to predicted CE values.
        """
        if self.calibrator is None:
            raise RuntimeError("Calibrator has not been fitted.")

        y_pred_norm = self._normalize_ce(y_pred)
        return self.calibrator.predict(y_pred_norm)

    def brier_score(self, y_true, y_pred):
        """
        Regression Brier score (lower is better).
        """
        y_true_norm = self._normalize_ce(y_true)
        y_pred_norm = self._normalize_ce(y_pred)

        return self._regression_brier(y_true_norm, y_pred_norm)

    def evaluate(self, y_true, y_pred):
        """
        Compare calibration error before and after isotonic calibration.
        """
        brier_before = self.brier_score(y_true, y_pred)
        y_calibrated = self.predict(y_pred)
        brier_after = self._regression_brier(
            self._normalize_ce(y_true),
            y_calibrated
        )

        return {
            "brier_before": brier_before,
            "brier_after": brier_after,
            "improvement": brier_before - brier_after
        }

    def save(self, path):
        """
        Save calibrator to disk.
        """
        if self.calibrator is None:
            raise RuntimeError("Nothing to save. Calibrator not fitted.")

        payload = {
            "calibrator": self.calibrator,
            "min_ce": self.min_ce,
            "max_ce": self.max_ce
        }
        joblib.dump(payload, path)

    def load(self, path):
        """
        Load calibrator from disk.
        """
        payload = joblib.load(path)
        self.calibrator = payload["calibrator"]
        self.min_ce = payload["min_ce"]
        self.max_ce = payload["max_ce"]