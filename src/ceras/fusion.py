import numpy as np
import pandas as pd


class CERASFusion:
    """
    Fusion engine for CERAS.

    Combines:
    - CEPM (cognitive strength)
    - calibrated CNN (behavioral patterns)
    - ANFIS (reasoning alignment)

    Produces:
    - fused CE score
    - readiness label
    - confidence score
    - diagnostic flags
    """

    def __init__(
        self,
        w_cepm=0.5,
        w_cnn=0.35,
        w_anfis=0.15,
        disagreement_threshold=0.25
    ):
        self.w_cepm = w_cepm
        self.w_cnn = w_cnn
        self.w_anfis = w_anfis
        self.disagreement_threshold = disagreement_threshold

    #Core Helpers
    @staticmethod
    def _to_numpy(x):
        return np.asarray(x, dtype=float)

    @staticmethod
    def _clip01(x):
        return np.clip(x, 0.0, 1.0)

    #Readiness Label
    @staticmethod
    def _readiness_label(score):
        score = float(score)

        if score >= 0.75:
            return "Highly Ready"
        elif score >= 0.60:
            return "Ready"
        elif score >= 0.45:
            return "Needs Support"
        else:
            return "At Risk"

    #Diagnostics 
    def _diagnostics(self, cepm, cnn, anfis):
        return {
            "concept_gap": bool(cepm < 0.45),
            "effort_gap": bool(cnn < 0.45),
            "strategy_gap": bool(anfis < 0.40),
            "high_disagreement": bool(
                abs(float(cepm) - float(cnn)) > self.disagreement_threshold
            )
        }

    #Confidence 
    def _confidence(self, cepm, cnn):
        cepm = float(self._clip01(cepm))
        cnn = float(self._clip01(cnn))
        return float(1.0 - abs(cepm - cnn))

    #Main Fusion API
    def fuse(
        self,
        student_ids,
        cepm_scores,
        cnn_scores,
        anfis_scores
    ) -> pd.DataFrame:

        cepm = self._clip01(self._to_numpy(cepm_scores))
        cnn = self._clip01(self._to_numpy(cnn_scores))
        anfis = self._clip01(self._to_numpy(anfis_scores))

        #Weighted Fusion
        fused_raw = (
            self.w_cepm * cepm +
            self.w_cnn * cnn +
            self.w_anfis * anfis
        )

        fused_raw = self._clip01(fused_raw)

        #Diagnostics & Confidence
        diagnostics = []
        confidence = []

        for c, b, a in zip(cepm, cnn, anfis):
            diagnostics.append(self._diagnostics(c, b, a))
            confidence.append(self._confidence(c, b))

        confidence = np.array(confidence, dtype=float)

        #ANFIS-aware Adjustment
        fused_adjusted = fused_raw.copy()

        for i, d in enumerate(diagnostics):
            if d["strategy_gap"]:
                fused_adjusted[i] -= 0.05
            elif (
                not d["concept_gap"]
                and not d["effort_gap"]
                and anfis[i] >= 0.50
            ):
                fused_adjusted[i] += 0.05

        fused_adjusted = self._clip01(fused_adjusted)

        #Readiness Labels
        labels = [self._readiness_label(s) for s in fused_adjusted]

        #Output Table
        return pd.DataFrame({
            "student_id": student_ids,
            "fused_ce_score": fused_adjusted.astype(float),
            "readiness_label": labels,
            "confidence": confidence.astype(float),
            "diagnostics": diagnostics
        })