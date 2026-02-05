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
    - human-readable insight
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
            "concept_gap": cepm < 0.45,
            "effort_gap": cnn < 0.45,
            "strategy_gap": anfis < 0.40,
            "high_disagreement": abs(cepm - cnn) > self.disagreement_threshold
        }

    #Confidence
    def _confidence(self, cepm, cnn):
        cepm = float(self._clip01(cepm))
        cnn = float(self._clip01(cnn))
        return 1.0 - abs(cepm - cnn)

    #Human-readable Insight
    @staticmethod
    def _build_insight(diagnostics, cepm, cnn, anfis):
        reasons = []

        if diagnostics["concept_gap"]:
            reasons.append(
                "weak conceptual understanding, indicating gaps in core knowledge"
            )

        if diagnostics["effort_gap"]:
            reasons.append(
                "low behavioral engagement, suggesting insufficient learning effort"
            )

        if diagnostics["strategy_gap"]:
            reasons.append(
                "ineffective learning strategies, reducing learning efficiency"
            )

        if diagnostics["high_disagreement"]:
            reasons.append(
                "inconsistent cognitive and behavioral signals, indicating unstable learning patterns"
            )

        if not reasons:
            return (
                "The learner demonstrates a strong cognitive foundation, "
                "consistent engagement, and well-aligned learning strategies."
            )

        return "The learner shows " + "; ".join(reasons) + "."

    #Main Fusion API
    def fuse(
        self,
        student_ids,
        cepm_scores,
        cnn_scores,
        anfis_scores
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
        student_ids : array-like
        cepm_scores : array-like
        cnn_scores : array-like (calibrated)
        anfis_scores : array-like
        """

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

        confidence = self._clip01(np.array(confidence))

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

        #Labels & Insights
        labels = [self._readiness_label(s) for s in fused_adjusted]

        insights = [
            self._build_insight(d, c, b, a)
            for d, c, b, a in zip(diagnostics, cepm, cnn, anfis)
        ]

        #Output Table
        return pd.DataFrame({
            "student_id": student_ids,
            "fused_ce_score": fused_adjusted,
            "readiness_label": labels,
            "confidence": confidence,
            "insight": insights,
            "diagnostics": diagnostics
        })