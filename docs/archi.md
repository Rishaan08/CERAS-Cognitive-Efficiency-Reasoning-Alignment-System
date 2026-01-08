# CERAS Dual-Pipeline Architecture

This document visualizes the core system architecture of CERAS, showing both the **data analysis pipeline** and the **reasoning engine pipeline**, which merge into a unified **inference engine** for cognitive readiness prediction.


```mermaid
flowchart LR
  %% =========================
  %% Source & ingest (unchanged)
  %% =========================
  A["Data Sources\n(Behavioral signals, Interaction logs,\nCognitive self-reports, Prompts & Gold answers)"]
    --> B["Ingest & Storage\n- Raw event store (time-series)\n- Prompt / response store\n- Audit logs"]

  %% =========================
  %% LEFT PIPELINE — SHARED PREPROCESSING
  %% =========================
  B --> B1["Data Quality & Contextual Enrichment\n- Context tagging (course, task type, difficulty)\n- Session validation & idle-time filtering\n- Meta-features (engagement duration, attention ratio)"]

  B1 --> B2["Behavioral Signal Fusion\n- Align multimodal signals (keystrokes, clicks, sensors)\n- Temporal synchronization & resampling\n- Missing-modality fallback / fusion embeddings"]

  B2 --> B3["Preprocessing Auditor\n- Anomaly detection on raw streams\n- Early drift detection (feature distribution checks)\n- Audit logs & flagging for manual review"]

  B3 --> C["Feature Engineering & Data Quality\n- Normalization & scaling\n- Temporal smoothing / windows\n- Sessionization & alignment\n- Embeddings & derived features\n- Missingness modelling & imputation\n- Outlier detection & audit"]

  %% =========================
  %% LEFT PIPELINE — SUBPIPELINE 1 (PISA → CEPM)
  %% =========================
  C --> D1["CEPM (PISA-based Cognitive Model)\n- PISA assessment features\n- LightGBM / XGBoost\n- Static CE prediction"]

  D1 --> E1["CE Score (PISA)\n- Cognitive efficiency baseline"]

  %% =========================
  %% LEFT PIPELINE — SUBPIPELINE 2 (OULAD/MEU → CNN)
  %% =========================
  C --> D2["CNN Behavioral Model\n- OULAD + MEU behavioral signals\n- Convolutional encoder\n- Behavioral CE + explainability + intention"]

  D2 --> E2["Behavioral CE Intelligence\n- Behavioral CE estimate\n- SHAP explainability\n- Latent intention embeddings"]

  %% =========================
  %% LEFT PIPELINE — CE FUSION & POSTPROCESS
  %% =========================
  E1 --> FUSE["CE Fusion Layer\n- Combine PISA CE + Behavioral CE\n- Confidence-aware fusion"]

  E2 --> FUSE

  FUSE --> E["Unified CE Score (0–100)\n+ CE time-series (per session/task)"]

  E --> E4["Calibration & Confidence Estimation\n- Platt / isotonic calibration\n- Per-session confidence intervals\n- Reliability reports"]

  E4 --> E5["Behavioral Insight Analyzer\n- Trend detection (fatigue, engagement drops)\n- Moving-window analytics & pattern clustering\n- Alerts for anomalous sessions"]

  E5 --> E6["CE Model Monitor & Retraining Trigger\n- Monitor feature / performance drift\n- Explainability drift tracking\n- Automated retraining / human review triggers"]

  E6 --> K

  %% =========================
  %% RIGHT PIPELINE — CAMRE-EDU 
  %% =========================
  B --> F["Student Reasoning Trace\n(time-stamped steps / tokenized trace)"]

  F --> G["Trace Processing & Step Segmentation\n- Tokenization\n- Step bundling heuristics\n- Remove noise / de-dup"]

  G --> H["Decomposer (LLM) — STRICT JSON output\n(Break query into atomic subtasks)"]

  H --> V["Subtask Verifier (LLM classifier)\n- Input: original query + JSON subtasks\n- Output: {approved: yes/no, confidence, revised_subtasks?}"]

  V -- yes --> I["CAMRE-EDU Reasoning Analysis Engine\n- ToT graph alignment\n- Embedding semantic scoring\n- Coverage / verifier agreement\n- Granularity / redundancy / coherence metrics"]

  V -- no --> H2["Decomposer (LLM) — re-run\n(Verifier feedback → refine JSON subtasks)"]

  H2 --> V

  I --> J["RDS: Reasoning Diagnostic Score (0.00–1.00)\n+ Breakpoints & Diagnostics (missing_concepts, per-step scores)"]

  %% -----------------------
  %% Tree-of-Thoughts subsystem 
  %% -----------------------
  H --> T0["Pipeline Normalizer\n- pipeline_to_candidates()\n- permissive parsing"]

  T0 --> T1["Candidate Enricher & Deduper"]

  T1 --> T2["Per-Candidate Scorer / Verifier"]

  T2 --> T3["Tree-of-Thoughts Builder"]

  T3 --> T4["Path-level ToT Beam Search (optional)"]
  T3 --> T5["Greedy Expansion (beam/depth)"]

  T5 --> T6["Duplicate Merge & Provenance"]
  T6 --> T7["Tree Store & Exports"]

  T7 --> VIZ["Visualizer & Exporter"]
  VIZ --> IMG["Tree images (.png)"]
  VIZ --> ARTIFACTS["Artifacts"]

  %% =========================
  %% FINAL INFERENCE
  %% =========================
  J --> K["Inference Engine\n(Combine CE & RDS → Learning Readiness + Explanations)"]
  T7 --> K
  VIZ --> K

  %% =========================
  %% MONITORING & AUDIT
  %% =========================
  K --> L["Research & Analytics Layer\n- CE / RDS time series\n- Task-level reports\n- Archetype clustering\n- Model monitoring & drift detection\n- Dashboards / exports"]

  B --> AUDIT["Provenance & Audit Store\n- model versions\n- data lineage\n- pipeline_debug.log"]

  D1 --> AUDIT
  D2 --> AUDIT
  T7 --> AUDIT
  I --> AUDIT

  %% =========================
  %% STYLING
  %% =========================
  style D1 fill:#f3f6ff,stroke:#3b6ef5
  style D2 fill:#f0fff4,stroke:#2e8b57
  style FUSE fill:#fff7e6,stroke:#e69500
  style AUDIT fill:#fff3e6,stroke:#b95
