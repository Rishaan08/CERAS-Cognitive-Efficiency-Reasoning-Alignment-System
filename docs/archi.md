# CERAS Dual-Pipeline Architecture

This document visualizes the core system architecture of CERAS, showing both the **data analysis pipeline** and the **reasoning engine pipeline**, which merge into a unified **inference engine** for cognitive readiness prediction.

flowchart LR
  %% =========================
  %% SOURCE & INGEST (UNCHANGED)
  %% =========================
  A["Data Sources\n- Behavioral signals\n- Interaction logs\n- Cognitive self-reports\n- Prompts & gold answers"] 
    --> B["Ingest & Storage\n- Raw event store (time-series)\n- Prompt / response store\n- Audit logs"]

  %% =========================
  %% LEFT PIPELINE – PREPROCESSING (SHARED)
  %% =========================
  B --> B1["Data Quality & Contextual Enrichment\n- Context tagging (course, task, difficulty)\n- Session validation & idle filtering\n- Meta-features (engagement, attention ratio)"]

  B1 --> B2["Behavioral Signal Fusion\n- Align multimodal signals\n- Temporal synchronization\n- Missing-modality fallback"]

  B2 --> B3["Preprocessing Auditor\n- Anomaly detection\n- Early drift detection\n- Audit flags"]

  B3 --> C["Feature Engineering & Data Quality\n- Normalization & scaling\n- Temporal windows\n- Sessionization\n- Missingness modelling\n- Outlier detection"]

  %% =========================
  %% LEFT PIPELINE SPLIT
  %% =========================
  C --> PISA_PIPE
  C --> OULAD_PIPE

  %% =========================
  %% SUB-PIPELINE 1: PISA → CEPM
  %% =========================
  PISA_PIPE["PISA 2022 Dataset\n(COG + QQQ)"] 
    --> PISA_PRE["PISA Preprocessing\n- Merge COG + QQQ\n- Drop audit / version columns\n- CE construction"]

  PISA_PRE --> PISA_FS["4-Layer Feature Selection\nMI → RFE → LASSO → Boruta"]

  PISA_FS --> CEPM["CEPM (PISA-based)\nLightGBM / XGBoost\nPure Cognitive Efficiency"]

  CEPM --> CE_PISA["CE Score (Population-level)\n+ Confidence"]

  %% =========================
  %% SUB-PIPELINE 2: OULAD / MEU → CNN
  %% =========================
  OULAD_PIPE["OULAD + MEU Dataset\n(features_final.parquet)"]
    --> OULAD_PRE["Behavioral Preprocessing\n- Scaling\n- Session alignment"]

  OULAD_PRE --> CNN["Behavioral CNN Model\nConv1D-based"]

  CNN --> CNN_OUT["Behavioral Cognitive Modeling Output\n- CE prediction\n- Model explainability (SHAP)\n- Intention & engagement patterns\n- Latent behavioral representations"]

  %% =========================
  %% CE FUSION & ALIGNMENT
  %% =========================
  CE_PISA --> FUSION
  CE_CNN --> FUSION

  FUSION["CE Fusion & Alignment Layer\n- Weighting / agreement checks\n- Confidence estimation\n- Final CE"]

  %% =========================
  %% POSTPROCESS (LEFT)
  %% =========================
  FUSION --> POST1["Calibration & Confidence Estimation\n- Platt / Isotonic\n- Reliability curves"]

  POST1 --> POST2["Behavioral Insight Analyzer\n- Trend detection\n- Engagement & fatigue signals"]

  POST2 --> POST3["CE Model Monitor\n- Feature drift\n- Performance drift\n- Retraining triggers"]

  %% =========================
  %% RIGHT PIPELINE (UNCHANGED)
  %% =========================
  B --> F["Student Reasoning Trace\n(time-stamped steps)"]

  F --> G["Trace Processing & Step Segmentation\n- Tokenization\n- Step bundling\n- Noise removal"]

  G --> H["Decomposer (LLM)\nSTRICT JSON subtasks"]

  H --> V["Subtask Verifier (LLM)\nApprove / Reject + Feedback"]

  V -- yes --> I["CAMRE-EDU Reasoning Engine\n- ToT alignment\n- Semantic coverage\n- Coherence & redundancy"]

  V -- no --> H2["Decomposer Re-run\n(Verifier feedback)"]
  H2 --> V

  I --> J["RDS: Reasoning Diagnostic Score\n(0.00–1.00)"]

  %% =========================
  %% TREE-OF-THOUGHTS SYSTEM
  %% =========================
  H --> T0["Pipeline Normalizer"]
  T0 --> T1["Candidate Enricher & Deduper"]
  T1 --> T2["Per-Candidate Scoring"]
  T2 --> T3["Tree-of-Thoughts Builder"]
  T3 --> T5["Greedy / Beam Expansion"]
  T5 --> T6["Duplicate Merge & Provenance"]
  T6 --> T7["Tree Store & Exports"]

  T7 --> VIZ["Visualizer & Exporter\nPNG / JSON / HTML"]
  VIZ --> IMG["Tree Visuals"]
  VIZ --> ARTIFACTS["Reasoning Artifacts"]

  %% =========================
  %% FINAL INFERENCE
  %% =========================
  J --> K["Inference Engine\n(CE + RDS → Learning Readiness)"]
  POST3 --> K
  T7 --> K

  %% =========================
  %% MONITORING & AUDIT
  %% =========================
  K --> L["Research & Analytics Layer\n- CE/RDS time-series\n- Archetypes\n- Dashboards"]

  B --> AUDIT["Provenance & Audit Store"]
  CEPM --> AUDIT
  CNN --> AUDIT
  I --> AUDIT
  T7 --> AUDIT

  %% =========================
  %% STYLING
  %% =========================
  style CEPM fill:#f3f6ff,stroke:#3b6ef5
  style CNN fill:#f0fff4,stroke:#2e8b57
  style FUSION fill:#fff7e6,stroke:#e69500
  style CER44 fill:#eef7ff,stroke:#2b7
  style CER45 fill:#eef7ff,stroke:#2b7
  style AUDIT fill:#fff3e6,stroke:#b95
