# CERAS Dual-Pipeline Architecture

This document visualizes the core system architecture of CERAS, showing both the **data analysis pipeline** and the **reasoning engine pipeline**, which merge into a unified **inference engine** for cognitive readiness prediction.

```mermaid
flowchart LR
  %% Source & ingest (unchanged)
  A["Data Sources\n(Behavioral signals, Interaction logs,\nCognitive self-reports, Prompts & Gold answers)"] --> B["Ingest & Storage\n- Raw event store (time-series)\n- Prompt / response store\n- Audit logs"]

  %% Left: Pre-feature additions 
  B --> B1["Data Quality & Contextual Enrichment\n- Context tagging (course, task type, difficulty)\n- Session validation & idle-time filtering\n- Meta-features (engagement duration, attention ratio)"]
  B1 --> B2["Behavioral Signal Fusion\n- Align multimodal signals (keystrokes, clicks, sensors)\n- Temporal synchronization & resampling\n- Missing-modality fallback / fusion embeddings"]
  B2 --> B3["Preprocessing Auditor\n- Anomaly detection on raw streams\n- Early drift detection (feature distribution checks)\n- Audit logs & flagging for manual review"]
  B3 --> C["Feature Engineering & Data Quality\n- Normalization & scaling\n- Temporal smoothing / windows\n- Sessionization & alignment\n- Embeddings & derived features\n- Missingness modelling & imputation\n- Outlier detection & audit"]

  C --> D["CEPM: Cognitive Efficiency Prediction Model\n(GRU / Temporal Encoder -> XGBoost / head)\n+ Explainability (SHAP)"]
  D --> E["CE Score (0–100)\n+ CE time-series (per session/task)"]

  %% Left: Post-score additions 
  E --> E1["Calibration & Confidence Estimation\n- Platt / isotonic calibration\n- Per-session confidence intervals\n- Reliability reports"]
  E1 --> E2["Behavioral Insight Analyzer\n- Trend detection (fatigue, engagement drops)\n- Moving-window analytics & pattern clustering\n- Alerts for anomalous sessions"]
  E2 --> E3["CE Model Monitor & Retraining Trigger\n- Monitor feature / performance drift\n- SHAP importance drift tracking\n- Automated retraining / human review triggers"]
  E3 --> K

  %% Right: Reasoning pipeline (CAMRE-EDU) - original flow
  B --> F["Student Reasoning Trace\n(time-stamped steps / tokenized trace)"]
  F --> G["Trace Processing & Step Segmentation\n- Tokenization\n- Step bundling heuristics\n- Remove noise / de-dup"]
  G --> H["Decomposer (LLM) — STRICT JSON output\n(Break query into atomic subtasks)"]

  %% New: Decomposer improvements & verifier loop
  H --> V["Subtask Verifier (LLM classifier)\n- Input: original query + JSON subtasks\n- Output: {approved: yes/no, confidence, revised_subtasks?}"]
  V -- yes --> I["CAMRE-EDU Reasoning Analysis Engine\n- ToT graph alignment\n- Embedding semantic scoring\n- Coverage / verifier agreement\n- Granularity / redundancy / coherence metrics"]
  V -- no --> H2["Decomposer (LLM) — re-run\n(Verifier feedback → refine JSON subtasks)"]
  H2 --> V

  I --> J["RDS: Reasoning Diagnostic Score (0.00–1.00)\n+ Breakpoints & Diagnostics (missing_concepts, per-step scores)"]

  %% -----------------------
  %% NEW SUBSYSTEM: Tree-of-Thoughts orchestration & candidate processing
  %% -----------------------
  H --> T0["Pipeline Normalizer\n- pipeline_to_candidates(): extract final_subtasks/suggestions/outputs\n- permissive parsing of nested metadata"]
  T0 --> T1["Candidate Enricher & Deduper\n- extract hidden text from metadata\n- dedupe_candidates_by_text()"]
  T1 --> T2["Per-Candidate Scorer / Verifier\n- run_inference_pipeline(candidate, auto_extend=False)\n- combined_reasoning_score(candidate)\n- attach per-candidate reasoning_snapshot"]
  T2 --> T3["Tree-of-Thoughts Builder\n- add_node(role=textual role + provenance)\n- attach scores, metadata, timestamps"]
  T3 --> T4["Path-level ToT Beam Search (optional)\n- path-level beam expansion\n- rerank full paths by summed combined_score"]
  T3 --> T5["Greedy Expansion (beam/depth)\n- expand children (beam), grandchildren (depth)\n- merge duplicate candidates during expansion"]

  %% Merge duplicates & provenance
  T5 --> T6["Duplicate Merge & Provenance\n- merge_duplicate_nodes_marking()\n- reparent children to best node\n- keep metadata.merged_into trace"]
  T6 --> T7["Tree Store & Exports\n- tree_of_thoughts_example.json\n- pipeline_debug.log\n- pipeline_1_output (in-memory dict)"]

  %% Visualization & outputs
  T7 --> VIZ["Visualizer & Exporter\n- pipeline_output_generator.py\n- Save: full PNG, substantive PNG\n- Save: pipeline_1_output (var), JSON snapshot, optional HTML viewer"]
  VIZ --> IMG["Tree images (.png) + interactive HTML viewer (optional)"]
  VIZ --> ARTIFACTS["Artifacts\n- top_nodes, best_path, reasoning_diagnostic\n- pipeline_debug.log\n- provenance metadata"]

  %% Merge with RDS & Inference
  J --> K["Inference Engine\n(Combine CE & RDS -> Learning Readiness + Explanations)"]
  T7 --> K
  VIZ --> K

  %% Monitoring & Research layer
  K --> L["Research & Analytics Layer\n- CE / RDS time series\n- Task-level reports\n- Archetype clustering\n- Model monitoring & drift detection\n- Dashboards / exports"]

  %% Audit & Provenance store link
  B --> AUDIT["Provenance & Audit Store\n- model versions, timestamps, node provenance\n- pipeline_debug.log\n- data lineage"]
  T7 --> AUDIT
  VIZ --> AUDIT
  D --> AUDIT
  I --> AUDIT

  %% Optional tooling & operator interfaces
  L --> OPS["Operator UI / QA\n- Inspect tree, click node -> metadata\n- Trigger re-run or manual edit\n- Export examples for retraining"]
  OPS --> D3["Retraining Workbench\n- sample selection, human labels, pipeline replay"]

  style T0 fill:#f8f1f1,stroke:#b44
  style T3 fill:#eef7ff,stroke:#2b7
  style VIZ fill:#f2fff0,stroke:#07a
  style AUDIT fill:#fff3e6,stroke:#b95
