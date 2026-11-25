# AquaSentinel UK — 300-Skill Engineering & Domain Capability Matrix

This matrix maps **300 concrete capabilities** across 6 core engineering and regulatory pillars directly to implemented files and symbols in `aquasentinel-uk`.

## 1. Real-Time Hydrology IoT Streaming & Data Contracts (001–050)
**Primary Scope**: `src/ingestion/live_hydrology_connectors.py` & `src/ingestion/hydrology_telemetry.py`

| ID | Capability | Repository File | Symbol / Artifact |
|---:|---|---|---|
| 001 | Live EA Level Measures Polling | `src/ingestion/live_hydrology_connectors.py` | `EA_LEVEL_MEASURES_URL / _fetch_ea_measures` |
| 002 | Live EA Rainfall Measures Polling | `src/ingestion/live_hydrology_connectors.py` | `EA_RAINFALL_MEASURES_URL / _fetch_ea_measures` |
| 003 | HTTPX Connection Pooling & Timeouts | `src/ingestion/live_hydrology_connectors.py` | `_fetch_ea_measures` |
| 004 | Tenacity Exponential Backoff Retries | `src/ingestion/live_hydrology_connectors.py` | `_fetch_ea_measures (@retry)` |
| 005 | Stop-After-3-Attempts Circuit Guard | `src/ingestion/live_hydrology_connectors.py` | `stop_after_attempt(3)` |
| 006 | HTTP Status Error Raising | `src/ingestion/live_hydrology_connectors.py` | `resp.raise_for_status` |
| 007 | Live Feed Healthy Status Tagging | `src/ingestion/live_hydrology_connectors.py` | `LIVE_EA_FEED_OK` |
| 008 | Graceful Degraded Feed Fallback | `src/ingestion/live_hydrology_connectors.py` | `LIVE_EA_FEED_DEGRADED` |
| 009 | Zero-Measurement Payload Guard | `src/ingestion/live_hydrology_connectors.py` | `fetch_live_uk_ea_hydrology_and_edm` |
| 010 | JSON LatestReading Extraction | `src/ingestion/live_hydrology_connectors.py` | `_extract_valid_ea_readings` |
| 011 | Finite Float Coercion Guard | `src/ingestion/live_hydrology_connectors.py` | `_extract_valid_ea_readings` |
| 012 | EA River Level Physical Plausibility Filter | `src/ingestion/live_hydrology_connectors.py` | `_extract_valid_ea_readings (0-12m)` |
| 013 | EA Rainfall Physical Plausibility Filter | `src/ingestion/live_hydrology_connectors.py` | `_extract_valid_ea_readings (0-100mm)` |
| 014 | Station Reference & Notation Parsing | `src/ingestion/live_hydrology_connectors.py` | `_extract_valid_ea_readings` |
| 015 | UTC ISO-8601 Timestamp Parsing | `src/ingestion/live_hydrology_connectors.py` | `pd.to_datetime(..., utc=True)` |
| 016 | Basin Inference from Gauge Metadata | `src/ingestion/live_hydrology_connectors.py` | `_infer_basin_from_text` |
| 017 | Thames Water (THM) CSO Calibration | `src/ingestion/live_hydrology_connectors.py` | `CALIBRATED_CSO_CATALOG` |
| 018 | United Utilities (UUW) CSO Calibration | `src/ingestion/live_hydrology_connectors.py` | `CALIBRATED_CSO_CATALOG` |
| 019 | Severn Trent (SVT) CSO Calibration | `src/ingestion/live_hydrology_connectors.py` | `CALIBRATED_CSO_CATALOG` |
| 020 | Yorkshire Water (YKS) CSO Calibration | `src/ingestion/live_hydrology_connectors.py` | `CALIBRATED_CSO_CATALOG` |
| 021 | Live Gauge Median Level Anchoring | `src/ingestion/live_hydrology_connectors.py` | `_build_calibrated_cso_telemetry` |
| 022 | Live Gauge Mean Rainfall Anchoring | `src/ingestion/live_hydrology_connectors.py` | `_build_calibrated_cso_telemetry` |
| 023 | 15-Minute Temporal Lookback Backfill | `src/ingestion/live_hydrology_connectors.py` | `_build_calibrated_cso_telemetry` |
| 024 | Power-Law Stage-Discharge Flow Synthesis | `src/ingestion/live_hydrology_connectors.py` | `_build_calibrated_cso_telemetry` |
| 025 | Turbidity & DO Inverse Coupling | `src/ingestion/live_hydrology_connectors.py` | `_build_calibrated_cso_telemetry` |
| 026 | Controlled Fault Injection for QA | `src/ingestion/live_hydrology_connectors.py` | `_build_calibrated_cso_telemetry` |
| 027 | Live Connector Attrs Telemetry Metadata | `src/ingestion/live_hydrology_connectors.py` | `fetch_live_uk_ea_hydrology_and_edm` |
| 028 | Pandera Schema Contract Definition | `src/ingestion/hydrology_telemetry.py` | `HydrologyEDMSchema` |
| 029 | 15-Min UTC Timestamp Schema Check | `src/ingestion/hydrology_telemetry.py` | `HydrologyEDMSchema (timestamp)` |
| 030 | UK Water Company Code Allowlist | `src/ingestion/hydrology_telemetry.py` | `VALID_WATER_COMPANY_CODES` |
| 031 | UK River Basin Allowlist | `src/ingestion/hydrology_telemetry.py` | `VALID_BASINS` |
| 032 | River Level Bounds Contract (0-12 m) | `src/ingestion/hydrology_telemetry.py` | `HydrologyEDMSchema (river_level_m)` |
| 033 | River Flow Bounds Contract (0-1500 m3/s) | `src/ingestion/hydrology_telemetry.py` | `HydrologyEDMSchema (flow_m3s)` |
| 034 | Dissolved Oxygen Bounds Contract (0-20 mg/L) | `src/ingestion/hydrology_telemetry.py` | `HydrologyEDMSchema (dissolved_oxygen_mg_l)` |
| 035 | Turbidity Bounds Contract (0-2000 NTU) | `src/ingestion/hydrology_telemetry.py` | `HydrologyEDMSchema (turbidity_ntu)` |
| 036 | 15-Min Rainfall Bounds Contract (0-100 mm) | `src/ingestion/hydrology_telemetry.py` | `HydrologyEDMSchema (rainfall_mm_15m)` |
| 037 | EDM Spill Binary Indicator Validation | `src/ingestion/hydrology_telemetry.py` | `HydrologyEDMSchema (edm_spill_active)` |
| 038 | EDM 15-Min Duration Bounds (0-15 min) | `src/ingestion/hydrology_telemetry.py` | `HydrologyEDMSchema (edm_duration_mins)` |
| 039 | Stuck/Flatlined Sensor Run Detection | `src/ingestion/hydrology_telemetry.py` | `flag_stuck_sensors` |
| 040 | Zero-Variance Window Tolerance Check | `src/ingestion/hydrology_telemetry.py` | `flag_stuck_sensors (tolerance=1e-9)` |
| 041 | Original Row Order Preservation | `src/ingestion/hydrology_telemetry.py` | `flag_stuck_sensors` |
| 042 | UK GDPR HMAC-SHA256 Pseudonymisation | `src/ingestion/hydrology_telemetry.py` | `pseudonymise_identifier` |
| 043 | Configurable GDPR Salt Environment Key | `src/ingestion/hydrology_telemetry.py` | `AQUASENTINEL_GDPR_SALT` |
| 044 | Raw Reporter & Landowner PII Stripping | `src/ingestion/hydrology_telemetry.py` | `apply_gdpr_pseudonymisation` |
| 045 | Lazy Pandera Row Failure Extraction | `src/ingestion/hydrology_telemetry.py` | `_extract_failure_reasons` |
| 046 | Dead-Letter Quarantine Parquet Writer | `src/ingestion/hydrology_telemetry.py` | `write_quarantine_parquet` |
| 047 | Quarantine Reason Audit Column | `src/ingestion/hydrology_telemetry.py` | `ingest_hydrology_telemetry` |
| 048 | DuckDB Analytical Table Persistence | `src/ingestion/hydrology_telemetry.py` | `persist_telemetry_duckdb` |
| 049 | Structured Ingestion Summary Dataclass | `src/ingestion/hydrology_telemetry.py` | `IngestionResult` |
| 050 | Post-Validation Clean Re-Verification | `src/ingestion/hydrology_telemetry.py` | `ingest_hydrology_telemetry` |

## 2. NetworkX Directed River Catchment Graphs & Routing (051–100)
**Primary Scope**: `src/analytics/river_network.py`

| ID | Capability | Repository File | Symbol / Artifact |
|---:|---|---|---|
| 051 | Directed Catchment Graph Construction | `src/analytics/river_network.py` | `build_uk_river_network` |
| 052 | CSO Outfall Node Attribution | `src/analytics/river_network.py` | `build_uk_river_network (node_type='cso')` |
| 053 | EA Monitoring Station Node Attribution | `src/analytics/river_network.py` | `build_uk_river_network (node_type='ea_station')` |
| 054 | Designated Bathing Water Node Attribution | `src/analytics/river_network.py` | `build_uk_river_network (node_type='bathing_water')` |
| 055 | Thames Catchment Topology Modeling | `src/analytics/river_network.py` | `CSO_THM_MOGDEN -> BW_THM_CHISWICK` |
| 056 | Chalk Stream Sensitivity Flagging | `src/analytics/river_network.py` | `CSO_THM_KENNET_CHALK (sensitive_site)` |
| 057 | Mersey Catchment Topology Modeling | `src/analytics/river_network.py` | `CSO_UUW_SALFORD -> BW_UUW_CROSBY` |
| 058 | Severn Catchment Topology Modeling | `src/analytics/river_network.py` | `CSO_SVT_WORCESTER -> BW_SVT_WESTON` |
| 059 | Aire Catchment Topology Modeling | `src/analytics/river_network.py` | `CSO_YKS_BRADFORD -> BW_YKS_HUMBER` |
| 060 | Geospatial Coordinate Node Metadata | `src/analytics/river_network.py` | `build_uk_river_network (lat/lon)` |
| 061 | Water Company Node Ownership Mapping | `src/analytics/river_network.py` | `build_uk_river_network (water_company_code)` |
| 062 | Catchment Type Stratification Attribute | `src/analytics/river_network.py` | `build_uk_river_network (catchment_type)` |
| 063 | Reach Distance Weighting (km) | `src/analytics/river_network.py` | `build_uk_river_network (distance_km)` |
| 064 | Mean Stream Velocity Parameterization (m/s) | `src/analytics/river_network.py` | `build_uk_river_network (mean_velocity_mps)` |
| 065 | Hydraulic Travel Time Edge Calculation | `src/analytics/river_network.py` | `build_uk_river_network (travel_time_hours)` |
| 066 | Shortest-Path Pollution Travel Time | `src/analytics/river_network.py` | `compute_pollution_travel_time` |
| 067 | Unknown Graph Node Validation Error | `src/analytics/river_network.py` | `compute_pollution_travel_time` |
| 068 | Dijkstra Multi-Hop Impact Discovery | `src/analytics/river_network.py` | `get_downstream_impacts` |
| 069 | Downstream Impact Dataclass Schema | `src/analytics/river_network.py` | `DownstreamImpact` |
| 070 | Cumulative Reach Distance Aggregation | `src/analytics/river_network.py` | `get_downstream_impacts (total_distance_km)` |
| 071 | Path Hop Count Calculation | `src/analytics/river_network.py` | `get_downstream_impacts (hops)` |
| 072 | Downstream Target Type Filtering | `src/analytics/river_network.py` | `get_downstream_impacts` |
| 073 | Arrival-Time Priority Sorting | `src/analytics/river_network.py` | `get_downstream_impacts` |
| 074 | Antecedent Precipitation Index Formula | `src/analytics/river_network.py` | `compute_antecedent_precipitation_index` |
| 075 | Configurable Exponential Recession Factor | `src/analytics/river_network.py` | `compute_antecedent_precipitation_index (decay_k)` |
| 076 | NaN-Safe Rainfall Series Handling | `src/analytics/river_network.py` | `compute_antecedent_precipitation_index` |
| 077 | Per-CSO Grouped Rolling Rainfall (24h) | `src/analytics/river_network.py` | `detect_dry_weather_spills (rainfall_24h_mm)` |
| 078 | 15-Min Rolling 96-Step Windowing | `src/analytics/river_network.py` | `detect_dry_weather_spills` |
| 079 | Per-CSO Chronological Merge Sorting | `src/analytics/river_network.py` | `detect_dry_weather_spills` |
| 080 | Dry-Weather Rainfall Threshold Check (<=0.25mm) | `src/analytics/river_network.py` | `detect_dry_weather_spills` |
| 081 | Dry-Weather API Threshold Check (<=2.0mm) | `src/analytics/river_network.py` | `detect_dry_weather_spills` |
| 082 | Minimum Spill Duration Gate (>=1.0 min) | `src/analytics/river_network.py` | `detect_dry_weather_spills` |
| 083 | Dry-Day Spill Anomaly Boolean Flagging | `src/analytics/river_network.py` | `detect_dry_weather_spills (dry_day_spill)` |
| 084 | Nearest Bathing Water Site Resolution | `src/analytics/river_network.py` | `detect_dry_weather_spills (downstream_target_site)` |
| 085 | Fallback to Nearest EA Station When No BW | `src/analytics/river_network.py` | `detect_dry_weather_spills` |
| 086 | Travel Time Feature Enrichment | `src/analytics/river_network.py` | `detect_dry_weather_spills (travel_time_to_target_h)` |
| 087 | Catchment Type Feature Enrichment | `src/analytics/river_network.py` | `detect_dry_weather_spills (catchment_type)` |
| 088 | Unknown CSO Graph Graceful Handling | `src/analytics/river_network.py` | `detect_dry_weather_spills` |
| 089 | Predecessor/Successor Neighbor Lookup | `src/app/main.py` | `_spatial_interpolate_reading` |
| 090 | Graph-Aware Sensor Dropout Spatial Fallback | `src/app/main.py` | `_spatial_interpolate_reading` |
| 091 | Multi-Basin Graph Inspection UI Selector | `src/app/dashboard.py` | `tab_network` |
| 092 | Downstream Impact Tabular Visualization | `src/app/dashboard.py` | `tab_network` |
| 093 | Sensitive Site Statutory Priority Linking | `src/analytics/river_network.py` | `DownstreamImpact.sensitive_site` |
| 094 | CSO-to-CSO Cascade Connectivity (Salford->Davyhulme) | `src/analytics/river_network.py` | `build_uk_river_network` |
| 095 | CSO-to-CSO Cascade Connectivity (Bradford->Knostrop) | `src/analytics/river_network.py` | `build_uk_river_network` |
| 096 | Tidal Estuary Route Modeling (Aire->Humber) | `src/analytics/river_network.py` | `build_uk_river_network` |
| 097 | API-Integrated Dry-Day Risk Scoring | `src/app/main.py` | `forecast_spills` |
| 098 | Graph Node & Edge Readiness Telemetry | `src/app/main.py` | `health` |
| 099 | Catchment Graph Verification Unit Tests | `tests/test_contracts_and_errors.py` | `test_river_network_travel_time_and_dry_day_spill` |
| 100 | End-to-End Graph Enrichment Pipeline Step | `src/run_pipeline.py` | `run_pipeline` |

## 3. Conformal Spill Forecasting & Sliced Error Analysis (101–150)
**Primary Scope**: `src/models/spill_and_vision.py` & `src/evaluation/error_analysis.py`

| ID | Capability | Repository File | Symbol / Artifact |
|---:|---|---|---|
| 101 | Multi-Horizon Forecasting (t+6h, t+12h, t+24h) | `src/models/spill_and_vision.py` | `HORIZONS` |
| 102 | 15-Min Step Horizon Mapping (24/48/96 steps) | `src/models/spill_and_vision.py` | `HORIZONS` |
| 103 | 14-Feature Hydrological Feature Matrix | `src/models/spill_and_vision.py` | `FEATURE_COLUMNS` |
| 104 | 1-Hour Lagged River Level Feature | `src/models/spill_and_vision.py` | `build_spill_feature_frame (river_level_lag_4)` |
| 105 | 1-Hour Lagged Flow Feature | `src/models/spill_and_vision.py` | `build_spill_feature_frame (flow_lag_4)` |
| 106 | 1-Hour Lagged Rainfall Feature | `src/models/spill_and_vision.py` | `build_spill_feature_frame (rainfall_lag_4)` |
| 107 | 1-Hour Rolling Turbidity Mean Feature | `src/models/spill_and_vision.py` | `build_spill_feature_frame (turbidity_roll_4)` |
| 108 | 1-Hour Rolling Dissolved Oxygen Mean Feature | `src/models/spill_and_vision.py` | `build_spill_feature_frame (do_roll_4)` |
| 109 | Urban Combined Sewer Indicator Encoding | `src/models/spill_and_vision.py` | `build_spill_feature_frame (is_urban_combined)` |
| 110 | Forward Spill Duration Summation Target | `src/models/spill_and_vision.py` | `build_spill_feature_frame (spill_duration_target_*)` |
| 111 | Forward Binary Spill Occurrence Target | `src/models/spill_and_vision.py` | `build_spill_feature_frame (spill_prob_target_*)` |
| 112 | Short-Series Boundary-Safe Target Windowing | `src/models/spill_and_vision.py` | `build_spill_feature_frame` |
| 113 | Basin-Grouped GroupKFold Cross-Validation | `src/models/spill_and_vision.py` | `train_conformal_spill_models (GroupKFold)` |
| 114 | Dynamic Safe Group Split Sizing | `src/models/spill_and_vision.py` | `_safe_group_splits` |
| 115 | LightGBM Spill Probability Classifier | `src/models/spill_and_vision.py` | `LGBMClassifier` |
| 116 | Single-Class Fold Graceful Probability Fallback | `src/models/spill_and_vision.py` | `train_conformal_spill_models` |
| 117 | Out-of-Fold Probability Calibration Tracking | `src/models/spill_and_vision.py` | `oof_prob` |
| 118 | LightGBM Spill Duration Base Regressor | `src/models/spill_and_vision.py` | `LGBMRegressor` |
| 119 | MAPIE CrossConformalRegressor (90% CI) | `src/models/spill_and_vision.py` | `CrossConformalRegressor` |
| 120 | Jackknife+ Conformal Conformity Scoring | `src/models/spill_and_vision.py` | `CrossConformalRegressor(method='plus')` |
| 121 | GroupKFold Conformal Calibration Alignment | `src/models/spill_and_vision.py` | `conformal_reg.fit_conformalize(..., groups=groups)` |
| 122 | Non-Negative Duration Prediction Clipping | `src/models/spill_and_vision.py` | `np.maximum(0.0, ...)` |
| 123 | Monotonic Conformal Lower/Upper Bounds | `src/models/spill_and_vision.py` | `interval_low / interval_high` |
| 124 | macOS OpenMP Thread-Safety Guard (n_jobs=1) | `src/models/spill_and_vision.py` | `OMP_NUM_THREADS / LGBM*(n_jobs=1)` |
| 125 | Horizon MAE & RMSE Computation | `src/models/spill_and_vision.py` | `mean_absolute_error / root_mean_squared_error` |
| 126 | Horizon PR-AUC & F1 Scoring | `src/models/spill_and_vision.py` | `average_precision_score / f1_score` |
| 127 | Empirical 90% Coverage Auditing | `src/models/spill_and_vision.py` | `coverage90_*` |
| 128 | Mean Conformal Interval Width Tracking | `src/models/spill_and_vision.py` | `interval_width_*` |
| 129 | HorizonModelResult Structured Container | `src/models/spill_and_vision.py` | `HorizonModelResult` |
| 130 | SpillModelSuite Unified Model Output | `src/models/spill_and_vision.py` | `SpillModelSuite` |
| 131 | Joblib Model Bundle Serialization | `src/models/spill_and_vision.py` | `data/processed/spill_models_bundle.pkl` |
| 132 | FastAPI Live Bundle Inference Loading | `src/app/main.py` | `_load_model_bundle` |
| 133 | Hydrological Heuristic Serving Fallback | `src/app/main.py` | `forecast_spills` |
| 134 | Sliced Error Analysis Framework | `src/evaluation/error_analysis.py` | `evaluate_error_slices` |
| 135 | Urban Combined Sewer Slice Audit | `src/evaluation/error_analysis.py` | `SLICE_NAMES[0]` |
| 136 | Rural Chalk Stream Slice Audit | `src/evaluation/error_analysis.py` | `SLICE_NAMES[1]` |
| 137 | Heavy Storm Event Slice Audit (>=10mm) | `src/evaluation/error_analysis.py` | `SLICE_NAMES[2]` |
| 138 | Adaptive Storm Quantile Fallback | `src/evaluation/error_analysis.py` | `evaluate_error_slices` |
| 139 | Dry-Weather Anomaly Slice Audit | `src/evaluation/error_analysis.py` | `SLICE_NAMES[3]` |
| 140 | Rare Dry-Day Spill PR-F1 Evaluation | `src/evaluation/error_analysis.py` | `_compute_slice_metrics (dry_day_pr_f1)` |
| 141 | Rare Dry-Day Spill PR-AUC Evaluation | `src/evaluation/error_analysis.py` | `_compute_slice_metrics (dry_day_pr_auc)` |
| 142 | Slice-Level 90% Conformal Coverage Audit | `src/evaluation/error_analysis.py` | `_compute_slice_metrics (conformal_coverage_90)` |
| 143 | Slice-Level PI Width Audit | `src/evaluation/error_analysis.py` | `_compute_slice_metrics (mean_ci90_width_mins)` |
| 144 | Empty Slice Zero-Division Safety | `src/evaluation/error_analysis.py` | `_compute_slice_metrics` |
| 145 | SliceMetrics Typed Dataclass | `src/evaluation/error_analysis.py` | `SliceMetrics` |
| 146 | GitHub Markdown Error Table Formatting | `src/evaluation/error_analysis.py` | `tabulate(..., tablefmt='github')` |
| 147 | Automated Error Report Generation | `src/evaluation/error_analysis.py` | `generate_error_analysis_report` |
| 148 | Diagnostic Operational Mitigation Notes | `src/evaluation/error_analysis.py` | `docs/ERROR_ANALYSIS_REPORT.md` |
| 149 | Conformal Coverage Verification Tests | `tests/test_contracts_and_errors.py` | `test_conformal_spill_forecasting_and_catchment_spill_net` |
| 150 | Error Analysis Report Integration Tests | `tests/test_contracts_and_errors.py` | `test_error_analysis_report_generation` |

## 4. PyTorch CatchmentSpillNet Spatio-Temporal & Plume Modeling (151–200)
**Primary Scope**: `src/models/spill_and_vision.py`

| ID | Capability | Repository File | Symbol / Artifact |
|---:|---|---|---|
| 151 | PyTorch nn.Module Architecture Definition | `src/models/spill_and_vision.py` | `CatchmentSpillNet` |
| 152 | Multi-Lag Sequence Feature Selection (6 channels) | `src/models/spill_and_vision.py` | `SEQ_FEATURE_COLUMNS` |
| 153 | Rainfall 15m Sequence Input Channel | `src/models/spill_and_vision.py` | `SEQ_FEATURE_COLUMNS (rainfall_mm_15m)` |
| 154 | River Level Sequence Input Channel | `src/models/spill_and_vision.py` | `SEQ_FEATURE_COLUMNS (river_level_m)` |
| 155 | River Flow Sequence Input Channel | `src/models/spill_and_vision.py` | `SEQ_FEATURE_COLUMNS (flow_m3s)` |
| 156 | Turbidity Sequence Input Channel | `src/models/spill_and_vision.py` | `SEQ_FEATURE_COLUMNS (turbidity_ntu)` |
| 157 | Dissolved Oxygen Sequence Input Channel | `src/models/spill_and_vision.py` | `SEQ_FEATURE_COLUMNS (dissolved_oxygen_mg_l)` |
| 158 | API Sequence Input Channel | `src/models/spill_and_vision.py` | `SEQ_FEATURE_COLUMNS (api_mm)` |
| 159 | 1D Temporal Convolution Layer (kernel=3) | `src/models/spill_and_vision.py` | `CatchmentSpillNet.temporal_conv` |
| 160 | GELU Non-Linear Temporal Activation | `src/models/spill_and_vision.py` | `CatchmentSpillNet.temporal_conv` |
| 161 | Batch-First GRU Recurrent Encoder | `src/models/spill_and_vision.py` | `CatchmentSpillNet.gru` |
| 162 | Final Hidden State Sequence Pooling | `src/models/spill_and_vision.py` | `CatchmentSpillNet.forward (h_n[-1])` |
| 163 | Static Catchment Metadata MLP Branch | `src/models/spill_and_vision.py` | `CatchmentSpillNet.static_mlp` |
| 164 | Travel Time Static Input Feature | `src/models/spill_and_vision.py` | `_build_sequence_tensors` |
| 165 | Urban Combined Sewer Static Flag | `src/models/spill_and_vision.py` | `_build_sequence_tensors` |
| 166 | Rolling Turbidity Static Context | `src/models/spill_and_vision.py` | `_build_sequence_tensors` |
| 167 | Rolling DO Static Context | `src/models/spill_and_vision.py` | `_build_sequence_tensors` |
| 168 | Temporal-Static Representation Concatenation | `src/models/spill_and_vision.py` | `torch.cat([temporal_repr, static_repr])` |
| 169 | Shared Fused Trunk with Dropout (0.1) | `src/models/spill_and_vision.py` | `CatchmentSpillNet.head` |
| 170 | Multi-Task Spill Probability Logit Head | `src/models/spill_and_vision.py` | `CatchmentSpillNet.spill_head` |
| 171 | Outfall Turbidity/Algal Plume Risk Head | `src/models/spill_and_vision.py` | `CatchmentSpillNet.plume_head` |
| 172 | Sigmoid Plume Probability Bounding (0-1) | `src/models/spill_and_vision.py` | `torch.sigmoid(self.plume_head(fused))` |
| 173 | Non-Negative Softplus Duration Head | `src/models/spill_and_vision.py` | `F.softplus(self.duration_head(fused))` |
| 174 | Outfall Plume Ground-Truth Proxy Synthesis | `src/models/spill_and_vision.py` | `build_spill_feature_frame (plume_risk_target)` |
| 175 | Turbidity Normalization Proxy Component | `src/models/spill_and_vision.py` | `build_spill_feature_frame` |
| 176 | DO Deficit Proxy Component | `src/models/spill_and_vision.py` | `build_spill_feature_frame` |
| 177 | Per-CSO Left-Padded Sequence Tensor Builder | `src/models/spill_and_vision.py` | `_build_sequence_tensors` |
| 178 | Zero-Mean Unit-Variance Sequence Scaling | `src/models/spill_and_vision.py` | `_build_sequence_tensors` |
| 179 | Epsilon Numerical Stability Guard (1e-5) | `src/models/spill_and_vision.py` | `_build_sequence_tensors` |
| 180 | Static Tensor Standardization | `src/models/spill_and_vision.py` | `_build_sequence_tensors` |
| 181 | Deterministic PyTorch RNG Seeding | `src/models/spill_and_vision.py` | `torch.manual_seed` |
| 182 | AdamW Optimizer with Weight Decay | `src/models/spill_and_vision.py` | `train_catchment_spill_net` |
| 183 | BCEWithLogitsLoss Spill Objective | `src/models/spill_and_vision.py` | `train_catchment_spill_net` |
| 184 | Weighted MSE Plume Risk Objective | `src/models/spill_and_vision.py` | `train_catchment_spill_net` |
| 185 | SmoothL1Loss (Huber) Duration Objective | `src/models/spill_and_vision.py` | `train_catchment_spill_net` |
| 186 | Mini-Batch Shuffled Epoch Iteration | `src/models/spill_and_vision.py` | `torch.randperm` |
| 187 | Adaptive Mini-Batch Size Clamping | `src/models/spill_and_vision.py` | `train_catchment_spill_net` |
| 188 | Train/Eval Mode Switching | `src/models/spill_and_vision.py` | `model.train() / model.eval()` |
| 189 | Gradient-Free Inference Context | `src/models/spill_and_vision.py` | `torch.no_grad()` |
| 190 | Plume Risk MAE Metric Computation | `src/models/spill_and_vision.py` | `torch_plume_mae` |
| 191 | Next-Window Spill Accuracy Tracking | `src/models/spill_and_vision.py` | `torch_spill_acc` |
| 192 | Final Multi-Task Loss Tracking | `src/models/spill_and_vision.py` | `torch_final_loss` |
| 193 | Feature Frame Plume Score Column Attachment | `src/models/spill_and_vision.py` | `pred_plume_risk` |
| 194 | PyTorch State Dict Bundle Persistence | `src/models/spill_and_vision.py` | `torch_model.state_dict()` |
| 195 | MLflow PyTorch Metric Logging | `src/models/spill_and_vision.py` | `mlflow.log_metrics(torch_metrics)` |
| 196 | Tensor Channel-First Conv Transpose Handling | `src/models/spill_and_vision.py` | `seq_x.transpose(1, 2)` |
| 197 | Tensor Squeeze Output Shape Alignment | `src/models/spill_and_vision.py` | `CatchmentSpillNet.forward` |
| 198 | Pipeline Summary Torch Metrics Reporting | `src/run_pipeline.py` | `run_pipeline (torch_metrics)` |
| 199 | Forward Tensor Shape & Bounds Assertions | `tests/test_contracts_and_errors.py` | `test_conformal_spill_forecasting_and_catchment_spill_net` |
| 200 | Integrated End-to-End Neural + Tree Suite | `src/models/spill_and_vision.py` | `train_conformal_spill_models` |

## 5. Guardrailed LangGraph Ofwat PR24 & Environment Act RAG (201–250)
**Primary Scope**: `src/agents/ofwat_copilot.py` & `data/regulatory_docs/`

| ID | Capability | Repository File | Symbol / Artifact |
|---:|---|---|---|
| 201 | Disk Statutory Corpus Loader | `src/agents/ofwat_copilot.py` | `load_ofwat_regulatory_corpus` |
| 202 | PDF Regulatory Text Extraction (pypdf) | `src/agents/ofwat_copilot.py` | `_extract_pdf_text (PdfReader)` |
| 203 | Markdown Statutory Section Parser | `src/agents/ofwat_copilot.py` | `_parse_markdown_sections` |
| 204 | Plain Text Statutory Document Ingestion | `src/agents/ofwat_copilot.py` | `load_ofwat_regulatory_corpus (.txt)` |
| 205 | Filename-Based Statutory Framework Inference | `src/agents/ofwat_copilot.py` | `_infer_framework_from_filename` |
| 206 | Deterministic Disk Doc ID Assignment | `src/agents/ofwat_copilot.py` | `_parse_markdown_sections` |
| 207 | Ofwat PR24 Final Determinations Corpus File | `data/regulatory_docs/ofwat_pr24_final_determinations.md` | `OFWAT-PR24-ODI-01 / PCD-02 / STORM-03` |
| 208 | Environment Act 2021 s.141A Corpus File | `data/regulatory_docs/environment_act_2021_section_141a.md` | `ENV-ACT-2021-S141A-01 / S141B-02` |
| 209 | EA EDM Permitting Rules Corpus File | `data/regulatory_docs/ea_edm_permitting_rules.md` | `EA-EDM-PERMIT-01 / DATA-02` |
| 210 | Builtin + Disk Corpus Deduplicated Merge | `src/agents/ofwat_copilot.py` | `load_ofwat_regulatory_corpus` |
| 211 | RegulatoryDocument Frozen Dataclass | `src/agents/ofwat_copilot.py` | `RegulatoryDocument` |
| 212 | BM25Okapi Sparse Lexical Indexing | `src/agents/ofwat_copilot.py` | `HybridRegulatoryRetriever` |
| 213 | TF-IDF Unigram+Bigram Vectorization | `src/agents/ofwat_copilot.py` | `TfidfVectorizer(ngram_range=(1, 2))` |
| 214 | Cosine Similarity Dense-Sparse Scoring | `src/agents/ofwat_copilot.py` | `cosine_similarity` |
| 215 | Max-Normalized Hybrid Fusion (0.55/0.45) | `src/agents/ofwat_copilot.py` | `HybridRegulatoryRetriever.retrieve` |
| 216 | Regex Legal Tokenization | `src/agents/ofwat_copilot.py` | `_tokenize` |
| 217 | Dynamic Custom Docs Dir Retrieval Support | `src/agents/ofwat_copilot.py` | `run_ofwat_compliance_copilot(docs_dir=...)` |
| 218 | LangGraph Typed State Schema | `src/agents/ofwat_copilot.py` | `CopilotState` |
| 219 | LangGraph Compiled Compliance StateGraph | `src/agents/ofwat_copilot.py` | `build_ofwat_copilot_graph` |
| 220 | Empty Query Input Guardrail | `src/agents/ofwat_copilot.py` | `_guardrail_node` |
| 221 | Prompt Injection & SQL/XSS Guardrail Regex | `src/agents/ofwat_copilot.py` | `_BLOCKED_PATTERNS / _guardrail_node` |
| 222 | Conditional Routing After Guardrail | `src/agents/ofwat_copilot.py` | `_route_after_guardrail` |
| 223 | Top-K Statutory Clause Retrieval Node | `src/agents/ofwat_copilot.py` | `_retrieve_node` |
| 224 | Relevance Score Filtering & Top-2 Floor | `src/agents/ofwat_copilot.py` | `_draft_compliance_node` |
| 225 | Inline Citation Bracket Formatting | `src/agents/ofwat_copilot.py` | `_draft_compliance_node` |
| 226 | Citation-Verifier Judge Node | `src/agents/ofwat_copilot.py` | `_citation_verifier_node` |
| 227 | Inline Citation Regex Extraction | `src/agents/ofwat_copilot.py` | `_citation_verifier_node` |
| 228 | Retrieved-Context Grounding Verification | `src/agents/ofwat_copilot.py` | `_citation_verifier_node` |
| 229 | Zero-Citation Rejection Trigger | `src/agents/ofwat_copilot.py` | `_citation_verifier_node` |
| 230 | Hallucinated Citation Detection | `src/agents/ofwat_copilot.py` | `_citation_verifier_node` |
| 231 | Conditional Fallback Routing After Judge | `src/agents/ofwat_copilot.py` | `_route_after_verifier` |
| 232 | Deterministic Statutory Fallback Node | `src/agents/ofwat_copilot.py` | `_deterministic_fallback_node` |
| 233 | Ofwat PR24 ODI Spill Count Calculator | `src/agents/ofwat_copilot.py` | `calculate_statutory_penalty_exposure` |
| 234 | Sensitive Site Spill Cap Halving (20 -> 10) | `src/agents/ofwat_copilot.py` | `calculate_statutory_penalty_exposure` |
| 235 | Sensitive Site ODI Rate Escalation (2x) | `src/agents/ofwat_copilot.py` | `calculate_statutory_penalty_exposure` |
| 236 | Excess Spill-Hour ODI Penalty (£1,250/hr) | `src/agents/ofwat_copilot.py` | `calculate_statutory_penalty_exposure` |
| 237 | Dry-Day Permit Breach Fine Exposure (£25k) | `src/agents/ofwat_copilot.py` | `calculate_statutory_penalty_exposure` |
| 238 | EDM Uptime Shortfall Penalty (<90%) | `src/agents/ofwat_copilot.py` | `calculate_statutory_penalty_exposure` |
| 239 | Environment Act s.141A Priority Site Flag | `src/agents/ofwat_copilot.py` | `breach_flags` |
| 240 | EA Dry-Weather Permit Breach Flag | `src/agents/ofwat_copilot.py` | `breach_flags` |
| 241 | Ofwat PR24 ODI Exceedance Flag | `src/agents/ofwat_copilot.py` | `breach_flags` |
| 242 | EA EDM Completeness Shortfall Flag | `src/agents/ofwat_copilot.py` | `breach_flags` |
| 243 | 3-Tier Statutory Compliance Classification | `src/agents/ofwat_copilot.py` | `COMPLIANT / AT_RISK / STATUTORY_BREACH` |
| 244 | Total GBP Regulatory Exposure Aggregation | `src/agents/ofwat_copilot.py` | `total_exposure_gbp` |
| 245 | Offline Zero-API-Key Deterministic Execution | `src/agents/ofwat_copilot.py` | `run_ofwat_compliance_copilot` |
| 246 | FastAPI Regulatory Audit Endpoint | `src/app/main.py` | `regulatory_audit` |
| 247 | Streamlit Interactive Copilot Audit Tab | `src/app/dashboard.py` | `tab_copilot` |
| 248 | Pipeline Automated Copilot Smoke Audit | `src/run_pipeline.py` | `run_pipeline` |
| 249 | PDF/MD/TXT Disk Corpus Unit Test | `tests/test_contracts_and_errors.py` | `test_load_ofwat_regulatory_corpus_pdf_md_txt` |
| 250 | Citation Verifier & Guardrail Unit Tests | `tests/test_contracts_and_errors.py` | `test_langgraph_ofwat_copilot_and_citation_verification` |

## 6. MLOps, Serving, Testing, CI/CD & Production Resilience (251–300)
**Primary Scope**: `pyproject.toml`, `Dockerfile`, `docker-compose.yml`, `src/app/`, `tests/`

| ID | Capability | Repository File | Symbol / Artifact |
|---:|---|---|---|
| 251 | Python 3.12 Virtual Environment Isolation | `pyproject.toml` | `requires-python >= 3.11 / uv venv` |
| 252 | PEP 621 Setuptools Package Discovery | `pyproject.toml` | `[tool.setuptools.packages.find]` |
| 253 | Explicit Dependency Pinning (httpx/tenacity/pypdf) | `pyproject.toml` | `[project.dependencies]` |
| 254 | Dev & UI Optional Dependency Groups | `pyproject.toml` | `[project.optional-dependencies]` |
| 255 | Pytest Configuration & Pythonpath Setup | `pyproject.toml` | `[tool.pytest.ini_options]` |
| 256 | Ruff Linter Ruleset (E/F/W/I/UP/B) | `pyproject.toml` | `[tool.ruff.lint]` |
| 257 | Multi-Stage Slim Python 3.12 Dockerfile | `Dockerfile` | `FROM python:3.12-slim` |
| 258 | LightGBM libgomp1 System Dependency | `Dockerfile` | `apt-get install libgomp1` |
| 259 | Container HTTP Healthcheck Probe | `Dockerfile` | `HEALTHCHECK curl /health` |
| 260 | Docker Compose pgvector Service (PG16) | `docker-compose.yml` | `services.pgvector` |
| 261 | Docker Compose MLflow Tracking Server | `docker-compose.yml` | `services.mlflow` |
| 262 | Docker Compose FastAPI Service Orchestration | `docker-compose.yml` | `services.api` |
| 263 | Persistent Named Volume for pgvector | `docker-compose.yml` | `volumes.pgvector_data` |
| 264 | GitHub Actions CI Workflow | `.github/workflows/ci.yml` | `AquaSentinel UK CI` |
| 265 | CI Automated Ruff Lint Gate | `.github/workflows/ci.yml` | `ruff check src tests` |
| 266 | CI Automated End-to-End Pipeline Gate | `.github/workflows/ci.yml` | `python -m src.run_pipeline` |
| 267 | CI Automated Pytest Suite Gate | `.github/workflows/ci.yml` | `pytest -v` |
| 268 | MLflow SQLite Backend Tracking URI | `src/models/spill_and_vision.py` | `sqlite:///mlruns/mlflow.db` |
| 269 | MLflow Experiment Auto-Creation | `src/models/spill_and_vision.py` | `mlflow.set_experiment` |
| 270 | MLflow CV Hyperparameter Logging | `src/models/spill_and_vision.py` | `mlflow.log_params` |
| 271 | MLflow Multi-Horizon & Torch Metric Logging | `src/models/spill_and_vision.py` | `mlflow.log_metrics` |
| 272 | CLI --live / --no-live BooleanOptionalAction | `src/run_pipeline.py` | `main (argparse)` |
| 273 | Live-by-Default Pipeline Entrypoint | `src/run_pipeline.py` | `run_pipeline(live=True)` |
| 274 | Offline Deterministic Synthetic Generator | `src/run_pipeline.py` | `generate_synthetic_uk_telemetry` |
| 275 | Raw CSV Snapshot Archival | `src/run_pipeline.py` | `data/raw/uk_hydrology_edm_15m.csv` |
| 276 | FastAPI Service Initialization & Metadata | `src/app/main.py` | `FastAPI(title='AquaSentinel UK API')` |
| 277 | Liveness & Readiness Endpoint (/health) | `src/app/main.py` | `health` |
| 278 | DuckDB Row Count Readiness Probe | `src/app/main.py` | `health (duckdb_rows)` |
| 279 | Quarantine Parquet Existence Probe | `src/app/main.py` | `health (quarantine_parquet_exists)` |
| 280 | Pydantic Telemetry Request Validation | `src/app/main.py` | `TelemetryReadingRequest` |
| 281 | Pydantic Multi-Horizon Forecast Response | `src/app/main.py` | `SpillForecastResponse` |
| 282 | Pydantic Regulatory Audit Request | `src/app/main.py` | `RegulatoryAuditRequest` |
| 283 | NaN/Inf/None Sensor Dropout Detection | `src/app/main.py` | `_is_missing` |
| 284 | Spatial Interpolation Tier 1: Graph Neighbors | `src/app/main.py` | `_spatial_interpolate_reading` |
| 285 | Spatial Interpolation Tier 2: DuckDB Basin Median | `src/app/main.py` | `_spatial_interpolate_reading` |
| 286 | Spatial Interpolation Tier 3: Basin Climatology | `src/app/main.py` | `BASIN_BASELINE_FALLBACK` |
| 287 | Interpolation Audit Trail Response Fields | `src/app/main.py` | `interpolated_fields / interpolation_source` |
| 288 | Streamlit Wide-Layout Dashboard | `src/app/dashboard.py` | `st.set_page_config` |
| 289 | KPI Telemetry & Quarantine Metrics Bar | `src/app/dashboard.py` | `col1 / col2 / col3` |
| 290 | Interactive Basin Filter & Quarantine Table | `src/app/dashboard.py` | `tab_telemetry` |
| 291 | Embedded Error Analysis Report Viewer | `src/app/dashboard.py` | `tab_report` |
| 292 | Gitignore Protection for Raw/Quarantine/MLruns | `.gitignore` | `data/raw/*, mlruns/` |
| 293 | Gitignore Tracking for Regulatory Docs | `.gitignore` | `data/regulatory_docs/` |
| 294 | Quarantine & GDPR Contract Pytest | `tests/test_contracts_and_errors.py` | `test_flatlined_corrupt_quarantine_and_gdpr_hashing` |
| 295 | Live EA Mocked HTTPX Ingestion Pytest | `tests/test_contracts_and_errors.py` | `test_live_ea_hydrology_connector_mocked_and_degraded` |
| 296 | Live EA Degraded Fallback Pytest | `tests/test_contracts_and_errors.py` | `test_live_ea_hydrology_connector_mocked_and_degraded` |
| 297 | Pipeline --live Degraded CLI Pytest | `tests/test_contracts_and_errors.py` | `test_run_pipeline_live_degraded_fallback` |
| 298 | FastAPI Sensor Dropout TestClient Pytest | `tests/test_contracts_and_errors.py` | `test_fastapi_health_forecast_dropout_and_regulatory_audit` |
| 299 | Quickstart & Architecture Documentation | `README.md` | `Quickstart / Docker Compose` |
| 300 | 300-Skill Engineering Capability Matrix | `docs/SKILLS_MATRIX.md` | `AquaSentinel UK 300-Skill Matrix` |
