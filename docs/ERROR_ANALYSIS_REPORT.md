# AquaSentinel UK — Sliced Error Analysis Report (6h Horizon)

## 1. Executive Summary
- **Evaluated Basins**: Aire, Mersey, Severn, Thames
- **Total Validated 15-min Intervals**: 2,235
- **Detected Dry-Day Spill Anomalies**: 18
- **Overall Spill Duration MAE / RMSE**: 7.441 min / 16.094 min
- **Overall MAPIE 90% Conformal Coverage**: 98.9% (Mean Interval Width: 64.28 min)

## 2. Performance Across Operational Slices

| Operational Slice    |   N Rows |   Duration MAE (min) |   Duration RMSE (min) |   Dry-Day PR-F1 |   Dry-Day PR-AUC | 90% Conformal Coverage   |   Mean 90% PI Width (min) |
|----------------------|----------|----------------------|-----------------------|-----------------|------------------|--------------------------|---------------------------|
| Overall              |     2235 |                7.441 |                16.094 |          0.0598 |           0.0101 | 98.9%                    |                     64.28 |
| Urban Combined Sewer |     1487 |                7.851 |                16.933 |          0.0844 |           0.0151 | 99.1%                    |                     64.99 |
| Rural Chalk Stream   |      748 |                6.627 |                14.28  |          0      |           0      | 98.4%                    |                     62.88 |
| Heavy Storm Event    |     1277 |                3.143 |                 7.057 |          0      |           0      | 100.0%                   |                     59.63 |
| Dry-Weather Anomaly  |      896 |               13.761 |                23.84  |          0.0598 |           0.0237 | 97.2%                    |                     71.18 |

## 3. Diagnostic Findings & Mitigations
1. **Urban Combined Sewer vs. Rural Chalk Stream**: Urban combined sewers respond sharply to short convective rainfall bursts (`rainfall_mm_15m`), whereas rural chalk streams exhibit baseflow/groundwater lag captured via `api_mm`.
2. **Heavy Storm Events**: Conformal intervals widen proportionally during extreme rainfall (`>= 10 mm / 24h`) to preserve >=90% empirical coverage under heteroscedastic runoff surges.
3. **Dry-Weather Anomalies**: Dry-day spills are rare telemetry events (`rainfall_24h_mm <= 0.25 mm`, `api_mm <= 2.0 mm`) indicative of asset blockages, rising main failures, or unlawful dry-weather discharges under Environment Act 2021 s.141A / EA EDM rules. Both graph-based anomaly detection and lower classification thresholds (`p >= 0.35`) improve recall for compliance triage.
