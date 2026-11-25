"""FastAPI service exposing /health, /forecast-spills (with spatial fallback), and /regulatory-audit."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import duckdb
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.agents.ofwat_copilot import run_ofwat_compliance_copilot
from src.analytics.river_network import (
    build_uk_river_network,
    compute_antecedent_precipitation_index,
    get_downstream_impacts,
)
from src.models.spill_and_vision import FEATURE_COLUMNS

DUCKDB_PATH = Path("data/processed/hydrology.duckdb")
QUARANTINE_PATH = Path("data/quarantine/hydrology_quarantine.parquet")
BUNDLE_PATH = Path("data/processed/spill_models_bundle.pkl")

BASIN_BASELINE_FALLBACK: dict[str, dict[str, float]] = {
    "Thames": {"river_level_m": 1.65, "flow_m3s": 42.0, "dissolved_oxygen_mg_l": 8.4, "turbidity_ntu": 28.0},
    "Mersey": {"river_level_m": 1.85, "flow_m3s": 36.0, "dissolved_oxygen_mg_l": 7.8, "turbidity_ntu": 35.0},
    "Severn": {"river_level_m": 2.10, "flow_m3s": 68.0, "dissolved_oxygen_mg_l": 8.9, "turbidity_ntu": 24.0},
    "Aire": {"river_level_m": 1.55, "flow_m3s": 29.0, "dissolved_oxygen_mg_l": 7.5, "turbidity_ntu": 38.0},
}

app = FastAPI(
    title="AquaSentinel UK API",
    version="0.1.0",
    description="UK Sewage Spill Forecasting, River Network Analytics & Ofwat PR24 Compliance API",
)

_RIVER_GRAPH = build_uk_river_network()


class TelemetryReadingRequest(BaseModel):
    """Single station/CSO telemetry reading (fields may drop out for spatial fallback)."""

    station_id: str = Field(..., examples=["EA_THM_TEDDINGTON"])
    cso_id: str = Field(..., examples=["CSO_THM_MOGDEN"])
    basin: str = Field(default="Thames")
    water_company_code: str = Field(default="THM")
    river_level_m: float | None = Field(default=None)
    flow_m3s: float | None = Field(default=None)
    dissolved_oxygen_mg_l: float | None = Field(default=None)
    turbidity_ntu: float | None = Field(default=None)
    rainfall_mm_15m: float = Field(default=0.0, ge=0.0)
    rainfall_24h_mm: float | None = Field(default=None)
    api_mm: float | None = Field(default=None)
    sensor_dropout: bool = Field(default=False)
    peer_readings: list[dict[str, Any]] = Field(default_factory=list)


class HorizonForecast(BaseModel):
    horizon: str
    spill_probability: float
    predicted_duration_mins: float
    ci90_low_mins: float
    ci90_high_mins: float


class SpillForecastResponse(BaseModel):
    cso_id: str
    basin: str
    spatial_interpolation_applied: bool
    interpolated_fields: list[str]
    interpolation_source: str
    dry_day_spill_risk: bool
    downstream_target_site: str
    travel_time_to_target_h: float
    forecasts: list[HorizonForecast]


class RegulatoryAuditRequest(BaseModel):
    query: str
    water_company_code: str = "THM"
    annual_spill_count: int = 28
    total_spill_hours: float = 110.0
    dry_day_spills: int = 1
    sensitive_site: bool = True
    edm_uptime_pct: float = 96.5


def _is_missing(val: float | None) -> bool:
    return val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val)))


def _spatial_interpolate_reading(
    req: TelemetryReadingRequest,
    duckdb_path: Path = DUCKDB_PATH,
) -> tuple[dict[str, float], list[str], str]:
    """Interpolate dropped-out sensors using graph neighbors, peer readings, or DuckDB basin medians."""
    sensor_fields = ("river_level_m", "flow_m3s", "dissolved_oxygen_mg_l", "turbidity_ntu")
    resolved: dict[str, float] = {}
    missing_fields: list[str] = []

    for field in sensor_fields:
        raw_val = getattr(req, field)
        if req.sensor_dropout or _is_missing(raw_val):
            missing_fields.append(field)
        else:
            resolved[field] = float(raw_val)  # type: ignore[arg-type]

    if not missing_fields:
        return resolved, [], "direct_sensor"

    source_used = "basin_climatology_fallback"
    peer_df = pd.DataFrame(req.peer_readings) if req.peer_readings else pd.DataFrame()

    # Check DuckDB for historical/neighbor readings in same basin
    if peer_df.empty and duckdb_path.exists():
        try:
            con = duckdb.connect(str(duckdb_path), read_only=True)
            try:
                peer_df = con.execute(
                    "SELECT station_id, cso_id, basin, river_level_m, flow_m3s, "
                    "dissolved_oxygen_mg_l, turbidity_ntu FROM telemetry_15m WHERE basin = ? LIMIT 500",
                    [req.basin],
                ).df()
            finally:
                con.close()
        except Exception:
            peer_df = pd.DataFrame()

    # Prefer graph neighbors (upstream/downstream stations in the same basin)
    neighbor_ids = set()
    if req.cso_id in _RIVER_GRAPH:
        neighbor_ids |= set(_RIVER_GRAPH.successors(req.cso_id))
        neighbor_ids |= set(_RIVER_GRAPH.predecessors(req.cso_id))

    basin_defaults = BASIN_BASELINE_FALLBACK.get(req.basin, BASIN_BASELINE_FALLBACK["Thames"])

    for field in missing_fields:
        val: float | None = None
        if not peer_df.empty and field in peer_df.columns:
            # First try direct graph neighbors
            if neighbor_ids and "station_id" in peer_df.columns:
                nbr_mask = peer_df["station_id"].isin(neighbor_ids) | peer_df.get(
                    "cso_id", pd.Series(dtype=str)
                ).isin(neighbor_ids)
                nbr_vals = pd.to_numeric(peer_df.loc[nbr_mask, field], errors="coerce").dropna()
                if not nbr_vals.empty:
                    val = float(nbr_vals.median())
                    source_used = "graph_neighbor_interpolation"
            if val is None:
                basin_vals = pd.to_numeric(peer_df[field], errors="coerce").dropna()
                if not basin_vals.empty:
                    val = float(basin_vals.median())
                    source_used = "duckdb_basin_spatial_interpolation"
        if val is None:
            val = float(basin_defaults[field])
        resolved[field] = round(val, 4)

    return resolved, missing_fields, source_used


def _load_model_bundle(bundle_path: Path = BUNDLE_PATH) -> dict[str, Any] | None:
    if bundle_path.exists():
        return joblib.load(bundle_path)
    return None


@app.get("/health")
def health() -> dict[str, Any]:
    """Liveness and data/model readiness check."""
    duckdb_rows = 0
    if DUCKDB_PATH.exists():
        try:
            con = duckdb.connect(str(DUCKDB_PATH), read_only=True)
            try:
                duckdb_rows = int(con.execute("SELECT COUNT(*) FROM telemetry_15m").fetchone()[0])
            finally:
                con.close()
        except Exception:
            duckdb_rows = 0

    return {
        "status": "ok",
        "service": "aquasentinel-uk",
        "duckdb_ready": DUCKDB_PATH.exists(),
        "duckdb_rows": duckdb_rows,
        "quarantine_parquet_exists": QUARANTINE_PATH.exists(),
        "model_bundle_loaded": BUNDLE_PATH.exists(),
        "graph_nodes": _RIVER_GRAPH.number_of_nodes(),
        "graph_edges": _RIVER_GRAPH.number_of_edges(),
    }


@app.post("/forecast-spills", response_model=SpillForecastResponse)
def forecast_spills(req: TelemetryReadingRequest) -> SpillForecastResponse:
    """Predict t+6h, t+12h, t+24h CSO spills with spatial interpolation fallback on sensor dropout."""
    resolved_sensors, missing_fields, interp_source = _spatial_interpolate_reading(req)
    interp_applied = len(missing_fields) > 0

    rain_24h = (
        float(req.rainfall_24h_mm)
        if req.rainfall_24h_mm is not None
        else float(req.rainfall_mm_15m) * 4.0
    )
    api_val = (
        float(req.api_mm)
        if req.api_mm is not None
        else float(compute_antecedent_precipitation_index(np.array([req.rainfall_mm_15m])).iloc[0])
    )

    impacts = get_downstream_impacts(_RIVER_GRAPH, req.cso_id)
    bw_impacts = [imp for imp in impacts if imp.target_type == "bathing_water"]
    chosen = bw_impacts[0] if bw_impacts else (impacts[0] if impacts else None)
    downstream_site = chosen.target_site if chosen else "UNKNOWN"
    travel_h = chosen.travel_time_hours if chosen else 4.0
    is_urban = (
        1.0
        if (
            req.cso_id in _RIVER_GRAPH
            and _RIVER_GRAPH.nodes[req.cso_id].get("catchment_type") == "Urban Combined Sewer"
        )
        else 0.0
    )

    feature_row = {
        "river_level_m": resolved_sensors["river_level_m"],
        "flow_m3s": resolved_sensors["flow_m3s"],
        "dissolved_oxygen_mg_l": resolved_sensors["dissolved_oxygen_mg_l"],
        "turbidity_ntu": resolved_sensors["turbidity_ntu"],
        "rainfall_mm_15m": float(req.rainfall_mm_15m),
        "api_mm": api_val,
        "rainfall_24h_mm": rain_24h,
        "river_level_lag_4": resolved_sensors["river_level_m"],
        "flow_lag_4": resolved_sensors["flow_m3s"],
        "rainfall_lag_4": float(req.rainfall_mm_15m),
        "turbidity_roll_4": resolved_sensors["turbidity_ntu"],
        "do_roll_4": resolved_sensors["dissolved_oxygen_mg_l"],
        "travel_time_to_target_h": travel_h,
        "is_urban_combined": is_urban,
    }
    X_pred = pd.DataFrame([feature_row])[FEATURE_COLUMNS]

    bundle = _load_model_bundle()
    forecasts: list[HorizonForecast] = []

    if bundle is not None:
        for h_name in ("6h", "12h", "24h"):
            h_art = bundle["horizons"][h_name]
            prob = float(h_art["classifier"].predict_proba(X_pred)[0, 1])
            dur, pis = h_art["conformal_regressor"].predict_interval(X_pred)
            dur_val = max(0.0, float(dur[0]))
            low_val = max(0.0, float(pis[0, 0, 0]))
            high_val = max(low_val, float(pis[0, 1, 0]))
            forecasts.append(
                HorizonForecast(
                    horizon=h_name,
                    spill_probability=round(prob, 4),
                    predicted_duration_mins=round(dur_val, 2),
                    ci90_low_mins=round(low_val, 2),
                    ci90_high_mins=round(high_val, 2),
                )
            )
    else:
        # Hydrological heuristic fallback when model bundle is not yet trained
        base_prob = float(np.clip(0.15 + 0.04 * rain_24h + 0.002 * resolved_sensors["turbidity_ntu"], 0.02, 0.98))
        for scale, h_name in ((1.0, "6h"), (1.4, "12h"), (1.9, "24h")):
            dur_val = round(base_prob * 45.0 * scale, 2)
            forecasts.append(
                HorizonForecast(
                    horizon=h_name,
                    spill_probability=round(min(0.99, base_prob * (0.9 + 0.08 * scale)), 4),
                    predicted_duration_mins=dur_val,
                    ci90_low_mins=round(max(0.0, dur_val * 0.5), 2),
                    ci90_high_mins=round(dur_val * 1.8 + 10.0, 2),
                )
            )

    dry_day_risk = bool(
        rain_24h <= 0.25 and api_val <= 2.0 and forecasts[0].spill_probability >= 0.35
    )

    return SpillForecastResponse(
        cso_id=req.cso_id,
        basin=req.basin,
        spatial_interpolation_applied=interp_applied,
        interpolated_fields=missing_fields,
        interpolation_source=interp_source,
        dry_day_spill_risk=dry_day_risk,
        downstream_target_site=downstream_site,
        travel_time_to_target_h=round(float(travel_h), 3),
        forecasts=forecasts,
    )


@app.post("/regulatory-audit")
def regulatory_audit(req: RegulatoryAuditRequest) -> dict[str, Any]:
    """Run guardrailed Ofwat PR24 / Environment Act 2021 LangGraph compliance copilot."""
    result = run_ofwat_compliance_copilot(
        query=req.query,
        telemetry_context={
            "water_company_code": req.water_company_code,
            "annual_spill_count": req.annual_spill_count,
            "total_spill_hours": req.total_spill_hours,
            "dry_day_spills": req.dry_day_spills,
            "sensitive_site": req.sensitive_site,
            "edm_uptime_pct": req.edm_uptime_pct,
        },
    )
    return {
        "guardrail_passed": result["guardrail_passed"],
        "citations_verified": result["citations_verified"],
        "cited_doc_ids": result["cited_doc_ids"],
        "verification_notes": result["verification_notes"],
        "penalty_exposure": result["penalty_exposure"],
        "answer": result["final_answer"],
    }
