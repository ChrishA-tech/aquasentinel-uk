"""End-to-end AquaSentinel UK live/offline data ingestion and pipeline runner."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.agents.ofwat_copilot import run_ofwat_compliance_copilot
from src.analytics.river_network import build_uk_river_network, detect_dry_weather_spills
from src.evaluation.error_analysis import generate_error_analysis_report
from src.ingestion.hydrology_telemetry import (
    ingest_hydrology_telemetry,
    persist_telemetry_duckdb,
)
from src.ingestion.live_hydrology_connectors import fetch_live_uk_ea_hydrology_and_edm
from src.models.spill_and_vision import train_conformal_spill_models

STATION_CATALOG: tuple[dict[str, str], ...] = (
    {
        "station_id": "EA_THM_TEDDINGTON",
        "cso_id": "CSO_THM_MOGDEN",
        "basin": "Thames",
        "water_company_code": "THM",
    },
    {
        "station_id": "EA_THM_READING",
        "cso_id": "CSO_THM_KENNET_CHALK",
        "basin": "Thames",
        "water_company_code": "THM",
    },
    {
        "station_id": "EA_UUW_WARRINGTON",
        "cso_id": "CSO_UUW_DAVYHULME",
        "basin": "Mersey",
        "water_company_code": "UUW",
    },
    {
        "station_id": "EA_SVT_TEWKESBURY",
        "cso_id": "CSO_SVT_WORCESTER",
        "basin": "Severn",
        "water_company_code": "SVT",
    },
    {
        "station_id": "EA_SVT_TEWKESBURY",
        "cso_id": "CSO_SVT_GLOUCESTER",
        "basin": "Severn",
        "water_company_code": "SVT",
    },
    {
        "station_id": "EA_YKS_BEAL_WEIR",
        "cso_id": "CSO_YKS_KNOSTROP",
        "basin": "Aire",
        "water_company_code": "YKS",
    },
)


def generate_synthetic_uk_telemetry(
    n_steps_per_cso: int = 384,  # 4 days at 15-min resolution -> 6 * 384 = 2304 rows
    corrupt_fraction: float = 0.03,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate realistic UK EA hydrology + Water Company EDM + rainfall with ~3% corrupted/flatlined rows."""
    rng = np.random.default_rng(seed)
    start_ts = datetime(2026, 10, 1, 0, 0, tzinfo=UTC)
    timestamps = [start_ts + timedelta(minutes=15 * i) for i in range(n_steps_per_cso)]

    rows: list[dict[str, Any]] = []
    for cso_idx, meta in enumerate(STATION_CATALOG):
        # Construct convective storm events and dry spells scaled to series length
        rainfall = np.zeros(n_steps_per_cso, dtype=float)
        # Storm 1 early in series (so >=96 dry steps can precede dry-weather anomaly)
        s1 = min(max(8, int(n_steps_per_cso * 0.05) + (cso_idx * 3) % 8), max(8, n_steps_per_cso - 24))
        s1_end = min(n_steps_per_cso, s1 + min(14, max(4, n_steps_per_cso // 20)))
        rainfall[s1:s1_end] = rng.uniform(2.0, 5.5, size=s1_end - s1)

        # Storm 2 late in series
        s2 = min(max(s1_end + 10, int(n_steps_per_cso * 0.68) + (cso_idx * 3) % 10), max(s1_end + 4, n_steps_per_cso - 16))
        s2_end = min(n_steps_per_cso, s2 + min(18, max(4, n_steps_per_cso // 16)))
        if s2_end > s2:
            rainfall[s2:s2_end] = rng.uniform(2.4, 6.8, size=s2_end - s2)

        # Light drizzle only near storm 2 so the dry window before s2 stays genuinely dry
        drizzle_pool = np.arange(max(s2_end, 0), n_steps_per_cso)
        if len(drizzle_pool) > 6:
            drizzle_idx = rng.choice(drizzle_pool, size=min(10, len(drizzle_pool) // 2), replace=False)
            rainfall[drizzle_idx] = rng.uniform(0.05, 0.35, size=len(drizzle_idx))

        river_level = np.zeros(n_steps_per_cso, dtype=float)
        flow = np.zeros(n_steps_per_cso, dtype=float)
        do_vals = np.zeros(n_steps_per_cso, dtype=float)
        turbidity = np.zeros(n_steps_per_cso, dtype=float)
        spill_active = np.zeros(n_steps_per_cso, dtype=int)
        spill_dur = np.zeros(n_steps_per_cso, dtype=float)

        api_state = 0.5
        lvl = 1.3 + 0.1 * cso_idx
        dry_start = max(s1_end + 98, int(n_steps_per_cso * 0.52))
        dry_end = min(s2 - 2, dry_start + 6)

        for t in range(n_steps_per_cso):
            api_state = 0.94 * api_state + rainfall[t]
            lvl = 0.85 * lvl + 0.15 * (1.2 + 0.12 * api_state) + rng.normal(0.0, 0.03)
            lvl = float(np.clip(lvl, 0.25, 8.5))
            flw = float(np.clip(12.0 + 14.0 * (lvl ** 1.4) + rng.normal(0.0, 1.5), 2.0, 600.0))

            storm_spill = (api_state > 6.5) or (rainfall[t] > 3.0)
            # Rare dry-weather spill anomaly (blockage/rising main fault during dry window)
            dry_anomaly = (dry_start <= t < dry_end) and (cso_idx % 2 == 0)

            is_spill = int(storm_spill or dry_anomaly)
            dur = (
                float(np.clip(6.0 + 1.1 * api_state + rng.normal(0.0, 1.5), 2.0, 15.0))
                if is_spill
                else 0.0
            )
            turb = float(
                np.clip(
                    18.0 + 4.5 * api_state + (85.0 if is_spill else 0.0) + rng.normal(0.0, 5.0),
                    2.0,
                    650.0,
                )
            )
            do = float(
                np.clip(
                    9.4 - 0.012 * turb - (1.4 if is_spill else 0.0) + rng.normal(0.0, 0.2),
                    1.5,
                    14.5,
                )
            )

            river_level[t] = round(lvl, 4)
            flow[t] = round(flw, 3)
            turbidity[t] = round(turb, 2)
            do_vals[t] = round(do, 3)
            spill_active[t] = is_spill
            spill_dur[t] = round(dur, 2)

        for t in range(n_steps_per_cso):
            rows.append(
                {
                    "timestamp": timestamps[t],
                    "station_id": meta["station_id"],
                    "cso_id": meta["cso_id"],
                    "basin": meta["basin"],
                    "water_company_code": meta["water_company_code"],
                    "river_level_m": river_level[t],
                    "flow_m3s": flow[t],
                    "dissolved_oxygen_mg_l": do_vals[t],
                    "turbidity_ntu": turbidity[t],
                    "rainfall_mm_15m": round(float(rainfall[t]), 3),
                    "edm_spill_active": int(spill_active[t]),
                    "edm_duration_mins": spill_dur[t],
                    "is_flatlined": False,
                    "reporter_id": f"citizen_reporter_{(t + cso_idx) % 25:03d}",
                    "landowner_id": f"riparian_owner_{cso_idx:02d}",
                }
            )

    df = pd.DataFrame(rows)

    # Inject ~3% corrupted/flatlined rows to exercise dead-letter quarantine
    n_total = len(df)
    n_corrupt = max(8, int(round(n_total * corrupt_fraction)))
    # 1. Flatlined block (e.g. 6 consecutive constant readings on first station)
    flatline_len = max(4, n_corrupt // 4)
    df.loc[10 : 10 + flatline_len - 1, "river_level_m"] = 2.3456

    # 2. Out-of-bounds physical violations & invalid water company codes
    remaining = max(4, n_corrupt - flatline_len)
    corrupt_indices = rng.choice(
        np.arange(30, n_total), size=remaining, replace=False
    )
    for k, idx in enumerate(corrupt_indices):
        mode = k % 4
        if mode == 0:
            df.loc[idx, "dissolved_oxygen_mg_l"] = 26.5  # > 20 mg/L bound
        elif mode == 1:
            df.loc[idx, "river_level_m"] = -1.25  # < 0 m bound
        elif mode == 2:
            df.loc[idx, "water_company_code"] = "BAD_CO"  # invalid UK water company code
        else:
            df.loc[idx, "flow_m3s"] = 2400.0  # > 1500 m3/s physical bound

    return df


def run_pipeline(live: bool = True) -> dict[str, Any]:
    """Execute end-to-end ingestion, network analytics, model training, evaluation, and copilot check."""
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_csv_path = raw_dir / "uk_hydrology_edm_15m.csv"

    if live:
        valid_df = fetch_live_uk_ea_hydrology_and_edm()
        valid_df.to_csv(raw_csv_path, index=False)
        persist_telemetry_duckdb(valid_df)
        feed_status = str(valid_df.attrs.get("feed_status", "LIVE_EA_FEED_OK"))
        total_raw_rows = int(valid_df.attrs.get("raw_rows", len(valid_df)))
        valid_rows = len(valid_df)
        quarantined_rows = int(valid_df.attrs.get("quarantined_rows", 0))
        quarantine_rate = (quarantined_rows / total_raw_rows) if total_raw_rows > 0 else 0.0
    else:
        raw_df = generate_synthetic_uk_telemetry()
        raw_df.to_csv(raw_csv_path, index=False)
        ingestion = ingest_hydrology_telemetry(raw_df)
        valid_df = ingestion.valid_df
        feed_status = "OFFLINE_SYNTHETIC"
        total_raw_rows = ingestion.total_rows
        valid_rows = ingestion.valid_rows
        quarantined_rows = ingestion.quarantined_rows
        quarantine_rate = ingestion.quarantine_rate

    graph = build_uk_river_network()
    enriched_df = detect_dry_weather_spills(valid_df, graph=graph)

    suite = train_conformal_spill_models(enriched_df)
    report_path = generate_error_analysis_report(suite.feature_df)

    dry_spills_detected = int(enriched_df["dry_day_spill"].sum())
    copilot_out = run_ofwat_compliance_copilot(
        query=(
            "Audit Ofwat PR24 ODI exposure and Environment Act 2021 Section 141A duties "
            "for detected dry-day spills impacting downstream bathing waters."
        ),
        telemetry_context={
            "water_company_code": "THM",
            "annual_spill_count": 34,
            "total_spill_hours": 115.0,
            "dry_day_spills": max(1, dry_spills_detected),
            "sensitive_site": True,
            "edm_uptime_pct": 100.0 * (1.0 - quarantine_rate),
        },
    )

    summary = {
        "feed_status": feed_status,
        "total_raw_rows": total_raw_rows,
        "valid_rows": valid_rows,
        "quarantined_rows": quarantined_rows,
        "quarantine_rate": round(quarantine_rate, 4),
        "dry_day_spills_detected": dry_spills_detected,
        "mlflow_run_id": suite.mlflow_run_id,
        "horizon_metrics": {h: res.metrics for h, res in suite.horizon_results.items()},
        "torch_metrics": suite.torch_metrics,
        "report_path": str(report_path),
        "copilot_verified": copilot_out["citations_verified"],
        "copilot_status": copilot_out["penalty_exposure"].get("compliance_status"),
    }
    return summary


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(
        description="Run the AquaSentinel UK telemetry, forecasting, and Ofwat compliance pipeline."
    )
    parser.add_argument(
        "--live",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fetch live UK Environment Agency hydrology telemetry (with offline fallback).",
    )
    args = parser.parse_args(argv)
    result = run_pipeline(live=args.live)
    print("=== AquaSentinel UK Pipeline Complete ===")
    for key, value in result.items():
        print(f"{key}: {value}")
    return result


if __name__ == "__main__":
    main()
