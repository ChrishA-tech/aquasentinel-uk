"""Hydrology and EDM telemetry ingestion with Pandera contracts, GDPR hashing, and quarantine."""

from __future__ import annotations

import hashlib
import hmac
import os
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pandera.pandas as pa

VALID_WATER_COMPANY_CODES: frozenset[str] = frozenset(
    {
        "THM",  # Thames Water
        "SVT",  # Severn Trent Water
        "UUW",  # United Utilities Water
        "SWW",  # South West Water
        "YKS",  # Yorkshire Water
        "ANGL",  # Anglian Water
        "NWT",  # Northumbrian Water
        "SRN",  # Southern Water
        "WSH",  # Dwr Cymru Welsh Water
        "WEX",  # Wessex Water
    }
)

VALID_BASINS: frozenset[str] = frozenset(
    {"Thames", "Mersey", "Severn", "Aire", "Trent", "Ouse", "Tyne", "Exe"}
)


HydrologyEDMSchema = pa.DataFrameSchema(
    columns={
        "timestamp": pa.Column(
            "datetime64[ns, UTC]",
            coerce=True,
            nullable=False,
        ),
        "station_id": pa.Column(str, pa.Check.str_length(min_value=2), nullable=False),
        "cso_id": pa.Column(str, pa.Check.str_length(min_value=2), nullable=False),
        "basin": pa.Column(str, pa.Check.isin(VALID_BASINS), nullable=False),
        "water_company_code": pa.Column(
            str, pa.Check.isin(VALID_WATER_COMPANY_CODES), nullable=False
        ),
        "river_level_m": pa.Column(
            float, pa.Check.in_range(0.0, 12.0), coerce=True, nullable=False
        ),
        "flow_m3s": pa.Column(
            float, pa.Check.in_range(0.0, 1500.0), coerce=True, nullable=False
        ),
        "dissolved_oxygen_mg_l": pa.Column(
            float, pa.Check.in_range(0.0, 20.0), coerce=True, nullable=False
        ),
        "turbidity_ntu": pa.Column(
            float, pa.Check.in_range(0.0, 2000.0), coerce=True, nullable=False
        ),
        "rainfall_mm_15m": pa.Column(
            float, pa.Check.in_range(0.0, 100.0), coerce=True, nullable=False
        ),
        "edm_spill_active": pa.Column(
            int, pa.Check.isin([0, 1]), coerce=True, nullable=False
        ),
        "edm_duration_mins": pa.Column(
            float, pa.Check.in_range(0.0, 15.0), coerce=True, nullable=False
        ),
        "is_flatlined": pa.Column(
            bool, pa.Check.equal_to(False), coerce=True, nullable=False
        ),
    },
    strict=False,
    coerce=True,
)


@dataclass
class IngestionResult:
    """Summary of telemetry ingestion, validation, and quarantine."""

    valid_df: pd.DataFrame
    quarantine_df: pd.DataFrame
    total_rows: int
    valid_rows: int
    quarantined_rows: int
    quarantine_rate: float
    duckdb_path: Path
    quarantine_path: Path


def pseudonymise_identifier(raw_id: object, salt: str | None = None) -> str:
    """Apply UK GDPR HMAC-SHA256 pseudonymisation to reporter/landowner identifiers."""
    if raw_id is None or (isinstance(raw_id, float) and np.isnan(raw_id)):
        return "ukgdpr_anonymous"
    raw_str = str(raw_id).strip()
    if not raw_str or raw_str.lower() in {"none", "nan", "null"}:
        return "ukgdpr_anonymous"
    secret = salt or os.environ.get("AQUASENTINEL_GDPR_SALT", "uk-ea-gdpr-default-salt-2026")
    digest = hmac.new(
        secret.encode("utf-8"),
        raw_str.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return f"ukgdpr_{digest}"


def apply_gdpr_pseudonymisation(
    df: pd.DataFrame,
    pii_columns: tuple[str, ...] = ("reporter_id", "landowner_id"),
    salt: str | None = None,
) -> pd.DataFrame:
    """Replace raw PII columns with HMAC-SHA256 pseudonymised hashes."""
    out = df.copy()
    for col in pii_columns:
        hash_col = col.removesuffix("_id") + "_hash"
        if col in out.columns:
            out[hash_col] = out[col].map(lambda val: pseudonymise_identifier(val, salt=salt))
            out = out.drop(columns=[col])
        elif hash_col not in out.columns:
            out[hash_col] = "ukgdpr_anonymous"
    return out


def flag_stuck_sensors(
    df: pd.DataFrame,
    window: int = 4,
    monitored_cols: tuple[str, ...] = ("river_level_m", "dissolved_oxygen_mg_l"),
    tolerance: float = 1e-9,
) -> pd.DataFrame:
    """Flag stuck/flatlined IoT sensors when readings remain constant across >= `window` steps."""
    out = df.copy()
    if "is_flatlined" not in out.columns:
        out["is_flatlined"] = False
    else:
        out["is_flatlined"] = out["is_flatlined"].fillna(False).astype(bool)

    if len(out) < window or "station_id" not in out.columns:
        return out

    out["_orig_order"] = np.arange(len(out))
    sort_cols = ["station_id", "timestamp"] if "timestamp" in out.columns else ["station_id"]
    out = out.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    flatline_mask = out["is_flatlined"].to_numpy(dtype=bool).copy()

    for _, grp in out.groupby("station_id", sort=False):
        idx = grp.index.to_numpy()
        if len(idx) < window:
            continue
        for col in monitored_cols:
            if col not in grp.columns:
                continue
            vals = pd.to_numeric(grp[col], errors="coerce").to_numpy(dtype=float)
            # Identify runs of consecutive identical finite values of length >= window
            run_start = 0
            for pos in range(1, len(vals) + 1):
                is_same = (
                    pos < len(vals)
                    and np.isfinite(vals[pos])
                    and np.isfinite(vals[pos - 1])
                    and abs(vals[pos] - vals[pos - 1]) <= tolerance
                )
                if not is_same:
                    run_len = pos - run_start
                    if run_len >= window:
                        flatline_mask[idx[run_start:pos]] = True
                    run_start = pos

    out["is_flatlined"] = flatline_mask
    out = out.sort_values("_orig_order").drop(columns=["_orig_order"]).reset_index(drop=True)
    return out


def _extract_failure_reasons(
    df: pd.DataFrame, err: pa.errors.SchemaErrors
) -> dict[int, list[str]]:
    """Map failing DataFrame row indices to human-readable Pandera contract violation reasons."""
    reasons: dict[int, list[str]] = {}
    fc = err.failure_cases
    if fc is None or fc.empty:
        return reasons

    for _, row in fc.iterrows():
        idx = row.get("index")
        col = row.get("column")
        check = row.get("check")
        if idx is None or (isinstance(idx, float) and np.isnan(idx)):
            continue
        try:
            int_idx = int(idx)
        except (TypeError, ValueError):
            continue
        if int_idx not in df.index:
            continue
        msg = f"{col}:{check}" if col is not None else str(check)
        reasons.setdefault(int_idx, [])
        if msg not in reasons[int_idx]:
            reasons[int_idx].append(msg)
    return reasons


def ingest_hydrology_telemetry(
    df: pd.DataFrame,
    duckdb_path: str | Path = "data/processed/hydrology.duckdb",
    quarantine_path: str | Path = "data/quarantine/hydrology_quarantine.parquet",
    gdpr_salt: str | None = None,
    flatline_window: int = 4,
    table_name: str = "telemetry_15m",
) -> IngestionResult:
    """Validate 15-min hydrology/EDM telemetry, quarantine invalid rows, and persist to DuckDB."""
    duckdb_file = Path(duckdb_path)
    quarantine_file = Path(quarantine_path)
    duckdb_file.parent.mkdir(parents=True, exist_ok=True)
    quarantine_file.parent.mkdir(parents=True, exist_ok=True)

    prepared = df.reset_index(drop=True).copy()
    if "timestamp" in prepared.columns:
        prepared["timestamp"] = pd.to_datetime(prepared["timestamp"], utc=True, errors="coerce")

    prepared = flag_stuck_sensors(prepared, window=flatline_window)
    prepared = apply_gdpr_pseudonymisation(prepared, salt=gdpr_salt)

    failure_reasons: dict[int, list[str]] = {}
    try:
        HydrologyEDMSchema.validate(prepared, lazy=True)
        bad_indices: set[int] = set()
    except pa.errors.SchemaErrors as err:
        failure_reasons = _extract_failure_reasons(prepared, err)
        bad_indices = set(failure_reasons.keys())
        # Fallback if a schema-level error lacks row indices
        if not bad_indices:
            bad_indices = set(prepared.index.tolist())

    # Catch any leftover flatlined flags explicitly even if schema coercion shifted something
    if "is_flatlined" in prepared.columns:
        flat_idx = set(prepared.index[prepared["is_flatlined"].fillna(False).astype(bool)].tolist())
        for idx in flat_idx:
            failure_reasons.setdefault(idx, []).append("is_flatlined:sensor_stuck")
        bad_indices |= flat_idx

    bad_mask = prepared.index.isin(bad_indices)
    quarantine_df = prepared.loc[bad_mask].copy()
    valid_df = prepared.loc[~bad_mask].copy().reset_index(drop=True)

    if not quarantine_df.empty:
        quarantine_df["quarantine_reason"] = [
            "; ".join(failure_reasons.get(idx, ["schema_violation"]))
            for idx in quarantine_df.index
        ]
        quarantine_df = quarantine_df.reset_index(drop=True)
    else:
        quarantine_df["quarantine_reason"] = pd.Series(dtype=str)

    if not valid_df.empty:
        valid_df = HydrologyEDMSchema.validate(valid_df, lazy=False)

    # Persist valid telemetry to DuckDB
    con = duckdb.connect(str(duckdb_file))
    try:
        con.register("_valid_telemetry", valid_df)
        con.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM _valid_telemetry")
        con.unregister("_valid_telemetry")
    finally:
        con.close()

    # Persist dead-letter quarantine to Parquet
    quarantine_to_write = quarantine_df.copy()
    if "timestamp" in quarantine_to_write.columns:
        quarantine_to_write["timestamp"] = quarantine_to_write["timestamp"].astype(str)
    quarantine_to_write.to_parquet(quarantine_file, index=False)

    total_rows = len(prepared)
    quarantined_rows = len(quarantine_df)
    valid_rows = len(valid_df)
    quarantine_rate = (quarantined_rows / total_rows) if total_rows > 0 else 0.0

    return IngestionResult(
        valid_df=valid_df,
        quarantine_df=quarantine_df,
        total_rows=total_rows,
        valid_rows=valid_rows,
        quarantined_rows=quarantined_rows,
        quarantine_rate=quarantine_rate,
        duckdb_path=duckdb_file,
        quarantine_path=quarantine_file,
    )
