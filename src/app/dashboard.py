"""Streamlit dashboard for AquaSentinel UK catchment spill risk & Ofwat PR24 compliance."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
import streamlit as st

from src.agents.ofwat_copilot import run_ofwat_compliance_copilot
from src.analytics.river_network import build_uk_river_network, get_downstream_impacts

DUCKDB_PATH = Path("data/processed/hydrology.duckdb")
QUARANTINE_PATH = Path("data/quarantine/hydrology_quarantine.parquet")
REPORT_PATH = Path("docs/ERROR_ANALYSIS_REPORT.md")

st.set_page_config(page_title="AquaSentinel UK", layout="wide")
st.title("AquaSentinel UK — Catchment Spill Risk & Ofwat PR24 Copilot")

col1, col2, col3 = st.columns(3)

if DUCKDB_PATH.exists():
    con = duckdb.connect(str(DUCKDB_PATH), read_only=True)
    try:
        df = con.execute("SELECT * FROM telemetry_15m ORDER BY timestamp DESC LIMIT 2000").df()
    finally:
        con.close()
    col1.metric("Validated 15m Telemetry Rows", f"{len(df):,}")
    col2.metric("Active Spill Intervals", f"{int(df['edm_spill_active'].sum()):,}")
else:
    df = pd.DataFrame()
    col1.metric("Validated 15m Telemetry Rows", "0 (Run pipeline)")
    col2.metric("Active Spill Intervals", "0")

if QUARANTINE_PATH.exists():
    q_df = pd.read_parquet(QUARANTINE_PATH)
    col3.metric("Quarantined Sensor Rows", f"{len(q_df):,}")
else:
    q_df = pd.DataFrame()
    col3.metric("Quarantined Sensor Rows", "0")

tab_network, tab_telemetry, tab_copilot, tab_report = st.tabs(
    ["Catchment Network", "Telemetry & Quarantine", "Ofwat PR24 Copilot", "Error Analysis"]
)

with tab_network:
    st.subheader("Directed Hydrological Catchment Graph")
    graph = build_uk_river_network()
    cso_nodes = [n for n, attrs in graph.nodes(data=True) if attrs.get("node_type") == "cso"]
    selected_cso = st.selectbox("Select Upstream CSO Outfall", cso_nodes)
    impacts = get_downstream_impacts(graph, selected_cso)
    if impacts:
        st.dataframe(pd.DataFrame([imp.__dict__ for imp in impacts]), use_container_width=True)

with tab_telemetry:
    st.subheader("Validated Telemetry Sample")
    if not df.empty:
        basin_filter = st.multiselect(
            "Filter River Basin",
            options=sorted(df["basin"].unique()),
            default=sorted(df["basin"].unique()),
        )
        filtered = df[df["basin"].isin(basin_filter)]
        st.dataframe(filtered.head(100), use_container_width=True)
    else:
        st.info("Run `python -m src.run_pipeline` to populate DuckDB telemetry.")

    st.subheader("Dead-Letter Quarantine (Flatlined / Out-of-Bounds Sensors)")
    if not q_df.empty:
        st.dataframe(q_df.head(50), use_container_width=True)

with tab_copilot:
    st.subheader("Ofwat PR24 & Environment Act 2021 Section 141A Compliance Copilot")
    query = st.text_area(
        "Compliance Question",
        value="What are the Ofwat PR24 ODI and Environment Act 2021 Section 141A penalties for dry-day spills into a chalk stream?",
    )
    c_code = st.selectbox("Water Company", ["THM", "SVT", "UUW", "SWW", "YKS"])
    spills = st.number_input("Annual Spill Count", min_value=0, value=32)
    hours = st.number_input("Total Spill Hours", min_value=0.0, value=120.0)
    dry_spills = st.number_input("Confirmed Dry-Day Spills", min_value=0, value=2)
    sensitive = st.checkbox("High-Priority Sensitive Site (Bathing Water / Chalk Stream)", value=True)
    uptime = st.slider("EDM Telemetry Uptime (%)", min_value=50.0, max_value=100.0, value=96.0)

    if st.button("Run Regulatory Audit"):
        out = run_ofwat_compliance_copilot(
            query=query,
            telemetry_context={
                "water_company_code": c_code,
                "annual_spill_count": int(spills),
                "total_spill_hours": float(hours),
                "dry_day_spills": int(dry_spills),
                "sensitive_site": sensitive,
                "edm_uptime_pct": float(uptime),
            },
        )
        st.success(f"Citations Verified: {out['citations_verified']} | {out['verification_notes']}")
        st.markdown(out["final_answer"])
        st.json(out["penalty_exposure"])

with tab_report:
    if REPORT_PATH.exists():
        st.markdown(REPORT_PATH.read_text(encoding="utf-8"))
    else:
        st.info("Run `python -m src.run_pipeline` to generate `docs/ERROR_ANALYSIS_REPORT.md`.")
