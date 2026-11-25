"""NetworkX directed UK river catchment graph, travel-time routing, API, and dry-day spill detection."""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DownstreamImpact:
    """Reachable downstream EA station or Bathing Water site from an upstream CSO."""

    source_cso: str
    target_site: str
    target_type: str
    basin: str
    travel_time_hours: float
    total_distance_km: float
    hops: int
    sensitive_site: bool


def build_uk_river_network() -> nx.DiGraph:
    """Build a directed hydrological network across Thames, Mersey, Severn, and Aire basins."""
    g = nx.DiGraph()

    nodes = [
        # Thames Basin (THM)
        ("CSO_THM_MOGDEN", {"node_type": "cso", "basin": "Thames", "water_company_code": "THM", "catchment_type": "Urban Combined Sewer", "lat": 51.466, "lon": -0.340, "sensitive_site": False}),
        ("CSO_THM_HAMMERSMITH", {"node_type": "cso", "basin": "Thames", "water_company_code": "THM", "catchment_type": "Urban Combined Sewer", "lat": 51.491, "lon": -0.224, "sensitive_site": False}),
        ("CSO_THM_KENNET_CHALK", {"node_type": "cso", "basin": "Thames", "water_company_code": "THM", "catchment_type": "Rural Chalk Stream", "lat": 51.412, "lon": -1.325, "sensitive_site": True}),
        ("EA_THM_TEDDINGTON", {"node_type": "ea_station", "basin": "Thames", "water_company_code": "THM", "catchment_type": "Urban Combined Sewer", "lat": 51.430, "lon": -0.321, "sensitive_site": False}),
        ("EA_THM_READING", {"node_type": "ea_station", "basin": "Thames", "water_company_code": "THM", "catchment_type": "Rural Chalk Stream", "lat": 51.458, "lon": -0.968, "sensitive_site": False}),
        ("BW_THM_WALLINGFORD", {"node_type": "bathing_water", "basin": "Thames", "water_company_code": "THM", "catchment_type": "Rural Chalk Stream", "lat": 51.600, "lon": -1.123, "sensitive_site": True}),
        ("BW_THM_CHISWICK", {"node_type": "bathing_water", "basin": "Thames", "water_company_code": "THM", "catchment_type": "Urban Combined Sewer", "lat": 51.487, "lon": -0.249, "sensitive_site": True}),

        # Mersey Basin (UUW)
        ("CSO_UUW_DAVYHULME", {"node_type": "cso", "basin": "Mersey", "water_company_code": "UUW", "catchment_type": "Urban Combined Sewer", "lat": 53.465, "lon": -2.357, "sensitive_site": False}),
        ("CSO_UUW_SALFORD", {"node_type": "cso", "basin": "Mersey", "water_company_code": "UUW", "catchment_type": "Urban Combined Sewer", "lat": 53.471, "lon": -2.289, "sensitive_site": False}),
        ("EA_UUW_WARRINGTON", {"node_type": "ea_station", "basin": "Mersey", "water_company_code": "UUW", "catchment_type": "Urban Combined Sewer", "lat": 53.385, "lon": -2.593, "sensitive_site": False}),
        ("BW_UUW_CROSBY", {"node_type": "bathing_water", "basin": "Mersey", "water_company_code": "UUW", "catchment_type": "Urban Combined Sewer", "lat": 53.485, "lon": -3.047, "sensitive_site": True}),

        # Severn Basin (SVT)
        ("CSO_SVT_WORCESTER", {"node_type": "cso", "basin": "Severn", "water_company_code": "SVT", "catchment_type": "Rural Chalk Stream", "lat": 52.192, "lon": -2.222, "sensitive_site": True}),
        ("CSO_SVT_GLOUCESTER", {"node_type": "cso", "basin": "Severn", "water_company_code": "SVT", "catchment_type": "Urban Combined Sewer", "lat": 51.867, "lon": -2.253, "sensitive_site": False}),
        ("EA_SVT_TEWKESBURY", {"node_type": "ea_station", "basin": "Severn", "water_company_code": "SVT", "catchment_type": "Rural Chalk Stream", "lat": 51.993, "lon": -2.162, "sensitive_site": False}),
        ("BW_SVT_WESTON", {"node_type": "bathing_water", "basin": "Severn", "water_company_code": "SVT", "catchment_type": "Urban Combined Sewer", "lat": 51.346, "lon": -2.982, "sensitive_site": True}),

        # Aire Basin (YKS)
        ("CSO_YKS_BRADFORD", {"node_type": "cso", "basin": "Aire", "water_company_code": "YKS", "catchment_type": "Urban Combined Sewer", "lat": 53.796, "lon": -1.759, "sensitive_site": False}),
        ("CSO_YKS_KNOSTROP", {"node_type": "cso", "basin": "Aire", "water_company_code": "YKS", "catchment_type": "Urban Combined Sewer", "lat": 53.776, "lon": -1.498, "sensitive_site": False}),
        ("EA_YKS_BEAL_WEIR", {"node_type": "ea_station", "basin": "Aire", "water_company_code": "YKS", "catchment_type": "Urban Combined Sewer", "lat": 53.726, "lon": -1.196, "sensitive_site": False}),
        ("BW_YKS_HUMBER", {"node_type": "bathing_water", "basin": "Aire", "water_company_code": "YKS", "catchment_type": "Urban Combined Sewer", "lat": 53.575, "lon": -0.056, "sensitive_site": True}),
    ]
    g.add_nodes_from(nodes)

    edges = [
        # Thames
        ("CSO_THM_KENNET_CHALK", "EA_THM_READING", 22.0, 0.72),
        ("EA_THM_READING", "BW_THM_WALLINGFORD", 14.0, 0.68),
        ("EA_THM_READING", "EA_THM_TEDDINGTON", 48.0, 0.85),
        ("CSO_THM_MOGDEN", "EA_THM_TEDDINGTON", 4.5, 0.90),
        ("EA_THM_TEDDINGTON", "CSO_THM_HAMMERSMITH", 12.0, 1.05),
        ("CSO_THM_HAMMERSMITH", "BW_THM_CHISWICK", 2.4, 1.10),
        # Mersey
        ("CSO_UUW_SALFORD", "CSO_UUW_DAVYHULME", 6.2, 0.85),
        ("CSO_UUW_DAVYHULME", "EA_UUW_WARRINGTON", 24.5, 0.95),
        ("EA_UUW_WARRINGTON", "BW_UUW_CROSBY", 38.0, 1.15),
        # Severn
        ("CSO_SVT_WORCESTER", "EA_SVT_TEWKESBURY", 25.0, 0.88),
        ("EA_SVT_TEWKESBURY", "CSO_SVT_GLOUCESTER", 18.0, 0.92),
        ("CSO_SVT_GLOUCESTER", "BW_SVT_WESTON", 64.0, 1.25),
        # Aire
        ("CSO_YKS_BRADFORD", "CSO_YKS_KNOSTROP", 19.5, 0.82),
        ("CSO_YKS_KNOSTROP", "EA_YKS_BEAL_WEIR", 26.0, 0.90),
        ("EA_YKS_BEAL_WEIR", "BW_YKS_HUMBER", 72.0, 1.10),
    ]

    for u, v, distance_km, velocity_mps in edges:
        travel_time_hours = (distance_km * 1000.0) / (velocity_mps * 3600.0)
        g.add_edge(
            u,
            v,
            distance_km=round(distance_km, 3),
            mean_velocity_mps=round(velocity_mps, 3),
            travel_time_hours=round(travel_time_hours, 3),
        )

    return g


def compute_pollution_travel_time(
    graph: nx.DiGraph,
    source_cso: str,
    target_site: str,
) -> float:
    """Compute shortest hydraulic travel time (hours) from upstream CSO to downstream site."""
    if source_cso not in graph or target_site not in graph:
        raise ValueError(f"Unknown node(s): {source_cso}, {target_site}")
    return float(
        nx.shortest_path_length(
            graph, source=source_cso, target=target_site, weight="travel_time_hours"
        )
    )


def get_downstream_impacts(graph: nx.DiGraph, cso_id: str) -> list[DownstreamImpact]:
    """Return all downstream EA stations and Bathing Water sites reachable from `cso_id`."""
    if cso_id not in graph:
        return []
    lengths = nx.single_source_dijkstra_path_length(graph, cso_id, weight="travel_time_hours")
    paths = nx.single_source_dijkstra_path(graph, cso_id, weight="travel_time_hours")
    impacts: list[DownstreamImpact] = []
    for target, travel_h in lengths.items():
        if target == cso_id:
            continue
        attrs = graph.nodes[target]
        if attrs.get("node_type") not in {"ea_station", "bathing_water"}:
            continue
        path = paths[target]
        dist_km = sum(
            graph[path[i]][path[i + 1]]["distance_km"] for i in range(len(path) - 1)
        )
        impacts.append(
            DownstreamImpact(
                source_cso=cso_id,
                target_site=target,
                target_type=str(attrs["node_type"]),
                basin=str(attrs["basin"]),
                travel_time_hours=round(float(travel_h), 3),
                total_distance_km=round(float(dist_km), 3),
                hops=len(path) - 1,
                sensitive_site=bool(attrs.get("sensitive_site", False)),
            )
        )
    impacts.sort(key=lambda item: item.travel_time_hours)
    return impacts


def compute_antecedent_precipitation_index(
    rainfall_series: pd.Series | np.ndarray,
    decay_k: float = 0.85,
) -> pd.Series:
    """Compute Antecedent Precipitation Index: API_t = k * API_{t-1} + P_t."""
    values = np.asarray(rainfall_series, dtype=float)
    api = np.zeros_like(values, dtype=float)
    prev = 0.0
    for i, p in enumerate(values):
        rain = 0.0 if np.isnan(p) else float(p)
        prev = decay_k * prev + rain
        api[i] = prev
    index = rainfall_series.index if isinstance(rainfall_series, pd.Series) else None
    return pd.Series(api, index=index, name="api_mm")


def detect_dry_weather_spills(
    df: pd.DataFrame,
    graph: nx.DiGraph | None = None,
    decay_k_15m: float = 0.96,
    dry_rain_24h_threshold_mm: float = 0.25,
    dry_api_threshold_mm: float = 2.0,
    min_spill_mins: float = 1.0,
) -> pd.DataFrame:
    """Compute rolling rainfall, API, dry-day spill anomaly flags, and downstream graph exposure."""
    g = graph or build_uk_river_network()
    out = df.copy()
    if "timestamp" in out.columns:
        out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
        out = out.sort_values(["cso_id", "timestamp"], kind="mergesort").reset_index(drop=True)

    api_values = np.zeros(len(out), dtype=float)
    rain_24h_values = np.zeros(len(out), dtype=float)

    for _, grp in out.groupby("cso_id", sort=False):
        idx = grp.index.to_numpy()
        rain = grp["rainfall_mm_15m"].astype(float)
        api_series = compute_antecedent_precipitation_index(rain, decay_k=decay_k_15m)
        rain_24h = rain.rolling(window=96, min_periods=1).sum()
        api_values[idx] = api_series.to_numpy()
        rain_24h_values[idx] = rain_24h.to_numpy()

    out["api_mm"] = np.round(api_values, 4)
    out["rainfall_24h_mm"] = np.round(rain_24h_values, 4)

    spill_active = out["edm_spill_active"].astype(int) == 1
    spill_dur = out["edm_duration_mins"].astype(float) >= min_spill_mins
    is_dry = (out["rainfall_24h_mm"] <= dry_rain_24h_threshold_mm) & (
        out["api_mm"] <= dry_api_threshold_mm
    )
    out["dry_day_spill"] = spill_active & spill_dur & is_dry

    # Enrich with downstream bathing water / EA travel time from graph
    downstream_site_map: dict[str, str] = {}
    downstream_travel_map: dict[str, float] = {}
    catchment_type_map: dict[str, str] = {}

    for cso_id in out["cso_id"].unique():
        if cso_id in g:
            catchment_type_map[cso_id] = str(
                g.nodes[cso_id].get("catchment_type", "Urban Combined Sewer")
            )
            impacts = get_downstream_impacts(g, cso_id)
            bw_impacts = [imp for imp in impacts if imp.target_type == "bathing_water"]
            chosen = bw_impacts[0] if bw_impacts else (impacts[0] if impacts else None)
            if chosen is not None:
                downstream_site_map[cso_id] = chosen.target_site
                downstream_travel_map[cso_id] = chosen.travel_time_hours
            else:
                downstream_site_map[cso_id] = "NONE"
                downstream_travel_map[cso_id] = -1.0
        else:
            catchment_type_map[cso_id] = "Urban Combined Sewer"
            downstream_site_map[cso_id] = "UNKNOWN"
            downstream_travel_map[cso_id] = -1.0

    out["catchment_type"] = out["cso_id"].map(catchment_type_map)
    out["downstream_target_site"] = out["cso_id"].map(downstream_site_map)
    out["travel_time_to_target_h"] = out["cso_id"].map(downstream_travel_map)
    return out
