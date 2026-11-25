"""Catchment-grouped LightGBM + MAPIE conformal spill forecaster and PyTorch CatchmentSpillNet."""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import joblib  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from lightgbm import LGBMClassifier, LGBMRegressor  # noqa: E402
from mapie.regression import CrossConformalRegressor  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    f1_score,
    mean_absolute_error,
    root_mean_squared_error,
)
from sklearn.model_selection import GroupKFold  # noqa: E402

HORIZONS: dict[str, int] = {
    "6h": 24,   # 24 * 15m = 6h
    "12h": 48,  # 48 * 15m = 12h
    "24h": 96,  # 96 * 15m = 24h
}

FEATURE_COLUMNS: list[str] = [
    "river_level_m",
    "flow_m3s",
    "dissolved_oxygen_mg_l",
    "turbidity_ntu",
    "rainfall_mm_15m",
    "api_mm",
    "rainfall_24h_mm",
    "river_level_lag_4",
    "flow_lag_4",
    "rainfall_lag_4",
    "turbidity_roll_4",
    "do_roll_4",
    "travel_time_to_target_h",
    "is_urban_combined",
]

SEQ_FEATURE_COLUMNS: list[str] = [
    "rainfall_mm_15m",
    "river_level_m",
    "flow_m3s",
    "turbidity_ntu",
    "dissolved_oxygen_mg_l",
    "api_mm",
]


class CatchmentSpillNet(nn.Module):
    """Spatio-temporal sequence + outfall plume risk network (1D-Conv + GRU + Plume MLP)."""

    def __init__(
        self,
        seq_in_features: int = len(SEQ_FEATURE_COLUMNS),
        static_in_features: int = 4,
        hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(seq_in_features, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.static_mlp = nn.Sequential(
            nn.Linear(static_in_features, 16),
            nn.GELU(),
        )
        fused_dim = hidden_dim + 16
        self.head = nn.Sequential(
            nn.Linear(fused_dim, 32),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.spill_head = nn.Linear(32, 1)
        self.plume_head = nn.Linear(32, 1)
        self.duration_head = nn.Linear(32, 1)

    def forward(
        self, seq_x: torch.Tensor, static_x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning (spill_logit, plume_risk_prob, expected_duration_mins)."""
        # seq_x: (B, T, F) -> conv expects (B, F, T)
        conv_out = self.temporal_conv(seq_x.transpose(1, 2)).transpose(1, 2)
        _, h_n = self.gru(conv_out)
        temporal_repr = h_n[-1]
        static_repr = self.static_mlp(static_x)
        fused = self.head(torch.cat([temporal_repr, static_repr], dim=-1))

        spill_logit = self.spill_head(fused).squeeze(-1)
        plume_risk = torch.sigmoid(self.plume_head(fused)).squeeze(-1)
        duration_mins = torch.nn.functional.softplus(self.duration_head(fused)).squeeze(-1)
        return spill_logit, plume_risk, duration_mins


@dataclass
class HorizonModelResult:
    """Trained models and conformal/classification predictions for one forecast horizon."""

    horizon: str
    classifier: LGBMClassifier
    conformal_regressor: CrossConformalRegressor
    pred_prob: np.ndarray
    pred_duration: np.ndarray
    interval_low: np.ndarray
    interval_high: np.ndarray
    metrics: dict[str, float]


@dataclass
class SpillModelSuite:
    """Complete trained suite of LightGBM+MAPIE horizon models and PyTorch CatchmentSpillNet."""

    feature_df: pd.DataFrame
    horizon_results: dict[str, HorizonModelResult]
    torch_model: CatchmentSpillNet
    torch_metrics: dict[str, float]
    mlflow_run_id: str | None
    bundle_path: Path | None


def build_spill_feature_frame(
    df: pd.DataFrame,
    horizons: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Create multi-lag hydrological features and forward t+6h/t+12h/t+24h spill targets."""
    horizons = horizons or HORIZONS
    out = df.copy()
    if "timestamp" in out.columns:
        out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
        out = out.sort_values(["cso_id", "timestamp"], kind="mergesort").reset_index(drop=True)

    if "api_mm" not in out.columns:
        out["api_mm"] = out["rainfall_mm_15m"].astype(float)
    if "rainfall_24h_mm" not in out.columns:
        out["rainfall_24h_mm"] = out["rainfall_mm_15m"].astype(float)
    if "travel_time_to_target_h" not in out.columns:
        out["travel_time_to_target_h"] = 4.0
    if "catchment_type" not in out.columns:
        out["catchment_type"] = "Urban Combined Sewer"
    if "dry_day_spill" not in out.columns:
        out["dry_day_spill"] = False

    out["is_urban_combined"] = (
        out["catchment_type"].astype(str) == "Urban Combined Sewer"
    ).astype(float)

    for col_name, src_col in [
        ("river_level_lag_4", "river_level_m"),
        ("flow_lag_4", "flow_m3s"),
        ("rainfall_lag_4", "rainfall_mm_15m"),
    ]:
        out[col_name] = (
            out.groupby("cso_id", sort=False)[src_col]
            .shift(4)
            .bfill()
            .fillna(out[src_col])
        )

    out["turbidity_roll_4"] = (
        out.groupby("cso_id", sort=False)["turbidity_ntu"]
        .transform(lambda s: s.rolling(4, min_periods=1).mean())
    )
    out["do_roll_4"] = (
        out.groupby("cso_id", sort=False)["dissolved_oxygen_mg_l"]
        .transform(lambda s: s.rolling(4, min_periods=1).mean())
    )

    # Forward horizon targets using reversed rolling over future window
    for h_name, steps in horizons.items():
        future_dur = np.zeros(len(out), dtype=float)
        future_prob = np.zeros(len(out), dtype=int)
        for _, grp in out.groupby("cso_id", sort=False):
            idx = grp.index.to_numpy()
            dur_vals = grp["edm_duration_mins"].to_numpy(dtype=float)
            act_vals = grp["edm_spill_active"].to_numpy(dtype=int)
            n = len(idx)
            h_dur = np.zeros(n, dtype=float)
            h_act = np.zeros(n, dtype=int)
            for i in range(n):
                end = min(n, i + 1 + steps)
                if i + 1 < end:
                    window_dur = dur_vals[i + 1 : end]
                    window_act = act_vals[i + 1 : end]
                else:
                    window_dur = dur_vals[i : i + 1]
                    window_act = act_vals[i : i + 1]
                h_dur[i] = float(np.sum(window_dur))
                h_act[i] = int(np.max(window_act) > 0)
            future_dur[idx] = h_dur
            future_prob[idx] = h_act

        out[f"spill_duration_target_{h_name}"] = np.round(future_dur, 4)
        out[f"spill_prob_target_{h_name}"] = future_prob

    # Plume risk ground-truth proxy (high turbidity + low DO + active spill)
    turb_norm = np.clip(out["turbidity_ntu"].to_numpy(dtype=float) / 300.0, 0.0, 1.0)
    do_deficit = np.clip((10.0 - out["dissolved_oxygen_mg_l"].to_numpy(dtype=float)) / 10.0, 0.0, 1.0)
    spill_act = out["edm_spill_active"].to_numpy(dtype=float)
    out["plume_risk_target"] = np.clip(0.45 * turb_norm + 0.35 * do_deficit + 0.20 * spill_act, 0.0, 1.0)
    return out


def _build_sequence_tensors(
    feature_df: pd.DataFrame,
    seq_len: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct `(N, seq_len, F_seq)` and `(N, 4)` tensors for CatchmentSpillNet."""
    n = len(feature_df)
    seq_arr = np.zeros((n, seq_len, len(SEQ_FEATURE_COLUMNS)), dtype=np.float32)

    # Normalize sequence features robustly
    raw_seq = feature_df[SEQ_FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    means = np.mean(raw_seq, axis=0, keepdims=True)
    stds = np.std(raw_seq, axis=0, keepdims=True) + 1e-5
    norm_seq = (raw_seq - means) / stds

    for _, grp in feature_df.groupby("cso_id", sort=False):
        idx = grp.index.to_numpy()
        g_vals = norm_seq[idx]
        for pos, global_i in enumerate(idx):
            start = max(0, pos - seq_len + 1)
            slice_vals = g_vals[start : pos + 1]
            pad_len = seq_len - len(slice_vals)
            if pad_len > 0:
                pad = np.repeat(slice_vals[:1], pad_len, axis=0)
                seq_arr[global_i] = np.vstack([pad, slice_vals])
            else:
                seq_arr[global_i] = slice_vals

    static_cols = feature_df[
        ["travel_time_to_target_h", "is_urban_combined", "turbidity_roll_4", "do_roll_4"]
    ].to_numpy(dtype=np.float32)
    s_means = np.mean(static_cols, axis=0, keepdims=True)
    s_stds = np.std(static_cols, axis=0, keepdims=True) + 1e-5
    norm_static = (static_cols - s_means) / s_stds

    return torch.from_numpy(seq_arr), torch.from_numpy(norm_static)


def train_catchment_spill_net(
    feature_df: pd.DataFrame,
    seq_len: int = 8,
    epochs: int = 12,
    lr: float = 3e-3,
    seed: int = 42,
) -> tuple[CatchmentSpillNet, dict[str, float], np.ndarray]:
    """Train PyTorch CatchmentSpillNet on multi-lag rainfall-runoff and outfall plume risk."""
    torch.manual_seed(seed)
    seq_x, static_x = _build_sequence_tensors(feature_df, seq_len=seq_len)

    y_spill = torch.tensor(
        feature_df["spill_prob_target_6h"].to_numpy(dtype=np.float32)
    )
    y_plume = torch.tensor(
        feature_df["plume_risk_target"].to_numpy(dtype=np.float32)
    )
    y_dur = torch.tensor(
        feature_df["spill_duration_target_6h"].to_numpy(dtype=np.float32)
    )

    model = CatchmentSpillNet(
        seq_in_features=seq_x.shape[-1],
        static_in_features=static_x.shape[-1],
        hidden_dim=32,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()
    huber = nn.SmoothL1Loss()

    batch_size = min(256, max(16, len(feature_df)))
    n = len(feature_df)

    model.train()
    final_loss = 0.0
    for _ in range(epochs):
        perm = torch.randperm(n)
        epoch_loss = 0.0
        batches = 0
        for start in range(0, n, batch_size):
            b_idx = perm[start : start + batch_size]
            optimizer.zero_grad()
            spill_logit, plume_pred, dur_pred = model(seq_x[b_idx], static_x[b_idx])
            loss = (
                bce(spill_logit, y_spill[b_idx])
                + 2.0 * mse(plume_pred, y_plume[b_idx])
                + 0.05 * huber(dur_pred, y_dur[b_idx])
            )
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            batches += 1
        final_loss = epoch_loss / max(1, batches)

    model.eval()
    with torch.no_grad():
        spill_logit, plume_pred, _ = model(seq_x, static_x)
        plume_np = plume_pred.numpy()
        plume_mae = float(np.mean(np.abs(plume_np - y_plume.numpy())))
        spill_acc = float(
            np.mean(((torch.sigmoid(spill_logit) >= 0.5).float() == y_spill).numpy())
        )

    metrics = {
        "torch_final_loss": round(final_loss, 4),
        "torch_plume_mae": round(plume_mae, 4),
        "torch_spill_acc": round(spill_acc, 4),
    }
    return model, metrics, plume_np


def _safe_group_splits(groups: np.ndarray, requested_splits: int = 4) -> int:
    n_unique = len(np.unique(groups))
    return max(2, min(requested_splits, n_unique))


def train_conformal_spill_models(
    df: pd.DataFrame,
    horizons: dict[str, int] | None = None,
    mlflow_tracking_uri: str | None = "sqlite:///mlruns/mlflow.db",
    bundle_path: str | Path | None = "data/processed/spill_models_bundle.pkl",
    seed: int = 42,
) -> SpillModelSuite:
    """Train GroupKFold LightGBM + MAPIE 90% conformal regressors and CatchmentSpillNet."""
    horizons = horizons or HORIZONS
    feature_df = build_spill_feature_frame(df, horizons=horizons)

    X = feature_df[FEATURE_COLUMNS].astype(float)
    groups = feature_df["basin"].astype(str).to_numpy()
    n_splits = _safe_group_splits(groups, requested_splits=4)
    gkf = GroupKFold(n_splits=n_splits)

    horizon_results: dict[str, HorizonModelResult] = {}

    if mlflow_tracking_uri:
        if mlflow_tracking_uri.startswith("sqlite:///"):
            db_rel = mlflow_tracking_uri.removeprefix("sqlite:///")
            Path(db_rel).parent.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment("aquasentinel-uk-spill-forecasting")

    run_id: str | None = None

    def _fit_all() -> None:
        nonlocal horizon_results
        for h_name in horizons:
            y_prob = feature_df[f"spill_prob_target_{h_name}"].to_numpy(dtype=int)
            y_dur = feature_df[f"spill_duration_target_{h_name}"].to_numpy(dtype=float)

            # Out-of-fold probabilities grouped by river basin
            oof_prob = np.zeros(len(feature_df), dtype=float)
            for tr_idx, va_idx in gkf.split(X, y_prob, groups=groups):
                fold_clf = LGBMClassifier(
                    n_estimators=80,
                    learning_rate=0.06,
                    num_leaves=15,
                    n_jobs=1,
                    random_state=seed,
                    verbose=-1,
                )
                if len(np.unique(y_prob[tr_idx])) < 2:
                    oof_prob[va_idx] = float(np.mean(y_prob[tr_idx]))
                else:
                    fold_clf.fit(X.iloc[tr_idx], y_prob[tr_idx])
                    oof_prob[va_idx] = fold_clf.predict_proba(X.iloc[va_idx])[:, 1]

            full_clf = LGBMClassifier(
                n_estimators=100,
                learning_rate=0.06,
                num_leaves=15,
                n_jobs=1,
                random_state=seed,
                verbose=-1,
            )
            full_clf.fit(X, y_prob)

            base_reg = LGBMRegressor(
                n_estimators=100,
                learning_rate=0.06,
                num_leaves=15,
                n_jobs=1,
                random_state=seed,
                verbose=-1,
            )
            conformal_reg = CrossConformalRegressor(
                estimator=base_reg,
                confidence_level=0.90,
                conformity_score="absolute",
                method="plus",
                cv=GroupKFold(n_splits=n_splits),
                random_state=seed,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                conformal_reg.fit_conformalize(X, y_dur, groups=groups)
                pred_dur, pred_pis = conformal_reg.predict_interval(X)

            pred_dur = np.maximum(0.0, np.asarray(pred_dur, dtype=float))
            interval_low = np.maximum(0.0, np.asarray(pred_pis[:, 0, 0], dtype=float))
            interval_high = np.maximum(interval_low, np.asarray(pred_pis[:, 1, 0], dtype=float))

            mae = float(mean_absolute_error(y_dur, pred_dur))
            rmse = float(root_mean_squared_error(y_dur, pred_dur))
            coverage_90 = float(np.mean((y_dur >= interval_low) & (y_dur <= interval_high)))
            mean_width = float(np.mean(interval_high - interval_low))
            pr_auc = (
                float(average_precision_score(y_prob, oof_prob))
                if len(np.unique(y_prob)) > 1
                else 0.0
            )
            pred_label = (oof_prob >= 0.5).astype(int)
            f1 = float(f1_score(y_prob, pred_label, zero_division=0))

            metrics = {
                f"mae_{h_name}": round(mae, 4),
                f"rmse_{h_name}": round(rmse, 4),
                f"coverage90_{h_name}": round(coverage_90, 4),
                f"interval_width_{h_name}": round(mean_width, 4),
                f"pr_auc_{h_name}": round(pr_auc, 4),
                f"f1_{h_name}": round(f1, 4),
            }

            feature_df[f"pred_prob_{h_name}"] = np.round(oof_prob, 4)
            feature_df[f"pred_duration_{h_name}"] = np.round(pred_dur, 4)
            feature_df[f"ci90_low_{h_name}"] = np.round(interval_low, 4)
            feature_df[f"ci90_high_{h_name}"] = np.round(interval_high, 4)

            horizon_results[h_name] = HorizonModelResult(
                horizon=h_name,
                classifier=full_clf,
                conformal_regressor=conformal_reg,
                pred_prob=oof_prob,
                pred_duration=pred_dur,
                interval_low=interval_low,
                interval_high=interval_high,
                metrics=metrics,
            )

    torch_model, torch_metrics, plume_scores = train_catchment_spill_net(
        feature_df, seed=seed
    )
    feature_df["pred_plume_risk"] = np.round(plume_scores, 4)

    if mlflow_tracking_uri:
        with mlflow.start_run(run_name="groupkfold_conformal_lgbm_catchmentnet") as run:
            run_id = run.info.run_id
            mlflow.log_params(
                {
                    "group_cv": "GroupKFold(basin)",
                    "n_splits": n_splits,
                    "confidence_level": 0.90,
                    "horizons": ",".join(horizons.keys()),
                    "n_rows": len(feature_df),
                }
            )
            _fit_all()
            for res in horizon_results.values():
                mlflow.log_metrics(res.metrics)
            mlflow.log_metrics(torch_metrics)
    else:
        _fit_all()

    saved_path: Path | None = None
    if bundle_path is not None:
        saved_path = Path(bundle_path)
        saved_path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "feature_columns": FEATURE_COLUMNS,
            "horizons": {
                h: {
                    "classifier": res.classifier,
                    "conformal_regressor": res.conformal_regressor,
                    "metrics": res.metrics,
                }
                for h, res in horizon_results.items()
            },
            "torch_state_dict": torch_model.state_dict(),
            "torch_metrics": torch_metrics,
        }
        joblib.dump(payload, saved_path)

    return SpillModelSuite(
        feature_df=feature_df,
        horizon_results=horizon_results,
        torch_model=torch_model,
        torch_metrics=torch_metrics,
        mlflow_run_id=run_id,
        bundle_path=saved_path,
    )
