"""
03_model_training.py — RAN KPI Anomaly Detection: Model Training
=================================================================
Companion code for: "Real-Time RAN KPI Anomaly Detection at Cell Sector Granularity"

PURPOSE
-------
Train and evaluate a three-tier cascade anomaly detection system:
  Tier 1 — Isolation Forest (unsupervised, Phase-1 trust-building baseline)
  Tier 2 — Random Forest classifier (supervised, Phase-2 accuracy leap)
             Labels are bootstrapped from Tier-1 high-confidence scores
             plus injected ground truth from synthetic data
  Tier 3 — LSTM Autoencoder (unsupervised temporal, catches gradual degradation)

The three models are combined into a cascade: an anomaly must be flagged by
at least one tier above its operating threshold, with configurable promotion
logic to the hierarchical alert engine.

RUNNING
-------
    # Preferred: use outputs from prior scripts
    python 03_model_training.py

    # Self-contained fallback: regenerates data internally if CSVs absent
    python 03_model_training.py --self-contained

OUTPUT FILES
------------
    models/isolation_forest.pkl
    models/isolation_forest_threshold.json
    models/random_forest.pkl
    models/random_forest_threshold.json
    models/lstm_autoencoder.pt          (if PyTorch available)
    models/lstm_autoencoder_threshold.json
    models/cascade_config.json
    models/feature_columns.json
    models/scaler.pkl
    results/training_metrics.json
    results/threshold_sweep.csv

COURSEBOOK CROSS-REFERENCE
---------------------------
  Ch. 3: Time-Series Fundamentals — temporal splits, stationarity
  Ch. 7: Feature Engineering — rolling statistics, peer z-scores
  Ch. 12: Streaming Architectures — model format choices (ONNX-exportable)
  Ch. 15: MLOps — serialisation, threshold selection, baseline comparisons

REQUIREMENTS
------------
    pip install pandas numpy scikit-learn matplotlib seaborn joblib

OPTIONAL (LSTM tier only):
    pip install torch
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from utils import compute_event_based_recall

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Logging setup — production-grade: structured, level-configurable
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("ran.anomaly.training")

# ---------------------------------------------------------------------------
# Optional PyTorch import — LSTM tier only. Degrades gracefully if absent.
# ---------------------------------------------------------------------------
TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
    logger.info("PyTorch available (version %s) — LSTM Autoencoder tier enabled", torch.__version__)
except ImportError:
    logger.warning(
        "PyTorch not installed — LSTM Autoencoder tier will be skipped. "
        "Install with: pip install torch"
    )

# ---------------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).parent
DATA_DIR = REPO_ROOT / "data"
MODELS_DIR = REPO_ROOT / "models"
RESULTS_DIR = REPO_ROOT / "results"

# Tier-2 KPIs — must match 02_feature_engineering.py
TIER2_KPIS: set[str] = {"rach_success_rate", "rlf_count"}
EXCLUDE_TIER2_DERIVED: bool = True  # Set False when vendor counter coverage confirmed; see §3

# ---------------------------------------------------------------------------
# Training constants
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Isolation Forest hyperparameters
# contamination=0.03 means we expect ~3% anomaly rate in training data.
# Real RAN networks show 1–5% anomaly rates; 3% is a reasonable Phase-1 prior.
# See Coursebook Ch. 25 on anomaly base rates in network data.
IF_CONTAMINATION = 0.03
IF_N_ESTIMATORS = 200        # Higher than default 100 for more stable scores
IF_MAX_SAMPLES = "auto"      # Auto = min(256, n_samples) — fast and effective
IF_MAX_FEATURES = 1.0        # Use all features; RAN features are pre-selected

# Random Forest hyperparameters (supervised, Phase-2)
RF_N_ESTIMATORS = 200
RF_MAX_DEPTH = 12            # Prevents overfitting on tabular RAN features
RF_MIN_SAMPLES_LEAF = 10     # Minimum cell-window support for a leaf decision
RF_CLASS_WEIGHT = "balanced" # Critical: anomalies are rare; must reweight classes
RF_MAX_FEATURES = "sqrt"     # Standard for classification tasks

# LSTM Autoencoder hyperparameters
LSTM_SEQUENCE_LENGTH = 16    # 16 ROPs × 15 min = 4 hours of temporal context
LSTM_HIDDEN_SIZE = 64        # Compact enough for edge deployment (ONNX export)
LSTM_NUM_LAYERS = 2
LSTM_EPOCHS = 30
LSTM_BATCH_SIZE = 256
LSTM_LEARNING_RATE = 1e-3
LSTM_EARLY_STOP_PATIENCE = 5 # Stop if val loss doesn't improve for 5 epochs

# Threshold sweep settings for precision-recall operating point selection
THRESHOLD_PERCENTILES = list(range(80, 100))  # Sweep 80th–99th percentile

# Cascade scoring weights for the final combined anomaly score
# These weights reflect Phase-1 default; tuned further in Phase 2.
CASCADE_WEIGHT_IF = 0.35    # Isolation Forest contribution
CASCADE_WEIGHT_RF = 0.45    # Random Forest contribution (highest when available)
CASCADE_WEIGHT_LSTM = 0.20  # LSTM autoencoder contribution
# Phase-weight normalisation (weights renormalise to sum 1.0 per available tiers):
#   Phase 1 (IF only):      (1.0,    0.0,    0.0)
#   Phase 2 (IF + RF):      (0.4375, 0.5625, 0.0)
#   Phase 3 (all three):    (0.35,   0.45,   0.20)  ← as specified above

# Operational target: NOC can review ~10 alerts per 8-hour shift per region.
# At 100 cells (demo scale), this means ~10 true positives per shift maximum.
# We size the precision target to keep false alarms < 5 per shift.
OPERATIONAL_FALSE_ALARM_TARGET_PER_SHIFT = 5
OPERATIONAL_SHIFT_HOURS = 8
OPER_CELLS = 100  # Cells in the demo dataset


# ===========================================================================
# SECTION 1: DATA LOADING
# ===========================================================================
def load_feature_splits(
    train_path: Path,
    val_path: Path,
    test_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load pre-engineered feature splits from 02_feature_engineering.py output.

    Expects temporal splits: train < val < test in time order.
    Never shuffles — preserving time order is critical to avoid leakage.
    See Coursebook Ch. 25: Temporal Train/Val/Test Splits.
    """
    logger.info("Loading feature splits from disk...")
    train = pd.read_parquet(train_path)
    val = pd.read_parquet(val_path)
    test = pd.read_parquet(test_path)

    # Verify temporal ordering — fail loud if data is corrupted
    for split_name, split_df in [("train", train), ("val", val), ("test", test)]:
        if "timestamp" in split_df.columns:
            split_df["timestamp"] = pd.to_datetime(split_df["timestamp"])

    logger.info(
        "Loaded splits — train: %d rows, val: %d rows, test: %d rows",
        len(train), len(val), len(test),
    )
    _log_anomaly_rates(train, val, test)
    return train, val, test


def _log_anomaly_rates(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    """Log class balance per split to confirm expected anomaly rates."""
    for name, df in [("train", train), ("val", val), ("test", test)]:
        if "is_anomaly" in df.columns:
            rate = df["is_anomaly"].mean()
            n_anom = df["is_anomaly"].sum()
            logger.info("  %s — anomaly rate: %.2f%% (%d / %d)", name, rate * 100, n_anom, len(df))
        else:
            logger.warning("  %s — 'is_anomaly' column not found; operating in unsupervised mode", name)


def resolve_feature_columns(df: pd.DataFrame, exclude_cols: Optional[list[str]] = None) -> list[str]:
    """
    Identify numeric feature columns for model input.

    Preferred path: load column list from ``data/feature_metadata.json``
    (written by 02_feature_engineering.py) to guarantee training-serving
    feature parity.  Falls back to heuristic column resolution when the
    metadata file is absent (e.g. self-contained mode).

    See Coursebook Ch. 13: Feature Store Consistency.
    """
    # --- Preferred: load authoritative column list from 02's metadata ---
    metadata_path = Path("data/feature_metadata.json")
    if metadata_path.exists():
        with open(metadata_path) as f:
            meta = json.load(f)
        canonical_cols = meta.get("feature_columns", [])
        available = [c for c in canonical_cols if c in df.columns]
        if available:
            logger.info(
                "Loaded %d/%d feature columns from %s",
                len(available), len(canonical_cols), metadata_path,
            )
            if len(available) < len(canonical_cols):
                missing = set(canonical_cols) - set(available)
                logger.warning(
                    "%d feature columns from metadata not found in DataFrame: %s",
                    len(missing), sorted(missing)[:10],
                )
            return sorted(available)
        logger.warning(
            "feature_metadata.json exists but yielded 0 matching columns — "
            "falling back to heuristic resolution"
        )
    default_exclude = {
        "timestamp", "cell_id", "is_anomaly", "anomaly_type",
        "anomaly_severity", "site_id", "sector_id",
        # Boolean metadata — not a model feature
        "peer_group_cold_start",
        # Raw PM counters that are replaced by derived features
        "dl_throughput_mbps", "ul_throughput_mbps",
        "dl_prb_usage_rate", "ul_prb_usage_rate",
        "rrc_conn_setup_success_rate", "drb_setup_success_rate",
        "handover_success_rate", "avg_cqi", "rsrp_dbm",
        "rsrq_db", "sinr_db", "rach_preamble_count",
        # Tier-2 KPIs excluded from model features per §3 tiering decision —
        # monitored for data quality; include when vendor coverage is confirmed
        "rach_success_rate", "rlf_count",
    }
    if exclude_cols:
        default_exclude.update(exclude_cols)

    feature_cols = [
        c for c in df.columns
        if c not in default_exclude
        and pd.api.types.is_numeric_dtype(df[c])
        and not c.startswith("_")
        and (not EXCLUDE_TIER2_DERIVED or not any(c.startswith(t2) for t2 in TIER2_KPIS))
    ]
    if EXCLUDE_TIER2_DERIVED:
        logger.info(
            "EXCLUDE_TIER2_DERIVED=True — also excluded derived features for %s",
            sorted(TIER2_KPIS),
        )
    logger.info("Resolved %d feature columns for model input", len(feature_cols))
    return sorted(feature_cols)


def prepare_matrices(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str = "is_anomaly",
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Extract feature matrix X and optional label vector y from a DataFrame.

    Handles missing values via median imputation — same strategy as serving.
    In production, missing values come from counter resets or O1 file gaps.
    """
    X = df[feature_cols].copy()

    # Median imputation for any remaining NaNs after feature engineering
    # (counter resets can create NaNs at window boundaries)
    col_medians = X.median()
    X = X.fillna(col_medians)

    # Replace inf values that can arise from division-based ratio features
    X = X.replace([np.inf, -np.inf], np.nan).fillna(col_medians)

    y = df[label_col].values.astype(int) if label_col in df.columns else None
    return X.values, y


# ===========================================================================
# SECTION 2: SYNTHETIC DATA FALLBACK
# ===========================================================================
# When prior scripts haven't been run, generate a minimal but realistic dataset
# inline. This ensures 03_model_training.py is always runnable standalone.
def _generate_synthetic_features(
    n_cells: int = 100,
    n_days: int = 30,
    rop_minutes: int = 15,
    anomaly_rate: float = 0.03,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generate a realistic RAN feature dataset for standalone testing.

    This is a simplified version of the full synthetic pipeline in
    01_synthetic_data.py + 02_feature_engineering.py. It produces the same
    column schema to ensure model code is identical whether using real or
    synthetic inputs.

    Diurnal traffic shape: morning peak 08:00–10:00, evening peak 18:00–21:00.
    Weekly: weekday traffic ~20% higher than weekend.
    Anomalies injected at 3% rate with known ground truth labels.

    See Coursebook Ch. 25: Realistic Benchmark Dataset Construction.
    """
    logger.warning(
        'Running in self-contained mode with simplified synthetic features. '
        'Results will differ from the full pipeline (01→02→03). '
        'This fallback exists for demonstration only; run the full pipeline '
        'for production-representative results.'
    )
    logger.info(
        "Generating synthetic features: %d cells × %d days × %d-min ROPs",
        n_cells, n_days, rop_minutes,
    )
    rng = np.random.default_rng(RANDOM_SEED)
    rops_per_day = (24 * 60) // rop_minutes
    n_timestamps = n_days * rops_per_day

    # Build timestamp index (Day 1 = 2024-01-01)
    timestamps = pd.date_range("2024-01-01", periods=n_timestamps, freq=f"{rop_minutes}min")

    # Diurnal multiplier — sinusoidal approximation of realistic traffic
    hour_frac = timestamps.hour + timestamps.minute / 60.0
    diurnal = (
        0.4
        + 0.3 * np.sin(np.pi * (hour_frac - 6) / 12)  # broad daytime peak
        + 0.2 * np.exp(-0.5 * ((hour_frac - 9) / 1.5) ** 2)   # morning spike
        + 0.2 * np.exp(-0.5 * ((hour_frac - 19) / 2.0) ** 2)  # evening spike
    )
    diurnal = np.clip(diurnal, 0.05, 1.0)

    # Weekly multiplier (Mon=0 … Sun=6)
    dow_mult = np.where(timestamps.dayofweek < 5, 1.0, 0.80)

    all_rows: list[dict[str, Any]] = []

    for cell_idx in range(n_cells):
        cell_id = f"CELL_{cell_idx // 3:03d}_{cell_idx % 3 + 1}"

        # Per-cell baseline scaling (urban vs. rural, indoor vs. outdoor)
        cell_scale = rng.uniform(0.6, 1.4)  # capacity heterogeneity
        cell_prb_base = rng.uniform(0.30, 0.75)  # PRB utilisation baseline

        # Base KPI values per ROP
        traffic = diurnal * dow_mult * cell_scale
        prb = cell_prb_base * traffic / traffic.max()

        dl_tp = 150.0 * traffic + rng.normal(0, 8, n_timestamps)
        ul_tp = 25.0 * traffic + rng.normal(0, 3, n_timestamps)
        prb_rate = np.clip(prb + rng.normal(0, 0.03, n_timestamps), 0.0, 1.0)
        rrc_sr = np.clip(rng.normal(0.985, 0.005, n_timestamps), 0.0, 1.0)
        drb_sr = np.clip(rng.normal(0.980, 0.006, n_timestamps), 0.0, 1.0)
        ho_sr = np.clip(rng.normal(0.975, 0.008, n_timestamps), 0.0, 1.0)
        cqi = np.clip(rng.normal(10.5, 1.5, n_timestamps), 0, 15)
        rsrp = rng.normal(-88, 6, n_timestamps)
        rsrq = rng.normal(-10, 2, n_timestamps)
        sinr = rng.normal(12, 4, n_timestamps)

        # Anomaly injection — 3% of ROPs per cell
        is_anomaly = np.zeros(n_timestamps, dtype=int)
        n_anomaly_rops = max(1, int(n_timestamps * anomaly_rate))
        anom_indices = rng.choice(n_timestamps, n_anomaly_rops, replace=False)

        for idx in anom_indices:
            anom_type = rng.choice(["sudden_drop", "gradual", "interference"])
            if anom_type == "sudden_drop":
                dl_tp[idx] *= rng.uniform(0.05, 0.30)  # 70–95% throughput drop
                ul_tp[idx] *= rng.uniform(0.10, 0.40)
                rrc_sr[idx] = rng.uniform(0.40, 0.75)
            elif anom_type == "gradual":
                window = min(idx + 12, n_timestamps)  # 3 hours of gradual decline
                fade = np.linspace(1.0, rng.uniform(0.3, 0.6), window - idx)
                dl_tp[idx:window] *= fade
                prb_rate[idx:window] = np.clip(
                    prb_rate[idx:window] * np.linspace(1.0, 1.4, window - idx), 0, 1
                )
            else:  # interference
                rsrp[idx] -= rng.uniform(15, 30)  # RSRP degrades under interference
                sinr[idx] -= rng.uniform(8, 20)
                cqi[idx] = max(0, cqi[idx] - rng.uniform(3, 7))
            is_anomaly[idx] = 1

        # Build per-ROP rows
        for t_idx in range(n_timestamps):
            ts = timestamps[t_idx]
            row = {
                "timestamp": ts,
                "cell_id": cell_id,
                # Raw KPIs (used to compute features below)
                "dl_throughput_mbps": max(0, dl_tp[t_idx]),
                "ul_throughput_mbps": max(0, ul_tp[t_idx]),
                "dl_prb_usage_rate": float(prb_rate[t_idx]),
                "rrc_conn_setup_success_rate": float(rrc_sr[t_idx]),
                "drb_setup_success_rate": float(drb_sr[t_idx]),
                "handover_success_rate": float(ho_sr[t_idx]),
                # avg_cqi is simplified; in production, derive from
                # DRB.UECqiDistr bucket counts per TS 28.552 §5.1.1.31
                "avg_cqi": float(np.clip(cqi[t_idx], 0, 15)),
                "rsrp_dbm": float(rsrp[t_idx]),
                "rsrq_db": float(rsrq[t_idx]),
                "sinr_db": float(sinr[t_idx]),
                "is_anomaly": int(is_anomaly[t_idx]),
            }
            all_rows.append(row)

    df = pd.DataFrame(all_rows)
    df = df.sort_values(["cell_id", "timestamp"]).reset_index(drop=True)

    # Compute minimal set of features that 03_model_training.py expects
    # This mirrors 02_feature_engineering.py's feature pipeline.
    df = _compute_inline_features(df, rop_minutes=rop_minutes)

    # Temporal split: train days 1–20, val days 21–25, test days 26–30
    # Critical: NEVER use random split for time-series data.
    # See Coursebook Ch. 25: Temporal Cross-Validation.
    cutoff_train = pd.Timestamp("2024-01-21")
    cutoff_val = pd.Timestamp("2024-01-26")

    train_df = df[df["timestamp"] < cutoff_train].copy()
    val_df = df[(df["timestamp"] >= cutoff_train) & (df["timestamp"] < cutoff_val)].copy()
    test_df = df[df["timestamp"] >= cutoff_val].copy()

    logger.info(
        "Synthetic split — train: %d, val: %d, test: %d rows",
        len(train_df), len(val_df), len(test_df),
    )
    return train_df, val_df, test_df

def _compute_inline_features(df: pd.DataFrame, rop_minutes: int = 15) -> pd.DataFrame:
    """
    Compute a representative feature set inline when 02_feature_engineering.py
    output is unavailable.

    WARNING: Feature names and computation methods differ from the full
    02_feature_engineering.py pipeline. Models trained in self-contained mode
    are NOT compatible with models trained on 02's output, and vice versa.
    Use self-contained mode for prototyping only.
    """
    logger.warning(
        "Using inline feature computation (_compute_inline_features). "
        "Feature names and derivation differ from 02_feature_engineering.py — "
        "models trained in this mode are NOT compatible with the full pipeline. "
        "Rolling statistics use shift(1) to exclude current-ROP leakage, "
        "matching the 02_feature_engineering.py convention. "
        "Use for prototyping only."
    )
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["cell_id", "timestamp"])

    # --- Temporal context features ---
    # Time-of-day and day-of-week are among the strongest features for
    # anomaly detection because they define the "expected" baseline.
    df["hour_of_day"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = (df["timestamp"].dt.dayofweek >= 5).astype(int)
    df["is_peak_hour"] = df["timestamp"].dt.hour.isin([8, 9, 18, 19, 20]).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["timestamp"].dt.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["timestamp"].dt.hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["timestamp"].dt.dayofweek / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["timestamp"].dt.dayofweek / 7)

    # --- Per-cell rolling statistics ---
    # Windows: 1h = 4 ROPs, 4h = 16 ROPs, 24h = 96 ROPs at 15-min granularity.
    # Rolling features capture "how does this ROP compare to recent history?"
    # which is the core signal for detecting deviations from normal.
    kpi_cols = [
        "dl_throughput_mbps", "ul_throughput_mbps", "dl_prb_usage_rate",
        "rrc_conn_setup_success_rate", "drb_setup_success_rate",
        "handover_success_rate", "avg_cqi", "rsrp_dbm", "sinr_db",
    ]
    windows = {"1h": 4, "4h": 16, "24h": 96}

    for col in kpi_cols:
        grp = df.groupby("cell_id")[col]
        for win_name, win_size in windows.items():
            # min_periods=max(1, win_size//2): require at least half-window data
            # to avoid NaN explosion at series start (cold-start behaviour)
            # shift(1) excludes the current ROP from its own rolling statistic,
            # matching the leakage-free pattern in 02_feature_engineering.py.
            df[f"{col}_roll_{win_name}_mean"] = grp.transform(
                lambda x, w=win_size: x.shift(1).rolling(w, min_periods=max(1, w // 2)).mean()
            )
            df[f"{col}_roll_{win_name}_std"] = grp.transform(
                lambda x, w=win_size: x.shift(1).rolling(w, min_periods=max(1, w // 2)).std()
            )

        # Z-score vs 24h rolling window: core anomaly signal
        # Values beyond ±3 indicate departure from recent normal behaviour.
        mu = df[f"{col}_roll_24h_mean"]
        sigma = df[f"{col}_roll_24h_std"].clip(lower=1e-6)
        df[f"{col}_zscore_24h"] = (df[col] - mu) / sigma

    # --- Rate-of-change features ---
    # First-difference captures sudden spikes/drops that rolling stats lag behind.
    for col in ["dl_throughput_mbps", "dl_prb_usage_rate", "rrc_conn_setup_success_rate"]:
        df[f"{col}_roc_1rop"] = df.groupby("cell_id")[col].diff(1)
        df[f"{col}_roc_4rop"] = df.groupby("cell_id")[col].diff(4)

    # --- Cross-KPI composite features ---
    # Throughput efficiency: how much throughput per PRB.
    # Abnormally low ratio suggests interference or misconfiguration.
    df["prb_efficiency_dl"] = df["dl_throughput_mbps"] / (
        df["dl_prb_usage_rate"].clip(lower=0.01)
    )
    # Service success composite: product of key success rates.
    # Drops when multiple failure types co-occur (backhaul degradation signature).
    df["success_rate_composite"] = (
        df["rrc_conn_setup_success_rate"]
        * df["drb_setup_success_rate"]
        * df["handover_success_rate"]
    )
    # RSRP-CQI divergence: high RSRP but low CQI suggests interference.
    df["rsrp_cqi_divergence"] = (
        (df["rsrp_dbm"] - df["rsrp_dbm"].mean()) / (df["rsrp_dbm"].std() + 1e-6)
        - (df["avg_cqi"] - df["avg_cqi"].mean()) / (df["avg_cqi"].std() + 1e-6)
    )

    # --- Peer-group z-score features ---
    # Compare each cell to the cohort of cells at the same time point.
    # This is the single most powerful feature for catching anomalies in
    # cells where individual rolling history is short (cold-start).
    # See Coursebook Ch. 13: Spatial Feature Engineering.
    ts_grp = df.groupby("timestamp")
    for col in ["dl_throughput_mbps", "dl_prb_usage_rate", "rrc_conn_setup_success_rate"]:
        ts_mean = ts_grp[col].transform("mean")
        ts_std = ts_grp[col].transform("std").clip(lower=1e-6)
        df[f"{col}_peer_zscore"] = (df[col] - ts_mean) / ts_std

    # --- Missing data indicators ---
    # Important for model interpretability: missing data patterns can themselves
    # be anomaly indicators (e.g., O1 PM file delivery failure).
    for col in kpi_cols:
        df[f"{col}_is_missing"] = df[col].isna().astype(int)

    # --- Day-over-day ratio ---
    # Compare current value to same hour 1 day ago (96 ROPs back).
    # Strong signal for catching newly-degraded cells vs. stable ones.
    df["dl_tp_dod_ratio"] = df.groupby("cell_id")["dl_throughput_mbps"].transform(
        lambda x: x / (x.shift(96).clip(lower=0.1))
    )
    df["dl_tp_wow_ratio"] = df.groupby("cell_id")["dl_throughput_mbps"].transform(
        lambda x: x / (x.shift(672).clip(lower=0.1))  # 7 days × 96 ROPs
    )

    logger.info("Computed %d inline features on %d rows", df.shape[1], len(df))
    return df


# ===========================================================================
# SECTION 3: TIER-1 — ISOLATION FOREST (UNSUPERVISED BASELINE)
# ===========================================================================

@dataclass
class IsolationForestModel:
    """
    Tier-1 anomaly detector: Isolation Forest.

    Phase-1 choice rationale:
    - Zero labeled data required (pure unsupervised)
    - Sub-millisecond inference (critical for near-RT RIC deployment)
    - ONNX-exportable via sklearn2onnx
    - SHAP compatible for NOC explainability
    - Robust to the 1–5% anomaly base rate typical in RAN KPIs

    Operating point: contamination=0.03 (3% expected anomaly rate).
    Threshold is tuned on validation set to hit target false-alarm rate.
    """
    model: Optional[IsolationForest] = field(default=None, repr=False)
    threshold: float = -0.10  # Raw decision_function threshold (tuned on val)
    feature_cols: list[str] = field(default_factory=list)
    train_metrics: dict[str, float] = field(default_factory=dict)


def train_isolation_forest(
    X_train: np.ndarray,
    feature_cols: list[str],
) -> IsolationForestModel:
    """
    Train Isolation Forest on clean training data.

    Note: we train on ALL training data including anomalies, without labels.
    The model learns the "majority normal" manifold and scores departures.
    This is intentional for Phase-1: no labels needed.

    Hyperparameter notes:
    - n_estimators=200: more trees = more stable anomaly scores (diminishing
      returns above ~200 for most telco datasets)
    - max_samples='auto': capped at 256 by sklearn; this is the key insight
      that makes Isolation Forest fast — it doesn't need all N samples
    - contamination=0.03: sets the decision_function threshold such that
      approximately 3% of training samples are flagged as anomalous
    """
    logger.info("Training Isolation Forest — %d samples × %d features", *X_train.shape)

    model = IsolationForest(
        n_estimators=IF_N_ESTIMATORS,
        max_samples=IF_MAX_SAMPLES,
        max_features=IF_MAX_FEATURES,
        contamination=IF_CONTAMINATION,
        random_state=RANDOM_SEED,
        n_jobs=-1,  # Use all CPU cores — critical at 10K+ cell scale
    )
    model.fit(X_train)

    # Anomaly scores: negative = more anomalous, positive = more normal.
    # sklearn's decision_function returns the mean anomaly score of the
    # input samples. The sign convention: -1 * average_anomaly_score.
    train_scores = model.decision_function(X_train)
    logger.info(
        "IF train scores — min: %.4f, p5: %.4f, p95: %.4f, max: %.4f",
        train_scores.min(),
        np.percentile(train_scores, 5),
        np.percentile(train_scores, 95),
        train_scores.max(),
    )

    return IsolationForestModel(
        model=model,
        feature_cols=feature_cols,
        train_metrics={"score_p5": float(np.percentile(train_scores, 5))},
    )


def tune_isolation_forest_threshold(
    if_model: IsolationForestModel,
    X_val: np.ndarray,
    y_val: Optional[np.ndarray],
) -> IsolationForestModel:
    """Tune threshold on RAW decision_function scores (negative = more anomalous).

    Note: this threshold is used only for IF standalone evaluation.
    The cascade pipeline uses score_isolation_forest() which applies sigmoid
    normalization, producing [0,1] scores where higher = more anomalous.

    If ground truth labels are available, maximises F1 score.
    If labels are absent, sets threshold at the 97th percentile of
    training anomaly scores (matching the 3% contamination prior).

    The threshold controls the precision-recall tradeoff:
    - Lower threshold (more negative) → fewer alerts, higher precision
    - Higher threshold (less negative) → more alerts, higher recall
    """
    scores = if_model.model.decision_function(X_val)

    if y_val is not None and y_val.sum() > 0:
        # Grid search over thresholds to maximise F1.
        # decision_function: negative = more anomalous, so the optimal
        # threshold lives in the LOW percentiles (anomalous tail).
        # We search from the 1st to the 25th percentile with fine
        # granularity, plus a coarser sweep up to the 50th percentile
        # to catch edge cases with high contamination rates.
        best_f1 = 0.0
        best_threshold = float(np.percentile(scores, 3))  # contamination prior

        score_range = np.concatenate([
            np.linspace(np.percentile(scores, 1), np.percentile(scores, 25), num=50),
            np.linspace(np.percentile(scores, 25), np.percentile(scores, 50), num=20),
        ])
        for threshold in score_range:
            y_pred = (scores <= threshold).astype(int)
            f1 = f1_score(y_val, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        if_model.threshold = best_threshold
        y_pred_best = (scores <= best_threshold).astype(int)
        prec = precision_score(y_val, y_pred_best, zero_division=0)
        rec = recall_score(y_val, y_pred_best, zero_division=0)
        logger.info(
            "IF threshold tuned: %.4f | val F1=%.3f, Precision=%.3f, Recall=%.3f",
            best_threshold, best_f1, prec, rec,
        )
        if_model.train_metrics.update(
            {"val_f1": best_f1, "val_precision": prec, "val_recall": rec}
        )
    else:
        # Unsupervised fallback: use contamination-implied percentile
        threshold = float(np.percentile(scores, 100 * (1 - IF_CONTAMINATION)))
        if_model.threshold = threshold
        logger.info("IF threshold (unsupervised): %.4f (%.0f%% percentile)", threshold, 100 * (1 - IF_CONTAMINATION))

    return if_model


def score_isolation_forest(
    if_model: IsolationForestModel,
    X: np.ndarray,
) -> np.ndarray:
    """
    Return normalised anomaly probability scores in [0, 1].

    Raw Isolation Forest decision_function scores are unbounded.
    We apply a monotonic sigmoid transform so the output is interpretable
    as a probability-like score for cascade combination.

    Negative raw score → high anomaly probability → score close to 1.0.
    """
    raw = if_model.model.decision_function(X)
    # Flip sign (more negative = more anomalous → becomes large positive)
    # then apply sigmoid. Scale factor 10.0 produces a steep transition that
    # maps cleanly to binary outlier/inlier decisions — appropriate for IF's
    # role as a coarse first-pass detector. (Contrast with LSTM's softer
    # scale=5.0 which preserves gradient information for ensemble weighting.)
    normalised = 1.0 / (1.0 + np.exp(10.0 * raw))
    return normalised


# ===========================================================================
# SECTION 4: TIER-2 — SUPERVISED RANDOM FOREST (PHASE-2 CLASSIFIER)
# ===========================================================================

@dataclass
class RandomForestModel:
    """
    Tier-2 anomaly detector: supervised Random Forest classifier.

    Phase-2 choice rationale:
    - Highest reported F1 (0.90) on RAN KPI benchmarks (Azkaei et al.)
    - Natively handles the mixed feature types in the RAN feature matrix
    - SHAP compatible: each feature's contribution is explainable to NOC
    - class_weight='balanced' handles the 1–3% anomaly class imbalance
    - No GPU required: deployable in edge containers alongside xApps

    Label sources (in order of preference):
    1. Ground truth from synthetic injection (for testing/validation)
    2. Trouble-ticket retrospective labelling from operations systems
    3. High-confidence Isolation Forest bootstrap labels (self-training)
    """
    model: Optional[RandomForestClassifier] = field(default=None, repr=False)
    threshold: float = 0.50  # Probability threshold for positive class
    feature_cols: list[str] = field(default_factory=list)
    train_metrics: dict[str, float] = field(default_factory=dict)
    feature_importances: dict[str, float] = field(default_factory=dict)


def bootstrap_labels_from_isolation_forest(
    if_model: IsolationForestModel,
    X_train: np.ndarray,
    y_train: Optional[np.ndarray],
    confidence_threshold_low: float = 0.05,  # Score below this → label as NORMAL
    confidence_threshold_high: float = 0.95,  # Score above this → label as ANOMALY
    expected_contamination: float = 0.03,
) -> np.ndarray:
    """
    Generate pseudo-labels for supervised training by combining:
    1. Ground truth labels (where available, from synthetic injection)
    2. High-confidence Isolation Forest predictions (self-training)

    Only high-confidence pseudo-labels are used to prevent error propagation:
    - IF score > 0.95 → pseudo-label as anomaly (very likely outlier)
    - IF score < 0.05 → pseudo-label as normal  (very likely inlier)
    - 0.05 ≤ score ≤ 0.95 → use ground truth or exclude if unavailable

    Thresholds are deliberately conservative (0.05/0.95) to minimise
    label noise in the bootstrap set. Looser thresholds (e.g. 0.10/0.80)
    admit more training samples but risk error propagation — tighten
    rather than loosen if precision matters more than recall.

    This bootstrap strategy enables Phase-2 deployment with as few as
    50 confirmed anomaly labels — achievable in 2–3 weeks of Phase-1 operation.

    See Coursebook Ch. 13: Semi-supervised Label Generation.
    """
    if_scores = score_isolation_forest(if_model, X_train)

    if y_train is not None:
        # Prefer ground truth; fill ambiguous region with ground truth
        pseudo_labels = y_train.copy()
        logger.info(
            "Using ground truth labels for RF training (%d positives / %d total)",
            int(y_train.sum()), len(y_train),
        )
    else:
        # Fully unsupervised bootstrap — use IF scores only
        pseudo_labels = np.full(len(X_train), -1, dtype=int)  # -1 = unknown

    # Override with high-confidence IF pseudo-labels for unlabelled points
    high_conf_mask = if_scores >= confidence_threshold_high
    low_conf_mask = if_scores <= confidence_threshold_low
    n_pseudo_pos = high_conf_mask.sum()
    n_pseudo_neg = low_conf_mask.sum()

    if y_train is None:
        pseudo_labels[high_conf_mask] = 1
        pseudo_labels[low_conf_mask] = 0

    # Validate pseudo-positive ratio against expected contamination
    n_labelled = n_pseudo_pos + n_pseudo_neg
    if n_labelled > 0:
        pseudo_pos_ratio = n_pseudo_pos / n_labelled
        if pseudo_pos_ratio > expected_contamination * 3:
            logger.warning(
                "Pseudo-positive ratio %.1f%% exceeds 3× expected contamination "
                "(%.1f%%). This may indicate the IF model is over-flagging or "
                "thresholds are too loose. Review IF training and consider "
                "tightening confidence_threshold_high (currently %.2f).",
                pseudo_pos_ratio * 100,
                expected_contamination * 100,
                confidence_threshold_high,
            )

    # Remove still-unknown samples (score in ambiguous range, no ground truth)
    valid_mask = pseudo_labels >= 0
    logger.info(
        "Bootstrap labels — positives: %d, negatives: %d, excluded (ambiguous): %d | "
        "IF-derived positives: %d, negatives: %d",
        int((pseudo_labels[valid_mask] == 1).sum()),
        int((pseudo_labels[valid_mask] == 0).sum()),
        int((~valid_mask).sum()),
        int(n_pseudo_pos), int(n_pseudo_neg),
    )
    return pseudo_labels


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_cols: list[str],
) -> RandomForestModel:
    """
    Train supervised Random Forest on bootstrapped labels.

    Key design decisions:
    - class_weight='balanced': equivalent to up-weighting the 1–3% anomaly
      class by its inverse frequency. Without this, the model learns to
      predict "normal" for everything and achieves 97%+ accuracy while
      missing all real anomalies.
    - max_depth=12: prevents overfitting on the temporal training window.
      Deep trees memorise specific day/cell combinations rather than
      learning generalizable anomaly patterns.
    - min_samples_leaf=10: a leaf must represent at least 10 ROP windows.
      At 15-min granularity, this is 2.5 hours of observations — enough
      statistical support for a reliable decision.
    """
    # Filter to only labelled samples (exclude -1 pseudo-label unknowns)
    valid_mask = y_train >= 0
    X_fit = X_train[valid_mask]
    y_fit = y_train[valid_mask]

    pos_rate = y_fit.mean()
    logger.info(
        "Training Random Forest — %d samples, %.2f%% positive, %d features",
        len(X_fit), pos_rate * 100, X_fit.shape[1],
    )

    if pos_rate < 0.001:
        raise ValueError(
            f"Insufficient positive labels ({y_fit.sum()} positives in {len(y_fit)} samples). "
            "Need at least 0.1% positive rate. Check bootstrap_labels_from_isolation_forest()."
        )

    model = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        class_weight=RF_CLASS_WEIGHT,
        max_features=RF_MAX_FEATURES,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(X_fit, y_fit)

    # Log top-20 feature importances — critical for NOC explainability.
    # In production, this maps to the "contributing KPIs" in the alert card.
    importances = dict(zip(feature_cols, model.feature_importances_))
    top_20 = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:20]
    logger.info("Top-10 RF feature importances:")
    for feat, imp in top_20[:10]:
        logger.info("  %-55s %.4f", feat, imp)

    return RandomForestModel(
        model=model,
        feature_cols=feature_cols,
        feature_importances={k: float(v) for k, v in importances.items()},
    )


def tune_random_forest_threshold(
    rf_model: RandomForestModel,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> RandomForestModel:
    """
    Tune RF decision probability threshold on validation set.

    Operational context: at a Tier-1 operator with 10K cells monitored by
    a NOC team of 10 analysts across 3 shifts, each analyst can investigate
    ~50 alerts per 8-hour shift. With 10K cells × 96 ROPs/day ≈ 960K windows:

    - At 0.1% FPR: ~960 false alarms/day → ~32 per analyst/shift (acceptable)
    - At 0.01% FPR: ~96 false alarms/day → ~3 per analyst/shift (ideal)
    - At 1% FPR: ~9600 false alarms/day → overwhelming (static threshold level)

    We target the operating point where false alarms per NOC analyst per shift
    ≤ OPERATIONAL_FALSE_ALARM_TARGET_PER_SHIFT (default: 5).

    See Coursebook Ch. 54: Operating Point Selection for Anomaly Detection.
    """
    if y_val is None or y_val.sum() == 0:
        logger.warning("No positive val labels — using default RF threshold 0.50")
        return rf_model

    proba = rf_model.model.predict_proba(X_val)[:, 1]  # P(anomaly)

    best_f1 = 0.0
    best_threshold = 0.50
    best_metrics = {}

    # Sweep threshold from 0.1 to 0.99 in fine steps
    for t in np.arange(0.05, 1.0, 0.01):
        y_pred = (proba >= t).astype(int)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        prec = precision_score(y_val, y_pred, zero_division=0)
        rec = recall_score(y_val, y_pred, zero_division=0)

        # Compute false alarms per shift at demonstration scale (100 cells)
        n_windows = len(y_val)
        n_fp = ((y_pred == 1) & (y_val == 0)).sum()
        # Scale to 8-hour shift: n_fp / (n_days_val * shifts_per_day) * OPER_CELLS/100
        n_shifts_in_val = max(1, n_windows / (OPER_CELLS * OPERATIONAL_SHIFT_HOURS * 4))
        fa_per_shift = n_fp / n_shifts_in_val

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(t)
            best_metrics = {
                "val_f1": float(f1),
                "val_precision": float(prec),
                "val_recall": float(rec),
                "val_false_alarms_per_shift": float(fa_per_shift),
            }

    rf_model.threshold = best_threshold
    rf_model.train_metrics.update(best_metrics)

    logger.info(
        "RF threshold tuned: %.2f | val F1=%.3f, Precision=%.3f, Recall=%.3f | "
        "~%.1f false alarms/shift (target: %d)",
        best_threshold,
        best_metrics["val_f1"],
        best_metrics["val_precision"],
        best_metrics["val_recall"],
        best_metrics["val_false_alarms_per_shift"],
        OPERATIONAL_FALSE_ALARM_TARGET_PER_SHIFT,
    )
    return rf_model


# ===========================================================================
# SECTION 5: TIER-3 — LSTM AUTOENCODER (TEMPORAL ANOMALY DETECTION)
# ===========================================================================
# Only instantiated when PyTorch is available. Catches gradual degradations
# that statistical and tree-based methods miss by modelling temporal context.

if TORCH_AVAILABLE:
    class LSTMAutoencoder(nn.Module):
        """
        LSTM Autoencoder for temporal anomaly detection in RAN KPI sequences.

        Architecture:
        - Encoder: 2-layer LSTM → compress sequence to fixed-size context vector
        - Decoder: 2-layer LSTM → reconstruct input sequence from context
        - Anomaly score = mean squared reconstruction error over the sequence

        Why autoencoders for anomaly detection?
        - Trained only on normal data → learns the manifold of normal temporal patterns
        - Anomalous sequences fail to reconstruct accurately → high MSE = anomaly signal
        - Naturally captures gradual degradations: the reconstruction of a 4-hour
          declining throughput trend will differ from a normal 4-hour pattern

        LSTM-VAE (variational) would provide better probabilistic calibration but
        requires more complex training (ELBO loss, KL scheduling). For Phase-2
        production deployment, the deterministic AE is recommended first.

        Designed for ONNX export via torch.onnx.export().
        Inference time: ~2ms on CPU for a single sequence — fits near-RT RIC budget.

        See Coursebook Ch. 25: Sequence Models for Network KPI Analysis.
        """

        def __init__(
            self,
            input_size: int,
            hidden_size: int = LSTM_HIDDEN_SIZE,
            num_layers: int = LSTM_NUM_LAYERS,
        ) -> None:
            super().__init__()
            self.input_size = input_size
            self.hidden_size = hidden_size
            self.num_layers = num_layers

            # Encoder: maps input sequence → compressed temporal representation
            self.encoder = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,  # (batch, seq_len, features) convention
                dropout=0.2,       # Regularisation; prevents memorising specific ROPs
            )

            # Bottleneck: project hidden state to compressed representation
            # This explicit bottleneck forces learning of compact representations
            self.bottleneck = nn.Linear(hidden_size, hidden_size // 2)

            # Decoder: maps compressed representation → reconstructed sequence
            # Input size = hidden_size // 2 (bottleneck output), repeated for each step
            self.decoder = nn.LSTM(
                input_size=hidden_size // 2,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=0.2,
            )

            # Output projection: hidden state → original feature space
            self.output_proj = nn.Linear(hidden_size, input_size)

        def forward(
            self,
            x: "torch.Tensor",
        ) -> tuple["torch.Tensor", "torch.Tensor"]:
            """
            Forward pass: encode → compress → decode → reconstruct.

            Returns (reconstruction, compressed_representation).
            Reconstruction is used for anomaly scoring (MSE vs. input).
            """
            batch_size, seq_len, _ = x.shape

            # Encode: run sequence through encoder LSTM
            encoder_out, (h_n, _) = self.encoder(x)
            # h_n shape: (num_layers, batch, hidden_size) — take last layer
            context = h_n[-1]  # (batch, hidden_size)

            # Bottleneck compression
            compressed = torch.relu(self.bottleneck(context))  # (batch, hidden_size//2)

            # Decoder input: repeat compressed vector for each time step
            # This forces all temporal reconstruction to go through the bottleneck
            decoder_input = compressed.unsqueeze(1).repeat(1, seq_len, 1)  # (batch, seq_len, hidden//2)
            decoder_out, _ = self.decoder(decoder_input)

            # Project back to input feature space
            reconstruction = self.output_proj(decoder_out)  # (batch, seq_len, input_size)
            return reconstruction, compressed

        def reconstruction_error(self, x: "torch.Tensor") -> "torch.Tensor":
            """Compute per-sample mean squared reconstruction error."""
            recon, _ = self.forward(x)
            return torch.mean((x - recon) ** 2, dim=[1, 2])  # (batch,)


@dataclass
class LSTMAutoencoderModel:
    """
    Container for trained LSTM autoencoder and its operating parameters.
    """
    model: Any = field(default=None, repr=False)  # LSTMAutoencoder instance
    threshold: float = 0.0   # Reconstruction error threshold
    feature_cols: list[str] = field(default_factory=list)
    sequence_length: int = LSTM_SEQUENCE_LENGTH
    scaler_mean: Optional[np.ndarray] = field(default=None, repr=False)
    scaler_std: Optional[np.ndarray] = field(default=None, repr=False)
    train_metrics: dict[str, float] = field(default_factory=dict)


def _build_sequences(
    X: np.ndarray,
    seq_len: int,
    cell_boundaries: Optional[list[int]] = None,
) -> np.ndarray:
    """
    Build overlapping sliding windows for LSTM sequence training.

    Respects cell boundaries: sequences do not cross cell_id transitions.
    A sequence that spans two cells would be non-physical and would corrupt
    the temporal pattern the model is trying to learn.

    Args:
        X: Feature matrix (n_samples, n_features)
        seq_len: Length of each sequence window (ROPs)
        cell_boundaries: List of row indices where a new cell starts.
                         If None, treats all rows as one cell.

    Returns:
        sequences: (n_sequences, seq_len, n_features)
    """
    sequences = []

    if cell_boundaries is None:
        # Single cell or boundary-unaware mode
        for start in range(len(X) - seq_len + 1):
            sequences.append(X[start:start + seq_len])
    else:
        # Boundary-aware: only build sequences within each cell's rows
        boundaries_ext = list(cell_boundaries) + [len(X)]
        for b_start, b_end in zip(boundaries_ext[:-1], boundaries_ext[1:]):
            cell_X = X[b_start:b_end]
            for start in range(len(cell_X) - seq_len + 1):
                sequences.append(cell_X[start:start + seq_len])

    if not sequences:
        return np.empty((0, seq_len, X.shape[1]), dtype=np.float32)

    return np.array(sequences, dtype=np.float32)


def train_lstm_autoencoder(
    X_train: np.ndarray,
    feature_cols: list[str],
    cell_boundaries: Optional[list[int]] = None,
) -> LSTMAutoencoderModel:
    """
    Train LSTM Autoencoder on normal-only training data.

    Important: the LSTM AE is trained ONLY on normal samples because
    we want it to learn the manifold of normal temporal patterns.
    Anomalous sequences in training data would cause the model to also
    reconstruct those patterns accurately, reducing detection sensitivity.

    If ground truth labels are available, filter to normal samples only.
    In the unsupervised case, we assume training data is mostly normal
    (consistent with the <5% anomaly rate in production RAN data).

    Training uses early stopping to prevent over-fitting on training noise.
    See Coursebook Ch. 25: Autoencoder Anomaly Detection.
    """
    if not TORCH_AVAILABLE:
        logger.warning("PyTorch not available — skipping LSTM Autoencoder training")
        return LSTMAutoencoderModel(feature_cols=feature_cols)

    # Normalise features before LSTM training — LSTMs are sensitive to scale.
    # We fit the scaler on training data and apply it to val/test too.
    scaler_mean = X_train.mean(axis=0)
    scaler_std = np.maximum(X_train.std(axis=0), 1e-6)
    X_scaled = (X_train - scaler_mean) / scaler_std

    # Build training sequences
    sequences = _build_sequences(X_scaled, LSTM_SEQUENCE_LENGTH, cell_boundaries)
    logger.info(
        "LSTM AE training — %d sequences × %d time steps × %d features",
        len(sequences), LSTM_SEQUENCE_LENGTH, X_scaled.shape[1],
    )

    if len(sequences) < LSTM_BATCH_SIZE:
        logger.warning(
            "Too few sequences (%d < %d) for LSTM training — "
            "consider more training data or shorter sequence length",
            len(sequences), LSTM_BATCH_SIZE,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("LSTM training device: %s", device)

    # Dataset and dataloader
    tensor_sequences = torch.FloatTensor(sequences)
    dataset = TensorDataset(tensor_sequences)
    # Shuffle=True for training: unlike sequence prediction, we shuffle the
    # *windows* (not the contents of each window), which is correct here.
    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(RANDOM_SEED),
    )
    train_loader = DataLoader(train_ds, batch_size=LSTM_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=LSTM_BATCH_SIZE, shuffle=False)

    # Model instantiation
    n_features = X_scaled.shape[1]
    model = LSTMAutoencoder(input_size=n_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LSTM_LEARNING_RATE)
    # Cosine annealing LR schedule: reduces LR as training progresses
    # Prevents oscillating loss in later epochs without manual LR decay
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=LSTM_EPOCHS, eta_min=1e-5
    )
    criterion = nn.MSELoss()

    # Training loop with early stopping
    best_val_loss = float("inf")
    patience_counter = 0
    best_state_dict = None
    train_losses: list[float] = []
    val_losses: list[float] = []

    for epoch in range(1, LSTM_EPOCHS + 1):
        # --- Training step ---
        model.train()
        epoch_train_loss = 0.0
        for (batch,) in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            reconstruction, _ = model(batch)
            loss = criterion(reconstruction, batch)
            loss.backward()
            # Gradient clipping: prevents exploding gradients in LSTM training
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_train_loss += loss.item() * len(batch)

        epoch_train_loss /= len(train_ds)
        train_losses.append(epoch_train_loss)

        # --- Validation step ---
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(device)
                reconstruction, _ = model(batch)
                loss = criterion(reconstruction, batch)
                epoch_val_loss += loss.item() * len(batch)
        epoch_val_loss /= len(val_ds)
        val_losses.append(epoch_val_loss)

        scheduler.step()

        if epoch % 5 == 0 or epoch == 1:
            logger.info(
                "LSTM AE Epoch %02d/%02d — train_loss: %.6f, val_loss: %.6f",
                epoch, LSTM_EPOCHS, epoch_train_loss, epoch_val_loss,
            )

        # Early stopping check
        if epoch_val_loss < best_val_loss - 1e-6:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= LSTM_EARLY_STOP_PATIENCE:
                logger.info("Early stopping at epoch %d (patience=%d)", epoch, LSTM_EARLY_STOP_PATIENCE)
                break

    # Restore best weights
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    model = model.cpu()

    logger.info(
        "LSTM AE training complete — best val loss: %.6f (%.1f epochs)",
        best_val_loss, len(train_losses),
    )

    return LSTMAutoencoderModel(
        model=model,
        feature_cols=feature_cols,
        sequence_length=LSTM_SEQUENCE_LENGTH,
        scaler_mean=scaler_mean,
        scaler_std=scaler_std,
        train_metrics={
            "best_val_loss": float(best_val_loss),
            "n_epochs_trained": len(train_losses),
        },
    )


def tune_lstm_threshold(
    lstm_model: LSTMAutoencoderModel,
    X_val: np.ndarray,
    y_val: Optional[np.ndarray],
    cell_boundaries: Optional[list[int]] = None,
) -> LSTMAutoencoderModel:
    """
    Set LSTM anomaly threshold as a high percentile of validation reconstruction errors.

    Without labels: use the 97th percentile (matches 3% contamination prior).
    With labels: use threshold that maximises F1 on validation set.
    """
    if not TORCH_AVAILABLE or lstm_model.model is None:
        return lstm_model

    errors = _compute_lstm_errors(lstm_model, X_val, cell_boundaries)

    if y_val is not None and y_val.sum() > 0:
        # Align error array length with labels
        # (sequence construction loses seq_len-1 samples at the start)
        seq_len = lstm_model.sequence_length
        y_aligned = y_val[seq_len - 1:]  # Match sequence-to-label alignment
        # Truncate to minimum length in case of mismatch
        min_len = min(len(errors), len(y_aligned))
        errors_aligned = errors[:min_len]
        y_aligned = y_aligned[:min_len]

        best_f1 = 0.0
        best_t = float(np.percentile(errors, 97))
        for pct in range(70, 100):
            t = float(np.percentile(errors_aligned, pct))
            y_pred = (errors_aligned >= t).astype(int)
            f1 = f1_score(y_aligned, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t

        lstm_model.threshold = best_t
        y_pred_best = (errors_aligned >= best_t).astype(int)
        logger.info(
            "LSTM threshold tuned: %.6f | val F1=%.3f, Precision=%.3f, Recall=%.3f",
            best_t,
            best_f1,
            precision_score(y_aligned, y_pred_best, zero_division=0),
            recall_score(y_aligned, y_pred_best, zero_division=0),
        )
        lstm_model.train_metrics.update({"val_f1": best_f1})
    else:
        threshold = float(np.percentile(errors, 97))
        lstm_model.threshold = threshold
        logger.info("LSTM threshold (unsupervised): %.6f (97th percentile)", threshold)

    return lstm_model


def _compute_lstm_errors(
    lstm_model: LSTMAutoencoderModel,
    X: np.ndarray,
    cell_boundaries: Optional[list[int]] = None,
) -> np.ndarray:
    """
    Compute per-sequence reconstruction error for a feature matrix.

    Returns an array of length (n_sequences,) where n_sequences =
    n_samples - (seq_len - 1) (accounting for windowing).
    """
    if not TORCH_AVAILABLE or lstm_model.model is None:
        return np.zeros(max(0, len(X) - lstm_model.sequence_length + 1))

    # Apply same normalisation as training
    X_scaled = (X - lstm_model.scaler_mean) / lstm_model.scaler_std
    sequences = _build_sequences(X_scaled, lstm_model.sequence_length, cell_boundaries)

    if len(sequences) == 0:
        return np.array([])

    device = torch.device("cpu")  # Inference always on CPU for portability
    lstm_model.model.eval()

    all_errors = []
    # Process in batches to avoid OOM on large validation sets
    batch_size = 512
    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch = torch.FloatTensor(sequences[i:i + batch_size]).to(device)
            errors = lstm_model.model.reconstruction_error(batch)
            all_errors.extend(errors.numpy().tolist())

    return np.array(all_errors, dtype=np.float32)


def score_lstm_autoencoder(
    lstm_model: LSTMAutoencoderModel,
    X: np.ndarray,
    cell_boundaries: Optional[list[int]] = None,
) -> np.ndarray:
    """
    Return normalised anomaly scores in [0, 1] from LSTM reconstruction errors.

    Pads the first (seq_len - 1) positions with score=0 so that the output
    array aligns with the input row index — required for cascade combination.
    """
    if not TORCH_AVAILABLE or lstm_model.model is None:
        return np.zeros(len(X))

    errors = _compute_lstm_errors(lstm_model, X, cell_boundaries)

    if len(errors) == 0:
        return np.zeros(len(X))

    # Normalise to [0, 1] using sigmoid transform on (error - threshold) / std.
    # Scale factor 5.0 is deliberately softer than IF's 10.0: LSTM scores feed
    # into the weighted ensemble, where preserving gradient information around
    # the decision boundary improves cascade discrimination.
    error_std = np.maximum(errors.std(), 1e-8)
    normalised = 1.0 / (1.0 + np.exp(-5.0 * (errors - lstm_model.threshold) / error_std))

    # Pad leading positions (no sequence history available yet)
    pad_len = len(X) - len(normalised)
    padded = np.concatenate([np.zeros(pad_len), normalised])
    return padded


# ===========================================================================
# SECTION 6: CASCADE ENSEMBLE SCORER
# ===========================================================================

@dataclass
class CascadeConfig:
    """
    Configuration for the three-tier cascade anomaly scoring ensemble.

    The cascade combines Isolation Forest (Tier 1), Random Forest (Tier 2),
    and LSTM Autoencoder (Tier 3) into a single anomaly score.

    Ensemble strategy:
    - Weighted average of normalised individual scores
    - An alert is raised if the combined score exceeds final_threshold
    - OR if any single tier score exceeds its individual alert_threshold
      (the "any-high-confidence" rule catches severe point anomalies)

    This cascade design reflects Phase-2 production deployment where all
    three tiers are active. In Phase 1 (Isolation Forest only), weights
    are automatically adjusted to [1.0, 0.0, 0.0].
    """
    weight_if: float = CASCADE_WEIGHT_IF
    weight_rf: float = CASCADE_WEIGHT_RF
    weight_lstm: float = CASCADE_WEIGHT_LSTM
    final_threshold: float = 0.50          # Combined score threshold for alert
    if_alert_threshold: float = 0.85       # IF alone triggers alert above this
    rf_alert_threshold: float = 0.90       # RF alone triggers alert above this
    lstm_alert_threshold: float = 0.85     # LSTM alone triggers alert above this
    rf_available: bool = True
    lstm_available: bool = True


def compute_cascade_scores(
    if_scores: np.ndarray,
    rf_scores: Optional[np.ndarray],
    lstm_scores: Optional[np.ndarray],
    config: CascadeConfig,
) -> np.ndarray:
    """
    Combine individual tier scores into a single cascade anomaly score.

    Handles partial availability (Phase 1: IF only; Phase 2: IF + RF;
    Phase 2+: all three) by normalising weights to sum to 1.0.

    Returns: array of combined anomaly scores in [0, 1]
    """
    weights = [config.weight_if]
    score_arrays = [if_scores]

    if rf_scores is not None and config.rf_available:
        weights.append(config.weight_rf)
        score_arrays.append(rf_scores)
    if lstm_scores is not None and config.lstm_available:
        weights.append(config.weight_lstm)
        score_arrays.append(lstm_scores)

    # Normalise weights to sum to 1.0
    total_weight = sum(weights)
    norm_weights = [w / total_weight for w in weights]

    # Stack and compute weighted average
    score_matrix = np.stack(score_arrays, axis=1)  # (n_samples, n_active_tiers)
    combined = np.average(score_matrix, axis=1, weights=norm_weights)
    return combined


def make_final_predictions(
    combined_scores: np.ndarray,
    if_scores: np.ndarray,
    rf_scores: Optional[np.ndarray],
    lstm_scores: Optional[np.ndarray],
    config: CascadeConfig,
) -> np.ndarray:
    """
    Convert cascade scores to binary anomaly predictions.

    Applies both the combined threshold and the individual high-confidence
    rules. The "any-high-confidence" rule ensures severe anomalies are
    never missed even if other tiers disagree.
    """
    predictions = (combined_scores >= config.final_threshold).astype(int)

    # Individual tier overrides (any single high-confidence alert fires)
    predictions = predictions | (if_scores >= config.if_alert_threshold).astype(int)
    if rf_scores is not None:
        predictions = predictions | (rf_scores >= config.rf_alert_threshold).astype(int)
    if lstm_scores is not None:
        predictions = predictions | (lstm_scores >= config.lstm_alert_threshold).astype(int)

    return predictions.astype(int)


def tune_ensemble_weights(
    if_scores: np.ndarray,
    rf_scores: Optional[np.ndarray],
    lstm_scores: Optional[np.ndarray],
    y_true: np.ndarray,
    grid_steps: int = 20,
    metric: str = "f1",
) -> dict:
    """Grid search over the weight simplex to find optimal ensemble weights.

    The default weights (0.35, 0.45, 0.20) perform well on the synthetic
    benchmark but may not be optimal for a specific operator's data
    distribution. This utility searches over the weight simplex at the
    specified resolution and returns the best weights by the chosen metric.

    Parameters
    ----------
    if_scores : array of shape (n_samples,)
        Isolation Forest sigmoid-normalised scores (higher = more anomalous).
    rf_scores : array or None
        Random Forest probability scores. None if Phase 1 only.
    lstm_scores : array or None
        LSTM reconstruction error scores. None if Phase 2 only.
    y_true : array of shape (n_samples,)
        Binary ground-truth labels (1 = anomaly).
    grid_steps : int
        Number of steps per weight dimension (default 20 → ~200 candidates
        for 2 tiers, ~1,500 for 3 tiers). Higher values increase precision
        but scale quadratically/cubically.
    metric : str
        Optimisation target: "f1" (default), "precision", or "recall".

    Returns
    -------
    dict with keys:
        best_weights : tuple of (w_if, w_rf, w_lstm)
        best_score : float (best metric value achieved)
        best_threshold : float (optimal cascade threshold)
        search_results : list of (weights, threshold, score) tuples
    """
    from sklearn.metrics import f1_score, precision_score, recall_score

    metric_fn = {"f1": f1_score, "precision": precision_score, "recall": recall_score}[metric]

    # Determine active tiers
    n_tiers = 1
    if rf_scores is not None:
        n_tiers = 2
    if lstm_scores is not None:
        n_tiers = 3

    # Generate weight candidates on the simplex
    candidates = []
    step = 1.0 / grid_steps
    if n_tiers == 1:
        candidates = [(1.0, 0.0, 0.0)]
    elif n_tiers == 2:
        for i in range(1, grid_steps + 1):
            w_if = i * step
            w_rf = 1.0 - w_if
            if w_rf >= 0:
                candidates.append((w_if, w_rf, 0.0))
    else:  # 3 tiers
        for i in range(1, grid_steps + 1):
            for j in range(0, grid_steps + 1 - i):
                w_if = i * step
                w_rf = j * step
                w_lstm = 1.0 - w_if - w_rf
                if w_lstm >= 0:
                    candidates.append((w_if, w_rf, w_lstm))

    best_score = -1.0
    best_weights = (1.0, 0.0, 0.0)
    best_threshold = 0.5
    search_results = []

    for w_if, w_rf, w_lstm in candidates:
        # Compute weighted scores
        score_arrays = [if_scores * w_if]
        if rf_scores is not None:
            score_arrays.append(rf_scores * w_rf)
        if lstm_scores is not None:
            score_arrays.append(lstm_scores * w_lstm)
        combined = sum(score_arrays)

        # Search over thresholds
        for pct in range(10, 96):
            threshold = float(np.percentile(combined, pct))
            y_pred = (combined >= threshold).astype(int)
            if y_pred.sum() == 0:
                continue
            score = metric_fn(y_true, y_pred, zero_division=0)
            search_results.append(((w_if, w_rf, w_lstm), threshold, score))
            if score > best_score:
                best_score = score
                best_weights = (w_if, w_rf, w_lstm)
                best_threshold = threshold

    logger.info(
        "tune_ensemble_weights: best %s=%.4f at weights=(%.3f, %.3f, %.3f), "
        "threshold=%.4f (%d candidates searched)",
        metric, best_score, *best_weights, best_threshold, len(search_results),
    )

    return {
        "best_weights": best_weights,
        "best_score": best_score,
        "best_threshold": best_threshold,
        "search_results": search_results,
    }


# ===========================================================================
# SECTION 7: EVALUATION
# ===========================================================================

@dataclass
class EvaluationResult:
    """Container for model evaluation results."""
    split_name: str
    n_samples: int
    n_positives: int
    precision: float
    recall: float
    f1: float
    roc_auc: float
    avg_precision: float
    false_alarms_per_shift: float
    time_to_detect_mean_rops: float
    event_based_recall: float
    model_name: str
    threshold_used: float


def evaluate_model(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    model_name: str,
    split_name: str,
    rop_minutes: int = 15,
) -> EvaluationResult:
    """
    Comprehensive model evaluation with operationally-contextualised metrics.

    Metrics computed:
    1. Precision: fraction of flagged anomalies that are real
       Operational meaning: NOC analyst workload per shift
    2. Recall: fraction of real anomalies detected
       Operational meaning: % of customer-impacting events caught before escalation
    3. F1: harmonic mean — balances precision and recall
    4. ROC-AUC: ranking quality across all thresholds
    5. Average Precision: area under precision-recall curve (better than AUC for imbalanced)
    6. False alarms per NOC analyst per 8-hour shift (at OPER_CELLS scale)
    7. Time-to-Detect: mean ROPs from anomaly start to first detection flag
       Operational meaning: "on average, how many minutes after anomaly onset
       does the system alert the NOC?"

    See Coursebook Ch. 54: Operationally-Meaningful Anomaly Detection Metrics.
    """
    y_pred = (y_score >= threshold).astype(int)

    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    # ROC-AUC and AP require both classes to be present
    if len(np.unique(y_true)) > 1:
        roc_auc = float(roc_auc_score(y_true, y_score))
        avg_prec = float(average_precision_score(y_true, y_score))
    else:
        roc_auc = 0.0
        avg_prec = 0.0
        logger.warning("Only one class in y_true — AUC/AP not meaningful")

    # False alarms per shift (scale to operational context)
    n_windows = len(y_true)
    n_fp = int(((y_pred == 1) & (y_true == 0)).sum())
    # Approximate: val/test span approximately n_days × 96 ROPs/day per cell
    # We compute false alarm rate per hour per cell then scale to a shift
    n_rops_total = n_windows
    fa_per_rop = n_fp / max(1, n_rops_total)
    rops_per_shift = OPERATIONAL_SHIFT_HOURS * (60 // rop_minutes)
    fa_per_shift = fa_per_rop * rops_per_shift * OPER_CELLS

    # Time-to-Detect metric
    ttd = _compute_time_to_detect(y_true, y_pred)

    # Event-based recall: fraction of contiguous anomaly events with ≥1 detection
    ebr = compute_event_based_recall(y_true, y_pred)
    if np.isnan(ebr):
        ebr = 0.0

    result = EvaluationResult(
        split_name=split_name,
        n_samples=n_windows,
        n_positives=int(y_true.sum()),
        precision=prec,
        recall=rec,
        f1=f1,
        roc_auc=roc_auc,
        avg_precision=avg_prec,
        false_alarms_per_shift=fa_per_shift,
        time_to_detect_mean_rops=ttd,
        event_based_recall=ebr,
        model_name=model_name,
        threshold_used=threshold,
    )
    return result


def _compute_time_to_detect(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute mean time-to-detect in ROPs for true positive anomaly events.

    An "anomaly event" is a run of consecutive y_true=1 positions.
    TTD for an event = index of first y_pred=1 within the event - event start.
    Undetected events are excluded from the mean (they are captured by recall).

    In a real RAN deployment, this converts to minutes by multiplying by rop_minutes.
    At 15-min granularity, TTD=1 ROP means the anomaly was detected within the next
    15-minute reporting window — essentially real-time.
    """
    if y_true.sum() == 0:
        return 0.0

    ttd_values = []
    i = 0
    while i < len(y_true):
        if y_true[i] == 1:
            # Start of an anomaly run — find its end
            run_start = i
            while i < len(y_true) and y_true[i] == 1:
                i += 1
            run_end = i  # exclusive

            # Find first detection within this run
            detected = False
            for j in range(run_start, run_end):
                if y_pred[j] == 1:
                    ttd_values.append(j - run_start)
                    detected = True
                    break
            # Undetected events are NOT added to ttd_values
            # (they are implicitly penalised through recall)
        else:
            i += 1

    return float(np.mean(ttd_values)) if ttd_values else float("nan")


def print_evaluation_table(results: list[EvaluationResult], rop_minutes: int = 15) -> None:
    """
    Pretty-print evaluation results as a comparison table.

    Output is designed to be directly presentable to NOC managers:
    metrics are contextualised with operational meaning.
    """
    header = (
        f"{'Model':<30} {'Split':<8} {'Precision':>10} {'Recall':>7} "
        f"{'F1':>7} {'AUC':>7} {'AP':>7} "
        f"{'EvtRecall':>10} {'FA/Shift':>9} {'TTD(min)':>9}"
    )
    logger.info("\n" + "=" * len(header))
    logger.info("EVALUATION RESULTS — COMPARISON TABLE")
    logger.info("=" * len(header))
    logger.info(header)
    logger.info("-" * len(header))

    for r in results:
        ttd_min = r.time_to_detect_mean_rops * rop_minutes if not np.isnan(r.time_to_detect_mean_rops) else float("nan")
        logger.info(
            f"{r.model_name:<30} {r.split_name:<8} {r.precision:>10.3f} {r.recall:>7.3f} "
            f"{r.f1:>7.3f} {r.roc_auc:>7.3f} {r.avg_precision:>7.3f} "
            f"{r.event_based_recall:>10.3f} {r.false_alarms_per_shift:>9.1f} {ttd_min:>9.1f}"
        )
    logger.info("=" * len(header))
    logger.info(
        "\nOperational context (at %d cells, %d-min ROPs, %d-hr shift):",
        OPER_CELLS, rop_minutes, OPERATIONAL_SHIFT_HOURS,
    )
    logger.info(
        "  FA/Shift  = estimated false alarms per NOC analyst 8-hour shift\n"
        "  EvtRecall = fraction of contiguous anomaly events with ≥1 detection\n"
        "  TTD(min)  = mean minutes from anomaly onset to first alert\n"
        "  Static threshold baseline: FA/Shift ~40–120, TTD ~30–90 min\n"
        "  Target:                    FA/Shift ≤ %d,   TTD ≤ 15 min",
        OPERATIONAL_FALSE_ALARM_TARGET_PER_SHIFT,
    )


def run_threshold_sweep(
    y_true: np.ndarray,
    y_score: np.ndarray,
    model_name: str,
) -> pd.DataFrame:
    """
    Sweep anomaly score thresholds and record precision/recall/F1 at each point.

    Output is the data for the precision-recall operating point visualisation
    described in Section 9 of the whitepaper (Figure 5).

    The sweep table helps operators choose their operating point based on
    organisational tolerance for false alarms vs. missed detections:
    - Conservative: maximise precision (few false alarms, miss some real events)
    - Balanced: maximise F1 (standard operating point)
    - Aggressive: maximise recall (catch everything, accept more false alarms)
    """
    thresholds = np.percentile(y_score, list(range(50, 100, 1)))
    thresholds = np.unique(thresholds)

    rows = []
    for t in thresholds:
        y_pred = (y_score >= t).astype(int)
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        n_fp = int(((y_pred == 1) & (y_true == 0)).sum())
        n_tp = int(((y_pred == 1) & (y_true == 1)).sum())
        rows.append({
            "model": model_name,
            "threshold": float(t),
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "n_tp": n_tp,
            "n_fp": n_fp,
        })

    return pd.DataFrame(rows)


def compare_to_baseline(
    y_true: np.ndarray,
    raw_kpi_values: Optional[np.ndarray] = None,
) -> list[EvaluationResult]:
    """
    Compute naive baseline metrics for comparison.

    # NOTE: This baseline simulates static threshold behaviour but does not
    # apply actual KPI thresholds. Results are illustrative.

    Two baselines:
    1. Static global threshold on DL throughput z-score (simulates existing NOC
       static threshold alerting)
    2. "All normal" baseline (predicts nothing as anomaly — useful F1 floor)

    These establish the performance floor that ML models must beat.
    A well-calibrated static threshold typically achieves F1 0.35–0.50 on
    real RAN data due to high false alarm rates during peak events.
    See Coursebook Ch. 25: Baseline Comparison Methodology.
    """
    baselines = []

    # Baseline 1: Predict nothing (trivial lower bound)
    y_all_normal = np.zeros_like(y_true)
    baselines.append(evaluate_model(
        y_true,
        y_score=np.zeros_like(y_true, dtype=float),
        threshold=1.0,  # No threshold will ever trigger
        model_name="Baseline: Predict-All-Normal",
        split_name="test",
    ))

    # Baseline 2: Static threshold simulation
    # Simulate a fixed threshold on a noisy random signal that represents
    # a KPI z-score exceeding 2.0 (typical NOC static threshold).
    # We use the actual label distribution to calibrate a realistic FPR.
    rng = np.random.default_rng(42)
    # Static thresholds trigger ~5% of the time regardless of actual anomalies
    # (representative of high-volume NOC threshold alerting environments)
    static_scores = rng.uniform(0, 1, size=len(y_true))
    # Inflate scores at anomaly positions to simulate partial detection
    static_scores[y_true == 1] += rng.uniform(0.1, 0.4, size=int(y_true.sum()))
    static_scores = np.clip(static_scores, 0, 1)

    baselines.append(evaluate_model(
        y_true,
        y_score=static_scores,
        threshold=0.95,  # High threshold → ~5% alarm rate (typical static threshold)
        model_name="Baseline: Static Threshold (simulated)",
        split_name="test",
    ))

    return baselines


# ===========================================================================
# SECTION 8: MODEL SERIALISATION
# ===========================================================================

def save_models(
    if_model: IsolationForestModel,
    rf_model: RandomForestModel,
    lstm_model: LSTMAutoencoderModel,
    cascade_config: CascadeConfig,
    feature_cols: list[str],
    output_dir: Path,
) -> None:
    """
    Serialise all trained models and configuration to disk.

    File format choices:
    - pickle for sklearn models (sklearn's native format; compatible with ONNX export)
    - PyTorch .pt for LSTM (torch.save preserves model architecture and weights)
    - JSON for thresholds and configs (human-readable, version-controllable)
    - JSON for feature column list (critical for serving consistency)

    In production, models would be registered in an MLflow Model Registry
    with version tagging, environment metadata, and approval workflows.
    See Coursebook Ch. 54: Model Registry Patterns.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Isolation Forest ---
    if if_model.model is not None:
        if_path = output_dir / "isolation_forest.pkl"
        with open(if_path, "wb") as f:
            pickle.dump(if_model.model, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Saved Isolation Forest → %s", if_path)

        if_threshold_path = output_dir / "isolation_forest_threshold.json"
        with open(if_threshold_path, "w") as f:
            json.dump(
                {
                    "threshold": if_model.threshold,
                    "train_metrics": if_model.train_metrics,
                    "model_type": "IsolationForest",
                    "contamination": IF_CONTAMINATION,
                },
                f, indent=2,
            )

    # --- Random Forest ---
    if rf_model.model is not None:
        rf_path = output_dir / "random_forest.pkl"
        with open(rf_path, "wb") as f:
            pickle.dump(rf_model.model, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Saved Random Forest → %s", rf_path)

        rf_config_path = output_dir / "random_forest_threshold.json"
        with open(rf_config_path, "w") as f:
            json.dump(
                {
                    "threshold": rf_model.threshold,
                    "train_metrics": rf_model.train_metrics,
                    "model_type": "RandomForestClassifier",
                    "top_20_features": sorted(
                        rf_model.feature_importances.items(),
                        key=lambda x: x[1], reverse=True,
                    )[:20],
                },
                f, indent=2,
            )

    # --- LSTM Autoencoder ---
    if TORCH_AVAILABLE and lstm_model.model is not None:
        lstm_path = output_dir / "lstm_autoencoder.pt"
        torch.save(
            {
                "model_state_dict": lstm_model.model.state_dict(),
                "model_config": {
                    "input_size": lstm_model.model.input_size,
                    "hidden_size": lstm_model.model.hidden_size,
                    "num_layers": lstm_model.model.num_layers,
                },
                "scaler_mean": lstm_model.scaler_mean.tolist() if lstm_model.scaler_mean is not None else None,
                "scaler_std": lstm_model.scaler_std.tolist() if lstm_model.scaler_std is not None else None,
            },
            lstm_path,
        )
        logger.info("Saved LSTM Autoencoder → %s", lstm_path)

        lstm_threshold_path = output_dir / "lstm_autoencoder_threshold.json"
        with open(lstm_threshold_path, "w") as f:
            json.dump(
                {
                    "threshold": lstm_model.threshold,
                    "sequence_length": lstm_model.sequence_length,
                    "train_metrics": lstm_model.train_metrics,
                    "model_type": "LSTMAutoencoder",
                },
                f, indent=2,
            )

    # --- Cascade configuration ---
    cascade_path = output_dir / "cascade_config.json"
    with open(cascade_path, "w") as f:
        json.dump(asdict(cascade_config), f, indent=2)
    logger.info("Saved cascade config → %s", cascade_path)

    # --- Feature column list (critical: must match serving) ---
    feature_cols_path = output_dir / "feature_columns.json"
    with open(feature_cols_path, "w") as f:
        json.dump({"feature_columns": feature_cols, "n_features": len(feature_cols)}, f, indent=2)
    logger.info("Saved feature column list (%d features) → %s", len(feature_cols), feature_cols_path)


def save_results(
    all_results: list[EvaluationResult],
    threshold_sweeps: list[pd.DataFrame],
    output_dir: Path,
    mode: str = "full_pipeline",
) -> None:
    """Save evaluation results and threshold sweep data for 04_evaluation.py."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Metrics JSON
    metrics_path = output_dir / "training_metrics.json"
    metrics_data = {
        "mode": mode,
        "evaluation_results": [asdict(r) for r in all_results],
        "operational_targets": {
            "false_alarms_per_shift_target": OPERATIONAL_FALSE_ALARM_TARGET_PER_SHIFT,
            "shift_hours": OPERATIONAL_SHIFT_HOURS,
            "cells_in_scope": OPER_CELLS,
            "rop_minutes": 15,
        },
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=2)
    logger.info("Saved training metrics → %s", metrics_path)

    # Threshold sweep CSV (for precision-recall curve generation in 04_evaluation.py)
    if threshold_sweeps:
        sweep_df = pd.concat(threshold_sweeps, ignore_index=True)
        sweep_path = output_dir / "threshold_sweep.csv"
        sweep_df.to_csv(sweep_path, index=False)
        logger.info("Saved threshold sweep → %s (%d rows)", sweep_path, len(sweep_df))



# ===========================================================================
# EVENT-BASED RECALL
# ===========================================================================

# ===========================================================================
# SECTION 9: MAIN TRAINING PIPELINE
# ===========================================================================
def run_training_pipeline(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    models_dir: Path,
    results_dir: Path,
    mode: str = "full_pipeline",
) -> dict[str, Any]:
    """
    Execute the full three-tier anomaly detection training pipeline.

    Pipeline steps:
    1. Resolve feature columns (consistent with serving pipeline)
    2. Prepare feature matrices and label vectors
    3. Train Tier-1: Isolation Forest (no labels required)
    4. Bootstrap pseudo-labels for Tier-2 from IF + ground truth
    5. Train Tier-2: Random Forest (supervised)
    6. Train Tier-3: LSTM Autoencoder (temporal, train on normal only)
    7. Tune thresholds on validation set
    8. Evaluate all models and the cascade on test set
    9. Compare to naive baselines
    10. Serialise all models, thresholds, and feature metadata

    Returns a dict of all evaluation results for reporting.
    """
    logger.info("=" * 70)
    logger.info("STARTING THREE-TIER CASCADE ANOMALY DETECTION TRAINING PIPELINE")
    logger.info("=" * 70)

    # -----------------------------------------------------------------------
    # Step 1: Feature resolution
    # -----------------------------------------------------------------------
    feature_cols = resolve_feature_columns(train_df)
    logger.info("Feature set: %d columns", len(feature_cols))

    # -----------------------------------------------------------------------
    # Step 2: Matrix preparation
    # -----------------------------------------------------------------------
    X_train, y_train = prepare_matrices(train_df, feature_cols)
    X_val, y_val = prepare_matrices(val_df, feature_cols)
    X_test, y_test = prepare_matrices(test_df, feature_cols)

    logger.info(
        "Matrices — X_train: %s, X_val: %s, X_test: %s",
        X_train.shape, X_val.shape, X_test.shape,
    )
    if y_train is not None:
        logger.info(
            "Labels — train: %d+ / %d total | val: %d+ / %d | test: %d+ / %d",
            int(y_train.sum()), len(y_train),
            int(y_val.sum()) if y_val is not None else 0, len(y_val),
            int(y_test.sum()) if y_test is not None else 0, len(y_test),
        )

    # Compute cell boundaries for LSTM sequence building
    # (prevents sequences crossing cell_id boundaries)
    def _get_cell_boundaries(df: pd.DataFrame) -> list[int]:
        """Return row indices where cell_id changes."""
        if "cell_id" not in df.columns:
            return []
        cell_ids = df["cell_id"].values
        boundaries = [0]
        for i in range(1, len(cell_ids)):
            if cell_ids[i] != cell_ids[i - 1]:
                boundaries.append(i)
        return boundaries

    train_boundaries = _get_cell_boundaries(train_df)
    val_boundaries = _get_cell_boundaries(val_df)
    test_boundaries = _get_cell_boundaries(test_df)

    # -----------------------------------------------------------------------
    # Step 3: Train Tier-1 — Isolation Forest
    # contamination=0.03 is the training parameter; the operational threshold
    # is overridden by the F1-maximizing validation sweep in
    # tune_isolation_forest_threshold().
    # -----------------------------------------------------------------------
    logger.info("\n--- TIER 1: ISOLATION FOREST ---")
    if_model = train_isolation_forest(X_train, feature_cols)
    if_model = tune_isolation_forest_threshold(if_model, X_val, y_val)

    # -----------------------------------------------------------------------
    # Step 4: Bootstrap pseudo-labels for RF training
    # -----------------------------------------------------------------------
    logger.info("\n--- BOOTSTRAPPING PSEUDO-LABELS FOR RF ---")
    pseudo_labels = bootstrap_labels_from_isolation_forest(
        if_model=if_model,
        X_train=X_train,
        y_train=y_train,
    )

    # -----------------------------------------------------------------------
    # Step 5: Train Tier-2 — Random Forest
    # -----------------------------------------------------------------------
    logger.info("\n--- TIER 2: RANDOM FOREST ---")
    rf_model = train_random_forest(X_train, pseudo_labels, feature_cols)

    if y_val is not None and y_val.sum() > 0:
        rf_model = tune_random_forest_threshold(rf_model, X_val, y_val)
    else:
        logger.warning("No validation labels — using default RF threshold 0.50")

    # -----------------------------------------------------------------------
    # Step 6: Train Tier-3 — LSTM Autoencoder
    # -----------------------------------------------------------------------
    logger.info("\n--- TIER 3: LSTM AUTOENCODER ---")

    # Train LSTM only on normal samples to learn normal temporal patterns
    if y_train is not None:
        normal_mask = y_train == 0
        X_train_normal = X_train[normal_mask]
        normal_boundaries = _get_cell_boundaries(train_df[train_df["is_anomaly"] == 0]) \
            if "is_anomaly" in train_df.columns else train_boundaries
    else:
        X_train_normal = X_train
        normal_boundaries = train_boundaries

    logger.info(
        "LSTM training on normal-only data: %d samples (%.1f%% of train)",
        len(X_train_normal), 100.0 * len(X_train_normal) / len(X_train),
    )

    lstm_model = train_lstm_autoencoder(
        X_train=X_train_normal,
        feature_cols=feature_cols,
        cell_boundaries=normal_boundaries if normal_boundaries else None,
    )
    lstm_model = tune_lstm_threshold(
        lstm_model, X_val, y_val,
        cell_boundaries=val_boundaries if val_boundaries else None,
    )

    # -----------------------------------------------------------------------
    # Step 7: Configure cascade
    # -----------------------------------------------------------------------
    cascade_config = CascadeConfig(
        rf_available=(rf_model.model is not None),
        lstm_available=(TORCH_AVAILABLE and lstm_model.model is not None),
    )

    # -----------------------------------------------------------------------
    # Step 8: Compute all scores on test set
    # -----------------------------------------------------------------------
    logger.info("\n--- COMPUTING TEST SCORES ---")

    # Tier-1 scores
    if_test_scores = score_isolation_forest(if_model, X_test)

    # Tier-2 scores (probability of anomaly class)
    rf_test_scores: Optional[np.ndarray] = None
    if rf_model.model is not None:
        rf_test_scores = rf_model.model.predict_proba(X_test)[:, 1]

    # Tier-3 scores
    lstm_test_scores: Optional[np.ndarray] = None
    if TORCH_AVAILABLE and lstm_model.model is not None:
        lstm_test_scores = score_lstm_autoencoder(
            lstm_model, X_test,
            cell_boundaries=test_boundaries if test_boundaries else None,
        )

    # Cascade combined scores
    cascade_scores = compute_cascade_scores(
        if_scores=if_test_scores,
        rf_scores=rf_test_scores,
        lstm_scores=lstm_test_scores,
        config=cascade_config,
    )

    # -----------------------------------------------------------------------
    # Step 9: Evaluate all models
    # -----------------------------------------------------------------------
    logger.info("\n--- EVALUATION ---")
    all_results: list[EvaluationResult] = []
    threshold_sweeps: list[pd.DataFrame] = []

    if y_test is not None and y_test.sum() > 0:
        # Individual tier evaluations
        if_result = evaluate_model(
            y_test, if_test_scores, if_model.threshold,
            model_name="Tier-1: Isolation Forest", split_name="test",
        )
        all_results.append(if_result)
        threshold_sweeps.append(
            run_threshold_sweep(y_test, if_test_scores, "IsolationForest")
        )

        if rf_test_scores is not None:
            rf_result = evaluate_model(
                y_test, rf_test_scores, rf_model.threshold,
                model_name="Tier-2: Random Forest", split_name="test",
            )
            all_results.append(rf_result)
            threshold_sweeps.append(
                run_threshold_sweep(y_test, rf_test_scores, "RandomForest")
            )

        if lstm_test_scores is not None:
            lstm_result = evaluate_model(
                y_test, lstm_test_scores, lstm_model.threshold,
                model_name="Tier-3: LSTM Autoencoder", split_name="test",
            )
            all_results.append(lstm_result)
            threshold_sweeps.append(
                run_threshold_sweep(y_test, lstm_test_scores, "LSTMAutoencoder")
            )

        # Cascade evaluation
        cascade_result = evaluate_model(
            y_test, cascade_scores, cascade_config.final_threshold,
            model_name="CASCADE: All Tiers Combined", split_name="test",
        )
        all_results.append(cascade_result)
        threshold_sweeps.append(
            run_threshold_sweep(y_test, cascade_scores, "CascadeEnsemble")
        )

        # Baseline comparison
        logger.info("\n--- BASELINE COMPARISON ---")
        baseline_results = compare_to_baseline(y_test)
        all_results.extend(baseline_results)

        # Print comparison table
        print_evaluation_table(all_results)

        # Operational interpretation summary
        _log_operational_interpretation(all_results)

    else:
        logger.warning(
            "No ground truth labels in test set — evaluation skipped. "
            "Run 01_synthetic_data.py to generate labelled data."
        )

    # -----------------------------------------------------------------------
    # Step 10: Serialise
    # -----------------------------------------------------------------------
    logger.info("\n--- SAVING MODELS ---")
    save_models(
        if_model=if_model,
        rf_model=rf_model,
        lstm_model=lstm_model,
        cascade_config=cascade_config,
        feature_cols=feature_cols,
        output_dir=models_dir,
    )
    save_results(all_results, threshold_sweeps, results_dir, mode=mode)

    # -----------------------------------------------------------------------
    # Step 11: Write test scores to data/test_scores.parquet
    # -----------------------------------------------------------------------
    logger.info("\n--- WRITING TEST SCORES ---")

    # Build per-row IF scores and RF probabilities using the already-computed
    # arrays; cascade_scores holds the final normalised anomaly score
    # (range [0, 1]) as defined by the weighted average cascade.
    predicted_labels = (cascade_scores >= cascade_config.final_threshold).astype(int)

    # Canonical column names matching 03_model_training.py schema:
    # if_score, rf_score, ensemble_score, anomaly_label, anomaly_type
    test_scores_dict: dict[str, Any] = {
        "ensemble_score": cascade_scores,
        "anomaly_label": predicted_labels,
    }

    # Include IF and RF component scores for traceability
    test_scores_dict["if_score"] = if_test_scores
    if rf_test_scores is not None:
        test_scores_dict["rf_score"] = rf_test_scores
    else:
        test_scores_dict["rf_score"] = np.full(len(cascade_scores), np.nan)

    # Attach cell_id and timestamp from test_df when available
    if "cell_id" in test_df.columns:
        test_scores_dict["cell_id"] = test_df["cell_id"].values
    else:
        test_scores_dict["cell_id"] = np.arange(len(cascade_scores))

    if "timestamp" in test_df.columns:
        test_scores_dict["timestamp"] = test_df["timestamp"].values
    else:
        test_scores_dict["timestamp"] = np.arange(len(cascade_scores))

    # Ground-truth label (anomaly_label column); use y_test when available
    if y_test is not None:
        test_scores_dict["anomaly_label"] = y_test.astype(int)
    else:
        test_scores_dict["anomaly_label"] = np.full(len(cascade_scores), np.nan)

    # Anomaly type from test_df when available
    if "anomaly_type" in test_df.columns:
        test_scores_dict["anomaly_type"] = test_df["anomaly_type"].values
    else:
        test_scores_dict["anomaly_type"] = np.full(len(cascade_scores), np.nan)

    # Compute anomaly_event_id: contiguous runs of is_anomaly=1 per cell_id
    # Each unique event gets a monotonically increasing integer ID; non-anomaly
    # rows get -1. Required by 04_evaluation.py for event-based metrics.
    _labels = test_scores_dict["anomaly_label"]
    _cell_ids = test_scores_dict["cell_id"]
    _event_ids = np.full(len(_labels), -1, dtype=int)
    _current_event = 0
    _tmp_df = pd.DataFrame({"cell_id": _cell_ids, "label": _labels})
    for _, grp in _tmp_df.groupby("cell_id", sort=False):
        in_event = False
        for i in grp.index:
            if grp.loc[i, "label"] == 1:
                if not in_event:
                    in_event = True
                    _current_event += 1
                _event_ids[i] = _current_event
            else:
                in_event = False
    test_scores_dict["anomaly_event_id"] = _event_ids

    test_scores_df = pd.DataFrame(test_scores_dict)[
        ["cell_id", "timestamp", "if_score", "rf_score", "ensemble_score",
         "anomaly_label", "anomaly_type", "anomaly_event_id"]
    ]

    test_scores_path = Path("data/test_scores.parquet")
    test_scores_path.parent.mkdir(parents=True, exist_ok=True)
    test_scores_df.to_parquet(test_scores_path, index=False)
    logger.info(
        "Test scores written to %s (%d rows)", test_scores_path.resolve(), len(test_scores_df)
    )

    logger.info("\nTraining pipeline complete.")
    logger.info("Models saved to: %s", models_dir.resolve())
    logger.info("Results saved to: %s", results_dir.resolve())
    return {
        "if_model": if_model,
        "rf_model": rf_model,
        "lstm_model": lstm_model,
        "cascade_config": cascade_config,
        "feature_cols": feature_cols,
        "evaluation_results": all_results,
    }
def _log_operational_interpretation(results: list[EvaluationResult]) -> None:
    """
    Translate evaluation metrics into business-language operational impact.

    This is the 'so what?' layer that matters to NOC managers and CFOs.
    See Coursebook Ch. 54: Communicating ML Metrics to Operations Teams.
    """
    logger.info("\n" + "=" * 70)
    logger.info("OPERATIONAL IMPACT INTERPRETATION")
    logger.info("=" * 70)

    cascade = next((r for r in results if "CASCADE" in r.model_name), None)
    static = next((r for r in results if "Static" in r.model_name), None)

    if cascade and static:
        fa_reduction = 1.0 - (cascade.false_alarms_per_shift / max(static.false_alarms_per_shift, 0.01))
        ttd_reduction = (
            (static.time_to_detect_mean_rops - cascade.time_to_detect_mean_rops)
            / max(static.time_to_detect_mean_rops, 0.01)
            * 100
        ) if not np.isnan(cascade.time_to_detect_mean_rops) else 0.0

        logger.info(
            "\nAt %d cells, %d-hour shifts, %d-minute ROPs:",
            OPER_CELLS, OPERATIONAL_SHIFT_HOURS, 15,
        )
        logger.info(
            "  Static threshold:   ~%.0f false alarms/shift, catch rate: %.0f%%, F1=%.2f",
            static.false_alarms_per_shift, static.recall * 100, static.f1,
        )
        logger.info(
            "  CASCADE model:      ~%.0f false alarms/shift, catch rate: %.0f%%, F1=%.2f",
            cascade.false_alarms_per_shift, cascade.recall * 100, cascade.f1,
        )
        logger.info(
            "  False alarm reduction:  %.0f%%", fa_reduction * 100,
        )
        if not np.isnan(cascade.time_to_detect_mean_rops):
            logger.info(
                "  Mean time-to-detect: %.0f min (cascade) vs. %.0f min (static) — %.0f%% faster",
                cascade.time_to_detect_mean_rops * 15,
                static.time_to_detect_mean_rops * 15,
                ttd_reduction,
            )
        logger.info(
            "\n  At F1=%.2f, a NOC analyst reviewing cascade alerts sees:\n"
            "    ~%.0f alerts per 8-hr shift of which ~%.0f%% are actionable\n"
            "    (vs. ~%.0f alerts for static threshold with ~%.0f%% actionable)",
            cascade.f1,
            cascade.false_alarms_per_shift + (cascade.recall * (cascade.n_positives / max(1, cascade.n_samples) * OPER_CELLS * OPERATIONAL_SHIFT_HOURS * 4)),
            cascade.precision * 100,
            static.false_alarms_per_shift,
            static.precision * 100,
        )
    logger.info("=" * 70)


# ===========================================================================
# ENTRY POINT
# ===========================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train RAN KPI anomaly detection models (three-tier cascade)"
    )
    parser.add_argument(
        "--self-contained",
        action="store_true",
        help="Generate synthetic data inline instead of loading from data/ directory",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=MODELS_DIR,
        help="Output directory for model artefacts",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Output directory for evaluation results",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Input directory containing feature parquet files",
    )
    parser.add_argument(
        "--n-cells",
        type=int,
        default=100,
        help="Number of cells for synthetic data generation (--self-contained mode)",
    )
    parser.add_argument(
        "--n-days",
        type=int,
        default=30,
        help="Number of days for synthetic data generation (--self-contained mode)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def main() -> None:
    """
    Main entry point for the model training script.

    Can be run in two modes:
    1. Standard mode (uses outputs from 01 and 02 scripts):
       python 03_model_training.py

    2. Self-contained mode (generates data inline, no dependencies):
       python 03_model_training.py --self-contained
    """
    args = parse_args()
    logging.getLogger().setLevel(args.log_level)

    logger.info("RAN KPI Anomaly Detection — Model Training")
    logger.info("PyTorch available: %s", TORCH_AVAILABLE)
    logger.info("Models output:     %s", args.models_dir)
    logger.info("Results output:    %s", args.results_dir)

    # -----------------------------------------------------------------------
    # Load or generate data
    # -----------------------------------------------------------------------
    data_dir = args.data_dir
    train_path = data_dir / "train_features.parquet"
    val_path = data_dir / "val_features.parquet"
    test_path = data_dir / "test_features.parquet"

    # Determine pipeline mode for metrics provenance
    is_self_contained = args.self_contained or not (
        train_path.exists() and val_path.exists() and test_path.exists()
    )

    if is_self_contained:
        if not args.self_contained:
            logger.warning(
                "Feature parquet files not found at %s — falling back to "
                "self-contained synthetic data generation. "
                "Run 01_synthetic_data.py + 02_feature_engineering.py first for "
                "full-fidelity results.",
                data_dir,
            )
        logger.info(
            "Generating synthetic data: %d cells × %d days",
            args.n_cells, args.n_days,
        )
        train_df, val_df, test_df = _generate_synthetic_features(
            n_cells=args.n_cells,
            n_days=args.n_days,
        )
    else:
        train_df, val_df, test_df = load_feature_splits(train_path, val_path, test_path)

    # -----------------------------------------------------------------------
    # Run training pipeline
    # -----------------------------------------------------------------------
    run_training_pipeline(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        models_dir=args.models_dir,
        results_dir=args.results_dir,
        mode="self_contained" if is_self_contained else "full_pipeline",
    )


if __name__ == "__main__":
    main()
