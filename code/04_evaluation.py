"""
04_evaluation.py — RAN KPI Anomaly Detection: Evaluation & Visualisation
=========================================================================
Companion code for:
  "Real-Time RAN KPI Anomaly Detection at Cell Sector Granularity"

PURPOSE
-------
Load the test-split scores produced by 03_model_training.py and perform a
thorough, operationally grounded evaluation:

  1. Per-model precision / recall / F1 at multiple thresholds
  2. Precision-Recall curves with annotated operating points
  3. ROC-AUC curves
  4. Confusion matrices (absolute + normalised)
  5. Time-to-detect (TTD) analysis — how many ROPs elapse before the first
     true positive fires after anomaly onset
  6. Per-cell-sector false-alarm-rate analysis
  7. Anomaly-type breakdown (sudden drop, gradual degradation, periodic
     interference, spatial correlation)
  8. Bootstrap confidence intervals on key metrics
  9. Operational interpretation table ("what does this score mean for a NOC
     analyst reviewing 10 000 cells per shift?")
  10. All figures saved as PNG; all scalar metrics saved as JSON

USAGE
-----
    python 04_evaluation.py [--data-dir DATA_DIR] [--output-dir OUTPUT_DIR]

    Defaults:
        --data-dir   ./data
        --output-dir ./evaluation_outputs

DEPENDENCIES
------------
    pip install pandas numpy scikit-learn matplotlib seaborn scipy

    Requires utils.py (shared constants and helpers) in the same directory
    or on PYTHONPATH.

PRIOR SCRIPTS
-------------
    01_synthetic_data.py  → data/pm_dataset.parquet + data/cell_metadata.parquet
                             data/anomaly_labels.parquet
    02_feature_engineering.py → data/test_features.parquet
                                data/feature_metadata.json
    03_model_training.py  → data/test_scores.parquet
                            data/model_artifacts/threshold_metrics.json
                            data/model_artifacts/*.pkl (model objects)
                            models/cascade_config.json

If none of the prior outputs are present the script regenerates them inline
from the same random seed so results are reproducible.

Coursebook cross-reference:
    Ch. 3  — Time-Series Fundamentals (temporal split rationale)
    Ch. 7  — Feature Engineering for Network PM Data
    Ch. 15 — MLOps: Evaluation Methodology & Production Metrics
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")          

# non-interactive backend — safe for headless servers
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from utils import compute_event_based_recall

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("04_evaluation")

# ---------------------------------------------------------------------------
# Constants — adjust to match your deployment targets
# ---------------------------------------------------------------------------
SEED = 42
_USING_FALLBACK_DATA = False  # set True when prior script outputs are absent
ROP_MINUTES = 15          # 15-minute reporting output period
ROPS_PER_SHIFT = 8 * 4    # NOC shift = 8 hours = 32 ROPs
CELLS_PER_NOC_DOMAIN = 10_000   # operator scale: 10 K cell sectors
SHIFTS_PER_DAY = 3

# Colour palette consistent across all figures
PALETTE = {
    "isolation_forest": "#2196F3",   # blue
    "random_forest":    "#4CAF50",   # green
    "ensemble":         "#FF9800",   # orange
    "baseline_static":  "#9E9E9E",   # grey
    "anomaly":          "#F44336",   # red
    "normal":           "#66BB6A",   # light green
}

# Operating-point annotations for the PR curve
# Each tuple: (model_key, threshold, label_for_chart)
ANNOTATED_OPERATING_POINTS = [
    ("isolation_forest", 0.30, "Conservative"),
    ("isolation_forest", 0.50, "Balanced"),
    ("random_forest",    0.40, "RF Default"),
    ("random_forest",    0.60, "RF Tight"),
    ("ensemble",         0.45, "Ensemble Rec."),
]

# Default thresholds — overridden at runtime by cascade_config.json when available.
# Referenced by charting functions for annotation lines.
ACTIVE_THRESHOLDS: dict[str, float] = {
    "if": 0.50,
    "rf": 0.40,
    "ensemble": 0.45,
}


# ============================================================================
# Section 1 — Data Loading Helpers
# ============================================================================


# ============================================================================
# Event-Based Recall
# ============================================================================


def _load_parquet_or_none(path: Path) -> Optional[pd.DataFrame]:
    """Return a DataFrame from *path* if it exists, else None."""
    if path.exists():
        df = pd.read_parquet(path)
        logger.info("Loaded %s  shape=%s", path.name, df.shape)
        return df
    return None


def load_evaluation_data(data_dir: Path) -> dict[str, pd.DataFrame | dict]:
    """
    Load test scores, features, and labels from the outputs of scripts 01-03.

    Falls back to inline regeneration if any file is missing so the script
    remains self-contained.

    Returns
    -------
    dict with keys:
        "scores"   : DataFrame with columns [cell_id, timestamp,
                       anomaly_label, anomaly_type, anomaly_event_id,
                       if_score, if_pred, rf_score, rf_pred,
                       ensemble_score, ensemble_pred, ...]
        "topology" : DataFrame with cell topology info
        "metadata" : dict (feature metadata from script 02)
    """
    scores_path   = data_dir / "test_scores.parquet"
    features_path = data_dir / "test_features.parquet"
    labels_path   = data_dir / "anomaly_labels.parquet"   # upstream: 01_synthetic_data.py
    topology_path = data_dir / "cell_metadata.parquet"     # upstream: 01_synthetic_data.py
    meta_path     = data_dir / "model_artifacts" / "threshold_metrics.json"

    scores_df   = _load_parquet_or_none(scores_path)
    features_df = _load_parquet_or_none(features_path)
    labels_df   = _load_parquet_or_none(labels_path)
    topology_df = _load_parquet_or_none(topology_path)

    metadata: dict = {}
    if meta_path.exists():
        with open(meta_path) as fh:
            metadata = json.load(fh)
        logger.info("Loaded threshold_metrics.json")

    # ------------------------------------------------------------------
    # Fallback: regenerate synthetic evaluation data when prior scripts
    # have not been run.  This keeps script 04 self-contained.
    # NOTE: labels_df is NOT required for the fallback gate — labels may
    # already be embedded in test_scores.parquet (the standard path).
    # ------------------------------------------------------------------
    if scores_df is None or features_df is None:
        logger.warning(
            "Prior script outputs not found — regenerating synthetic "
            "evaluation data inline (see _generate_fallback_scores)"
        )
        scores_df, topology_df = _generate_fallback_scores()
        global _USING_FALLBACK_DATA
        _USING_FALLBACK_DATA = True
    else:
        _USING_FALLBACK_DATA = False
        # Normalise label column name: upstream uses "is_anomaly",
        # evaluation expects "anomaly_label".
        if "is_anomaly" in scores_df.columns and "anomaly_label" not in scores_df.columns:
            scores_df["anomaly_label"] = scores_df["is_anomaly"].astype(int)
            logger.info("Renamed is_anomaly → anomaly_label in scores DataFrame")
        # Merge label columns into scores if they were stored separately.
        # In the standard 01→02→03→04 pipeline, 03_model_training.py
        # writes anomaly_label directly into test_scores.parquet, so this
        # merge path is a safety net for alternative label workflows.
        if "anomaly_label" not in scores_df.columns and labels_df is not None:
            merge_cols = ["cell_id", "timestamp"]
            label_cols = [c for c in labels_df.columns if c not in merge_cols]
            scores_df = scores_df.merge(
                labels_df[merge_cols + label_cols], on=merge_cols, how="left"
            )
            # Normalise after merge
            if "is_anomaly" in scores_df.columns and "anomaly_label" not in scores_df.columns:
                scores_df["anomaly_label"] = scores_df["is_anomaly"].astype(int)

    # Ensure timestamp is datetime
    if "timestamp" in scores_df.columns:
        scores_df["timestamp"] = pd.to_datetime(scores_df["timestamp"])

    # ------------------------------------------------------------------
    # Load trained thresholds from cascade_config.json if available.
    # Falls back to sensible defaults when model artifacts are absent
    # (e.g. self-contained / first-run evaluation).
    # ------------------------------------------------------------------
    cascade_cfg_path = data_dir / "models" / "cascade_config.json"
    if not cascade_cfg_path.exists():
        cascade_cfg_path = Path("models") / "cascade_config.json"
    if cascade_cfg_path.exists():
        with open(cascade_cfg_path) as fh:
            cascade_cfg = json.load(fh)
        if_threshold  = cascade_cfg.get("isolation_forest_threshold", 0.50)
        rf_threshold  = cascade_cfg.get("random_forest_threshold", 0.40)
        ens_threshold = cascade_cfg.get("cascade_threshold", 0.45)
        logger.info(
            "Loaded thresholds from %s: IF=%.2f, RF=%.2f, ensemble=%.2f",
            cascade_cfg_path, if_threshold, rf_threshold, ens_threshold,
        )
    else:
        if_threshold, rf_threshold, ens_threshold = 0.50, 0.40, 0.45
        logger.warning(
            "cascade_config.json not found — using default thresholds: "
            "IF=%.2f, RF=%.2f, ensemble=%.2f",
            if_threshold, rf_threshold, ens_threshold,
        )

    # Update module-level dict so charting functions use trained thresholds
    ACTIVE_THRESHOLDS["if"] = if_threshold
    ACTIVE_THRESHOLDS["rf"] = rf_threshold
    ACTIVE_THRESHOLDS["ensemble"] = ens_threshold

    # ------------------------------------------------------------------
    # Derive prediction columns from scores if absent.
    # 03_model_training.py writes score columns (if_score, rf_score,
    # ensemble_score) but may not write pred columns. Apply trained
    # thresholds to produce binary predictions for evaluation.
    # ------------------------------------------------------------------
    for model_name, score_col, threshold in [
        ("if", "if_score", if_threshold),
        ("rf", "rf_score", rf_threshold),
        ("ensemble", "ensemble_score", ens_threshold),
    ]:
        pred_col = f"{model_name}_pred"
        if score_col in scores_df.columns and pred_col not in scores_df.columns:
            scores_df[pred_col] = (scores_df[score_col] >= threshold).astype(int)
            logger.info(
                "Derived %s from %s at threshold %.2f",
                pred_col, score_col, threshold,
            )

    # Fill in static-threshold baseline column if absent
    if "baseline_pred" not in scores_df.columns:
        scores_df = _add_static_threshold_baseline(scores_df)

    return {"scores": scores_df, "topology": topology_df, "metadata": metadata}


def _add_static_threshold_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simulate a na¼ve static-threshold baseline.

    In practice a static threshold is tuned once globally and never adapts to
    cell-specific context.  We model it as: flag any row whose anomaly_score
    (raw isolation forest score before calibration) exceeds a fixed global
    percentile.

    This baseline deliberately over-fires on busy cells and misses gradual
    degradations — consistent with documented operator experience.

    # See Coursebook Ch. 25: Baseline Comparisons for Time-Series
    """
    rng = np.random.default_rng(SEED + 99)
    n = len(df)

    # Static threshold fires on ~12% of rows (typical false-alarm rate for
    # uncalibrated global thresholds at operator scale)
    if "if_score" in df.columns:
        raw = df["if_score"].values
        threshold = np.percentile(raw, 88)          # top-12% flagged
        df["baseline_pred"] = (raw >= threshold).astype(int)
    else:
        # Pure synthetic fallback
        df["baseline_pred"] = (rng.random(n) > 0.88).astype(int)

    return df


# ============================================================================
# Section 2 — Fallback Synthetic Data Generator
# ============================================================================

def _generate_fallback_scores(
    n_cells: int = 80,
    test_days: int = 5,
    anomaly_rate: float = 0.04,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate a realistic synthetic score DataFrame when prior scripts are absent.

    The synthetic data mirrors what scripts 01-03 would produce for the test
    split (days 26-30 in the 30-day dataset).

    Returns
    -------
    (scores_df, topology_df)
    """
    logger.info(
        "Generating fallback synthetic scores: n_cells=%d, test_days=%d",
        n_cells, test_days,
    )
    rng = np.random.default_rng(SEED)

    # --- Topology ---
    cell_ids = [
        f"CELL_{site:03d}_{sector}"
        for site in range(1, n_cells // 3 + 1)
        for sector in ["A", "B", "C"]
    ][:n_cells]

    cell_types = rng.choice(
        ["urban_macro", "urban_micro", "suburban", "rural"],
        size=n_cells,
        p=[0.40, 0.25, 0.25, 0.10],
    )
    vendors = rng.choice(["Ericsson", "Nokia", "Samsung"], size=n_cells, p=[0.45, 0.40, 0.15])
    topology_df = pd.DataFrame(
        {"cell_id": cell_ids, "cell_type": cell_types, "vendor": vendors}
    )

    # --- Timestamps (15-min ROPs for test_days) ---
    rops_per_day = 24 * 4    # 96 ROPs / day
    total_rops   = rops_per_day * test_days
    start_ts     = pd.Timestamp("2024-01-26 00:00:00")
    timestamps   = pd.date_range(start_ts, periods=total_rops, freq="15min")

    records = []
    for cell_id in cell_ids:
        for ts in timestamps:
            records.append({"cell_id": cell_id, "timestamp": ts})

    df = pd.DataFrame(records)
    n = len(df)
    logger.info("Fallback frame: %d rows  (%d cells × %d ROPs)", n, n_cells, total_rops)

    # --- Diurnal traffic pattern (affects KPI base level) ---
    hour_vec = df["timestamp"].dt.hour.values
    diurnal  = 0.5 + 0.5 * np.sin((hour_vec - 6) / 24 * 2 * np.pi)

    # --- Ground-truth anomaly labels ---
    anomaly_types = [
        "sudden_drop", "gradual_degradation",
        "periodic_interference", "spatial_correlation",
    ]
    df["anomaly_label"]    = 0
    df["anomaly_type"]     = "none"
    df["anomaly_event_id"] = -1

    # Inject anomaly events — each event spans 4-12 ROPs in a random cell
    n_events = int(n * anomaly_rate / 8)   # approximate event count
    event_id = 0
    for _ in range(n_events):
        c_idx    = rng.integers(0, n_cells)
        cell_hit = cell_ids[c_idx]
        t_start  = rng.integers(0, total_rops - 12)
        duration = rng.integers(4, 13)
        a_type   = rng.choice(anomaly_types)

        mask = (
            (df["cell_id"] == cell_hit) &
            (df["timestamp"] >= timestamps[t_start]) &
            (df["timestamp"] <  timestamps[min(t_start + duration, total_rops - 1)])
        )
        df.loc[mask, "anomaly_label"]    = 1
        df.loc[mask, "anomaly_type"]     = a_type
        df.loc[mask, "anomaly_event_id"] = event_id
        event_id += 1

    # --- Simulate model scores ---
    # True-positive rows: boosted score; false-negative: some escape detection
    true_anom = df["anomaly_label"].values == 1

    # Isolation Forest raw anomaly score (higher = more anomalous, 0-1 range)
    if_noise  = rng.normal(0, 0.08, n)
    if_base   = 0.15 + 0.10 * diurnal + if_noise
    if_base   = np.clip(if_base, 0, 1)
    # Boost true anomalies — not perfectly, leaves ~25% FN
    boost_mask = true_anom & (rng.random(n) > 0.25)
    if_base[boost_mask] += rng.uniform(0.25, 0.55, boost_mask.sum())
    if_base = np.clip(if_base, 0, 1)
    df["if_score"] = if_base

    # Random Forest probability
    rf_noise = rng.normal(0, 0.06, n)
    rf_base  = 0.08 + rf_noise
    rf_base  = np.clip(rf_base, 0, 1)
    boost_mask_rf = true_anom & (rng.random(n) > 0.10)   # fewer FN
    rf_base[boost_mask_rf] += rng.uniform(0.35, 0.60, boost_mask_rf.sum())
    rf_base = np.clip(rf_base, 0, 1)
    df["rf_score"] = rf_base

    # Ensemble = weighted average
    df["ensemble_score"] = 0.35 * df["if_score"] + 0.65 * df["rf_score"]

    # Convert scores to predictions at default thresholds.
    # Fallback data has no cascade_config.json — defaults are appropriate here.
    df["if_pred"]       = (df["if_score"]       >= 0.50).astype(int)
    df["rf_pred"]       = (df["rf_score"]        >= 0.40).astype(int)
    df["ensemble_pred"] = (df["ensemble_score"]  >= 0.45).astype(int)

    # Static baseline
    df = _add_static_threshold_baseline(df)

    # --- Anomaly onset time (for TTD analysis) ---
    # First ROP index within each anomaly event
    onset_map: dict[int, pd.Timestamp] = {}
    for eid, grp in df[df["anomaly_event_id"] >= 0].groupby("anomaly_event_id"):
        onset_map[int(eid)] = grp["timestamp"].min()
    df["anomaly_onset"] = df["anomaly_event_id"].map(onset_map)

    return df, topology_df


def _add_fallback_watermark(fig) -> None:
    """Overlay a diagonal watermark when evaluation uses synthetic fallback data."""
    fig.text(
        0.5, 0.5,
        "SYNTHETIC FALLBACK DATA\nRun full pipeline for production-representative results",
        transform=fig.transFigure,
        fontsize=18, color="red", alpha=0.25,
        ha="center", va="center", rotation=30,
        fontweight="bold", zorder=999,
    )


# ============================================================================
# Section 3 — Core Metric Helpers
# ============================================================================

def binary_metrics_at_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """
    Compute precision, recall, F1, FPR at a fixed decision threshold.

    Parameters
    ----------
    y_true    : ground-truth binary labels (0/1)
    scores    : continuous anomaly scores (higher = more anomalous)
    threshold : decision boundary

    Returns
    -------
    dict with keys: precision, recall, f1, fpr, fnr, tn, fp, fn, tp, support
    """
    y_pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec  = float(recall_score(y_true, y_pred, zero_division=0))
    f1   = float(f1_score(y_true, y_pred, zero_division=0))
    fpr  = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr  = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    return dict(
        threshold=threshold,
        precision=prec, recall=rec, f1=f1,
        fpr=fpr, fnr=fnr,
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
        support=int(y_true.sum()),
    )


def sweep_thresholds(
    y_true: np.ndarray,
    scores: np.ndarray,
    n_points: int = 200,
) -> pd.DataFrame:
    """
    Compute binary metrics across a grid of thresholds.

    Returns a DataFrame with one row per threshold; sorted by threshold.
    """
    thresholds = np.linspace(scores.min(), scores.max(), n_points)
    rows = [binary_metrics_at_threshold(y_true, scores, t) for t in thresholds]
    return pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)


def bootstrap_ci(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    metric: str = "f1",
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = SEED,
) -> tuple[float, float, float]:
    """
    Bootstrap confidence interval for a scalar metric at a fixed threshold.

    Parameters
    ----------
    metric : one of {"f1", "precision", "recall", "fpr", "roc_auc", "pr_auc"}

    Returns
    -------
    (point_estimate, lower_bound, upper_bound)
    """
    rng = np.random.default_rng(seed)
    n   = len(y_true)

    def _compute(yt: np.ndarray, sc: np.ndarray) -> float:
        if metric == "roc_auc":
            if len(np.unique(yt)) < 2:
                return float("nan")
            return float(roc_auc_score(yt, sc))
        if metric == "pr_auc":
            return float(average_precision_score(yt, sc))
        m = binary_metrics_at_threshold(yt, sc, threshold)
        return m[metric]

    point = _compute(y_true, scores)
    boot_vals: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        v   = _compute(y_true[idx], scores[idx])
        if not np.isnan(v):
            boot_vals.append(v)

    alpha = 1.0 - ci
    lower = float(np.percentile(boot_vals, 100 * alpha / 2))
    upper = float(np.percentile(boot_vals, 100 * (1 - alpha / 2)))
    return point, lower, upper


def time_to_detect(
    df: pd.DataFrame,
    pred_col: str,
) -> pd.Series:
    """
    For each anomaly event, compute the number of ROPs from anomaly onset to
    the first True Positive prediction by the given model.

    A "missed" event (no TP throughout the event window) gets TTD = NaN.

    Returns
    -------
    pd.Series of TTD values in units of ROPs (one entry per event)
    """
    results: dict[int, float] = {}
    anomalous = df[df["anomaly_event_id"] >= 0].copy()

    for eid, grp in anomalous.groupby("anomaly_event_id"):
        grp_sorted = grp.sort_values("timestamp")
        onset      = grp_sorted["timestamp"].iloc[0]
        tp_rows    = grp_sorted[grp_sorted[pred_col] == 1]
        if tp_rows.empty:
            results[int(eid)] = float("nan")
        else:
            first_det = tp_rows["timestamp"].iloc[0]
            delta     = (first_det - onset).total_seconds() / (ROP_MINUTES * 60)
            results[int(eid)] = max(0.0, delta)   # 0 means detected in first ROP

    return pd.Series(results, name="ttd_rops")


def per_cell_false_alarm_rate(
    df: pd.DataFrame,
    pred_col: str,
    rops_per_day: int = 96,   # 24h × 4 rops/h
) -> pd.Series:
    """
    Compute false alarms per cell per day for the given prediction column.

    A false alarm is a TP=0 row where the model fires (pred=1 but label=0).

    Returns
    -------
    pd.Series indexed by cell_id
    """
    test_days = df["timestamp"].dt.normalize().nunique()
    if test_days == 0:
        test_days = 1

    grp = df.groupby("cell_id").apply(
        lambda g: ((g[pred_col] == 1) & (g["anomaly_label"] == 0)).sum()
    )
    far = grp / test_days
    far.name = "false_alarms_per_day"
    return far


def operational_context(
    precision: float,
    recall: float,
    f1: float,
    model_name: str,
    cells_scale: int = CELLS_PER_NOC_DOMAIN,
    anomaly_rate: float = 0.04,
    rops_per_day: int = 96,
) -> dict[str, Any]:
    """
    Translate statistical metrics into NOC-meaningful quantities.

    Assumes:
    - `cells_scale`  : number of cell sectors in the NOC domain
    - `anomaly_rate` : fraction of ROPs that are truly anomalous
    - `rops_per_day` : ROPs per cell per day

    Returns a dictionary of operational quantities the whitepaper's
    Section 8 uses directly.

    # See Coursebook Ch. 54: Operationalising ML Metrics
    """
    total_rops_day  = cells_scale * rops_per_day
    true_anoms_day  = total_rops_day * anomaly_rate
    true_normals_day = total_rops_day - true_anoms_day

    # Model fires
    tp_day = recall    * true_anoms_day
    fp_day = (1 - precision) / precision * tp_day if precision > 0 else float("inf")

    alerts_day            = tp_day + fp_day
    alerts_per_shift      = alerts_day / SHIFTS_PER_DAY
    false_alarms_per_shift = fp_day / SHIFTS_PER_DAY

    # Detection coverage
    missed_anoms_day  = true_anoms_day - tp_day

    # Static baseline comparison (12% FAR, ~50% recall)
    static_fp_day     = true_normals_day * 0.12
    static_fp_per_shift = static_fp_day / SHIFTS_PER_DAY
    far_reduction_pct = 100 * (1 - fp_day / max(static_fp_day, 1))

    return {
        "model":                    model_name,
        "precision":                round(precision, 3),
        "recall":                   round(recall, 3),
        "f1":                       round(f1, 3),
        "cells_in_domain":          cells_scale,
        "total_rops_per_day":       int(total_rops_day),
        "true_anomalies_per_day":   int(true_anoms_day),
        "tp_per_day":               int(tp_day),
        "fp_per_day":               int(fp_day),
        "missed_per_day":           int(missed_anoms_day),
        "alerts_per_noc_shift":     int(alerts_per_shift),
        "false_alarms_per_shift":   int(false_alarms_per_shift),
        "false_alarms_static_shift":int(static_fp_per_shift),
        "false_alarm_reduction_pct":round(far_reduction_pct, 1),
        "note": (
            f"At this operating point a NOC analyst overseeing {cells_scale:,} cells "
            f"would handle ≈{int(alerts_per_shift):,} alerts/shift "
            f"({int(false_alarms_per_shift):,} false alarms) "
            f"vs {int(static_fp_per_shift):,} false alarms with static thresholds "
            f"— a {round(far_reduction_pct,1)}% reduction."
        ),
    }


# ============================================================================
# Section 4 — Figure: Precision-Recall Curves
# ============================================================================

def plot_precision_recall_curves(
    df: pd.DataFrame,
    output_dir: Path,
) -> dict[str, float]:
    """
    Figure 1 — Precision-Recall curves for all three models + static baseline.

    Annotates five operating points with operational interpretations.
    Saves figure as 'fig01_precision_recall.png'.

    Returns
    -------
    dict mapping model_key → AP (average precision)
    """
    logger.info("Plotting precision-recall curves …")
    y_true = df["anomaly_label"].values

    models = {
        "isolation_forest": ("if_score",       "Isolation Forest"),
        "random_forest":    ("rf_score",        "Random Forest"),
        "ensemble":         ("ensemble_score",  "Ensemble (IF+RF)"),
    }

    fig, ax = plt.subplots(figsize=(9, 7))
    ap_scores: dict[str, float] = {}

    # Static baseline is a single point (no score, just binary prediction)
    static_prec = precision_score(y_true, df["baseline_pred"].values, zero_division=0)
    static_rec  = recall_score(y_true, df["baseline_pred"].values, zero_division=0)
    ax.scatter(
        static_rec, static_prec,
        marker="X", s=180, zorder=6,
        color=PALETTE["baseline_static"],
        label=f"Static Threshold (prec={static_prec:.2f}, rec={static_rec:.2f})",
    )

    for key, (score_col, label) in models.items():
        scores = df[score_col].values
        prec, rec, thresholds = precision_recall_curve(y_true, scores)
        ap = average_precision_score(y_true, scores)
        ap_scores[key] = ap

        ax.plot(
            rec, prec,
            linewidth=2.0,
            color=PALETTE[key],
            label=f"{label}  AP={ap:.3f}",
        )

        # Annotate operating points
        for m_key, thr, op_label in ANNOTATED_OPERATING_POINTS:
            if m_key != key:
                continue
            # Find closest threshold in the precision_recall_curve arrays
            # thresholds is length n-1; prec/rec are length n
            if len(thresholds) == 0:
                continue
            idx = np.argmin(np.abs(thresholds - thr))
            p_val, r_val = prec[idx], rec[idx]

            ax.scatter(r_val, p_val, s=90, zorder=7, color=PALETTE[key], edgecolors="black", linewidths=0.8)
            ax.annotate(
                f"{op_label}\n(thr={thr:.2f})",
                xy=(r_val, p_val),
                xytext=(r_val + 0.03, p_val - 0.05),
                fontsize=7.5,
                arrowprops=dict(arrowstyle="->", color="grey", lw=0.8),
                color=PALETTE[key],
            )

    # No-skill baseline line
    no_skill = y_true.mean()
    ax.axhline(no_skill, linestyle="--", color="black", linewidth=0.8, alpha=0.5,
               label=f"No-skill baseline (prevalence={no_skill:.3f})")

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(
        "Figure 1 — Precision-Recall Curves\n"
        "RAN KPI Anomaly Detection · Test Split (Days 26-30)",
        fontsize=11,
    )
    ax.legend(loc="upper right", fontsize=8.5)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    out = output_dir / "fig01_precision_recall.png"
    fig.tight_layout()
    if _USING_FALLBACK_DATA:
        _add_fallback_watermark(fig)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out)
    return ap_scores


# ============================================================================
# Section 5 — Figure: ROC Curves
# ============================================================================

def plot_roc_curves(
    df: pd.DataFrame,
    output_dir: Path,
) -> dict[str, float]:
    """
    Figure 2 — ROC curves for all models.
    Saves 'fig02_roc_curves.png'.

    Returns
    -------
    dict mapping model_key → AUC
    """
    logger.info("Plotting ROC curves …")
    y_true = df["anomaly_label"].values

    models = {
        "isolation_forest": ("if_score",       "Isolation Forest"),
        "random_forest":    ("rf_score",        "Random Forest"),
        "ensemble":         ("ensemble_score",  "Ensemble (IF+RF)"),
    }

    fig, ax = plt.subplots(figsize=(7, 6))
    auc_scores: dict[str, float] = {}

    for key, (score_col, label) in models.items():
        scores   = df[score_col].values
        fpr, tpr, _ = roc_curve(y_true, scores)
        roc_auc  = auc(fpr, tpr)
        auc_scores[key] = roc_auc
        ax.plot(fpr, tpr, linewidth=2.0, color=PALETTE[key],
                label=f"{label}  AUC={roc_auc:.3f}")

    # Static baseline operating point
    static_tn, static_fp, static_fn, static_tp = confusion_matrix(
        y_true, df["baseline_pred"].values, labels=[0, 1]
    ).ravel()
    s_fpr = static_fp / (static_fp + static_tn + 1e-9)
    s_tpr = static_tp / (static_tp + static_fn + 1e-9)
    ax.scatter(s_fpr, s_tpr, s=140, marker="X", color=PALETTE["baseline_static"],
               zorder=6, label=f"Static Threshold (FPR={s_fpr:.2f}, TPR={s_tpr:.2f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5, label="Random classifier")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(
        "Figure 2 — ROC Curves\n"
        "RAN KPI Anomaly Detection · Test Split (Days 26-30)",
        fontsize=11,
    )
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(0, 1.01)
    ax.set_ylim(0, 1.01)
    ax.grid(True, alpha=0.3)

    out = output_dir / "fig02_roc_curves.png"
    fig.tight_layout()
    if _USING_FALLBACK_DATA:
        _add_fallback_watermark(fig)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out)
    return auc_scores


# ============================================================================
# Section 6 — Figure: Confusion Matrices
# ============================================================================

def plot_confusion_matrices(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Figure 3 — 2×4 grid of confusion matrices (absolute + normalised)
    for Isolation Forest, Random Forest, Ensemble, and Static Baseline.
    Saves 'fig03_confusion_matrices.png'.
    """
    logger.info("Plotting confusion matrices …")
    y_true = df["anomaly_label"].values

    models = [
        ("isolation_forest", "if_pred",        "Isolation Forest"),
        ("random_forest",    "rf_pred",        "Random Forest"),
        ("ensemble",         "ensemble_pred",  "Ensemble (IF+RF)"),
        ("baseline_static",  "baseline_pred",  "Static Threshold Baseline"),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    fig.suptitle(
        "Figure 3 — Confusion Matrices (Top: absolute, Bottom: normalised by true class)\n"
        "Test Split (Days 26-30)",
        fontsize=11,
    )

    for col_idx, (model_key, pred_col, label) in enumerate(models):
        y_pred = df[pred_col].values
        cm     = confusion_matrix(y_true, y_pred, labels=[0, 1])
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        for row_idx, (matrix, fmt, vmax) in enumerate(
            [(cm, "d", cm.max()), (cm_norm, ".2f", 1.0)]
        ):
            ax = axes[row_idx][col_idx]
            sns.heatmap(
                matrix,
                annot=True,
                fmt=fmt,
                cmap="Blues" if model_key != "baseline_static" else "Greys",
                ax=ax,
                vmin=0,
                vmax=vmax,
                xticklabels=["Pred Normal", "Pred Anomaly"],
                yticklabels=["True Normal", "True Anomaly"],
                linewidths=0.5,
                cbar=False,
            )
            ax.set_xlabel("Predicted", fontsize=9)
            ax.set_ylabel("Actual", fontsize=9)
            if row_idx == 0:
                ax.set_title(label, fontsize=10, color=PALETTE[model_key])
            ax.tick_params(labelsize=8)

    out = output_dir / "fig03_confusion_matrices.png"
    fig.tight_layout()
    if _USING_FALLBACK_DATA:
        _add_fallback_watermark(fig)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out)


# ============================================================================
# Section 7 — Figure: Time-to-Detect Analysis
# ============================================================================

def plot_time_to_detect(
    df: pd.DataFrame,
    output_dir: Path,
) -> dict[str, dict[str, float]]:
    """
    Figure 4 — Time-to-detect histograms for all three models.

    TTD is measured in ROPs from anomaly onset to first true-positive detection.
    Saves 'fig04_time_to_detect.png'.

    Returns
    -------
    dict mapping model_key → {median_rops, mean_rops, p90_rops, missed_pct}
    """
    logger.info("Computing time-to-detect …")

    pred_cols = {
        "isolation_forest": "if_pred",
        "random_forest":    "rf_pred",
        "ensemble":         "ensemble_pred",
    }

    ttd_stats: dict[str, dict[str, float]] = {}
    all_ttd: dict[str, pd.Series] = {}

    for key, pred_col in pred_cols.items():
        ttd = time_to_detect(df, pred_col)
        all_ttd[key] = ttd
        missed_pct = 100.0 * ttd.isna().mean()
        ttd_valid  = ttd.dropna()
        ttd_stats[key] = {
            "median_rops": float(ttd_valid.median()) if not ttd_valid.empty else float("nan"),
            "mean_rops":   float(ttd_valid.mean())   if not ttd_valid.empty else float("nan"),
            "p90_rops":    float(ttd_valid.quantile(0.90)) if not ttd_valid.empty else float("nan"),
            "missed_pct":  round(missed_pct, 1),
            "n_events":    len(ttd),
        }
        # Convert to minutes for logging
        med_min = ttd_stats[key]["median_rops"] * ROP_MINUTES
        logger.info(
            "TTD %s: median=%.1f ROPs (%.0f min)  missed=%.1f%%",
            key, ttd_stats[key]["median_rops"], med_min, missed_pct,
        )

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    fig.suptitle(
        "Figure 4 — Time-to-Detect Distribution by Model\n"
        f"(1 ROP = {ROP_MINUTES} minutes · missed events excluded from histogram)",
        fontsize=11,
    )

    for ax, (key, label) in zip(
        axes,
        [("isolation_forest", "Isolation Forest"),
         ("random_forest",    "Random Forest"),
         ("ensemble",         "Ensemble (IF+RF)")],
    ):
        ttd_valid = all_ttd[key].dropna()
        stats_k   = ttd_stats[key]

        if not ttd_valid.empty:
            ax.hist(
                ttd_valid.values,
                bins=min(20, max(5, len(ttd_valid) // 5)),
                color=PALETTE[key],
                edgecolor="white",
                alpha=0.85,
            )
        ax.axvline(
            stats_k["median_rops"], color="black", linestyle="--",
            linewidth=1.2, label=f"Median={stats_k['median_rops']:.1f} ROPs"
        )
        ax.axvline(
            stats_k["p90_rops"], color="red", linestyle=":",
            linewidth=1.2, label=f"P90={stats_k['p90_rops']:.1f} ROPs"
        )

        # Convert axis ticks to minutes
        rop_ticks = ax.get_xticks()
        ax.set_xticklabels(
            [f"{int(t * ROP_MINUTES)}m" for t in rop_ticks], fontsize=8
        )

        ax.set_title(
            f"{label}\nMissed: {stats_k['missed_pct']}%  "
            f"N={stats_k['n_events']} events",
            fontsize=10,
        )
        ax.set_xlabel("Time to First TP Detection", fontsize=9)
        ax.set_ylabel("Event Count" if ax == axes[0] else "", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    out = output_dir / "fig04_time_to_detect.png"
    fig.tight_layout()
    if _USING_FALLBACK_DATA:
        _add_fallback_watermark(fig)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out)
    return ttd_stats


# ============================================================================
# Section 8 — Figure: Per-Cell False Alarm Rate
# ============================================================================

def plot_per_cell_false_alarm_rate(
    df: pd.DataFrame,
    topology_df: Optional[pd.DataFrame],
    output_dir: Path,
) -> None:
    """
    Figure 5 — Distribution of false-alarm-rate (FAR) per cell per day.

    Coloured by cell type when topology is available.
    Saves 'fig05_per_cell_far.png'.

    The key insight: static thresholds have a heavy right tail (a small number
    of "hot" cells generate a disproportionate share of false alarms) whereas
    the ML models have a more uniform distribution.

    # See Coursebook Ch. 13: Spatial features and cell-type clustering
    """
    logger.info("Computing per-cell false alarm rates …")

    pred_configs = [
        ("baseline_pred",  "Static Threshold", PALETTE["baseline_static"]),
        ("if_pred",        "Isolation Forest",  PALETTE["isolation_forest"]),
        ("rf_pred",        "Random Forest",     PALETTE["random_forest"]),
        ("ensemble_pred",  "Ensemble (IF+RF)",  PALETTE["ensemble"]),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=False)
    fig.suptitle(
        "Figure 5 — Per-Cell False Alarm Rate (per day)\n"
        "Distribution across all cell sectors in test split",
        fontsize=11,
    )

    for ax, (pred_col, label, colour) in zip(axes, pred_configs):
        far = per_cell_false_alarm_rate(df, pred_col)

        # Merge cell type for colouring if topology available
        if topology_df is not None and "cell_id" in topology_df.columns and "cell_type" in topology_df.columns:
            ct_map = topology_df.set_index("cell_id")["cell_type"].to_dict()
            cell_types = far.index.map(ct_map).fillna("unknown")
            unique_types = cell_types.unique()
            type_palette = sns.color_palette("Set2", len(unique_types))
            type_colours = dict(zip(unique_types, type_palette))
            bar_colours  = [type_colours.get(ct, "grey") for ct in cell_types]
        else:
            bar_colours = colour
            unique_types = []
            type_colours = {}

        ax.bar(
            range(len(far)),
            far.sort_values(ascending=False).values,
            color=bar_colours if isinstance(bar_colours, list) else colour,
            edgecolor="none",
            width=1.0,
        )
        ax.axhline(
            far.median(), color="black", linestyle="--", linewidth=1.2,
            label=f"Median={far.median():.2f}/day"
        )
        ax.axhline(
            far.quantile(0.95), color="red", linestyle=":", linewidth=1.0,
            label=f"P95={far.quantile(0.95):.2f}/day"
        )

        # Add cell-type legend if available
        if unique_types is not None and len(unique_types) > 0:
            patches = [
                mpatches.Patch(color=type_colours[ct], label=ct)
                for ct in unique_types
            ]
            ax.legend(handles=patches + [
                mpatches.Patch(color="black", label=f"Median={far.median():.2f}"),
            ], fontsize=7, loc="upper right")
        else:
            ax.legend(fontsize=8)

        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Cell sectors (sorted by FAR)", fontsize=9)
        ax.set_ylabel("False Alarms / Day" if ax == axes[0] else "", fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

    out = output_dir / "fig05_per_cell_far.png"
    fig.tight_layout()
    if _USING_FALLBACK_DATA:
        _add_fallback_watermark(fig)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out)


# ============================================================================
# Section 9 — Figure: Anomaly-Type Breakdown
# ============================================================================

def plot_anomaly_type_breakdown(
    df: pd.DataFrame,
    output_dir: Path,
) -> dict[str, dict[str, float]]:
    """
    Figure 6 — Recall by anomaly type for each model.

    This is one of the most operationally important charts: it shows that
    Isolation Forest excels at sudden drops (point anomalies) but struggles
    with gradual degradations, while the LSTM-based ensemble (proxied here by
    the weighted ensemble score) catches more of the subtle patterns.

    Saves 'fig06_anomaly_type_breakdown.png'.

    Returns
    -------
    dict of {anomaly_type: {model: recall}}
    """
    logger.info("Computing anomaly-type breakdown …")

    anomaly_types = [t for t in df["anomaly_type"].unique() if t != "none"]
    pred_configs  = {
        "Isolation Forest": "if_pred",
        "Random Forest":    "rf_pred",
        "Ensemble (IF+RF)": "ensemble_pred",
        "Static Threshold": "baseline_pred",
    }

    breakdown: dict[str, dict[str, float]] = {at: {} for at in anomaly_types}

    for label, pred_col in pred_configs.items():
        for a_type in anomaly_types:
            sub = df[df["anomaly_type"] == a_type]
            if sub.empty or sub["anomaly_label"].sum() == 0:
                breakdown[a_type][label] = 0.0
                continue
            breakdown[a_type][label] = float(
                recall_score(sub["anomaly_label"].values, sub[pred_col].values, zero_division=0)
            )

    # Build a DataFrame for easy plotting
    rows = []
    for a_type, model_recalls in breakdown.items():
        for model_label, recall_val in model_recalls.items():
            rows.append({"Anomaly Type": a_type, "Model": model_label, "Recall": recall_val})
    plot_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(11, 6))
    bar_width = 0.18
    anomaly_type_labels = sorted(anomaly_types)
    x_positions = np.arange(len(anomaly_type_labels))
    model_order = list(pred_configs.keys())

    for m_idx, model_label in enumerate(model_order):
        vals = [
            plot_df[(plot_df["Anomaly Type"] == at) & (plot_df["Model"] == model_label)]["Recall"].values
            for at in anomaly_type_labels
        ]
        vals_flat = [v[0] if len(v) > 0 else 0.0 for v in vals]
        ax.bar(
            x_positions + m_idx * bar_width,
            vals_flat,
            width=bar_width,
            label=model_label,
            color=list(PALETTE.values())[m_idx],
            edgecolor="white",
        )

    ax.set_xticks(x_positions + bar_width * (len(model_order) - 1) / 2)
    ax.set_xticklabels(
        [at.replace("_", " ").title() for at in anomaly_type_labels],
        fontsize=10,
    )
    ax.set_xlabel("Anomaly Type", fontsize=11)
    ax.set_ylabel("Recall", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_title(
        "Figure 6 — Recall by Anomaly Type\n"
        "Gradual degradation is harder for all models; ensemble closes the gap",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    out = output_dir / "fig06_anomaly_type_breakdown.png"
    fig.tight_layout()
    if _USING_FALLBACK_DATA:
        _add_fallback_watermark(fig)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out)
    return breakdown


# ============================================================================
# Section 10 — Figure: Time-Series Overlay
# ============================================================================

def plot_time_series_overlay(
    df: pd.DataFrame,
    output_dir: Path,
    n_cells: int = 3,
) -> None:
    """
    Figure 7 — Time-series overlay showing raw KPI trajectory (proxy: ensemble
    score), ground-truth anomaly windows, and model predictions for a sample
    of cells.

    Saves 'fig07_timeseries_overlay.png'.

    This is the "show me where it fires" chart that NOC engineers always ask for.
    """
    logger.info("Plotting time-series overlay for %d sample cells …", n_cells)

    # Pick cells that have at least one real anomaly
    cells_with_anomaly = (
        df[df["anomaly_label"] == 1]["cell_id"].value_counts()
        .head(n_cells).index.tolist()
    )
    if not cells_with_anomaly:
        logger.warning("No anomaly cells found — skipping time-series overlay")
        return

    fig, axes = plt.subplots(n_cells, 1, figsize=(16, 4 * n_cells), sharex=False)
    if n_cells == 1:
        axes = [axes]

    fig.suptitle(
        "Figure 7 — Anomaly Score Time-Series Overlay\n"
        "Shaded = true anomaly window · Lines = model anomaly scores",
        fontsize=11,
    )

    for ax, cell_id in zip(axes, cells_with_anomaly):
        cell_df = df[df["cell_id"] == cell_id].sort_values("timestamp")

        ax.plot(
            cell_df["timestamp"], cell_df["if_score"],
            color=PALETTE["isolation_forest"], lw=1.2, alpha=0.8,
            label="IF Score"
        )
        ax.plot(
            cell_df["timestamp"], cell_df["rf_score"],
            color=PALETTE["random_forest"], lw=1.2, alpha=0.8,
            label="RF Score"
        )
        ax.plot(
            cell_df["timestamp"], cell_df["ensemble_score"],
            color=PALETTE["ensemble"], lw=1.8,
            label="Ensemble Score"
        )

        # Shade true anomaly windows
        anomaly_mask = cell_df["anomaly_label"].values == 1
        for i in range(len(cell_df) - 1):
            if anomaly_mask[i]:
                ax.axvspan(
                    cell_df["timestamp"].iloc[i],
                    cell_df["timestamp"].iloc[i + 1],
                    alpha=0.25, color=PALETTE["anomaly"], zorder=0,
                )

        # Mark ensemble prediction fires
        fire_mask = cell_df["ensemble_pred"].values == 1
        fire_times = cell_df["timestamp"].values[fire_mask]
        fire_scores = cell_df["ensemble_score"].values[fire_mask]
        ax.scatter(
            fire_times, fire_scores,
            marker="v", color=PALETTE["ensemble"], s=40, zorder=5,
            label="Ensemble Fires"
        )

        ax.set_ylabel("Anomaly Score", fontsize=9)
        ax.set_title(f"Cell: {cell_id}", fontsize=10)
        ax.set_ylim(-0.05, 1.10)
        _ens_t = ACTIVE_THRESHOLDS["ensemble"]
        ax.axhline(_ens_t, color=PALETTE["ensemble"], linestyle=":", lw=0.8, alpha=0.7,
                   label=f"Ensemble Threshold ({_ens_t:.2f})")
        ax.legend(fontsize=7.5, loc="upper left")
        ax.grid(True, alpha=0.3)

    out = output_dir / "fig07_timeseries_overlay.png"
    fig.tight_layout()
    if _USING_FALLBACK_DATA:
        _add_fallback_watermark(fig)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out)


# ============================================================================
# Section 11 — Figure: Bootstrap CI Summary
# ============================================================================

def plot_bootstrap_ci_summary(
    df: pd.DataFrame,
    output_dir: Path,
    n_boot: int = 500,
) -> dict[str, Any]:
    """
    Figure 8 — Bar chart with 95% bootstrap confidence intervals for F1, AUC,
    and AP across all models.

    Saves 'fig08_bootstrap_ci.png'.

    Bootstrap CIs are critical for production sign-off: a model with F1=0.85
    but 95% CI [0.71, 0.93] is far less reliable than one with F1=0.82 and
    CI [0.80, 0.84].

    # See Coursebook Ch. 54: Statistical Significance in Evaluation
    """
    logger.info("Computing bootstrap confidence intervals (n_boot=%d) …", n_boot)

    y_true = df["anomaly_label"].values
    model_configs = {
        "Isolation Forest": ("if_score",       ACTIVE_THRESHOLDS["if"]),
        "Random Forest":    ("rf_score",        ACTIVE_THRESHOLDS["rf"]),
        "Ensemble (IF+RF)": ("ensemble_score",  ACTIVE_THRESHOLDS["ensemble"]),
    }
    metrics = ["f1", "precision", "recall", "roc_auc", "pr_auc"]
    ci_results: dict[str, Any] = {}

    # Collect data
    plot_data: list[dict] = []
    for model_label, (score_col, thr) in model_configs.items():
        scores = df[score_col].values
        ci_results[model_label] = {}
        for metric in metrics:
            point, lo, hi = bootstrap_ci(
                y_true, scores, thr,
                metric=metric, n_boot=n_boot,
            )
            ci_results[model_label][metric] = {"point": point, "lo": lo, "hi": hi}
            plot_data.append({
                "Model": model_label,
                "Metric": metric.upper().replace("_", "-"),
                "Point": point, "Lo": lo, "Hi": hi,
            })
            logger.info(
                "  %-25s  %-10s  %.3f  [%.3f, %.3f]",
                model_label, metric, point, lo, hi,
            )

    plot_df = pd.DataFrame(plot_data)

    # One subplot per metric
    fig, axes = plt.subplots(1, len(metrics), figsize=(18, 6))
    fig.suptitle(
        "Figure 8 — Bootstrap 95% Confidence Intervals\n"
        f"(n_boot={n_boot} · Test split, Days 26-30)",
        fontsize=11,
    )

    model_colours = [
        PALETTE["isolation_forest"],
        PALETTE["random_forest"],
        PALETTE["ensemble"],
    ]

    for ax, metric_str in zip(axes, [m.upper().replace("_", "-") for m in metrics]):
        sub = plot_df[plot_df["Metric"] == metric_str].reset_index(drop=True)
        bars = ax.bar(
            sub["Model"],
            sub["Point"],
            color=model_colours[:len(sub)],
            edgecolor="white",
            width=0.55,
        )
        # Error bars
        yerr_lo = sub["Point"] - sub["Lo"]
        yerr_hi = sub["Hi"] - sub["Point"]
        ax.errorbar(
            range(len(sub)),
            sub["Point"],
            yerr=[yerr_lo.values, yerr_hi.values],
            fmt="none",
            color="black",
            capsize=5,
            linewidth=1.5,
        )

        ax.set_ylim(0, 1.08)
        ax.set_title(metric_str, fontsize=10)
        ax.set_xticklabels(sub["Model"], rotation=15, ha="right", fontsize=8)
        ax.set_ylabel("Score" if ax == axes[0] else "", fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

        # Annotate point estimates
        for bar_obj, point_val in zip(bars, sub["Point"]):
            ax.text(
                bar_obj.get_x() + bar_obj.get_width() / 2,
                point_val + 0.02,
                f"{point_val:.3f}",
                ha="center", va="bottom", fontsize=8,
            )

    out = output_dir / "fig08_bootstrap_ci.png"
    fig.tight_layout()
    if _USING_FALLBACK_DATA:
        _add_fallback_watermark(fig)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out)
    return ci_results


# ============================================================================
# Section 12 — Threshold Sweep Table & Operating Point Analysis
# ============================================================================

def compute_threshold_sweep_table(
    df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """
    For each model, compute precision/recall/F1/FAR across 200 thresholds.

    Returns a dict mapping model_key → sweep DataFrame.
    These DataFrames power both Figure 1 (PR curve) and the operating-point
    recommendation table.
    """
    logger.info("Computing threshold sweep tables …")
    y_true = df["anomaly_label"].values

    configs = {
        "isolation_forest": "if_score",
        "random_forest":    "rf_score",
        "ensemble":         "ensemble_score",
    }

    sweeps: dict[str, pd.DataFrame] = {}
    for key, col in configs.items():
        sweep_df = sweep_thresholds(y_true, df[col].values)
        sweeps[key] = sweep_df
        logger.debug("  %s: %d threshold points", key, len(sweep_df))

    return sweeps


def select_operating_points(
    sweeps: dict[str, pd.DataFrame],
    target_recall: float = 0.90,
) -> dict[str, dict[str, float]]:
    """
    For each model, select the threshold that maximises F1, and separately the
    threshold that achieves at least `target_recall` while maximising precision.

    Returns
    -------
    dict of model_key → {"max_f1": {...}, "target_recall": {...}}
    """
    ops: dict[str, dict[str, float]] = {}
    for key, sweep_df in sweeps.items():
        # Best F1 threshold
        best_f1_row = sweep_df.loc[sweep_df["f1"].idxmax()]

        # Best precision at target recall
        eligible = sweep_df[sweep_df["recall"] >= target_recall]
        if not eligible.empty:
            best_rec_row = eligible.loc[eligible["precision"].idxmax()]
        else:
            best_rec_row = sweep_df.loc[sweep_df["recall"].idxmax()]

        ops[key] = {
            "max_f1": best_f1_row.to_dict(),
            f"target_recall_{int(target_recall*100)}": best_rec_row.to_dict(),
        }
        logger.info(
            "Operating point [%s]  max-F1: thr=%.3f  p=%.3f  r=%.3f  f1=%.3f  "
            "| recall≥%.0f%%: thr=%.3f  p=%.3f  r=%.3f  f1=%.3f",
            key,
            best_f1_row["threshold"], best_f1_row["precision"],
            best_f1_row["recall"],    best_f1_row["f1"],
            target_recall * 100,
            best_rec_row["threshold"], best_rec_row["precision"],
            best_rec_row["recall"],    best_rec_row["f1"],
        )

    return ops


# ============================================================================
# Section 13 — Metrics Summary + Operational Interpretation
# ============================================================================

def compile_metrics_summary(
    df: pd.DataFrame,
    sweeps: dict[str, pd.DataFrame],
    operating_points: dict[str, dict[str, float]],
    ap_scores: dict[str, float],
    auc_scores: dict[str, float],
    ttd_stats: dict[str, dict[str, float]],
    ci_results: dict[str, Any],
    anomaly_type_breakdown: dict[str, dict[str, float]],
) -> dict[str, Any]:
    """
    Compile all scalar metrics into a single JSON-serialisable dictionary.

    The output is saved as 'metrics_summary.json' and is the authoritative
    record for the whitepaper's Section 8 ("Evaluation and Operational Impact").
    """
    logger.info("Compiling metrics summary …")
    y_true = df["anomaly_label"].values
    actual_anomaly_rate = float(y_true.mean()) if len(y_true) > 0 else 0.04

    model_configs = {
        "isolation_forest": ("if_score",       "if_pred",        "Isolation Forest"),
        "random_forest":    ("rf_score",        "rf_pred",        "Random Forest"),
        "ensemble":         ("ensemble_score",  "ensemble_pred",  "Ensemble (IF+RF)"),
        "baseline_static":  (None,              "baseline_pred",  "Static Threshold"),
    }

    summary: dict[str, Any] = {
        "dataset_info": {
            "total_rows":          len(df),
            "n_cells":             df["cell_id"].nunique(),
            "n_anomaly_rows":      int(y_true.sum()),
            "anomaly_rate_pct":    round(100.0 * y_true.mean(), 3),
            "n_anomaly_events":    int(df[df["anomaly_event_id"] >= 0]["anomaly_event_id"].nunique()),
            "test_days":           df["timestamp"].dt.normalize().nunique(),
            "rop_minutes":         ROP_MINUTES,
        },
        "models": {},
    }

    for key, (score_col, pred_col, label) in model_configs.items():
        y_pred = df[pred_col].values

        # Basic metrics at default threshold
        m = binary_metrics_at_threshold(y_true, y_pred.astype(float), 0.5)

        model_entry: dict[str, Any] = {
            "label":      label,
            "default_threshold_metrics": {
                "precision": m["precision"],
                "recall":    m["recall"],
                "f1":        m["f1"],
                "fpr":       m["fpr"],
                "fnr":       m["fnr"],
                "tn": m["tn"], "fp": m["fp"],
                "fn": m["fn"], "tp": m["tp"],
            },
        }

        if score_col is not None:
            scores = df[score_col].values
            model_entry["roc_auc"]        = round(roc_auc_score(y_true, scores), 4)
            model_entry["average_precision"] = round(ap_scores.get(key, float("nan")), 4)

        # Operating points
        if key in operating_points:
            model_entry["operating_points"] = operating_points[key]

        # TTD
        if key in ttd_stats:
            model_entry["time_to_detect"] = ttd_stats[key]

        # Bootstrap CIs
        if label in ci_results:
            model_entry["bootstrap_95ci"] = ci_results[label]

        # Operational context at the max-F1 operating point
        if key in operating_points and "max_f1" in operating_points[key]:
            op = operating_points[key]["max_f1"]
            op_ctx = operational_context(
                precision=op["precision"],
                recall=op["recall"],
                f1=op["f1"],
                model_name=label,
                anomaly_rate=actual_anomaly_rate,
            )
            model_entry["operational_context_at_max_f1"] = op_ctx

        # Anomaly type recall
        if key != "baseline_static":
            at_key = {
                "isolation_forest": "Isolation Forest",
                "random_forest":    "Random Forest",
                "ensemble":         "Ensemble (IF+RF)",
            }.get(key)
            if at_key:
                model_entry["recall_by_anomaly_type"] = {
                    a_type: anomaly_type_breakdown.get(a_type, {}).get(at_key, float("nan"))
                    for a_type in anomaly_type_breakdown
                }

        summary["models"][key] = model_entry

    return summary


# ============================================================================
# Section 14 — Figure: Threshold-Sweep F1 / Precision / Recall
# ============================================================================

def plot_threshold_sweep(
    sweeps: dict[str, pd.DataFrame],
    output_dir: Path,
) -> None:
    """
    Figure 9 — F1, Precision, and Recall as a function of decision threshold
    for all three models.

    Helps operators choose operating points without needing to read a full
    PR curve.  Saves 'fig09_threshold_sweep.png'.
    """
    logger.info("Plotting threshold-sweep curves …")

    model_order = [
        ("isolation_forest", "Isolation Forest"),
        ("random_forest",    "Random Forest"),
        ("ensemble",         "Ensemble (IF+RF)"),
    ]
    metric_cols = ["f1", "precision", "recall"]
    line_styles = ["-", "--", ":"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    fig.suptitle(
        "Figure 9 — Metric vs Decision Threshold\n"
        "(Vertical line = max-F1 operating point)",
        fontsize=11,
    )

    for ax, (key, label) in zip(axes, model_order):
        sweep_df = sweeps[key]
        for metric, ls in zip(metric_cols, line_styles):
            ax.plot(
                sweep_df["threshold"], sweep_df[metric],
                lw=1.8, linestyle=ls,
                color=PALETTE[key],
                label=metric.capitalize(),
            )

        # Mark max-F1 threshold
        best_idx = sweep_df["f1"].idxmax()
        best_thr = sweep_df.loc[best_idx, "threshold"]
        best_f1  = sweep_df.loc[best_idx, "f1"]
        ax.axvline(best_thr, color="black", linestyle="--", lw=1.0, alpha=0.7,
                   label=f"Max-F1={best_f1:.3f}\n@thr={best_thr:.3f}")

        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Decision Threshold", fontsize=9)
        ax.set_ylabel("Score" if ax == axes[0] else "", fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    out = output_dir / "fig09_threshold_sweep.png"
    fig.tight_layout()
    if _USING_FALLBACK_DATA:
        _add_fallback_watermark(fig)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out)


# ============================================================================
# Section 15 — Figure: Operational Impact Summary (Scorecard)
# ============================================================================

def plot_operational_scorecard(
    summary: dict[str, Any],
    output_dir: Path,
) -> None:
    """
    Figure 10 — Visual scorecard comparing models on five operational KPIs:
      - False alarms per NOC shift
      - Detection rate (recall at max-F1 threshold)
      - Median time to detect (minutes)
      - False alarm reduction vs static threshold (%)
      - Missed anomalies per day at operator scale

    This is the 'management summary' chart — designed for the NOC manager, not
    the data scientist.

    Saves 'fig10_operational_scorecard.png'.
    """
    logger.info("Plotting operational scorecard …")

    # Extract data from the pre-computed summary
    model_keys   = ["isolation_forest", "random_forest", "ensemble"]
    model_labels = ["Isolation Forest", "Random Forest", "Ensemble (IF+RF)"]
    ttd_key      = "time_to_detect"

    scorecard_rows = []
    for key, label in zip(model_keys, model_labels):
        m = summary["models"].get(key, {})
        op_ctx = m.get("operational_context_at_max_f1", {})
        ttd    = m.get(ttd_key, {})
        op_pts = m.get("operating_points", {})
        mf1    = op_pts.get("max_f1", {})

        scorecard_rows.append({
            "Model": label,
            "False Alarms\n/ Shift":          op_ctx.get("false_alarms_per_shift", float("nan")),
            "Detection Rate\n(Recall %)":      round(mf1.get("recall", float("nan")) * 100, 1),
            "Median TTD\n(minutes)":           round(ttd.get("median_rops", float("nan")) * ROP_MINUTES, 0),
            "FA Reduction\nvs Static (%)":     op_ctx.get("false_alarm_reduction_pct", float("nan")),
            "Missed Anomalies\n/ Day":         op_ctx.get("missed_per_day", float("nan")),
        })

    sc_df = pd.DataFrame(scorecard_rows).set_index("Model")

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis("off")

    col_labels = list(sc_df.columns)
    row_labels = list(sc_df.index)
    cell_data  = sc_df.values

    table = ax.table(
        cellText=[
            [f"{v:.0f}" if not (isinstance(v, float) and np.isnan(v)) else "N/A"
             for v in row]
            for row in cell_data
        ],
        rowLabels=row_labels,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        bbox=[0, 0.05, 1, 0.85],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    # Colour the header row
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#1565C0")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Colour model rows by model palette
    for i, (key, colour) in enumerate(
        zip(model_keys, [PALETTE["isolation_forest"], PALETTE["random_forest"], PALETTE["ensemble"]])
    ):
        # Row labels column is index -1
        table[i + 1, -1].set_facecolor(colour)
        table[i + 1, -1].set_text_props(color="white", fontweight="bold")
        for j in range(len(col_labels)):
            table[i + 1, j].set_facecolor(colour + "33")   # 20% opacity hex suffix

    ax.set_title(
        "Figure 10 — Operational Scorecard\n"
        f"(Scale: {CELLS_PER_NOC_DOMAIN:,} cell sectors, {SHIFTS_PER_DAY} shifts/day, "
        f"4% baseline anomaly rate)",
        fontsize=11, pad=20,
    )

    out = output_dir / "fig10_operational_scorecard.png"
    fig.tight_layout()
    if _USING_FALLBACK_DATA:
        _add_fallback_watermark(fig)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out)


# ============================================================================
# Section 16 — Error Analysis
# ============================================================================

def run_error_analysis(
    df: pd.DataFrame,
    topology_df: Optional[pd.DataFrame],
    output_dir: Path,
) -> dict[str, Any]:
    """
    Characterise false positives and false negatives for the ensemble model.

    Produces a text report and saves 'fig11_error_analysis.png'.

    Key questions:
    - Are FPs concentrated in certain cell types (urban, stadium)?
    - Are FNs concentrated in certain anomaly types (gradual degradation)?
    - Is there a time-of-day pattern in FP generation?

    # See Coursebook Ch. 54: Error Analysis for Time-Series Models
    """
    logger.info("Running error analysis …")

    y_true = df["anomaly_label"].values
    y_pred = df["ensemble_pred"].values

    # Categorise each row
    df = df.copy()
    df["error_type"] = "TN"
    df.loc[(y_pred == 1) & (y_true == 1), "error_type"] = "TP"
    df.loc[(y_pred == 1) & (y_true == 0), "error_type"] = "FP"
    df.loc[(y_pred == 0) & (y_true == 1), "error_type"] = "FN"

    results: dict[str, Any] = {}

    # 1. FP by hour of day
    fp_df = df[df["error_type"] == "FP"].copy()
    fn_df = df[df["error_type"] == "FN"].copy()

    fp_by_hour = fp_df.groupby(fp_df["timestamp"].dt.hour).size()
    fn_by_hour = fn_df.groupby(fn_df["timestamp"].dt.hour).size()

    results["fp_total"] = int(len(fp_df))
    results["fn_total"] = int(len(fn_df))
    results["fp_hour_distribution"] = fp_by_hour.to_dict()

    # 2. FN by anomaly type
    fn_by_type = fn_df.groupby("anomaly_type").size()
    results["fn_by_anomaly_type"] = fn_by_type.to_dict()

    # 3. Cell-type breakdown of FP rate
    if topology_df is not None and "cell_type" in topology_df.columns:
        merged = df.merge(
            topology_df[["cell_id", "cell_type", "vendor"]].drop_duplicates("cell_id"),
            on="cell_id", how="left",
        )
        fp_by_type = (
            merged[merged["error_type"] == "FP"]
            .groupby("cell_type").size()
            / merged.groupby("cell_type").size()
        ).rename("fp_rate")
        results["fp_rate_by_cell_type"] = fp_by_type.to_dict()
    else:
        merged = df.copy()
        merged["cell_type"] = "unknown"
        merged["vendor"]    = "unknown"
        results["fp_rate_by_cell_type"] = {}

    # --- Plotting ---
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle(
        "Figure 11 — Error Analysis: Ensemble Model\n"
        "(False Positive and False Negative characterisation)",
        fontsize=11,
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.4)

    # Top-left: FP by hour
    ax1 = fig.add_subplot(gs[0, 0])
    hours = sorted(fp_by_hour.index)
    ax1.bar(hours, [fp_by_hour.get(h, 0) for h in hours],
            color=PALETTE["anomaly"], edgecolor="white", alpha=0.85)
    ax1.set_title("FP Count by Hour of Day", fontsize=10)
    ax1.set_xlabel("Hour", fontsize=9)
    ax1.set_ylabel("FP Count", fontsize=9)
    ax1.grid(True, alpha=0.3, axis="y")

    # Top-middle: FN by anomaly type
    ax2 = fig.add_subplot(gs[0, 1])
    if not fn_by_type.empty:
        ax2.barh(
            [t.replace("_", " ").title() for t in fn_by_type.index],
            fn_by_type.values,
            color=PALETTE["isolation_forest"], edgecolor="white", alpha=0.85,
        )
    ax2.set_title("FN Count by Anomaly Type\n(What the model misses)", fontsize=10)
    ax2.set_xlabel("FN Count", fontsize=9)
    ax2.grid(True, alpha=0.3, axis="x")

    # Top-right: FP rate by cell type
    ax3 = fig.add_subplot(gs[0, 2])
    fp_ct = results.get("fp_rate_by_cell_type", {})
    if fp_ct:
        ct_keys = list(fp_ct.keys())
        ct_vals = list(fp_ct.values())
        ax3.barh(
            [k.replace("_", " ").title() for k in ct_keys],
            ct_vals,
            color=PALETTE["random_forest"], edgecolor="white", alpha=0.85,
        )
        ax3.set_title("FP Rate by Cell Type\n(FP / total rows per type)", fontsize=10)
        ax3.set_xlabel("FP Rate", fontsize=9)
        ax3.grid(True, alpha=0.3, axis="x")
    else:
        ax3.text(0.5, 0.5, "No topology data", ha="center", va="center", fontsize=12)
        ax3.set_title("FP Rate by Cell Type", fontsize=10)

    # Bottom-left: Score distribution for TP/FP/TN/FN
    ax4 = fig.add_subplot(gs[1, 0:2])
    for et, colour, alpha in [("TP", PALETTE["normal"], 0.7),
                               ("FN", PALETTE["anomaly"], 0.7),
                               ("FP", "#FF9800", 0.6),
                               ("TN", PALETTE["baseline_static"], 0.4)]:
        sub = df[df["error_type"] == et]["ensemble_score"]
        if not sub.empty:
            ax4.hist(sub.values, bins=40, alpha=alpha, color=colour,
                     label=f"{et} (n={len(sub)})", edgecolor="none", density=True)
    _ens_t = ACTIVE_THRESHOLDS["ensemble"]
    ax4.axvline(_ens_t, color="black", linestyle="--", lw=1.2, label=f"Threshold={_ens_t:.2f}")
    ax4.set_title("Ensemble Score Distribution by Error Type", fontsize=10)
    ax4.set_xlabel("Ensemble Anomaly Score", fontsize=9)
    ax4.set_ylabel("Density", fontsize=9)
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    # Bottom-right: FP + FN by vendor
    ax5 = fig.add_subplot(gs[1, 2])
    if "vendor" in merged.columns and merged["vendor"].nunique() > 1:
        vend_fp = merged[merged["error_type"] == "FP"].groupby("vendor").size()
        vend_fn = merged[merged["error_type"] == "FN"].groupby("vendor").size()
        vend_all = merged.groupby("vendor").size()
        vend_fp_rate = (vend_fp / vend_all).fillna(0)
        vend_fn_rate = (vend_fn / merged[merged["anomaly_label"] == 1].groupby("vendor").size()).fillna(0)

        x = np.arange(len(vend_fp_rate))
        ax5.bar(x - 0.2, vend_fp_rate.values, width=0.35,
                color=PALETTE["anomaly"], alpha=0.8, label="FP Rate")
        ax5.bar(x + 0.2, vend_fn_rate.reindex(vend_fp_rate.index).fillna(0).values,
                width=0.35, color=PALETTE["isolation_forest"], alpha=0.8, label="FN Rate")
        ax5.set_xticks(x)
        ax5.set_xticklabels(vend_fp_rate.index, fontsize=9)
        ax5.set_title("FP / FN Rate by Vendor\n(Multi-vendor normalisation quality)", fontsize=10)
        ax5.set_ylabel("Rate", fontsize=9)
        ax5.legend(fontsize=8)
        ax5.grid(True, alpha=0.3, axis="y")
        results["fp_rate_by_vendor"] = vend_fp_rate.to_dict()
        results["fn_rate_by_vendor"] = vend_fn_rate.to_dict()
    else:
        ax5.text(0.5, 0.5, "Single vendor\nor no vendor data", ha="center", va="center", fontsize=11)
        ax5.set_title("FP / FN Rate by Vendor", fontsize=10)

    out = output_dir / "fig11_error_analysis.png"
    if _USING_FALLBACK_DATA:
        _add_fallback_watermark(fig)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out)
    return results


# ============================================================================
# Section 17 — JSON Output
# ============================================================================

def save_metrics_json(
    summary: dict[str, Any],
    error_analysis: dict[str, Any],
    ci_results: dict[str, Any],
    operating_points: dict[str, dict[str, float]],
    output_dir: Path,
) -> None:
    """
    Save all scalar metrics to JSON for downstream consumption by dashboards,
    CI/CD evaluation gates, and the MLOps model registry.

    The JSON schema is designed to be ingested by MLflow log_metrics() and
    compatible with TM Forum AI Model Management APIs.

    # See Coursebook Ch. 54: Model Registry and Evaluation Gates
    """

    def _make_serialisable(obj: Any) -> Any:
        """Recursively coerce numpy types to Python native types."""
        if isinstance(obj, dict):
            return {k: _make_serialisable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_make_serialisable(v) for v in obj]
        if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        if isinstance(obj, (np.float64, np.float32)):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)
        if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return None
        return obj

    full_output = {
        "evaluation_config": {
            "seed":                   SEED,
            "rop_minutes":            ROP_MINUTES,
            "rops_per_shift":         ROPS_PER_SHIFT,
            "cells_per_noc_domain":   CELLS_PER_NOC_DOMAIN,
            "shifts_per_day":         SHIFTS_PER_DAY,
            "n_bootstrap_samples":    500,
            "bootstrap_ci":           0.95,
        },
        "metrics_summary":  summary,
        "error_analysis":   error_analysis,
        "bootstrap_ci":     ci_results,
        "operating_points": {
            k: {
                op_name: {
                    kk: (None if (isinstance(vv, float) and (np.isnan(vv) or np.isinf(vv))) else vv)
                    for kk, vv in op_dict.items()
                }
                for op_name, op_dict in ops.items()
            }
            for k, ops in operating_points.items()
        },
    }

    # ------------------------------------------------------------------
    # Data provenance flag — critical for downstream consumers
    # ------------------------------------------------------------------
    if _USING_FALLBACK_DATA:
        full_output["data_source"] = "synthetic_fallback"
        full_output["_WARNING"] = (
            "These metrics were computed on synthetic fallback data, NOT on "
            "the trained model's actual predictions. Run the full pipeline "
            "(01→02→03→04) for production-representative results."
        )
        logger.warning(
            "Saving metrics from SYNTHETIC FALLBACK data — not production-representative."
        )
    else:
        full_output["data_source"] = "full_pipeline"

    serialisable = _make_serialisable(full_output)

    out_path = output_dir / "metrics_summary.json"
    with open(out_path, "w") as fh:
        json.dump(serialisable, fh, indent=2)
    logger.info("Saved metrics_summary.json  (%d bytes)", out_path.stat().st_size)


# ============================================================================
# Section 18 — Text Report
# ============================================================================

def print_evaluation_report(
    summary: dict[str, Any],
    operating_points: dict[str, dict[str, float]],
    ttd_stats: dict[str, dict[str, float]],
) -> None:
    """
    Print a human-readable evaluation report to stdout (captured by CI logs).

    Structured to mirror the whitepaper's Section 8 narrative flow.
    """
    sep = "=" * 72
    actual_anomaly_rate = summary.get("anomaly_rate_pct", 4.0) / 100.0

    logger.info("\n%s", sep)
    logger.info("  RAN KPI ANOMALY DETECTION — EVALUATION REPORT")
    logger.info("  Test Split  |  Temporal split: Days 26-30 of 30-day dataset")
    logger.info("%s", sep)

    ds = summary["dataset_info"]
    logger.info(
        "Dataset:  %d rows  |  %d cells  |  %.2f%% anomaly rate  |  %d anomaly events",
        ds["total_rows"], ds["n_cells"], ds["anomaly_rate_pct"], ds["n_anomaly_events"],
    )

    logger.info("\n%s", "-" * 72)
    logger.info("  MODEL PERFORMANCE SUMMARY (default operating points)")
    logger.info("%s", "-" * 72)

    header = f"{'Model':<25}  {'Precision':>9}  {'Recall':>7}  {'F1':>7}  {'ROC-AUC':>8}  {'AP':>7}"
    logger.info(header)
    logger.info("-" * len(header))

    model_keys = ["baseline_static", "isolation_forest", "random_forest", "ensemble"]
    for key in model_keys:
        m = summary["models"].get(key, {})
        dt = m.get("default_threshold_metrics", {})
        roc = m.get("roc_auc", float("nan"))
        ap  = m.get("average_precision", float("nan"))
        logger.info(
            "%-25s  %9.3f  %7.3f  %7.3f  %8s  %7s",
            m.get("label", key),
            dt.get("precision", float("nan")),
            dt.get("recall",    float("nan")),
            dt.get("f1",        float("nan")),
            f"{roc:.3f}" if not np.isnan(roc) else "  N/A  ",
            f"{ap:.3f}"  if not np.isnan(ap)  else "  N/A  ",
        )

    logger.info("\n%s", "-" * 72)
    logger.info("  OPTIMAL OPERATING POINTS (Max-F1 threshold)")
    logger.info("%s", "-" * 72)

    for key in ["isolation_forest", "random_forest", "ensemble"]:
        ops = operating_points.get(key, {}).get("max_f1", {})
        if not ops:
            continue
        label = summary["models"][key]["label"]
        logger.info(
            "%-25s  threshold=%.3f  precision=%.3f  recall=%.3f  f1=%.3f  fpr=%.4f",
            label, ops.get("threshold", float("nan")),
            ops.get("precision", float("nan")), ops.get("recall", float("nan")),
            ops.get("f1", float("nan")),         ops.get("fpr", float("nan")),
        )

    logger.info("\n%s", "-" * 72)
    logger.info("  TIME-TO-DETECT (ROPs from anomaly onset to first TP)")
    logger.info("%s", "-" * 72)

    for key, ttd in ttd_stats.items():
        label = summary["models"].get(key, {}).get("label", key)
        logger.info(
            "%-25s  median=%.1f ROPs (%.0f min)  P90=%.1f ROPs (%.0f min)  missed=%.1f%%",
            label,
            ttd.get("median_rops", float("nan")),
            ttd.get("median_rops", float("nan")) * ROP_MINUTES,
            ttd.get("p90_rops",    float("nan")),
            ttd.get("p90_rops",    float("nan")) * ROP_MINUTES,
            ttd.get("missed_pct",  float("nan")),
        )

    logger.info("\n%s", "-" * 72)
    logger.info("  OPERATIONAL CONTEXT (Scale: %d cells, 3 shifts/day)", CELLS_PER_NOC_DOMAIN)
    logger.info("%s", "-" * 72)

    for key in ["baseline_static", "isolation_forest", "random_forest", "ensemble"]:
        m   = summary["models"].get(key, {})
        ctx = m.get("operational_context_at_max_f1", {})
        if not ctx:
            # Build a rough context for the baseline from default metrics
            dt  = m.get("default_threshold_metrics", {})
            ctx = operational_context(
                precision=dt.get("precision", 0.5),
                recall=dt.get("recall", 0.5),
                f1=dt.get("f1", 0.5),
                model_name=m.get("label", key),
                anomaly_rate=actual_anomaly_rate,
            )
        logger.info("")
        logger.info("  [%s]", m.get("label", key))
        logger.info("  %s", ctx.get("note", ""))

    logger.info("\n%s\n", sep)


# ============================================================================
# Section 19 — Argument Parsing & Main
# ============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="04_evaluation.py — RAN KPI Anomaly Detection: Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing outputs from scripts 01-03",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation_outputs"),
        help="Directory for figures and JSON metrics",
    )
    parser.add_argument(
        "--n-boot",
        type=int,
        default=500,
        help="Number of bootstrap resamples for confidence intervals",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def main() -> None:
    """
    Orchestrate the full evaluation pipeline.

    Step 1  Load data (or regenerate synthetically if prior scripts absent)
    Step 2  Compute threshold sweep tables
    Step 3  Select operating points
    Step 4  Generate all 11 figures
    Step 5  Bootstrap confidence intervals
    Step 6  Error analysis
    Step 7  Compile and save JSON summary
    Step 8  Print text report
    """
    args = parse_args()

    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", args.output_dir.resolve())

    # ------------------------------------------------------------------
    # Step 1 — Load Data
    # ------------------------------------------------------------------
    data = load_evaluation_data(args.data_dir)
    df          = data["scores"]
    topology_df = data["topology"]

    logger.info(
        "Evaluation dataset: %d rows | %d cells | anomaly_rate=%.2f%%",
        len(df), df["cell_id"].nunique(),
        100.0 * df["anomaly_label"].mean(),
    )

    # ------------------------------------------------------------------
    # Step 2 — Threshold Sweep Tables
    # ------------------------------------------------------------------
    sweeps = compute_threshold_sweep_table(df)

    # ------------------------------------------------------------------
    # Step 3 — Select Operating Points
    # ------------------------------------------------------------------
    operating_points = select_operating_points(sweeps, target_recall=0.90)

    # ------------------------------------------------------------------
    # Step 4 — Figures 1-2: PR / ROC curves
    # ------------------------------------------------------------------
    ap_scores  = plot_precision_recall_curves(df, args.output_dir)
    auc_scores = plot_roc_curves(df, args.output_dir)

    # ------------------------------------------------------------------
    # Figure 3: Confusion Matrices
    # ------------------------------------------------------------------
    plot_confusion_matrices(df, args.output_dir)

    # ------------------------------------------------------------------
    # Figure 4: Time-to-Detect
    # ------------------------------------------------------------------
    ttd_stats = plot_time_to_detect(df, args.output_dir)

    # ------------------------------------------------------------------
    # Figure 5: Per-cell FAR
    # ------------------------------------------------------------------
    plot_per_cell_false_alarm_rate(df, topology_df, args.output_dir)

    # ------------------------------------------------------------------
    # Figure 6: Anomaly-type breakdown
    # ------------------------------------------------------------------
    anomaly_type_breakdown = plot_anomaly_type_breakdown(df, args.output_dir)

    # ------------------------------------------------------------------
    # Figure 7: Time-series overlay
    # ------------------------------------------------------------------
    plot_time_series_overlay(df, args.output_dir, n_cells=3)

    # ------------------------------------------------------------------
    # Figure 8: Bootstrap CI  (computationally heavier — logged)
    # ------------------------------------------------------------------
    logger.info("Starting bootstrap CI computation (n_boot=%d) …", args.n_boot)
    ci_results = plot_bootstrap_ci_summary(df, args.output_dir, n_boot=args.n_boot)

    # ------------------------------------------------------------------
    # Figure 9: Threshold sweep
    # ------------------------------------------------------------------
    plot_threshold_sweep(sweeps, args.output_dir)

    # ------------------------------------------------------------------
    # Step 5 — Compile Metrics Summary
    # ------------------------------------------------------------------
    summary = compile_metrics_summary(
        df=df,
        sweeps=sweeps,
        operating_points=operating_points,
        ap_scores=ap_scores,
        auc_scores=auc_scores,
        ttd_stats=ttd_stats,
        ci_results=ci_results,
        anomaly_type_breakdown=anomaly_type_breakdown,
    )

    # ------------------------------------------------------------------
    # Figure 10: Operational Scorecard (requires compiled summary)
    # ------------------------------------------------------------------
    plot_operational_scorecard(summary, args.output_dir)

    # ------------------------------------------------------------------
    # Step 6 — Error Analysis + Figure 11
    # ------------------------------------------------------------------
    error_analysis = run_error_analysis(df, topology_df, args.output_dir)

    # ------------------------------------------------------------------
    # Step 7 — Save JSON
    # ------------------------------------------------------------------
    save_metrics_json(summary, error_analysis, ci_results, operating_points, args.output_dir)

    # ------------------------------------------------------------------
    # Step 8 — Text Report
    # ------------------------------------------------------------------
    print_evaluation_report(summary, operating_points, ttd_stats)

    logger.info(
        "Evaluation complete.  %d figures + 1 JSON saved to: %s",
        11, args.output_dir.resolve(),
    )


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    main()
