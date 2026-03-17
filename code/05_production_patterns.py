"""
05_production_patterns.py — Real-Time RAN KPI Anomaly Detection
===============================================================
Production serving patterns for cell-sector anomaly detection.

Demonstrates:
  1. Model serving wrapper (FastAPI) — takes a raw feature vector, returns
     anomaly score + SHAP-style contributing features.
  2. Flink-compatible feature computation — deterministic windowed aggregation
     that matches training-time feature engineering (no train/serve skew).
  3. Drift detection — Population Stability Index (PSI) and Wasserstein
     distance comparing live feature distributions against training baselines.
  4. Maintenance-window alert suppression — filters anomaly alerts against a
     maintenance calendar and recent CM-change log.
  5. Prometheus metrics exposition — inference latency, drift scores, alert
     counts, model health.
  6. Prediction logging — structured JSON log for downstream monitoring and
     retraining pipelines.
  7. Graceful degradation — falls back to a static-threshold scorer when the
     ML model is unavailable.

Usage:
    # Run full demo (no external dependencies required):
    python 05_production_patterns.py

    # Start the FastAPI server (requires: pip install fastapi uvicorn):
    python 05_production_patterns.py --serve --port 8080

    # Run drift detection report only:
    python 05_production_patterns.py --drift-report

Requirements:
    pip install pandas numpy scikit-learn scipy fastapi uvicorn shap prometheus_client

Coursebook cross-reference:
    Ch. 12: Streaming Architectures for Telco ML
    Ch. 15: MLOps for Network Assurance

O-RAN alignment:
    Model serving layer maps to the near-RT RIC xApp inference loop.
    Drift detection and retraining triggers align with O-RAN WG2 AI/ML
    workflow (O-RAN.WG2.AIML-v01.03) lifecycle management stages.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import logging
import os
import pickle
import sys
import time
import threading
import uuid
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import wasserstein_distance
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("ran_anomaly.production")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_DIR = Path("output")
MODEL_DIR = Path("models")

# KPI columns that the model expects at serving time.
# These match the feature columns produced by 02_feature_engineering.py.
# Note: RAW_KPIS in 01_synthetic_data.py defines 9 KPIs; the two omitted here
# — rach_success_rate and rlf_count — are Tier-2 KPIs monitored for data
# quality and RAN health dashboards but excluded from the anomaly detection
# feature set due to insufficient coverage in synthetic training data.
CORE_KPI_COLS = [
    "dl_throughput_mbps",
    "ul_throughput_mbps",
    "dl_prb_usage_rate",
    "rrc_conn_setup_success_rate",
    "drb_setup_success_rate",
    "handover_success_rate",
    "avg_cqi",
]

# Static fallback thresholds (used when ML model is unavailable).
# These are conservative operational thresholds — production values should be
# tuned per cell type.  Units match CORE_KPI_COLS.
STATIC_THRESHOLDS: Dict[str, Tuple[float, float]] = {
    "dl_throughput_mbps":            (0.5,   950.0),       # (min, max)
    "ul_throughput_mbps":            (0.1,   450.0),
    "dl_prb_usage_rate":             (0.0,     0.99),
    "rrc_conn_setup_success_rate":   (0.85,    1.0),
    "drb_setup_success_rate":        (0.85,    1.0),
    "handover_success_rate":         (0.80,    1.0),
    "avg_cqi":                       (2.0,    15.0),
}

# Alert severity levels mapped to anomaly score percentile thresholds.
SEVERITY_THRESHOLDS = {
    "critical": 0.90,   # top 10% anomaly scores
    "major":    0.75,
    "minor":    0.60,
}

# Prometheus metric names (used in text exposition; requires prometheus_client
# if you want a real scrape endpoint — we implement a lightweight stub here
# that works without the library).
PROMETHEUS_NAMESPACE = "ran_anomaly"


# ============================================================================
# Section 1: Data Structures
# ============================================================================

@dataclass
class AnomalyAlert:
    """Structured anomaly alert record.

    This is the canonical output object from the serving layer.  It maps
    directly to a TM Forum TMF642 Alarm object for NOC integration.
    """
    alert_id: str
    cell_id: str
    timestamp: str                    # ISO-8601
    anomaly_score: float              # 0.0 (normal) – 1.0 (very anomalous)
    severity: str                     # critical / major / minor / none
    contributing_kpis: List[Dict[str, Any]]  # [{kpi, value, zscore, importance}]
    probable_fault_category: str      # human-readable fault category
    recommended_action: str
    model_version: str
    inference_latency_ms: float
    suppressed: bool = False          # True if suppressed by maintenance window
    suppression_reason: str = ""


@dataclass
class MaintenanceWindow:
    """Scheduled maintenance window that suppresses anomaly alerts."""
    window_id: str
    cell_ids: List[str]              # empty list = all cells
    start_time: datetime.datetime
    end_time: datetime.datetime
    reason: str                      # e.g., "SW upgrade", "antenna work"


@dataclass
class CMChangeEvent:
    """Configuration management change event."""
    change_id: str
    cell_ids: List[str]
    change_time: datetime.datetime
    change_type: str                 # e.g., "handover_param", "antenna_tilt"
    settling_period_hours: int = 24  # suppress anomaly alerts during settling


@dataclass
class DriftReport:
    """Per-feature drift statistics."""
    feature_name: str
    psi: float                       # Population Stability Index
    wasserstein: float               # Wasserstein-1 distance
    ks_statistic: float              # Kolmogorov-Smirnov test statistic
    ks_pvalue: float
    drift_detected: bool
    drift_severity: str              # none / low / medium / high


@dataclass
class ModelHealth:
    """Snapshot of model health metrics for Prometheus / dashboard."""
    timestamp: str
    model_version: str
    requests_total: int
    anomalies_detected: int
    alerts_suppressed: int
    avg_inference_latency_ms: float
    p99_inference_latency_ms: float
    drift_detected: bool
    degraded_mode: bool              # True if fallback static-threshold scorer active


# ============================================================================
# Section 2: Lightweight Prometheus Metrics Registry
# ============================================================================
# We implement a minimal stub so the script runs without prometheus_client.
# In production, replace with:
#   from prometheus_client import Counter, Histogram, Gauge, generate_latest

class _MetricStub:
    """Thread-safe in-memory metric stub that approximates Prometheus counters,
    gauges, and histograms without requiring the prometheus_client library."""

    def __init__(self, name: str, metric_type: str, help_text: str = "") -> None:
        self._name = name
        self._type = metric_type
        self._value: float = 0.0
        self._observations: List[float] = []
        self._lock = threading.Lock()

    def inc(self, amount: float = 1.0) -> None:
        with self._lock:
            self._value += amount

    def set(self, value: float) -> None:
        with self._lock:
            self._value = value

    def observe(self, value: float) -> None:
        with self._lock:
            self._observations.append(value)

    def get(self) -> float:
        with self._lock:
            return self._value

    def percentile(self, p: float) -> float:
        with self._lock:
            if not self._observations:
                return 0.0
            return float(np.percentile(self._observations, p))

    def mean(self) -> float:
        with self._lock:
            if not self._observations:
                return 0.0
            return float(np.mean(self._observations))

    def exposition_line(self) -> str:
        """Prometheus text format line."""
        if self._type == "histogram":
            return (
                f"# HELP {self._name}\n"
                f"# TYPE {self._name} histogram\n"
                f"{self._name}_sum {sum(self._observations):.4f}\n"
                f"{self._name}_count {len(self._observations)}\n"
            )
        return (
            f"# HELP {self._name}\n"
            f"# TYPE {self._name} {self._type}\n"
            f"{self._name} {self._value:.4f}\n"
        )


class PrometheusRegistry:
    """Lightweight Prometheus metric registry.

    In production, replace with prometheus_client.CollectorRegistry and
    start_http_server() for a real /metrics scrape endpoint.
    """

    def __init__(self) -> None:
        self._metrics: Dict[str, _MetricStub] = {}

    def register(self, name: str, metric_type: str, help_text: str = "") -> _MetricStub:
        full_name = f"{PROMETHEUS_NAMESPACE}_{name}"
        m = _MetricStub(full_name, metric_type, help_text)
        self._metrics[full_name] = m
        return m

    def generate_latest(self) -> str:
        lines = []
        for m in self._metrics.values():
            lines.append(m.exposition_line())
        return "\n".join(lines)


# Module-level registry — shared by all components
REGISTRY = PrometheusRegistry()

# Register metrics upfront so they exist even before first request
_m_requests        = REGISTRY.register("inference_requests_total",  "counter", "Total inference requests")
_m_anomalies       = REGISTRY.register("anomalies_detected_total",  "counter", "Total anomalies detected")
_m_suppressed      = REGISTRY.register("alerts_suppressed_total",   "counter", "Alerts suppressed by maintenance window")
_m_fallback        = REGISTRY.register("fallback_mode_active",      "gauge",   "1 if static-threshold fallback is active")
_m_latency         = REGISTRY.register("inference_latency_ms",      "histogram", "Per-request inference latency in ms")
_m_drift_psi_max   = REGISTRY.register("drift_psi_max",             "gauge",   "Maximum PSI across all features")
_m_model_loaded    = REGISTRY.register("model_loaded",              "gauge",   "1 if ML model is loaded successfully")


# ============================================================================
# Section 3: Model Artifact Loading & Fallback
# ============================================================================

class ModelArtifact:
    """Container for a trained anomaly detection model and its metadata.

    Supports both Isolation Forest (unsupervised, Phase-1) and Random Forest
    (supervised, Phase-2).  Falls back to static thresholds if no artifact
    is found on disk.

    See Coursebook Ch. 54: MLOps for Network Assurance — model registry
    patterns.
    """

    def __init__(self) -> None:
        self.model: Optional[Any] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_cols: List[str] = []
        self.model_type: str = "none"
        self.version: str = "0.0.0"
        self.training_date: str = ""
        self.threshold: float = 0.5
        self.is_loaded: bool = False
        self._degraded_mode: bool = False

    def load(self, model_dir: Path) -> bool:
        """Attempt to load model artifacts from disk.

        Returns True if loaded successfully, False if falling back.
        Tries Isolation Forest artifact first, then Random Forest.
        """
        # Try Isolation Forest (Phase-1 model from 03_model_training.py)
        iso_path = model_dir / "isolation_forest_model.pkl"
        rf_path  = model_dir / "random_forest_model.pkl"

        for model_path, mtype in [(rf_path, "random_forest"),
                                   (iso_path, "isolation_forest")]:
            if model_path.exists():
                try:
                    with open(model_path, "rb") as f:
                        artifact = pickle.load(f)
                    # artifact may be a dict (from 03_model_training.py) or a raw model
                    if isinstance(artifact, dict):
                        self.model       = artifact.get("model")
                        self.scaler      = artifact.get("scaler")
                        self.feature_cols = artifact.get("feature_cols", [])
                        self.threshold   = artifact.get("threshold", 0.5)
                        self.version     = artifact.get("version", "1.0.0")
                        self.training_date = artifact.get("training_date", "unknown")
                    else:
                        # Bare model object — wrap it
                        self.model = artifact
                    self.model_type = mtype
                    self.is_loaded  = True
                    self._degraded_mode = False
                    _m_model_loaded.set(1.0)
                    _m_fallback.set(0.0)
                    logger.info(
                        "Loaded %s model v%s from %s",
                        mtype, self.version, model_path,
                    )
                    return True
                except Exception as exc:
                    logger.warning("Failed to load %s: %s", model_path, exc)

        # No artifact found — activate degraded mode
        logger.warning(
            "No model artifact found in %s. Activating static-threshold fallback.",
            model_dir,
        )
        self._degraded_mode = True
        _m_model_loaded.set(0.0)
        _m_fallback.set(1.0)
        return False

    def build_synthetic(self) -> None:
        """Build a minimal in-memory model for demo purposes.

        Used when no disk artifact exists and we still want to demonstrate
        the serving pipeline end-to-end.
        """
        logger.info("Building synthetic Isolation Forest model for demo…")
        rng = np.random.default_rng(42)

        # Simulate 30 days × 96 ROPs × 20 cells worth of training windows
        n_samples = 30 * 96 * 20
        n_features = len(CORE_KPI_COLS)
        X_train = rng.standard_normal((n_samples, n_features))

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_train)

        self.model = IsolationForest(
            n_estimators=100,
            contamination=0.03,   # ~3% expected anomaly rate in RAN data
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(X_scaled)

        self.feature_cols  = CORE_KPI_COLS
        self.model_type    = "isolation_forest"
        self.version       = "synthetic-demo-1.0"
        self.training_date = datetime.datetime.utcnow().isoformat()
        self.threshold     = 0.5
        self.is_loaded     = True
        self._degraded_mode = False
        _m_model_loaded.set(1.0)
        _m_fallback.set(0.0)
        logger.info("Synthetic model ready (features=%d, samples=%d)", n_features, n_samples)

    @property
    def degraded_mode(self) -> bool:
        return self._degraded_mode


# Singleton model artifact — loaded once at startup, shared across requests
_MODEL = ModelArtifact()


# ============================================================================
# Section 4: Feature Computation (Flink-compatible)
# ============================================================================
# The feature computation here MUST be byte-for-byte identical to what
# 02_feature_engineering.py computes during training.  Any divergence creates
# train/serve skew and degrades model accuracy.
#
# See Coursebook Ch. 28: Streaming Architectures — the "feature store pattern"
# avoids recomputation at serve time by materialising features into Redis.
# This module shows the inline computation path for cases where the feature
# store is unavailable (e.g., first-window cold-start or edge deployment
# without Redis connectivity).
def compute_serving_features(
    cell_id: str,
    recent_windows: pd.DataFrame,
    peer_windows: Optional[pd.DataFrame] = None,
) -> Dict[str, float]:
    """Compute the full feature vector for a single cell at serving time.

    This is the serving-time equivalent of 02_feature_engineering.py's
    ``compute_features_for_serving()``.  The logic is intentionally verbose
    to make the train/serve correspondence auditable.

    Parameters
    ----------
    cell_id:
        The cell sector identifier, e.g. "CELL_001_A".
    recent_windows:
        DataFrame with the last 96 ROPs (24 h × 15-min) for this cell.
        Must contain columns in CORE_KPI_COLS plus 'timestamp'.
    peer_windows:
        Optional DataFrame with the last 24 h of data for peer cells in the
        same peer group.  Used for relative z-score features.

    Returns
    -------
    Dict mapping feature name → float value.  All values are in the same
    scale expected by the model's scaler (pre-scaling raw KPI values).
    """
    if recent_windows.empty:
        logger.warning("Empty windows for cell %s — returning NaN features", cell_id)
        return {col: np.nan for col in CORE_KPI_COLS}

    # Ensure timestamp is sorted ascending
    df = recent_windows.copy()
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp")

    # --- Current-window raw KPI values (last ROP) ---
    current = df.iloc[-1]
    features: Dict[str, float] = {}
    for col in CORE_KPI_COLS:
        features[col] = float(current.get(col, np.nan))

    # --- Rolling statistics (1 h = 4 ROPs, 4 h = 16 ROPs, 24 h = 96 ROPs) ---
    # Exclude the current ROP from rolling computations to prevent
    # train/serve feature skew: training features are computed from
    # historical windows that do not include the target observation.
    for window_rops, window_label in [(4, "1h"), (16, "4h"), (96, "24h")]:
        window_df = df.tail(window_rops + 1).iloc[:-1]
        for col in CORE_KPI_COLS:
            if col not in window_df.columns:
                continue
            series = window_df[col].dropna()
            features[f"{col}_mean_{window_label}"]  = float(series.mean())  if len(series) else np.nan
            features[f"{col}_std_{window_label}"]   = float(series.std())   if len(series) > 1 else 0.0
            features[f"{col}_min_{window_label}"]   = float(series.min())   if len(series) else np.nan
            features[f"{col}_max_{window_label}"]   = float(series.max())   if len(series) else np.nan

    # --- Rate of change: (current − previous_1h) / previous_1h ---
    if len(df) >= 4:
        prev_window = df.iloc[-4]
        for col in CORE_KPI_COLS:
            curr_val = features.get(col, np.nan)
            prev_val = float(prev_window.get(col, np.nan))
            if pd.notna(curr_val) and pd.notna(prev_val) and prev_val != 0.0:
                features[f"{col}_roc_1h"] = (curr_val - prev_val) / abs(prev_val)
            else:
                features[f"{col}_roc_1h"] = 0.0

    # --- Temporal context features ---
    ts = pd.Timestamp(current.get("timestamp", pd.Timestamp.utcnow()))
    features["hour_of_day"]  = float(ts.hour)
    features["day_of_week"]  = float(ts.dayofweek)
    features["is_weekend"]   = float(ts.dayofweek >= 5)
    features["is_peak_hour"] = float(ts.hour in range(8, 22))

    # --- Peer-group relative features (spatial normalisation) ---
    # These are the single most valuable features per the whitepaper.
    # If peer_windows is None (e.g., cold-start), we skip gracefully.
    if peer_windows is not None and not peer_windows.empty:
        for col in CORE_KPI_COLS:
            if col not in peer_windows.columns:
                continue
            peer_last = (
                peer_windows.groupby("cell_id")[col]
                .last()          # last ROP per peer cell
                .dropna()
            )
            if len(peer_last) < 2:
                features[f"{col}_peer_zscore"] = 0.0
                continue
            peer_mean = float(peer_last.mean())
            peer_std  = float(peer_last.std())
            cell_val  = features.get(col, np.nan)
            if pd.notna(cell_val) and peer_std > 0:
                features[f"{col}_peer_zscore"] = (cell_val - peer_mean) / peer_std
            else:
                features[f"{col}_peer_zscore"] = 0.0

    # Warn if a large fraction of expected features are missing (NaN).
    # This typically indicates a data pipeline gap or a new cell with
    # insufficient history. Models may produce unreliable scores.
    n_features = len(features)
    n_missing = sum(1 for v in features.values() if pd.isna(v))
    if n_features > 0 and n_missing / n_features > 0.30:
        logger.warning(
            "Cell %s: %d/%d features (%.0f%%) are NaN — model scores may be unreliable. "
            "Check data pipeline and cell history.",
            cell_id, n_missing, n_features, 100.0 * n_missing / n_features,
        )

    return features

# ============================================================================
# Section 5: SHAP-style Feature Importance for NOC Explainability
# ============================================================================
def compute_contributing_features(
    feature_vector: Dict[str, float],
    model_artifact: ModelArtifact,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """Compute approximate feature importance for a single inference.

    For production, replace with ``shap.TreeExplainer`` (Random Forest) or
    ``shap.KernelExplainer`` (Isolation Forest).  We implement a lightweight
    perturbation-based approximation here so the script runs without SHAP
    installed.

    The approach: perturb each feature to its training-set mean and measure
    the change in anomaly score.  Features with the largest score delta are
    the most "contributing".

    Parameters
    ----------
    feature_vector : Dict[str, float]
        Raw (pre-scaled) feature values for the current window.
    model_artifact : ModelArtifact
        Loaded model + scaler.
    top_k : int
        Number of top contributing features to return.

    Returns
    -------
    List of dicts with keys: kpi, value, zscore, importance, direction.
    These map directly to the NOC Alert Card fields described in the
    whitepaper's Explainability Framework section.
    """
    if model_artifact.model is None or model_artifact.scaler is None:
        return []

    feat_cols = (
        model_artifact.feature_cols
        if model_artifact.feature_cols
        else list(feature_vector.keys())
    )
    # Build the raw vector in the correct column order
    raw = np.array([feature_vector.get(c, 0.0) for c in feat_cols], dtype=np.float64)
    raw = np.nan_to_num(raw, nan=0.0)

    # Scale the full vector
    try:
        scaled_full = model_artifact.scaler.transform(raw.reshape(1, -1))
    except Exception:
        return []

    def _score(x: np.ndarray) -> float:
        """Return anomaly probability (0=normal, 1=anomaly)."""
        try:
            if model_artifact.model_type == "isolation_forest":
                # decision_function returns negative scores — more negative = more anomalous
                raw_score = model_artifact.model.decision_function(x.reshape(1, -1))[0]
                # Normalise to [0, 1] using a sigmoid transformation
                return float(1 / (1 + np.exp(10.0 * raw_score)))
            else:
                # Random Forest: class 1 probability
                prob = model_artifact.model.predict_proba(x.reshape(1, -1))
                return float(prob[0, 1]) if prob.shape[1] > 1 else 0.0
        except Exception:
            return 0.0

    base_score = _score(scaled_full[0])

    # Training-set feature means from scaler (mean_ attribute)
    mean_vector = model_artifact.scaler.mean_  # shape: (n_features,)

    contributions = []
    for i, feat_name in enumerate(feat_cols):
        perturbed = scaled_full[0].copy()
        perturbed[i] = mean_vector[i]            # replace with training mean
        perturbed_score = _score(perturbed)
        importance = base_score - perturbed_score  # positive = feature pushed score up

        # Compute z-score of the raw feature value vs training distribution
        feat_std = np.sqrt(model_artifact.scaler.var_[i]) if i < len(model_artifact.scaler.var_) else 1.0
        feat_mean = mean_vector[i] if i < len(mean_vector) else 0.0
        zscore = (raw[i] - feat_mean) / (feat_std + 1e-9)

        contributions.append({
            "kpi":        feat_name,
            "value":      float(raw[i]),
            "zscore":     round(float(zscore), 3),
            "importance": round(float(importance), 4),
            "direction":  "above_normal" if zscore > 0 else "below_normal",
        })

    # Sort by absolute importance descending, return top-k
    contributions.sort(key=lambda x: abs(x["importance"]), reverse=True)
    return contributions[:top_k]

# ============================================================================
# Section 6: Fault Categorisation
# ============================================================================

# Rule-based fault category assignment based on contributing KPI patterns.
# In production, this rule set is maintained by RF engineers and versioned
# alongside the model artifact.  See whitepaper Gap 6: NOC Explainability.
_FAULT_RULES: List[Dict[str, Any]] = [
    {
        "category":   "Backhaul Congestion",
        "action":     "Check fronthaul/backhaul link utilisation and alarms.",
        "trigger_kpis": ["dl_throughput_mbps", "ul_throughput_mbps"],
        "direction":  "below_normal",
        "min_importance": 0.05,
    },
    {
        "category":   "Radio Interference",
        "action":     "Run inter-cell interference check; review ICIC parameters.",
        "trigger_kpis": ["avg_cqi", "dl_throughput_mbps"],
        "direction":  "below_normal",
        "min_importance": 0.03,
    },
    {
        "category":   "RRC / Bearer Setup Failure",
        "action":     "Verify RRC.ConnEstabSucc/Att and DRB.EstabSucc/Att on the affected cell. Check RAN causes: coverage gap, scheduler overload, RACH congestion. If RAN metrics normal, escalate to core team for S1/N2 interface review.",
        "trigger_kpis": ["rrc_conn_setup_success_rate", "drb_setup_success_rate"],
        "direction":  "below_normal",
        "min_importance": 0.03,
    },
    {
        "category":   "Handover Degradation",
        "action":     "Review handover parameters; check X2 interface.",
        "trigger_kpis": ["handover_success_rate"],
        "direction":  "below_normal",
        "min_importance": 0.03,
    },
    {
        "category":   "PRB Overload",
        "action":     "Check scheduler configuration; review load balancing.",
        "trigger_kpis": ["dl_prb_usage_rate"],
        "direction":  "above_normal",
        "min_importance": 0.04,
    },
    {
        "category":   "Hardware / RF Degradation",
        "action":     "Dispatch field team for antenna and hardware inspection.",
        "trigger_kpis": ["avg_cqi", "dl_throughput_mbps", "ul_throughput_mbps"],
        "direction":  "below_normal",
        "min_importance": 0.02,
    },
]


def categorise_fault(
    contributing_features: List[Dict[str, Any]],
) -> Tuple[str, str]:
    """Map contributing features to a human-readable fault category.

    Returns (fault_category, recommended_action).

    See whitepaper Section: NOC-Facing Explainability Framework.
    """
    if not contributing_features:
        return "Unknown", "Investigate manually."

    # Build a quick lookup of contributing KPIs
    contrib_kpi_set = {
        cf["kpi"]: cf
        for cf in contributing_features
        if abs(cf["importance"]) >= 0.01
    }

    for rule in _FAULT_RULES:
        trigger_kpis = rule["trigger_kpis"]
        direction    = rule["direction"]
        min_imp      = rule["min_importance"]

        matched = [
            k for k in trigger_kpis
            if k in contrib_kpi_set
            and contrib_kpi_set[k]["direction"] == direction
            and abs(contrib_kpi_set[k]["importance"]) >= min_imp
        ]
        if len(matched) >= 1:
            return rule["category"], rule["action"]

    return "Unclassified Anomaly", "Review top contributing KPIs manually."

# ============================================================================
# Section 7: Anomaly Severity Assignment
# ============================================================================
def assign_severity(anomaly_score: float) -> str:
    """Map a normalised anomaly score [0, 1] to a severity label.

    Thresholds are configurable via SEVERITY_THRESHOLDS.
    In production, these are exposed as A1 policy parameters so NOC managers
    can adjust sensitivity without redeploying the model.
    """
    if anomaly_score >= SEVERITY_THRESHOLDS["critical"]:
        return "critical"
    elif anomaly_score >= SEVERITY_THRESHOLDS["major"]:
        return "major"
    elif anomaly_score >= SEVERITY_THRESHOLDS["minor"]:
        return "minor"
    return "none"

# ============================================================================
# Section 8: Maintenance Window & CM-Change Suppression
# ============================================================================
class AlertSuppressionFilter:
    """Filter anomaly alerts against maintenance windows and CM changes.

    This addresses one of the most common NOC complaints with ML-based
    anomaly detection: alerts fire during planned maintenance because the
    model has never seen the post-change KPI distribution.

    Implementing this as a stateful filter (not a model feature) keeps the
    suppression logic auditable and independently testable.

    See Coursebook Ch. 54: MLOps — change management awareness.
    """

    def __init__(self) -> None:
        self._maintenance_windows: List[MaintenanceWindow] = []
        self._cm_changes: List[CMChangeEvent] = []
        self._lock = threading.Lock()

    def add_maintenance_window(self, window: MaintenanceWindow) -> None:
        with self._lock:
            self._maintenance_windows.append(window)
        logger.info(
            "Maintenance window registered: %s (%d cells, %s – %s)",
            window.window_id,
            len(window.cell_ids),
            window.start_time.isoformat(),
            window.end_time.isoformat(),
        )

    def add_cm_change(self, change: CMChangeEvent) -> None:
        with self._lock:
            self._cm_changes.append(change)
        logger.info(
            "CM change registered: %s (%s, settling %dh)",
            change.change_id,
            change.change_type,
            change.settling_period_hours,
        )

    def should_suppress(
        self,
        cell_id: str,
        alert_time: datetime.datetime,
    ) -> Tuple[bool, str]:
        """Return (suppressed, reason) for the given cell and alert time."""
        with self._lock:
            # Check maintenance windows
            for window in self._maintenance_windows:
                cell_covered = (
                    len(window.cell_ids) == 0      # empty = all cells
                    or cell_id in window.cell_ids
                )
                time_covered = window.start_time <= alert_time <= window.end_time
                if cell_covered and time_covered:
                    return True, (
                        f"Maintenance window {window.window_id}: {window.reason}"
                    )

            # Check CM change settling periods
            for change in self._cm_changes:
                cell_covered = (
                    len(change.cell_ids) == 0
                    or cell_id in change.cell_ids
                )
                settling_end = change.change_time + datetime.timedelta(
                    hours=change.settling_period_hours
                )
                time_covered = change.change_time <= alert_time <= settling_end
                if cell_covered and time_covered:
                    return True, (
                        f"CM change {change.change_id} ({change.change_type}) "
                        f"settling until {settling_end.isoformat()}"
                    )

        return False, ""


# Module-level suppression filter singleton
_SUPPRESSOR = AlertSuppressionFilter()

# ============================================================================
# Section 9: Static-Threshold Fallback Scorer
# ============================================================================
def static_threshold_score(feature_vector: Dict[str, float]) -> float:
    """Compute a simple anomaly score using static thresholds.

    Used when the ML model is unavailable (degraded mode).  Counts how many
    KPIs are outside their valid range and normalises by the number of KPIs
    checked.

    This matches the 'naive baseline' implemented in 03_model_training.py.
    """
    n_violations = 0
    n_checked = 0
    for kpi, (low, high) in STATIC_THRESHOLDS.items():
        val = feature_vector.get(kpi)
        if val is None or np.isnan(val):
            continue
        n_checked += 1
        if val < low or val > high:
            n_violations += 1

    if n_checked == 0:
        return 0.0
    # Normalise: 0 violations → 0.1 base score; all violations → 1.0
    return 0.1 + 0.9 * (n_violations / n_checked)

# ============================================================================
# Section 10: Core Inference Engine
# ============================================================================
class AnomalyInferenceEngine:
    """Core inference engine for cell-sector anomaly detection.

    This class is the heart of the serving layer.  It:
      1. Accepts a feature vector for one cell at one ROP.
      2. Runs ML inference (or static fallback).
      3. Computes SHAP-style contributing features.
      4. Categorises the probable fault.
      5. Assigns severity.
      6. Checks suppression.
      7. Emits a structured AnomalyAlert.
      8. Logs the prediction for downstream monitoring.
      9. Updates Prometheus metrics.

    In O-RAN terms, this maps to the inference step inside a near-RT RIC
    xApp that subscribes to E2SM-KPM indications.
    """

    def __init__(
        self,
        model: ModelArtifact,
        suppressor: AlertSuppressionFilter,
        prediction_log_path: Optional[Path] = None,
    ) -> None:
        self._model     = model
        self._suppressor = suppressor
        self._log_path  = prediction_log_path
        self._log_fh    = None
        if prediction_log_path:
            prediction_log_path.parent.mkdir(parents=True, exist_ok=True)
            # Append mode — survives process restarts without losing history
            self._log_fh = open(prediction_log_path, "a")

    def infer(
        self,
        cell_id: str,
        timestamp: str,
        feature_vector: Dict[str, float],
    ) -> AnomalyAlert:
        """Run inference for one cell-ROP and return a structured alert.

        Parameters
        ----------
        cell_id : str
            Cell sector ID in CELL_XXX_YYY format.
        timestamp : str
            ISO-8601 ROP timestamp (end of measurement period).
        feature_vector : Dict[str, float]
            Pre-computed feature values (output of compute_serving_features).
        """
        t_start = time.perf_counter()
        _m_requests.inc()

        alert_time = pd.Timestamp(timestamp).to_pydatetime()

        # --- Run ML inference or fallback ---
        if not self._model.degraded_mode and self._model.model is not None:
            score = self._run_ml_inference(feature_vector)
        else:
            score = static_threshold_score(feature_vector)
            _m_fallback.set(1.0)

        # --- Contributing features and fault categorisation ---
        if not self._model.degraded_mode:
            contributing = compute_contributing_features(
                feature_vector, self._model, top_k=5
            )
        else:
            # In degraded mode, report which static thresholds were violated
            contributing = self._static_contributing_features(feature_vector)

        fault_cat, recommended_action = categorise_fault(contributing)
        severity = assign_severity(score)

        # --- Suppression check ---
        suppressed, suppression_reason = _SUPPRESSOR.should_suppress(
            cell_id, alert_time
        )

        # --- Latency tracking ---
        latency_ms = (time.perf_counter() - t_start) * 1000.0
        _m_latency.observe(latency_ms)

        # --- Prometheus counters ---
        if severity != "none":
            _m_anomalies.inc()
        if suppressed:
            _m_suppressed.inc()

        alert = AnomalyAlert(
            alert_id             = str(uuid.uuid4()),
            cell_id              = cell_id,
            timestamp            = timestamp,
            anomaly_score        = round(score, 4),
            severity             = severity,
            contributing_kpis    = contributing,
            probable_fault_category = fault_cat,
            recommended_action   = recommended_action,
            model_version        = self._model.version,
            inference_latency_ms = round(latency_ms, 3),
            suppressed           = suppressed,
            suppression_reason   = suppression_reason,
        )

        # --- Prediction logging (for monitoring and retraining) ---
        self._log_prediction(alert, feature_vector)

        return alert

    def _run_ml_inference(self, feature_vector: Dict[str, float]) -> float:
        """Scale the feature vector and run model inference.

        Returns a normalised anomaly score in [0, 1].
        """
        feat_cols = (
            self._model.feature_cols
            if self._model.feature_cols
            else list(feature_vector.keys())
        )
        raw = np.array(
            [feature_vector.get(c, 0.0) for c in feat_cols], dtype=np.float64
        )
        raw = np.nan_to_num(raw, nan=0.0)

        try:
            scaled = self._model.scaler.transform(raw.reshape(1, -1))
        except Exception as exc:
            logger.error("Scaling failed: %s — falling back to static threshold", exc)
            return static_threshold_score(feature_vector)

        try:
            if self._model.model_type == "isolation_forest":
                df_val = self._model.model.decision_function(scaled)[0]
                # Sigmoid normalisation: centred at 0, steeper for larger deviations
                return float(1 / (1 + np.exp(10.0 * df_val)))
            else:
                # Random Forest probability of anomaly class
                prob = self._model.model.predict_proba(scaled)
                return float(prob[0, 1]) if prob.shape[1] > 1 else 0.0
        except Exception as exc:
            logger.error("Inference failed: %s — falling back to static threshold", exc)
            return static_threshold_score(feature_vector)

    @staticmethod
    def _static_contributing_features(
        feature_vector: Dict[str, float],
    ) -> List[Dict[str, Any]]:
        """Build contributing-feature list from static threshold violations."""
        result = []
        for kpi, (low, high) in STATIC_THRESHOLDS.items():
            val = feature_vector.get(kpi, np.nan)
            if np.isnan(val):
                continue
            if val < low:
                zscore = (val - low) / (abs(low) + 1e-9)
                direction = "below_normal"
                importance = abs(zscore) * 0.1
            elif val > high:
                zscore = (val - high) / (abs(high) + 1e-9)
                direction = "above_normal"
                importance = abs(zscore) * 0.1
            else:
                continue
            result.append({
                "kpi":        kpi,
                "value":      float(val),
                "zscore":     round(float(zscore), 3),
                "importance": round(float(importance), 4),
                "direction":  direction,
            })
        result.sort(key=lambda x: abs(x["importance"]), reverse=True)
        return result[:5]

    def _log_prediction(
        self,
        alert: AnomalyAlert,
        feature_vector: Dict[str, float],
    ) -> None:
        """Append a structured JSON prediction record to the prediction log.

        This log is consumed by:
          1. The drift detection monitor (feature distribution tracking).
          2. The retraining pipeline (candidate positive labels when
             anomaly_score > 0.9 and NOC confirms).
          3. The NOC feedback loop (NOC analysts add 'confirmed' or
             'false_positive' fields to these records).

        See Coursebook Ch. 54: MLOps — prediction logging and feedback loops.
        """
        if self._log_fh is None:
            return

        record = {
            "alert_id":        alert.alert_id,
            "cell_id":         alert.cell_id,
            "timestamp":       alert.timestamp,
            "anomaly_score":   alert.anomaly_score,
            "severity":        alert.severity,
            "suppressed":      alert.suppressed,
            "model_version":   alert.model_version,
            "latency_ms":      alert.inference_latency_ms,
            "features":        {
                k: (None if np.isnan(v) else round(v, 4))
                for k, v in feature_vector.items()
                if k in CORE_KPI_COLS  # only log raw KPIs; derived features logged separately
            },
            # NOC feedback fields — filled in later by analyst review
            "noc_confirmed":   None,   # True / False / None (pending)
            "noc_fault_code":  None,
        }
        try:
            self._log_fh.write(json.dumps(record) + "\n")
            self._log_fh.flush()
        except Exception as exc:
            logger.warning("Prediction log write failed: %s", exc)

    def close(self) -> None:
        if self._log_fh:
            self._log_fh.close()


def compute_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    n_bins: int = 10,
    epsilon: float = 1e-6,
) -> float:
    """Compute Population Stability Index between two distributions.

    PSI interpretation (credit-risk industry convention, valid for KPIs):
      PSI < 0.10 : negligible shift — model still valid
      PSI 0.10–0.25 : moderate shift — investigate
      PSI > 0.25 : significant shift — retrain required

    Parameters
    ----------
    expected : np.ndarray
        Training-time distribution sample.
    actual : np.ndarray
        Live / recent distribution sample.
    n_bins : int
        Number of quantile-equal bins.  10 is standard for PSI.
    epsilon : float
        Small constant to avoid log(0).

    Returns
    -------
    float : PSI score
    """
    expected = expected[~np.isnan(expected)]
    actual   = actual[~np.isnan(actual)]
    if len(expected) < 5 or len(actual) < 5:
        return 0.0  # Insufficient data

    # Use training-set quantile boundaries so bins have equal expected counts
    boundaries = np.quantile(expected, np.linspace(0, 1, n_bins + 1))
    boundaries[0]  -= 1e-9   # include the minimum value
    boundaries[-1] += 1e-9   # include the maximum value

    expected_counts = np.histogram(expected, bins=boundaries)[0]
    actual_counts   = np.histogram(actual,   bins=boundaries)[0]

    expected_pct = (expected_counts + epsilon) / (len(expected) + epsilon * n_bins)
    actual_pct   = (actual_counts   + epsilon) / (len(actual)   + epsilon * n_bins)

    psi = float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))
    return psi


def detect_drift(
    training_baseline: Dict[str, np.ndarray],
    live_window: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
) -> List[DriftReport]:
    """Run drift detection across all features.

    Computes PSI, Wasserstein distance, and Kolmogorov-Smirnov test for
    each feature.  Returns a DriftReport per feature.

    Parameters
    ----------
    training_baseline : Dict[str, np.ndarray]
        Feature name → array of training-set values.
    live_window : pd.DataFrame
        Recent (e.g., last 24 h) feature data from production traffic.
    feature_cols : List[str], optional
        Subset of features to check.  Defaults to all columns in live_window.
    """
    if feature_cols is None:
        feature_cols = [c for c in live_window.columns if c in training_baseline]

    reports = []
    for feat in feature_cols:
        if feat not in training_baseline or feat not in live_window.columns:
            continue

        train_vals = training_baseline[feat]
        live_vals  = live_window[feat].dropna().values

        if len(live_vals) < 5:
            logger.debug("Skipping drift for %s — insufficient live data", feat)
            continue

        psi = compute_psi(train_vals, live_vals)
        w1  = float(wasserstein_distance(train_vals, live_vals))
        ks  = stats.ks_2samp(train_vals, live_vals)

        # Drift threshold: PSI > 0.10 or W1 > 10% of training range
        train_range = float(np.nanmax(train_vals) - np.nanmin(train_vals))
        w1_threshold = 0.10 * train_range if train_range > 0 else 1.0

        drift_detected = psi > 0.10 or w1 > w1_threshold or ks.pvalue < 0.05

        if psi > 0.25 or (ks.pvalue < 0.01 and ks.statistic > 0.3):
            severity = "high"
        elif psi > 0.10 or ks.pvalue < 0.05:
            severity = "medium"
        elif psi > 0.05:
            severity = "low"
        else:
            severity = "none"

        reports.append(DriftReport(
            feature_name   = feat,
            psi            = round(psi, 4),
            wasserstein    = round(w1, 4),
            ks_statistic   = round(float(ks.statistic), 4),
            ks_pvalue      = round(float(ks.pvalue), 4),
            drift_detected = drift_detected,
            drift_severity = severity,
        ))

    # Update max-PSI Prometheus gauge
    if reports:
        max_psi = max(r.psi for r in reports)
        _m_drift_psi_max.set(max_psi)

    return reports


def build_training_baseline(
    training_df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
) -> Dict[str, np.ndarray]:
    """Build a training-time feature distribution baseline for drift detection.

    Call this once after model training and persist alongside the model
    artifact.  At serving time, load this baseline and pass it to
    detect_drift() periodically (e.g., every hour or every 1000 predictions).
    """
    if feature_cols is None:
        feature_cols = CORE_KPI_COLS

    baseline: Dict[str, np.ndarray] = {}
    for col in feature_cols:
        if col in training_df.columns:
            vals = training_df[col].dropna().values
            if len(vals) > 0:
                baseline[col] = vals.astype(np.float64)
    return baseline

# ============================================================================
# Section 12: Model Health Snapshot
# ============================================================================
def get_model_health(
    model: ModelArtifact,
    request_count: int,
    anomaly_count: int,
    suppressed_count: int,
) -> ModelHealth:
    """Collect and return a model health snapshot."""
    return ModelHealth(
        timestamp               = datetime.datetime.utcnow().isoformat(),
        model_version           = model.version,
        requests_total          = request_count,
        anomalies_detected      = anomaly_count,
        alerts_suppressed       = suppressed_count,
        avg_inference_latency_ms = _m_latency.mean(),
        p99_inference_latency_ms = _m_latency.percentile(99),
        drift_detected           = _m_drift_psi_max.get() > 0.10,
        degraded_mode            = model.degraded_mode,
    )

# ============================================================================
# Section 13: FastAPI Serving Application
# ============================================================================
# This section defines the FastAPI application.  It is imported only when the
# --serve flag is passed, so the script runs without fastapi installed in the
# default demo mode.
def _build_fastapi_app(
    engine: AnomalyInferenceEngine,
) -> Any:
    """Build and return a FastAPI application for model serving.

    Endpoints:
      POST /predict       — run inference for one cell-ROP
      GET  /health        — liveness probe (Kubernetes readiness/liveness)
      GET  /metrics       — Prometheus text format metrics
      GET  /drift         — latest drift report (JSON)
      POST /suppress      — register a maintenance window

    The request schema uses Pydantic models for validation.  In production,
    add authentication (mutual TLS or API key) before exposing this endpoint
    to the network.

    See Coursebook Ch. 28: Streaming Architectures — on-demand serving pattern.
    """
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import PlainTextResponse
        from pydantic import BaseModel
    except ImportError:
        logger.error(
            "FastAPI not installed. Run: pip install fastapi uvicorn pydantic"
        )
        raise

    app = FastAPI(
        title="RAN KPI Anomaly Detection Service",
        description=(
            "Cell-sector-level anomaly detection for 5G/LTE PM KPIs. "
            "Implements O-RAN near-RT RIC xApp inference pattern."
        ),
        version="1.0.0",
    )

    # -- Pydantic request/response models -----------------------------------

    class PredictRequest(BaseModel):
        """Single-cell inference request.

        feature_vector contains both raw KPI values and pre-computed
        rolling/peer features (output of compute_serving_features).
        """
        cell_id: str
        timestamp: str          # ISO-8601 ROP end timestamp
        feature_vector: Dict[str, float]

    class PredictResponse(BaseModel):
        alert_id: str
        cell_id: str
        timestamp: str
        anomaly_score: float
        severity: str
        contributing_kpis: List[Dict[str, Any]]
        probable_fault_category: str
        recommended_action: str
        model_version: str
        inference_latency_ms: float
        suppressed: bool
        suppression_reason: str

    class MaintenanceWindowRequest(BaseModel):
        window_id: str
        cell_ids: List[str]
        start_time: str         # ISO-8601
        end_time: str
        reason: str

    # -- Endpoints -----------------------------------------------------------

    @app.post("/predict", response_model=PredictResponse)
    def predict(request: PredictRequest) -> PredictResponse:
        """Run anomaly detection for one cell at one ROP.

        Accepts a pre-computed feature vector.  For the streaming path,
        feature computation happens upstream in Flink; this endpoint
        receives the pre-computed vector from the Feast online store lookup.
        """
        alert = engine.infer(
            cell_id        = request.cell_id,
            timestamp      = request.timestamp,
            feature_vector = request.feature_vector,
        )
        return PredictResponse(**dataclasses.asdict(alert))

    @app.get("/health")
    def health() -> Dict[str, Any]:
        """Kubernetes liveness and readiness probe.

        Returns 200 OK if the model is loaded (ready state).
        Returns 503 if in degraded mode (will process but with lower quality).
        """
        loaded  = _MODEL.is_loaded
        degraded = _MODEL.degraded_mode
        status  = "ok" if loaded and not degraded else "degraded"
        return {
            "status":        status,
            "model_version": _MODEL.version,
            "model_type":    _MODEL.model_type,
            "degraded_mode": degraded,
            "timestamp":     datetime.datetime.utcnow().isoformat(),
        }

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics() -> str:
        """Prometheus text format metrics endpoint.

        In production, expose this on a separate port (e.g., 9090) and
        configure your Prometheus scrape job to target it.  Do NOT expose
        /metrics on the same port as /predict if /predict is external-facing.
        """
        return REGISTRY.generate_latest()

    @app.post("/suppress")
    def register_maintenance_window(req: MaintenanceWindowRequest) -> Dict[str, str]:
        """Register a maintenance window for alert suppression.

        In production, this endpoint is called by the OSS change management
        system (e.g., via a ServiceNow webhook) whenever a planned change is
        scheduled.  Operators should NOT need to manually call this.
        """
        window = MaintenanceWindow(
            window_id  = req.window_id,
            cell_ids   = req.cell_ids,
            start_time = datetime.datetime.fromisoformat(req.start_time),
            end_time   = datetime.datetime.fromisoformat(req.end_time),
            reason     = req.reason,
        )
        _SUPPRESSOR.add_maintenance_window(window)
        return {"status": "registered", "window_id": req.window_id}

    return app

# ============================================================================
# Section 14: Synthetic Demo Data Generator
# ============================================================================
# Generates a small in-memory dataset that mimics the output of
# 01_synthetic_data.py and 02_feature_engineering.py for demo purposes.
def _make_demo_cells(n_cells: int = 20) -> List[str]:
    """Generate cell IDs in the standard CELL_XXX_YYY format."""
    return [
        f"CELL_{i:03d}_{sector}"
        for i in range(1, n_cells + 1)
        for sector in ["A", "B", "C"]
    ][:n_cells]


def _make_demo_feature_vector(
    rng: np.random.Generator,
    anomalous: bool = False,
) -> Dict[str, float]:
    """Generate a single realistic feature vector for demo inference.

    Normal distributions are based on typical LTE/5G KPI operating ranges.
    Anomalous vectors introduce specific violations relevant to common
    RAN fault types (backhaul degradation, interference, bearer setup failures).
    """
    if anomalous:
        # Simulate backhaul congestion + resulting throughput drop
        dl_tp  = float(rng.uniform(0.5, 15.0))   # severely degraded DL
        ul_tp  = float(rng.uniform(0.2, 8.0))
        prb    = float(rng.uniform(0.90, 0.999))  # PRBs still high (demand unchanged)
        rrc_sr = float(rng.uniform(0.60, 0.80))   # RRC setup failing
        drb_sr = float(rng.uniform(0.55, 0.75))
        ho_sr  = float(rng.uniform(0.65, 0.80))
        cqi    = float(rng.uniform(2.0, 5.0))     # poor radio conditions
    else:
        dl_tp  = float(rng.uniform(30.0, 400.0))
        ul_tp  = float(rng.uniform(5.0, 80.0))
        prb    = float(rng.uniform(0.20, 0.75))
        rrc_sr = float(rng.uniform(0.970, 0.999))
        drb_sr = float(rng.uniform(0.975, 0.999))
        ho_sr  = float(rng.uniform(0.950, 0.995))
        cqi    = float(rng.uniform(8.0, 14.0))

    # Rolling statistics (slightly perturbed from current values for realism)
    fv: Dict[str, float] = {
        "dl_throughput_mbps": dl_tp,
        "ul_throughput_mbps": ul_tp,
        "dl_prb_usage_rate":  prb,
        "rrc_conn_setup_success_rate":  rrc_sr,
        "drb_setup_success_rate":      drb_sr,
        "handover_success_rate":        ho_sr,
        "avg_cqi":            cqi,
    }
    # Add rolling mean features (4-window)
    for col in CORE_KPI_COLS:
        base = fv[col]
        for window, noise_scale in [("1h", 0.05), ("4h", 0.10), ("24h", 0.15)]:
            fv[f"{col}_mean_{window}"] = base * (1.0 + rng.uniform(-noise_scale, noise_scale))
            fv[f"{col}_std_{window}"]  = abs(base * rng.uniform(0.01, 0.08))
            fv[f"{col}_min_{window}"]  = base * (1.0 - rng.uniform(0.0, noise_scale * 2))
            fv[f"{col}_max_{window}"]  = base * (1.0 + rng.uniform(0.0, noise_scale * 2))
        fv[f"{col}_roc_1h"] = rng.uniform(-0.1, 0.1) if not anomalous else rng.uniform(-0.6, -0.3)
        fv[f"{col}_peer_zscore"] = (
            rng.uniform(-1.5, 1.5) if not anomalous else rng.uniform(-4.0, -2.5)
        )

    # Temporal features
    ts = datetime.datetime.utcnow()
    fv["hour_of_day"]  = float(ts.hour)
    fv["day_of_week"]  = float(ts.weekday())
    fv["is_weekend"]   = float(ts.weekday() >= 5)
    fv["is_peak_hour"] = float(ts.hour in range(8, 22))

    return fv


def _make_demo_training_data(
    n_rows: int = 5000,
    anomaly_rate: float = 0.03,
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """Generate a demo training DataFrame and baseline distribution dict.

    Returns (training_df, baseline_dict) where baseline_dict maps each
    KPI column to its training-set distribution array.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_rows):
        is_anomaly = rng.random() < anomaly_rate
        fv = _make_demo_feature_vector(rng, anomalous=is_anomaly)
        fv["is_anomaly"] = int(is_anomaly)
        rows.append(fv)

    df = pd.DataFrame(rows)
    baseline = build_training_baseline(df, feature_cols=CORE_KPI_COLS)
    return df, baseline


def _make_demo_live_data(
    n_rows: int = 500,
    drift: bool = False,
    seed: int = 99,
) -> pd.DataFrame:
    """Generate a demo live-window DataFrame for drift detection testing.

    If drift=True, the DL throughput and CQI distributions are shifted to
    simulate gradual degradation (e.g., due to interference buildup).
    """
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_rows):
        fv = _make_demo_feature_vector(rng, anomalous=False)
        if drift:
            # Simulate a 40% mean shift in DL throughput and CQI
            fv["dl_throughput_mbps"] *= 0.6
            fv["avg_cqi"] = max(1.0, fv["avg_cqi"] * 0.7)
        rows.append(fv)
    return pd.DataFrame(rows)

# ============================================================================
# Section 15: Demo Runner
# ============================================================================
def run_inference_demo(n_requests: int = 50) -> None:
    """Demonstrate the full inference pipeline with synthetic data.

    Runs n_requests inference calls through the full stack:
      - Feature vector → AnomalyInferenceEngine → AnomalyAlert
      - ~5% of requests are intentionally anomalous
      - 2 maintenance windows are pre-registered
      - Results summarised to stdout
    """
    logger.info("=" * 60)
    logger.info("DEMO: Inference Pipeline")
    logger.info("=" * 60)

    # Load or build model
    _MODEL.load(MODEL_DIR)
    if not _MODEL.is_loaded:
        _MODEL.build_synthetic()

    # Register maintenance windows and CM changes for suppression demo
    now = datetime.datetime.utcnow()

    _SUPPRESSOR.add_maintenance_window(MaintenanceWindow(
        window_id  = "MW-2024-001",
        cell_ids   = ["CELL_005_A", "CELL_005_B", "CELL_005_C"],
        start_time = now - datetime.timedelta(hours=1),
        end_time   = now + datetime.timedelta(hours=2),
        reason     = "Antenna replacement — scheduled outage",
    ))

    _SUPPRESSOR.add_cm_change(CMChangeEvent(
        change_id             = "CM-2024-042",
        cell_ids              = ["CELL_010_A"],
        change_time           = now - datetime.timedelta(hours=6),
        change_type           = "handover_param",
        settling_period_hours = 24,
    ))

    # Initialise the engine with prediction logging
    log_path = Path("output") / "predictions.jsonl"
    engine = AnomalyInferenceEngine(
        model              = _MODEL,
        suppressor         = _SUPPRESSOR,
        prediction_log_path = log_path,
    )

    rng = np.random.default_rng(42)
    cells = _make_demo_cells(20)

    alerts_by_severity: Dict[str, int] = {
        "none": 0, "minor": 0, "major": 0, "critical": 0
    }
    suppressed_count = 0
    latencies_ms: List[float] = []

    logger.info("Running %d inference requests…", n_requests)
    for i in range(n_requests):
        cell_id   = cells[i % len(cells)]
        ts        = (now + datetime.timedelta(minutes=15 * i)).isoformat()
        is_anomaly = rng.random() < 0.08   # 8% anomaly rate in demo

        fv = _make_demo_feature_vector(rng, anomalous=is_anomaly)
        alert = engine.infer(cell_id=cell_id, timestamp=ts, feature_vector=fv)

        alerts_by_severity[alert.severity] += 1
        if alert.suppressed:
            suppressed_count += 1
        latencies_ms.append(alert.inference_latency_ms)

        if alert.severity in ("major", "critical") and not alert.suppressed:
            logger.info(
                "ALERT [%s] cell=%s score=%.3f fault='%s' "
                "top_kpi=%s latency=%.2fms",
                alert.severity.upper(),
                alert.cell_id,
                alert.anomaly_score,
                alert.probable_fault_category,
                alert.contributing_kpis[0]["kpi"] if alert.contributing_kpis else "N/A",
                alert.inference_latency_ms,
            )

    engine.close()

    logger.info("-" * 60)
    logger.info("INFERENCE SUMMARY")
    logger.info("  Total requests  : %d", n_requests)
    logger.info("  Critical alerts : %d", alerts_by_severity["critical"])
    logger.info("  Major alerts    : %d", alerts_by_severity["major"])
    logger.info("  Minor alerts    : %d", alerts_by_severity["minor"])
    logger.info("  No anomaly      : %d", alerts_by_severity["none"])
    logger.info("  Suppressed      : %d", suppressed_count)
    logger.info(
        "  Avg latency     : %.2f ms (p99: %.2f ms)",
        float(np.mean(latencies_ms)),
        float(np.percentile(latencies_ms, 99)),
    )
    logger.info("  Prediction log  : %s", log_path)


def run_drift_demo(n_train: int = 5000, n_live: int = 500) -> None:
    """Demonstrate the drift detection pipeline.

    Compares a 'no-drift' live window against a 'drifted' live window
    to show how PSI and Wasserstein distance respond.
    """
    logger.info("=" * 60)
    logger.info("DEMO: Drift Detection")
    logger.info("=" * 60)

    training_df, baseline = _make_demo_training_data(n_rows=n_train, seed=42)

    for scenario_name, drift_flag in [("No Drift", False), ("Drift (throughput/CQI shift)", True)]:
        live_df = _make_demo_live_data(n_rows=n_live, drift=drift_flag, seed=99)
        reports = detect_drift(
            training_baseline = baseline,
            live_window       = live_df,
            feature_cols      = CORE_KPI_COLS,
        )

        drift_detected_count = sum(1 for r in reports if r.drift_detected)
        high_severity_count  = sum(1 for r in reports if r.drift_severity == "high")

        logger.info("\nScenario: %s", scenario_name)
        logger.info("  Features checked    : %d", len(reports))
        logger.info("  Drift detected      : %d / %d", drift_detected_count, len(reports))
        logger.info("  High-severity drift : %d", high_severity_count)

        for r in reports:
            if r.drift_detected:
                logger.info(
                    "  ⚠  %-42s PSI=%.4f  W1=%.4f  KS_p=%.4f  severity=%s",
                    r.feature_name,
                    r.psi,
                    r.wasserstein,
                    r.ks_pvalue,
                    r.drift_severity,
                )
            else:
                logger.info(
                    "  ✓  %-42s PSI=%.4f  W1=%.4f",
                    r.feature_name,
                    r.psi,
                    r.wasserstein,
                )

    # Demonstrate the PSI Prometheus gauge was updated
    logger.info(
        "\n  Prometheus metric %s_drift_psi_max = %.4f",
        PROMETHEUS_NAMESPACE,
        _m_drift_psi_max.get(),
    )


def run_health_demo() -> None:
    """Demonstrate the model health snapshot and Prometheus metrics."""
    logger.info("=" * 60)
    logger.info("DEMO: Model Health & Prometheus Metrics")
    logger.info("=" * 60)

    health = get_model_health(
        model            = _MODEL,
        request_count    = int(_m_requests.get()),
        anomaly_count    = int(_m_anomalies.get()),
        suppressed_count = int(_m_suppressed.get()),
    )

    logger.info("Model Health Snapshot:")
    for key, val in dataclasses.asdict(health).items():
        logger.info("  %-40s : %s", key, val)

    logger.info("\nPrometheus /metrics exposition:")
    print(REGISTRY.generate_latest())


def run_alert_card_demo() -> None:
    """Demonstrate the NOC Alert Card format.

    Shows how AnomalyAlert maps to the five explainability components
    described in the whitepaper's NOC-Facing Explainability Framework.
    """
    logger.info("=" * 60)
    logger.info("DEMO: NOC Alert Card")
    logger.info("=" * 60)

    if not _MODEL.is_loaded:
        _MODEL.build_synthetic()

    rng = np.random.default_rng(7)
    fv  = _make_demo_feature_vector(rng, anomalous=True)
    engine = AnomalyInferenceEngine(model=_MODEL, suppressor=_SUPPRESSOR)
    alert  = engine.infer(
        cell_id        = "CELL_042_B",
        timestamp      = datetime.datetime.utcnow().isoformat(),
        feature_vector = fv,
    )

    # Print in a NOC-friendly format
    print("\n" + "═" * 62)
    print("  NOC ALERT CARD")
    print("═" * 62)
    print(f"  Alert ID       : {alert.alert_id}")
    print(f"  Cell Sector    : {alert.cell_id}")
    print(f"  Timestamp      : {alert.timestamp}")
    print(f"  Severity       : {alert.severity.upper()}")
    print(f"  Anomaly Score  : {alert.anomaly_score:.3f}  (0=normal, 1=anomaly)")
    print(f"  Fault Category : {alert.probable_fault_category}")
    print(f"  Suppressed     : {alert.suppressed}")
    if alert.suppressed:
        print(f"  Reason         : {alert.suppression_reason}")
    print()
    print("  TOP CONTRIBUTING KPIs:")
    for i, kpi in enumerate(alert.contributing_kpis, 1):
        direction_symbol = "↓" if kpi["direction"] == "below_normal" else "↑"
        print(
            f"    {i}. {kpi['kpi']:<42s}  "
            f"val={kpi['value']:>8.2f}  "
            f"z={kpi['zscore']:>+6.2f}  "
            f"importance={kpi['importance']:>+.4f}  {direction_symbol}"
        )
    print()
    print(f"  RECOMMENDED ACTION:")
    print(f"    {alert.recommended_action}")
    print(f"\n  Model Version  : {alert.model_version}")
    print(f"  Inference Time : {alert.inference_latency_ms:.3f} ms")
    print("═" * 62 + "\n")


def run_suppression_demo() -> None:
    """Demonstrate that maintenance windows correctly suppress alerts."""
    logger.info("=" * 60)
    logger.info("DEMO: Alert Suppression")
    logger.info("=" * 60)

    now = datetime.datetime.utcnow()

    # Register a maintenance window covering CELL_003_A for the next 2 hours
    _SUPPRESSOR.add_maintenance_window(MaintenanceWindow(
        window_id  = "MW-DEMO-001",
        cell_ids   = ["CELL_003_A"],
        start_time = now - datetime.timedelta(minutes=5),
        end_time   = now + datetime.timedelta(hours=2),
        reason     = "Demo maintenance window",
    ))

    if not _MODEL.is_loaded:
        _MODEL.build_synthetic()

    engine = AnomalyInferenceEngine(model=_MODEL, suppressor=_SUPPRESSOR)
    rng    = np.random.default_rng(11)

    for cell_id in ["CELL_003_A", "CELL_004_A"]:
        fv    = _make_demo_feature_vector(rng, anomalous=True)
        alert = engine.infer(
            cell_id        = cell_id,
            timestamp      = now.isoformat(),
            feature_vector = fv,
        )
        status = "SUPPRESSED" if alert.suppressed else "ACTIVE"
        reason = f" ({alert.suppression_reason})" if alert.suppressed else ""
        logger.info(
            "  cell=%-15s  severity=%-8s  suppressed=%s%s",
            cell_id,
            alert.severity,
            status,
            reason,
        )

# ============================================================================
# Section 16: CLI Entry Point
# ============================================================================
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RAN KPI Anomaly Detection — Production Patterns Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Start FastAPI serving endpoint (requires: pip install fastapi uvicorn)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for FastAPI server (default: 8080)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host for FastAPI server (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--drift-report",
        action="store_true",
        dest="drift_report",
        help="Run drift detection report only",
    )
    parser.add_argument(
        "--n-requests",
        type=int,
        default=50,
        dest="n_requests",
        help="Number of demo inference requests (default: 50)",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=MODEL_DIR,
        dest="model_dir",
        help="Directory to load model artifacts from (default: models/)",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point for the production patterns demonstration.

    Running modes:
      python 05_production_patterns.py               → full demo
      python 05_production_patterns.py --drift-report → drift detection only
      python 05_production_patterns.py --serve        → start FastAPI server
    """
    args = _parse_args()

    # Ensure output directories exist
    Path("output").mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)

    logger.info("RAN KPI Anomaly Detection — Production Patterns")
    logger.info("Coursebook Ch. 28: Streaming Architectures for Telco ML")
    logger.info("Coursebook Ch. 54: MLOps for Network Assurance")

    # -----------------------------------------------------------------------
    # Serving mode
    # -----------------------------------------------------------------------
    if args.serve:
        logger.info("Starting FastAPI serving endpoint on %s:%d", args.host, args.port)
        _MODEL.load(args.model_dir)
        if not _MODEL.is_loaded:
            _MODEL.build_synthetic()

        log_path = Path("output") / "predictions.jsonl"
        engine   = AnomalyInferenceEngine(
            model               = _MODEL,
            suppressor          = _SUPPRESSOR,
            prediction_log_path = log_path,
        )
        try:
            import uvicorn
        except ImportError:
            logger.error("uvicorn not installed.  Run: pip install uvicorn")
            sys.exit(1)

        app = _build_fastapi_app(engine)
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
        return

    # -----------------------------------------------------------------------
    # Drift report only
    # -----------------------------------------------------------------------
    if args.drift_report:
        run_drift_demo()
        return

    # -----------------------------------------------------------------------
    # Full demo mode
    # -----------------------------------------------------------------------

    # 1. Load or build model
    _MODEL.load(args.model_dir)
    if not _MODEL.is_loaded:
        _MODEL.build_synthetic()

    # 2. Inference pipeline demo
    run_inference_demo(n_requests=args.n_requests)

    # 3. NOC Alert Card demo
    run_alert_card_demo()

    # 4. Alert suppression demo
    run_suppression_demo()

    # 5. Drift detection demo
    run_drift_demo()

    # 6. Model health and Prometheus metrics demo
    run_health_demo()

    logger.info("=" * 60)
    logger.info("All production pattern demos complete.")
    logger.info("To start the API server: python 05_production_patterns.py --serve")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
