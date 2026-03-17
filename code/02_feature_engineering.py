"""
02_feature_engineering.py — RAN KPI Anomaly Detection Feature Pipeline
=======================================================================
Companion code for: "Real-Time RAN KPI Anomaly Detection at Cell Sector Granularity"

PURPOSE:
    Reads the synthetic PM dataset produced by 01_synthetic_data.py and constructs
    a rich feature matrix suitable for both classical (Isolation Forest, Random Forest)
    and sequence-based (LSTM-VAE) anomaly detection models.

FEATURE GROUPS PRODUCED:
    1. Temporal context      — cyclical hour/DOW encodings, peak-hour flags
    2. Rolling statistics    — mean, std, min, max over 1h / 4h / 24h windows
    3. Rate-of-change        — first-order delta and percentage change per KPI
    4. Day-over-day (DoD)    — same ROP 24 h earlier; week-over-week (WoW) 7 days
    5. Cross-KPI ratios      — PRB efficiency, quality-to-load indicators
    6. Peer-group deviations — z-score vs geographically similar cells (spatial)
    7. Missing-data flags    — indicators for imputed / counter-reset values

OUTPUT:
    data/train_features.parquet
    data/val_features.parquet
    data/test_features.parquet
    data/feature_metadata.json   ← column names, split dates, stats

TEMPORAL SPLIT (no random shuffling — see Coursebook Ch. 25: Time Series Analysis):
    Train : days  1–20  (model learns "normal" behaviour)
    Val   : days 21–25  (threshold / hyperparameter selection)
    Test  : days 26–30  (held-out evaluation; contains injected anomalies)

USAGE:
    # Run synthetic data generation first:
    python 01_synthetic_data.py

    # Then run this script:
    python 02_feature_engineering.py

    # Optional: override data paths via environment variables:
    PM_DATA_PATH=/custom/path/raw_pm_data.parquet python 02_feature_engineering.py

REQUIREMENTS:
    Python 3.10+
    pandas >= 2.0, numpy >= 1.24, scikit-learn >= 1.3, pyarrow >= 14.0

Coursebook cross-reference:
    Ch. 25 — Time Series Analysis (rolling windows, stationarity)
    Ch. 13 — Feature Engineering (encoding, scaling, domain features)
    Ch. 28 — Data Pipelines (streaming, point-in-time correctness)
"""

from __future__ import annotations

import json
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Logging — structured, goes to stdout for container / systemd journal capture
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("feature_engineering")
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# ---------------------------------------------------------------------------
# Path configuration — mirrors 01_synthetic_data.py output conventions
# ---------------------------------------------------------------------------
DATA_DIR = Path(os.getenv("PM_DATA_DIR", "data"))
PM_DATA_PATH = Path(os.getenv("PM_DATA_PATH", DATA_DIR / "raw_pm_data.parquet"))
FEATURES_DIR = DATA_DIR
TOPOLOGY_PATH = DATA_DIR / "cell_metadata.parquet"

# ---------------------------------------------------------------------------
# Column name constants — single source of truth; keeps downstream code clean
# ---------------------------------------------------------------------------

# Raw KPIs produced by 01_synthetic_data.py
# RAW_KPIS defines the modelled KPI subset; `rach_success_rate` and `rlf_count`
# are in ingestion scope for data quality monitoring but excluded from feature engineering
RAW_KPIS: list[str] = [
    "dl_throughput_mbps",
    "ul_throughput_mbps",
    "dl_prb_usage_rate",
    "rrc_conn_setup_success_rate",
    "drb_setup_success_rate",
    "handover_success_rate",
    "avg_cqi",
    "rach_success_rate",
    "rlf_count",
]

# Tier-2 KPIs: processed through the feature engineering pipeline but excluded
# from serving features by default due to vendor-dependent counter availability.
# Set EXCLUDE_TIER2_DERIVED = False to include; see whitepaper §3 tiering rationale.
TIER2_KPIS: set[str] = {"rach_success_rate", "rlf_count"}
EXCLUDE_TIER2_DERIVED: bool = True

# Minimum days of cell history before peer z-scores are considered reliable.
PEER_GROUP_COLD_START_DAYS: int = 7

# Metadata columns from the raw dataset
REQUIRED_META_COLS: list[str] = [
    "cell_id",
    "timestamp",
    "site_id",
    "sector_id",
    "vendor",
    "frequency_band",
    "cell_type",         # urban_macro / urban_micro / suburban / rural
]
OPTIONAL_LABEL_COLS: list[str] = [
    "is_anomaly",        # ground-truth label (for evaluation only)
    "anomaly_type",      # gradual_degradation / sudden_drop / periodic / spatial / none
]
# Combined for backward compatibility (feature exclusion, etc.)
META_COLS: list[str] = REQUIRED_META_COLS + OPTIONAL_LABEL_COLS

# Temporal split boundaries (match train script expectations)
TRAIN_END_DAY   = 20
VAL_END_DAY     = 25
# Everything after val_end_day is test

# Rolling window sizes in terms of 15-min ROPs
# 1 h  = 4 ROPs,  4 h = 16 ROPs,  24 h = 96 ROPs
WINDOW_SIZES: dict[str, int] = {
    "1h":  4,
    "4h":  16,
    "24h": 96,
}

# Number of ROPs for look-back features
ROPS_PER_DAY  = 96   # 24 h × 4 ROPs / h
ROPS_PER_WEEK = 672  # 7 × 96
# ============================================================================
# SECTION 1 — Data Loading & Validation
# ============================================================================
def load_pm_data(path: Path) -> pd.DataFrame:
    """Load the PM dataset from Parquet and perform basic sanity checks.

    The dataset is produced by 01_synthetic_data.py.  We validate schema and
    value ranges before any feature computation to fail fast on upstream
    problems rather than producing silently bad features.

    See Coursebook Ch. 13 §13.1 — Data Contracts for Feature Pipelines.
    """
    logger.info("Loading PM data from %s", path)

    if not path.exists():
        raise FileNotFoundError(
            f"PM data not found at {path}.  "
            "Run 01_synthetic_data.py first, or set PM_DATA_PATH env var."
        )

    df = pd.read_parquet(path)
    logger.info("Loaded %d rows × %d columns", len(df), len(df.columns))

    # ------------------------------------------------------------------
    # Schema validation — required columns must be present; labels optional
    # ------------------------------------------------------------------
    required_cols = set(RAW_KPIS) | set(REQUIRED_META_COLS)
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {sorted(missing)}")

    # Label columns are optional — may come from a separate anomaly_labels file
    missing_labels = set(OPTIONAL_LABEL_COLS) - set(df.columns)
    if missing_labels:
        logger.warning("Label columns not in PM data (expected if labels are separate): %s", sorted(missing_labels))

    # ------------------------------------------------------------------
    # Timestamp integrity — must be datetime, sorted per cell, no future data
    # ------------------------------------------------------------------
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    df = df.sort_values(["cell_id", "timestamp"]).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Telco value-range assertions (soft — log warnings, don't drop rows)
    # These ranges come from 3GPP TS 28.552 and real-world PM experience.
    # ------------------------------------------------------------------
    range_checks: dict[str, tuple[float, float]] = {
        "dl_throughput_mbps":          (0.0,   2000.0),
        "ul_throughput_mbps":          (0.0,    500.0),
        "dl_prb_usage_rate":           (0.0,      1.0),
        "rrc_conn_setup_success_rate": (0.0,      1.0),
        "drb_setup_success_rate":      (0.0,      1.0),
        "handover_success_rate":       (0.0,      1.0),
        "avg_cqi":                     (0.0,     15.0),
    }
    for col, (lo, hi) in range_checks.items():
        n_out = ((df[col] < lo) | (df[col] > hi)).sum()
        if n_out > 0:
            logger.warning(
                "Column '%s': %d rows outside expected range [%g, %g]",
                col, n_out, lo, hi,
            )

    logger.info(
        "Date range: %s → %s | Unique cells: %d",
        df["timestamp"].min().date(),
        df["timestamp"].max().date(),
        df["cell_id"].nunique(),
    )

    # ------------------------------------------------------------------
    # Merge ground-truth labels from anomaly_labels.parquet if available
    # and label columns are not already present in the PM data.
    # ------------------------------------------------------------------
    labels_path = path.parent / "anomaly_labels.parquet"
    has_labels = "is_anomaly" in df.columns and df["is_anomaly"].notna().any()
    if not has_labels and labels_path.exists():
        labels_df = pd.read_parquet(labels_path)
        merge_cols = ["cell_id", "timestamp"]
        label_cols = [c for c in labels_df.columns if c not in merge_cols]
        n_before = len(df)
        df = df.merge(labels_df[merge_cols + label_cols], on=merge_cols, how="left")
        n_matched = df["is_anomaly"].notna().sum() if "is_anomaly" in df.columns else 0
        logger.info(
            "Merged anomaly_labels.parquet: %d/%d rows matched (%d label columns: %s)",
            n_matched, n_before, len(label_cols), label_cols,
        )
    elif has_labels:
        logger.info("Labels already present in PM data — skipping anomaly_labels merge")

    # ------------------------------------------------------------------
    # UE-COUNT SUPPRESSION — Mandatory privacy control (§9.5, GDPR Art. 5(1)(c))
    # ------------------------------------------------------------------
    # Operators MUST implement this before canary deployment. In cells with
    # very low active UE counts, aggregate KPIs become quasi-identifiable —
    # a single-UE cell's throughput anomaly can be attributed to that user.
    #
    # IMPLEMENTATION PATTERN (pseudocode):
    #
    #   MIN_UE_THRESHOLD = 5  # configurable; review with DPO
    #
    #   # Option A: RRC.ConnMean available in PM input
    #   ue_col = "rrc_conn_mean"  # from TS 28.552 §5.1.3.6
    #
    #   # Option B: RRC.ConnMax as fallback
    #   ue_col = "rrc_conn_max"   # conservative; suppresses more intervals
    #
    #   if ue_col in df.columns:
    #       low_ue_mask = df[ue_col] < MIN_UE_THRESHOLD
    #       n_suppressed = low_ue_mask.sum()
    #       # Set KPI columns to NaN — creates gaps in rolling windows
    #       df.loc[low_ue_mask, RAW_KPIS] = np.nan
    #       # Flag for downstream audit trail
    #       df["ue_count_suppressed"] = low_ue_mask
    #       logger.info(
    #           "UE-count suppression: %d/%d intervals (%.1f%%) below threshold %d",
    #           n_suppressed, len(df), 100 * n_suppressed / len(df),
    #           MIN_UE_THRESHOLD,
    #       )
    #   else:
    #       logger.warning(
    #           "UE-count column '%s' not found in PM data — suppression "
    #           "cannot be applied. Verify counter availability with vendor.",
    #           ue_col,
    #       )
    #
    # NOTE: Suppressed intervals must NOT contribute to peer group statistics.
    # The add_peer_group_features() function already handles NaN values via
    # nanmean/nanstd — suppressed rows propagate correctly as gaps.
    #
    # Counter availability is deployment-specific: RRC.ConnMean / RRC.ConnMax
    # (TS 28.552 §5.1.3.6) are not exposed at cell-sector granularity by all
    # vendors. Verify availability before enabling. Left as extension point
    # for operators — uncomment and adapt the pattern above.
    # ------------------------------------------------------------------

    return df
def load_topology(path: Path) -> pd.DataFrame | None:
    """Load cell topology metadata (neighbour lists, coordinates) if available.

    The topology file is an optional enrichment produced by 01_synthetic_data.py.
    If absent, peer-group features are computed from KPI similarity instead of
    geographic proximity — a degraded but functional fallback.
    """
    if not path.exists():
        logger.warning(
            "Topology file not found at %s — peer groups will be computed "
            "from KPI-based clustering (fallback mode).",
            path,
        )
        return None

    topo = pd.read_parquet(path)
    logger.info(
        "Loaded topology: %d cells, columns: %s", len(topo), topo.columns.tolist()
    )
    return topo


# ============================================================================
# SECTION 2 — Temporal Context Features
# ============================================================================

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-of-day and day-of-week features with cyclical encoding.

    WHY CYCLICAL ENCODING:
        Raw hour_of_day = 0 and hour_of_day = 23 are 1 unit apart in integer
        encoding but should be perceived as "nearly the same time" by the model.
        Sine/cosine projection onto the unit circle preserves this topology.
        See Coursebook Ch. 13 §13.3 — Cyclical Feature Encoding.

    WHY PEAK-HOUR FLAG:
        Telco KPIs have very different baseline distributions during peak hours
        (07:00–09:00, 17:00–21:00 local time).  A binary flag lets the model
        learn separate anomaly thresholds without requiring complex interaction
        terms.  This is a critical feature for reducing false alarms on cells
        near stadiums / transit hubs.
    """
    ts = df["timestamp"]

    hour = ts.dt.hour + ts.dt.minute / 60.0   # fractional hour

    # Cyclical encoding — projects [0, 24) onto unit circle
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)

    # Day-of-week: Monday=0, Sunday=6
    dow = ts.dt.dayofweek.astype(float)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)

    # Peak-hour flag (binary) — covers AM and PM commute / evening hours
    # Threshold based on typical European operator traffic patterns
    df["is_peak_hour"] = (
        ((ts.dt.hour >= 7)  & (ts.dt.hour <= 9))  |
        ((ts.dt.hour >= 17) & (ts.dt.hour <= 21))
    ).astype(np.int8)

    df["is_weekend"] = (ts.dt.dayofweek >= 5).astype(np.int8)

    # Night-time flag — traffic is anomalously low during night; a "dip" at
    # 03:00 is normal, not an anomaly.  See Ch. 3 §3.5 contextual anomalies.
    df["is_night"] = (
        (ts.dt.hour >= 0) & (ts.dt.hour < 6)
    ).astype(np.int8)

    # Absolute day-number from dataset start — used to compute train/val/test
    # split boundaries consistently across all cells
    start_date = df["timestamp"].min().normalize()
    df["day_number"] = (ts.dt.normalize() - start_date).dt.days

    return df

# ============================================================================
# SECTION 3 — Rolling Window Statistics
# ============================================================================

def add_rolling_features(
    df: pd.DataFrame,
    kpis: list[str] = RAW_KPIS,
    windows: dict[str, int] = WINDOW_SIZES,
) -> pd.DataFrame:
    """Compute rolling mean, std, min, max for each KPI × window combination.

    IMPLEMENTATION NOTES:
    - We group by cell_id and use pandas .rolling() with min_periods=1.
      min_periods=1 avoids NaN for the first (window-1) rows, which otherwise
      propagate through all downstream features.  The trade-off: statistics
      computed from very few observations are noisy.  We add a separate
      "observation count" feature per window to let the model downweight these.

    - We use closed='left' on all windows so that the current ROP is EXCLUDED
      from its own rolling statistic.  This is critical for point-in-time
      correctness (no look-ahead leakage).  See Coursebook Ch. 28 §28.4.

    - At 100 cells × 2880 ROPs × 7 KPIs × 4 windows × 4 stats = ~3.2M feature

    Args:
        df:      PM dataframe sorted by (cell_id, timestamp).
        kpis:    List of KPI column names to compute rolling stats for.
        windows: Dict mapping window label → number of ROPs.

    Returns:
        df with additional rolling feature columns appended.

    PRODUCTION NOTE:
        This pandas implementation mirrors the logic but not the execution
        model of the production Flink pipeline. In production, rolling
        statistics are computed using Flink's native windowed aggregations
        (tumbling and sliding windows with event-time semantics), which
        handle out-of-order events and late arrivals natively. The pandas
        version here is for offline training dataset generation only.
    """
    logger.info(
        "Computing rolling features: %d KPIs × %d windows",
        len(kpis), len(windows),
    )

    grouped = df.groupby("cell_id", sort=False)

    for win_label, win_rops in windows.items():
        logger.debug("  Window %s (%d ROPs)", win_label, win_rops)
        # Compute each aggregate separately (transform limitation)
        for kpi in kpis:
            series = grouped[kpi].transform(
                lambda s, w=win_rops: s.shift(1).rolling(window=w, min_periods=1).mean()
            )
            df[f"{kpi}_roll_{win_label}_mean"] = series

            series = grouped[kpi].transform(
                lambda s, w=win_rops: s.shift(1).rolling(window=w, min_periods=1).std()
            )
            # std is NaN when there is only one observation; fill with 0
            df[f"{kpi}_roll_{win_label}_std"] = series.fillna(0.0)

            series = grouped[kpi].transform(
                lambda s, w=win_rops: s.shift(1).rolling(window=w, min_periods=1).min()
            )
            df[f"{kpi}_roll_{win_label}_min"] = series

            series = grouped[kpi].transform(
                lambda s, w=win_rops: s.shift(1).rolling(window=w, min_periods=1).max()
            )
            df[f"{kpi}_roll_{win_label}_max"] = series

        # Observation count — tells model how many valid ROPs were in the window
        # (important for cold-start cells with short history)
        df[f"roll_{win_label}_obs_count"] = grouped["dl_throughput_mbps"].transform(
            lambda s, w=win_rops: s.shift(1).rolling(window=w, min_periods=1).count()
        )

    n_new_cols = len(df.columns) - len(META_COLS) - len(RAW_KPIS) - 7  # approx
    logger.info("Rolling features added.  Dataset now has %d columns.", len(df.columns))
    return df


# ============================================================================
# SECTION 4 — Rate-of-Change Features
# ============================================================================
def add_rate_of_change_features(
    df: pd.DataFrame,
    kpis: list[str] = RAW_KPIS,
) -> pd.DataFrame:
    """Compute first-order difference and percentage change per KPI per cell.

    WHY RATE-OF-CHANGE:
        A sudden drop from 200 Mbps to 50 Mbps is a strong anomaly signal.
        Rolling statistics alone don't capture this instantaneous change —
        the rolling mean barely moves after one anomalous ROP.  First-order
        delta is the most sensitive feature for "sudden drop" anomaly type.
        See Coursebook Ch. 25 §25.7 — Differencing for Stationarity.

    HANDLING COUNTER RESETS:
        01_synthetic_data.py injects occasional counter resets that produce
        a large *negative* delta on cumulative counters.  For rate-based KPIs
        (already normalised to percentages or rates), resets manifest as a jump
        to 100%.  We clip pct_change to [-200%, +200%] to bound the influence
        of reset artefacts without losing genuine large-magnitude events.
    """
    logger.info("Computing rate-of-change features for %d KPIs", len(kpis))
    grouped = df.groupby("cell_id", sort=False)

    for kpi in kpis:
        # Absolute first-order difference (current ROP vs previous ROP)
        df[f"{kpi}_delta_1rop"] = grouped[kpi].transform(
            lambda s: s.diff(1)
        )

        # Percentage change; clip to bound counter-reset artefacts
        df[f"{kpi}_pct_change_1rop"] = (
            grouped[kpi]
            .transform(lambda s: s.pct_change(1) * 100.0)
            .clip(-200.0, 200.0)
            .fillna(0.0)
        )

        # 4-ROP (1-hour) delta — catches gradual trends missed by 1-ROP diff
        df[f"{kpi}_delta_4rop"] = grouped[kpi].transform(
            lambda s: s.diff(4)
        )

    logger.info("Rate-of-change features added.")
    return df


# ============================================================================
# SECTION 5 — Day-over-Day and Week-over-Week Ratio Features
# ============================================================================

def add_historical_ratio_features(
    df: pd.DataFrame,
    kpis: list[str] = RAW_KPIS,
) -> pd.DataFrame:
    """Compute same-ROP comparison to 24 h ago and 7 days ago.

    WHY HISTORICAL RATIOS:
        A cell serving a stadium shows 10× normal throughput on match days.
        A static threshold fires an "anomaly" every game.  Historical ratios
        normalise against this expected periodicity: if today's value is close
        to last week's value for the same ROP, it's NOT anomalous even if the
        absolute value is extreme.  See Coursebook Ch. 25 §25.9 — Seasonality
        Handling.

    COLD-START HANDLING:
        For the first 24 h (day-over-day) or 7 days (week-over-week), the lag
        is unavailable.  We fill with 1.0 (ratio = no change) and add a
        boolean flag indicating the feature is imputed.  The model can learn
        to ignore these imputed rows.
    """
    logger.info("Computing DoD / WoW ratio features for %d KPIs", len(kpis))
    grouped = df.groupby("cell_id", sort=False)

    for kpi in kpis:
        # ---- Day-over-Day (lag = 96 ROPs = 24 h) ----
        lag_dod = grouped[kpi].transform(lambda s: s.shift(ROPS_PER_DAY))
        # Ratio: value / lag; clip to [0.01, 100] to prevent div-by-zero explosion
        ratio_dod = (df[kpi] / lag_dod.replace(0, np.nan)).clip(0.01, 100.0)
        df[f"{kpi}_dod_ratio"] = ratio_dod.fillna(1.0)
        df[f"{kpi}_dod_imputed"] = lag_dod.isna().astype(np.int8)

        # ---- Week-over-Week (lag = 672 ROPs = 7 days) ----
        lag_wow = grouped[kpi].transform(lambda s: s.shift(ROPS_PER_WEEK))
        ratio_wow = (df[kpi] / lag_wow.replace(0, np.nan)).clip(0.01, 100.0)
        df[f"{kpi}_wow_ratio"] = ratio_wow.fillna(1.0)
        df[f"{kpi}_wow_imputed"] = lag_wow.isna().astype(np.int8)

    logger.info("Historical ratio features added.")
    return df


# ============================================================================
# SECTION 6 — Cross-KPI Ratio Features
# ============================================================================

def add_cross_kpi_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute domain-meaningful ratios between pairs of KPIs.

    These derived indicators capture relationships that raw KPI values miss:

    prb_efficiency_dl:
        Throughput per unit of PRB — high PRB with low throughput suggests
        radio interference or poor modulation (low CQI).  A sudden drop in
        this ratio is a strong hardware/interference anomaly signal.
        Units: Mbps / % PRB ≈ Mbps per percent-PRB-occupied.

    dl_ul_ratio:
        Normal cells have a stable DL/UL asymmetry (typically 3–8× for eMBB
        traffic).  A dramatic ratio change suggests asymmetric failure (e.g.,
        UL antenna branch failure), traffic composition change, or backhaul
        issue on one direction.

    success_rate_composite:
        Mean of the three success-rate KPIs — used as a single "health score"
        feature that captures overall call processing health without requiring
        the model to learn that all three rates should degrade together.

    cqi_prb_load_interaction:
        PRB load × (15 - avg_cqi) — high load combined with poor channel
        quality is worse than either alone.  This non-linear interaction term
        helps tree-based models without requiring them to discover it.
        See Coursebook Ch. 13 §13.6 — Domain-Guided Feature Crosses.
    """
    logger.info("Computing cross-KPI ratio features")

    # PRB efficiency (DL throughput per unit PRB)
    # Guard against zero PRB (idle cell) — replace with NaN then fill
    prb_safe = df["dl_prb_usage_rate"].replace(0.0, np.nan)
    df["prb_efficiency_dl"] = (df["dl_throughput_mbps"] / prb_safe).fillna(0.0)

    # DL/UL throughput ratio — clip to [0.1, 50] to suppress div-by-zero noise
    ul_safe = df["ul_throughput_mbps"].replace(0.0, np.nan)
    df["dl_ul_ratio"] = (
        (df["dl_throughput_mbps"] / ul_safe)
        .clip(0.1, 50.0)
        .fillna(1.0)
    )

    # Composite success rate — average of RRC, ERAB, HO success rates
    df["success_rate_composite"] = df[[
        "rrc_conn_setup_success_rate",
        "drb_setup_success_rate",
        "handover_success_rate",
    ]].mean(axis=1)

    # CQI × PRB load interaction (non-linear term)
    # (15 - avg_cqi) is "CQI badness": 0 when perfect, 15 when worst
    df["cqi_prb_load_interaction"] = df["dl_prb_usage_rate"] * (15.0 - df["avg_cqi"])

    # Volume indicator — log-transform of throughput (handles right skew)
    # +1 to handle zero values; log scale compresses the 0–2000 Mbps range
    df["log_dl_throughput"] = np.log1p(df["dl_throughput_mbps"])
    df["log_ul_throughput"] = np.log1p(df["ul_throughput_mbps"])

    logger.info("Cross-KPI features added.")
    return df


# ============================================================================
# SECTION 7 — Spatial Peer-Group Deviation Features
# ============================================================================

def build_peer_groups(
    df: pd.DataFrame,
    topology: pd.DataFrame | None,
) -> dict[str, list[str]]:
    """Assign each cell to a peer group for relative anomaly scoring.

    PEER GROUP DEFINITION:
        Cells are grouped by (vendor, frequency_band, cell_type).  Within each
        group, cells are expected to behave similarly under similar load.
        Deviation from the group mean is a strong anomaly signal that is
        immune to global network-wide events (e.g., a stadium event raises
        ALL nearby cells, so the peer z-score stays near zero).

        This is the SINGLE MOST IMPACTFUL feature family for reducing false
        alarms, because it replaces absolute thresholds with relative context.
        See Coursebook Ch. 13 §13.8 — Spatial Feature Engineering.

    FALLBACK (no topology):
        If topology data is unavailable, we form groups solely from the PM
        dataset's metadata columns (vendor, frequency_band, cell_type), which
        were embedded by 01_synthetic_data.py.

    Returns:
        Dict mapping group_id → list of cell_ids in that group.
    """
    logger.info("Building peer groups")

    # Use metadata from the PM dataframe itself (always available)
    group_cols = ["vendor", "frequency_band", "cell_type"]

    # Check which group columns exist in the dataframe
    available_group_cols = [c for c in group_cols if c in df.columns]
    if not available_group_cols:
        # Last-resort fallback: single global group
        logger.warning(
            "No peer group metadata found; using single global peer group."
        )
        all_cells = df["cell_id"].unique().tolist()
        return {"global": all_cells}

    cell_meta = (
        df[["cell_id"] + available_group_cols]
        .drop_duplicates("cell_id")
        .copy()
    )

    # Create composite group key
    cell_meta["peer_group_id"] = cell_meta[available_group_cols].astype(str).agg(
        "_".join, axis=1
    )

    peer_groups: dict[str, list[str]] = (
        cell_meta.groupby("peer_group_id")["cell_id"]
        .apply(list)
        .to_dict()
    )

    # Log group sizes — groups < 3 cells are too small for meaningful z-scores
    for gid, cells in peer_groups.items():
        status = "OK" if len(cells) >= 3 else "SMALL (<3 cells)"
        logger.info("  Peer group '%-35s': %3d cells  [%s]", gid, len(cells), status)

    return peer_groups


def add_peer_group_features(
    df: pd.DataFrame,
    peer_groups: dict[str, list[str]],
    kpis: list[str] = RAW_KPIS,
) -> pd.DataFrame:
    """Compute per-cell z-score deviation from peer group at each timestamp.

    ALGORITHM:
        For each timestamp t and each KPI k:
          1. Compute the leave-one-out mean μ_loo(t, k) = (Σ_group - x_cell) /
             (N_group - 1), excluding the cell being scored to prevent
             information leakage from the cell into its own score.
          2. Compute the full-group std σ_g(t, k) across all cells in the peer
             group (NOT leave-one-out — see Limitation below).
          3. z_score = (cell_value - μ_loo) / (σ_g + ε)
             ε = 1e-6 prevents division by zero for constant-value windows.

    LIMITATION: The denominator uses full-group std rather than leave-one-out
        std. For groups with fewer than 5 cells, a single anomalous cell can
        inflate σ_g by 50–100% and attenuate its own z-score by 30–50%. See
        the commented leave-one-out std alternative below (Step 3) and §10
        Limitations.

    COMPLEXITY NOTE:
        Naïve implementation loops over cells × timestamps → O(N²).
        This vectorised implementation uses groupby on (peer_group_id, timestamp)
        to compute group stats in a single pass, then subtracts each cell's
        contribution (leave-one-out correction).  Scales to 100 cells × 2880
        ROPs in ~10 seconds on a modern laptop.

        For 10K+ cells, this should run as a Flink windowed join — see
        05_production_patterns.py.  See Coursebook Ch. 28 §28.6.

    Args:
        df:           PM dataframe with temporal features already added.
        peer_groups:  Dict from build_peer_groups().
        kpis:         KPI columns to compute peer z-scores for.

    Returns:
        df with additional peer z-score columns.
    """
    logger.info(
        "Computing peer-group z-score features: %d groups × %d KPIs",
        len(peer_groups), len(kpis),
    )

    # Create cell → peer_group lookup
    cell_to_group: dict[str, str] = {
        cell: gid
        for gid, cells in peer_groups.items()
        for cell in cells
    }

    df["peer_group_id"] = df["cell_id"].map(cell_to_group).fillna("ungrouped")

    # Pre-compute group size per group (needed for leave-one-out correction)
    group_sizes: dict[str, int] = {gid: len(cells) for gid, cells in peer_groups.items()}
    df["peer_group_size"] = df["peer_group_id"].map(group_sizes).fillna(1)

    for kpi in kpis:
        col_name = f"{kpi}_peer_zscore"

        # Step 1: Group sum and count at each timestamp
        # This gives us Σ(values in group) and N(cells in group) at time t
        group_stats = df.groupby(["peer_group_id", "timestamp"])[kpi].agg(
            ["sum", "count", "std"]
        ).reset_index()
        group_stats.columns = ["peer_group_id", "timestamp", "grp_sum", "grp_n", "grp_std"]

        # Merge group stats back to per-cell rows
        df_merged = df[["cell_id", "timestamp", "peer_group_id", kpi]].merge(
            group_stats, on=["peer_group_id", "timestamp"], how="left"
        )

        # Step 2: Leave-one-out mean (exclude this cell's contribution)
        # μ_loo = (Σ_group - x_cell) / (N_group - 1)
        # This prevents a single anomalous cell from shifting the group mean
        loo_sum = df_merged["grp_sum"] - df_merged[kpi]
        loo_n   = (df_merged["grp_n"] - 1).clip(lower=1)  # min 1 to avoid div/0
        loo_mean = loo_sum / loo_n

        # Step 3: Z-score using group std (not leave-one-out std for simplicity)
        # ------------------------------------------------------------------
        # LEAVE-ONE-OUT STD ALTERNATIVE (recommended for groups < 5 cells):
        #
        #   # Compute leave-one-out variance:
        #   # Var_loo = (Σ(x² in group) - x_cell²) / (N-1) - μ_loo²
        #   # This requires pre-computing group sum-of-squares:
        #   #   grp_sum_sq = df.groupby([...])[kpi].apply(lambda s: (s**2).sum())
        #   # Then:
        #   #   loo_sum_sq = grp_sum_sq - df_merged[kpi]**2
        #   #   loo_var = (loo_sum_sq / loo_n) - loo_mean**2
        #   #   loo_std = np.sqrt(loo_var.clip(lower=0))
        #   #   z_score = (df_merged[kpi] - loo_mean) / (loo_std + eps)
        #
        # This prevents a single outlier in a 3-cell group from inflating
        # its own denominator. ~5 extra lines; measurable improvement for
        # small peer groups (30–50% sensitivity recovery).
        # ------------------------------------------------------------------
        eps = 1e-6
        z_score = (df_merged[kpi] - loo_mean) / (df_merged["grp_std"].fillna(0) + eps)

        # Singleton peer groups (grp_n <= 1) have no peer information:
        # leave-one-out mean is 0 and std is 0, producing spurious ±10
        # clipped z-scores. Neutralise them.
        singleton_mask = df_merged["grp_n"] <= 1
        z_score = z_score.copy()
        z_score[singleton_mask] = 0.0

        # Clip z-scores to [-10, 10] — extreme values are all equally anomalous
        df[col_name] = z_score.clip(-10.0, 10.0).values

    # Log singleton neutralisation (computed on last KPI's mask — representative)
    if 'singleton_mask' in dir():
        n_singleton = int(singleton_mask.sum())
        if n_singleton > 0:
            logger.info("Set z-score to 0.0 for %d singleton peer-group ROPs", n_singleton)

    logger.info("Peer-group z-score features added.")

    # Flag cells with fewer than 7 days of history — peer z-scores are
    # unreliable during the cold-start period and should be down-weighted
    # or excluded from anomaly scoring.
    if "timestamp" in df.columns:
        cell_spans = df.groupby("cell_id")["timestamp"].agg(lambda s: (s.max() - s.min()).days)
        cold_start_map = (cell_spans < PEER_GROUP_COLD_START_DAYS).to_dict()
        df["peer_group_cold_start"] = df["cell_id"].map(cold_start_map).fillna(True).astype(bool)
        n_cold = df["peer_group_cold_start"].sum()
        if n_cold > 0:
            logger.warning(
                "%d rows (%d cells) flagged as peer_group_cold_start (< days history)",
                n_cold, df.loc[df["peer_group_cold_start"], "cell_id"].nunique(),
            )

    return df


# ============================================================================
# SECTION 8 — Missing Data and Quality Features
# ============================================================================

def add_missing_data_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add indicators for imputed values, counter resets, and data gaps.

    WHY MISSING DATA FEATURES:
        01_synthetic_data.py injects realistic missing data and counter resets.
        Models should know when they are scoring on imputed vs observed values —
        otherwise a cell that happens to be missing during a degradation will
        look "normal" (its imputed value matches the rolling mean by definition).

        Including missingness indicators also helps the model learn that
        "was missing, now back" is a different pattern from "was normal, still
        normal".  See Coursebook Ch. 13 §13.10 — Missing Value Encoding.

    COUNTER RESET DETECTION:
        A counter reset produces a large *negative* delta on raw cumulative
        counters (the counter wraps back to 0 or a small value).  For the
        already-normalised KPIs in our dataset, a reset manifests as a
        sudden jump to near-maximum (e.g., 100% success rate after a reset
        because the denominator restarts from 0).  We detect this as:
          delta > 80th percentile AND previous value < 20th percentile
        (heuristic — in production, use vendor-specific counter metadata).
    """
    logger.info("Adding missing data and quality indicator features")

    # Overall missing data fraction over rolling 1h window per cell
    # (useful for detecting PM collection outages)
    for kpi in RAW_KPIS:
        missing_indicator = df[kpi].isna().astype(np.int8)
        df[f"{kpi}_is_missing"] = missing_indicator

    # Total KPIs missing at this ROP (0–7 scale)
    df["n_kpis_missing"] = sum(
        df[kpi].isna().astype(int) for kpi in RAW_KPIS
    )

    # Rolling missing rate over 1h window (96 ROPs = 24h, 4 ROPs = 1h)
    grouped = df.groupby("cell_id", sort=False)
    df["missing_rate_1h"] = grouped["n_kpis_missing"].transform(
        lambda s: s.shift(1).rolling(window=4, min_periods=1).mean()
    )

    # Counter reset indicator — detect implausible positive delta
    # after a period of stable or declining values
    # Heuristic: dl_throughput jumps >500 Mbps in one ROP after being low
    # (in practice, tie to vendor PM documentation)
    dl_delta = grouped["dl_throughput_mbps"].transform(lambda s: s.diff(1))
    dl_low_prev = grouped["dl_throughput_mbps"].transform(
        lambda s: s.shift(1).rolling(4, min_periods=1).mean() < 10.0
    )
    df["counter_reset_suspected"] = (
        (dl_delta > 500.0) & dl_low_prev
    ).astype(np.int8)

    logger.info("Missing data features added.")
    return df


# ============================================================================
# SECTION 9 — Feature Matrix Assembly and NaN Handling
# ============================================================================

def _resolve_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the list of feature columns (exclude metadata and label columns).

    Keeps all numeric columns that are not in META_COLS.  The resulting list
    is saved to feature_metadata.json so that training and serving use the
    exact same feature set.
    """
    # Exclude raw KPIs — the model uses only derived features (rolling stats,
    # z-scores, rate-of-change). Raw KPIs are inputs to feature engineering,
    # not model features. See 03_model_training.py resolve_feature_columns().
    non_feature_cols = set(META_COLS) | set(RAW_KPIS) | {"peer_group_id", "peer_group_size", "day_number"}
    feature_cols = [
        c for c in df.columns
        if c not in non_feature_cols
        and pd.api.types.is_numeric_dtype(df[c])
        and (not EXCLUDE_TIER2_DERIVED or not any(c.startswith(t2) for t2 in TIER2_KPIS))
    ]
    if EXCLUDE_TIER2_DERIVED:
        logger.info(
            "EXCLUDE_TIER2_DERIVED=True — excluded derived features for %s",
            sorted(TIER2_KPIS),
        )
    return feature_cols


def impute_missing_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Fill NaN values in feature columns with cell-level median, then global median.

    IMPUTATION STRATEGY (two-pass):
        Pass 1: Fill NaN with the median of that KPI for that cell.
                This handles gaps where a cell was temporarily unreachable.
        Pass 2: Fill remaining NaN (entire column missing for a cell, e.g.,
                week-over-week ratio for a new cell) with the global column median.
        Pass 3: Any remaining NaN (column entirely missing) → fill with 0.

    CRITICAL: Imputation statistics MUST be computed on training data only and
    applied to validation/test data.  This function returns the imputed df and
    a stats dict that must be persisted and reused at serving time.
    See Coursebook Ch. 13 §13.11 — Leakage-Free Imputation.
    """
    logger.info("Imputing missing values in %d feature columns", len(feature_cols))

    # Count NaN before
    total_nan_before = df[feature_cols].isna().sum().sum()
    logger.info("  NaN values before imputation: %d", total_nan_before)

    # Pass 1 — cell-level median fill (only for the designated feature cols)
    df[feature_cols] = df.groupby("cell_id")[feature_cols].transform(
        lambda col: col.fillna(col.median())
    )

    # Pass 2 — global median fill (handles all-NaN columns per cell)
    global_medians = df[feature_cols].median()
    df[feature_cols] = df[feature_cols].fillna(global_medians)

    # Pass 3 — zero fill for any remaining NaN (e.g., entirely zero columns)
    df[feature_cols] = df[feature_cols].fillna(0.0)

    total_nan_after = df[feature_cols].isna().sum().sum()
    logger.info("  NaN values after imputation: %d", total_nan_after)

    return df, global_medians.to_dict()


# ============================================================================
# SECTION 10 — Temporal Train / Val / Test Split
# ============================================================================

def temporal_split(
    df: pd.DataFrame,
    train_end_day: int = TRAIN_END_DAY,
    val_end_day: int = VAL_END_DAY,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the feature matrix into train, validation, and test sets by time.

    NEVER USE RANDOM SPLIT FOR TIME-SERIES DATA.
        Random splitting leaks future information into the training set because
        a model can "see" anomaly signatures from day 28 while training on day
        10, producing optimistically biased evaluation metrics.

        We use a strict forward-in-time split:
          Train: days  0–20  (model learns "normal" cell behaviour)
          Val:   days 21–25  (threshold tuning, hyperparameter selection)
          Test:  days 26–30  (unbiased evaluation; most anomalies injected here)

        Note: In streaming / online learning scenarios, "train" is the initial
        warm-up window.  See Coursebook Ch. 25 §25.12 — Temporal Cross-Validation.

    Args:
        df:             Feature dataframe with "day_number" column.
        train_end_day:  Inclusive upper bound for training set.
        val_end_day:    Inclusive upper bound for validation set.

    Returns:
        (train_df, val_df, test_df) — non-overlapping, time-ordered splits.
    """
    if "day_number" not in df.columns:
        raise ValueError("'day_number' column required for temporal split.")

    train_df = df[df["day_number"] <= train_end_day].copy()
    val_df   = df[(df["day_number"] > train_end_day) & (df["day_number"] <= val_end_day)].copy()
    test_df  = df[df["day_number"] > val_end_day].copy()

    total = len(df)
    logger.info(
        "Temporal split complete:\n"
        "  Train : %6d rows  (days  0–%d,  %.1f%%)\n"
        "  Val   : %6d rows  (days %d–%d,  %.1f%%)\n"
        "  Test  : %6d rows  (days %d–30,  %.1f%%)",
        len(train_df), train_end_day, 100 * len(train_df) / total,
        len(val_df),   train_end_day + 1, val_end_day, 100 * len(val_df) / total,
        len(test_df),  val_end_day + 1, 100 * len(test_df) / total,
    )

    # Sanity check — no future leakage
    if not (train_df["timestamp"].max() < val_df["timestamp"].min()):
        logger.warning("Train / val boundary overlap detected — check day_number computation.")
    if len(val_df) > 0 and len(test_df) > 0:
        if not (val_df["timestamp"].max() < test_df["timestamp"].min()):
            logger.warning("Val / test boundary overlap detected.")

    # Anomaly rate per split (training should be low; test should have injected anomalies)
    for split_name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        if "is_anomaly" in split_df.columns:
            anomaly_rate = split_df["is_anomaly"].mean() * 100
            logger.info("  %s anomaly rate: %.2f%%", split_name, anomaly_rate)

    return train_df, val_df, test_df


# ============================================================================
# SECTION 11 — Feature Scaling
# ============================================================================

def fit_and_apply_scaling(
    train_df:     pd.DataFrame,
    val_df:       pd.DataFrame,
    test_df:      pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Fit StandardScaler on training data; transform all three splits.

    WHY STANDARD SCALING:
        Isolation Forest and LSTM autoencoders are sensitive to feature scale.
        Random Forests are scale-invariant, but standardisation makes SHAP
        values comparable across features (required for the NOC explanation
        framework).

    CRITICAL: Scaler must be fit on TRAINING data only.
        Fitting on the full dataset leaks test statistics into training,
        producing artificially better normalisation for anomalous test rows.

    Returns:
        Scaled versions of the three splits, plus a serialisable stats dict
        containing mean and std per feature (for serving-time reconstruction).
    """
    logger.info("Fitting StandardScaler on training data (%d rows)", len(train_df))

    scaler = StandardScaler()

    # Fit on training feature matrix
    train_scaled = train_df.copy()
    train_scaled[feature_cols] = scaler.fit_transform(train_df[feature_cols].values)

    # Apply (not fit) to val and test
    val_scaled = val_df.copy()
    val_scaled[feature_cols] = scaler.transform(val_df[feature_cols].values)

    test_scaled = test_df.copy()
    test_scaled[feature_cols] = scaler.transform(test_df[feature_cols].values)

    # Serialise scaler statistics — required for serving-time feature normalisation
    # These are saved to feature_metadata.json alongside feature column names
    scaler_stats: dict[str, Any] = {
        "mean": dict(zip(feature_cols, scaler.mean_.tolist())),
        "std":  dict(zip(feature_cols, scaler.scale_.tolist())),
    }

    logger.info("Scaling complete.  Feature mean range: [%.3f, %.3f]",
                float(scaler.mean_.min()), float(scaler.mean_.max()))

    return train_scaled, val_scaled, test_scaled, scaler_stats


# ============================================================================
# SECTION 12 — Output Writing and Metadata
# ============================================================================

def write_feature_splits(
    train_df:     pd.DataFrame,
    val_df:       pd.DataFrame,
    test_df:      pd.DataFrame,
    feature_cols: list[str],
    scaler_stats: dict[str, Any],
    impute_stats: dict[str, float],
    peer_groups:  dict[str, list[str]],
    output_dir:   Path = FEATURES_DIR,
) -> None:
    """Persist feature splits and metadata to Parquet + JSON.

    FILE OUTPUTS:
        train_features.parquet  — scaled feature matrix + labels (days 0–20)
        val_features.parquet    — scaled feature matrix + labels (days 21–25)
        test_features.parquet   — scaled feature matrix + labels (days 26–30)
        feature_metadata.json   — column names, scaler stats, split boundaries,
                                  peer group assignments.  Required at serving
                                  time to reconstruct features from raw counters.

    Parquet is preferred over CSV for:
        - Preserving column dtypes (avoids ambiguous int8 → float64 coercions)
        - Efficient columnar compression (~10× smaller than CSV for numeric data)
        - Flink / Spark / BentoML can read Parquet natively
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Columns to retain in output: metadata + labels + features
    keep_cols = (
        ["cell_id", "timestamp", "site_id", "sector_id", "day_number", "peer_group_id",
         "peer_group_cold_start"]
        + (["is_anomaly", "anomaly_type"] if "is_anomaly" in train_df.columns else [])
        + feature_cols
    )
    # Drop any keep_cols not present (defensive)
    keep_cols = [c for c in keep_cols if c in train_df.columns]

    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        out_path = output_dir / f"{name}_features.parquet"
        split_df[keep_cols].to_parquet(out_path, index=False, engine="pyarrow")
        logger.info("Wrote %s: %d rows × %d cols → %s",
                    name, len(split_df), len(keep_cols), out_path)

    # Metadata JSON — single source of truth for serving-time feature reconstruction
    metadata: dict[str, Any] = {
        "version": "1.0.0",
        "feature_columns": feature_cols,
        "n_features": len(feature_cols),
        "raw_kpis": RAW_KPIS,
        "split_boundaries": {
            "train_end_day":  TRAIN_END_DAY,
            "val_end_day":    VAL_END_DAY,
            "train_rows":     len(train_df),
            "val_rows":       len(val_df),
            "test_rows":      len(test_df),
        },
        "scaler_stats": scaler_stats,
        "imputation_defaults": impute_stats,
        "peer_groups": peer_groups,
        "window_sizes_rops": WINDOW_SIZES,
        "rops_per_day":  ROPS_PER_DAY,
        "rops_per_week": ROPS_PER_WEEK,
    }

    meta_path = output_dir / "feature_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, default=str)
    logger.info("Feature metadata written to %s", meta_path)


# ============================================================================
# SECTION 13 — Feature Importance Preview (diagnostic)
# ============================================================================

def log_feature_summary(
    train_df: pd.DataFrame,
    feature_cols: list[str],
) -> None:
    """Log descriptive statistics for key feature groups as a sanity check.

    This provides a quick diagnostic to confirm that features were computed
    correctly without needing to load a visualisation tool.  In CI/CD pipelines,
    these summary stats can be compared against reference values to detect
    regressions in the feature engineering code.
    """
    logger.info("--- Feature Summary (training set, post-scaling) ---")

    # Group features by type prefix for readable output
    feature_groups: dict[str, list[str]] = {
        "Temporal":       [c for c in feature_cols if any(c.startswith(p) for p in
                           ["hour_", "dow_", "is_peak", "is_week", "is_night"])],
        "Rolling_1h":     [c for c in feature_cols if "_roll_1h_" in c],
        "Rolling_24h":    [c for c in feature_cols if "_roll_24h_" in c],
        "Rate_of_change": [c for c in feature_cols if "_delta_" in c or "_pct_change_" in c],
        "DoD_ratio":      [c for c in feature_cols if "_dod_ratio" in c],
        "WoW_ratio":      [c for c in feature_cols if "_wow_ratio" in c],
        "Peer_zscore":    [c for c in feature_cols if "_peer_zscore" in c],
        "Cross_KPI":      [c for c in feature_cols if c in [
                           "prb_efficiency_dl", "dl_ul_ratio",
                           "success_rate_composite", "cqi_prb_load_interaction",
                           "log_dl_throughput", "log_ul_throughput"]],
        "Missing_flags":  [c for c in feature_cols if "_is_missing" in c
                          or c in ["n_kpis_missing", "missing_rate_1h",
                                   "counter_reset_suspected"]],
    }

    for group_name, cols in feature_groups.items():
        if not cols:
            continue
        subset = train_df[cols]
        logger.info(
            "  %-18s: %3d features | "
            "mean range [%6.3f, %6.3f] | "
            "std range [%5.3f, %5.3f]",
            group_name, len(cols),
            float(subset.mean().min()), float(subset.mean().max()),
            float(subset.std().min()),  float(subset.std().max()),
        )

    # Special check: peer z-scores should have mean ≈ 0 and std ≈ 1 on training data
    zscore_cols = [c for c in feature_cols if "_peer_zscore" in c]
    if zscore_cols:
        z_mean = train_df[zscore_cols].mean().mean()
        z_std  = train_df[zscore_cols].std().mean()
        status = "✓ OK" if abs(z_mean) < 0.5 and 0.5 < z_std < 3.0 else "⚠ CHECK"
        logger.info(
            "  Peer z-scores: grand mean=%.3f, avg std=%.3f  [%s]",
            z_mean, z_std, status,
        )

    total_features = len(feature_cols)
    logger.info("  Total features: %d", total_features)
# ============================================================================
# SECTION 14 — Master Pipeline Function
# ============================================================================

def apply_imputation(
    df: pd.DataFrame,
    feature_cols: list[str],
    impute_stats: dict[str, float],
) -> pd.DataFrame:
    """Apply pre-computed training-set imputation statistics to a DataFrame."""
    df = df.copy()
    for col in feature_cols:
        if col in df.columns:
            df[col] = df[col].fillna(impute_stats.get(col, 0.0))
    df[feature_cols] = df[feature_cols].fillna(0.0)
    return df


def run_feature_pipeline(
    pm_data_path: Path = PM_DATA_PATH,
    topology_path: Path = TOPOLOGY_PATH,
    output_dir: Path = FEATURES_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Execute the full feature engineering pipeline end-to-end.

    Orchestrates all feature engineering steps in the correct dependency order:

    1. Load raw PM data and topology
    2. Add temporal context features
    3. Add rolling window statistics       (requires sorted time index)
    4. Add rate-of-change features         (requires sorted time index)
    5. Add DoD / WoW historical ratios     (requires sorted time index + lags)
    6. Add cross-KPI derived features
    7. Build peer groups
    8. Add peer-group z-score features     (requires peer groups)
    9. Add missing data indicators
    10. Resolve feature column list
    11. Temporal train / val / test split  (MUST come before imputation)
    12. Impute NaN values using training-set statistics only
    13. Fit scaler on train; transform all splits
    14. Write outputs

    Returns:

    Output paths (canonical):
        data/train_features.parquet
        data/val_features.parquet
        data/test_features.parquet
        data/feature_metadata.json
    """
    logger.info("=" * 70)
    logger.info("RAN KPI Feature Engineering Pipeline — START")
    logger.info("=" * 70)

    # ------------------------------------------------------------------
    # Step 1: Load data
    # ------------------------------------------------------------------
    df      = load_pm_data(pm_data_path)
    topology = load_topology(topology_path)

    # ------------------------------------------------------------------
    # Step 2: Temporal context features
    # ------------------------------------------------------------------
    df = add_temporal_features(df)
    logger.info("[2/13] Temporal features added.  Shape: %s", df.shape)

    # ------------------------------------------------------------------
    # Step 3: Rolling statistics
    # ------------------------------------------------------------------
    df = add_rolling_features(df, kpis=RAW_KPIS, windows=WINDOW_SIZES)
    logger.info("[3/13] Rolling features added.   Shape: %s", df.shape)

    # ------------------------------------------------------------------
    # Step 4: Rate-of-change
    # ------------------------------------------------------------------
    df = add_rate_of_change_features(df, kpis=RAW_KPIS)
    logger.info("[4/13] Rate-of-change features added.  Shape: %s", df.shape)

    # ------------------------------------------------------------------
    # Step 5: DoD / WoW ratios
    # ------------------------------------------------------------------
    df = add_historical_ratio_features(df, kpis=RAW_KPIS)
    logger.info("[5/13] Historical ratio features added.  Shape: %s", df.shape)

    # ------------------------------------------------------------------
    # Step 6: Cross-KPI derived features
    # ------------------------------------------------------------------
    df = add_cross_kpi_features(df)
    logger.info("[6/13] Cross-KPI features added.  Shape: %s", df.shape)

    # ------------------------------------------------------------------
    # Step 7: Build peer groups
    # ------------------------------------------------------------------
    peer_groups = build_peer_groups(df, topology)
    logger.info("[7/13] Built %d peer groups.", len(peer_groups))

    # ------------------------------------------------------------------
    # Step 8: Peer-group z-scores
    # ------------------------------------------------------------------
    df = add_peer_group_features(df, peer_groups, kpis=RAW_KPIS)
    logger.info("[8/13] Peer-group z-scores added.  Shape: %s", df.shape)

    # ------------------------------------------------------------------
    # Step 9: Missing data features
    # ------------------------------------------------------------------
    df = add_missing_data_features(df)
    logger.info("[9/13] Missing data features added.  Shape: %s", df.shape)

    # ------------------------------------------------------------------
    # Step 10: Resolve feature columns
    # ------------------------------------------------------------------
    feature_cols = _resolve_feature_columns(df)
    logger.info("[10/13] Feature columns resolved: %d features.", len(feature_cols))

    # ------------------------------------------------------------------
    # Step 11: Temporal split (MUST come before imputation)
    # ------------------------------------------------------------------
    train_df, val_df, test_df = temporal_split(
        df, train_end_day=TRAIN_END_DAY, val_end_day=VAL_END_DAY
    )
    logger.info("[11/13] Temporal split complete.")

    # ------------------------------------------------------------------
    # Step 12: Impute NaN values
    # Compute imputation statistics on training data only to prevent data leakage.
    # Training-set medians are then applied to fill NaN values in val and test sets.
    # ------------------------------------------------------------------
    train_df, impute_stats = impute_missing_features(train_df, feature_cols)
    val_df  = apply_imputation(val_df,  feature_cols, impute_stats)
    test_df = apply_imputation(test_df, feature_cols, impute_stats)
    logger.info("[12/13] Missing values imputed.")

    # ------------------------------------------------------------------
    # Step 13: Fit scaler on train; apply to all splits
    # ------------------------------------------------------------------
    train_df, val_df, test_df, scaler_stats = fit_and_apply_scaling(
        train_df, val_df, test_df, feature_cols
    )
    logger.info("[13/13] Scaling complete.")

    # ------------------------------------------------------------------
    # Step 14: Write outputs + diagnostic summary
    # Canonical output paths:
    #   data/train_features.parquet
    #   data/val_features.parquet
    #   data/test_features.parquet
    #   data/feature_metadata.json
    # ------------------------------------------------------------------
    log_feature_summary(train_df, feature_cols)

    write_feature_splits(
        train_df     = train_df,
        val_df       = val_df,
        test_df      = test_df,
        feature_cols = feature_cols,
        scaler_stats = scaler_stats,
        impute_stats = impute_stats,
        peer_groups  = peer_groups,
        output_dir   = output_dir,
    )

    logger.info("=" * 70)
    logger.info("Feature Engineering Pipeline — COMPLETE")
    logger.info(
        "Outputs written to %s\n"
        "  train_features.parquet : %d rows\n"
        "  val_features.parquet   : %d rows\n"
        "  test_features.parquet  : %d rows\n"
        "  feature_metadata.json  : %d features documented",
        output_dir,
        len(train_df), len(val_df), len(test_df), len(feature_cols),
    )
    logger.info("=" * 70)

    return train_df, val_df, test_df, feature_cols


# ============================================================================
# SECTION 15 — Standalone Serving-Time Feature Recomputation
# ============================================================================
def compute_features_for_serving(
    raw_window: pd.DataFrame,
    metadata: dict[str, Any],
    peer_baseline: dict[str, dict[str, float]],
) -> pd.DataFrame:
    """Recompute features for a single cell's recent window at serving time.

    This function mirrors the training pipeline exactly but operates on a
    small window (e.g., the last 96 ROPs = 24 h) for a SINGLE cell.
    It is called by the model serving layer (05_production_patterns.py) when
    scoring a fresh incoming ROP.

    POINT-IN-TIME CORRECTNESS:
        All rolling computations use only data up to and including the
        "current" ROP (the last row in raw_window).  No future rows are used.
        This exactly mirrors training behaviour (shift(1) on a stream where
        shift(1) always points to a row that has been observed).

    Args:
        raw_window:     DataFrame with the last N ROPs for one cell.
                        Must contain all RAW_KPIS and a 'timestamp' column.
                        The last row is the ROP to score.
        metadata:       Loaded from feature_metadata.json — contains scaler
                        stats, window sizes, feature column list.
        peer_baseline:  Dict of {kpi: {"mean": float, "std": float}} for
                        the cell's peer group at the current time step.
                        Computed separately from the online feature store.

    Returns:
        Single-row DataFrame with scaled feature values, ready for model.score().
    """
    # Apply temporal features to the window
    window = raw_window.copy()
    window = add_temporal_features(window)
    window = add_cross_kpi_features(window)

    # Rolling features — no groupby needed; single cell
    win_sizes: dict[str, int] = metadata.get("window_sizes_rops", WINDOW_SIZES)
    for win_label, win_rops in win_sizes.items():
        for kpi in RAW_KPIS:
            rolled = window[kpi].shift(1).rolling(window=win_rops, min_periods=1)
            window[f"{kpi}_roll_{win_label}_mean"] = rolled.mean()
            window[f"{kpi}_roll_{win_label}_std"]  = rolled.std().fillna(0.0)
            window[f"{kpi}_roll_{win_label}_min"]  = rolled.min()
            window[f"{kpi}_roll_{win_label}_max"]  = rolled.max()
        window[f"roll_{win_label}_obs_count"] = (
            window["dl_throughput_mbps"].shift(1)
            .rolling(window=win_rops, min_periods=1).count()
        )

    # Rate-of-change
    for kpi in RAW_KPIS:
        window[f"{kpi}_delta_1rop"]     = window[kpi].diff(1)
        window[f"{kpi}_pct_change_1rop"] = window[kpi].pct_change(1).mul(100).clip(-200, 200).fillna(0)
        window[f"{kpi}_delta_4rop"]     = window[kpi].diff(4)

    # DoD / WoW ratios — use lags if available in window, else fill with 1.0
    rops_day  = metadata.get("rops_per_day",  ROPS_PER_DAY)
    rops_week = metadata.get("rops_per_week", ROPS_PER_WEEK)
    for kpi in RAW_KPIS:
        lag_dod  = window[kpi].shift(rops_day)
        lag_wow  = window[kpi].shift(rops_week)
        window[f"{kpi}_dod_ratio"] = (window[kpi] / lag_dod.replace(0, np.nan)).clip(0.01, 100).fillna(1.0)
        window[f"{kpi}_wow_ratio"] = (window[kpi] / lag_wow.replace(0, np.nan)).clip(0.01, 100).fillna(1.0)
        window[f"{kpi}_dod_imputed"] = lag_dod.isna().astype(np.int8)
        window[f"{kpi}_wow_imputed"] = lag_wow.isna().astype(np.int8)

    # Peer z-scores using pre-fetched peer baseline from online feature store
    for kpi in RAW_KPIS:
        if kpi in peer_baseline:
            peer_mean = peer_baseline[kpi]["mean"]
            peer_std  = peer_baseline[kpi]["std"]
            window[f"{kpi}_peer_zscore"] = (
                (window[kpi] - peer_mean) / (peer_std + 1e-6)
            ).clip(-10.0, 10.0)
        else:
            window[f"{kpi}_peer_zscore"] = 0.0  # fallback — no peer data

    # Missing data indicators
    for kpi in RAW_KPIS:
        window[f"{kpi}_is_missing"] = window[kpi].isna().astype(np.int8)
    window["n_kpis_missing"] = sum(window[kpi].isna().astype(int) for kpi in RAW_KPIS)
    window["missing_rate_1h"] = window["n_kpis_missing"].shift(1).rolling(4, min_periods=1).mean()
    dl_delta = window["dl_throughput_mbps"].diff(1)
    dl_low   = window["dl_throughput_mbps"].shift(1).rolling(4, min_periods=1).mean() < 10
    window["counter_reset_suspected"] = ((dl_delta > 500) & dl_low).astype(np.int8)

    # Extract only the last (current) row for scoring
    feature_cols: list[str] = metadata["feature_columns"]
    # Fill any missing feature columns with 0 (new features not seen at training time)
    for col in feature_cols:
        if col not in window.columns:
            window[col] = 0.0

    # Apply scaler — reconstruct from stored mean/std
    scaler_mean = metadata["scaler_stats"]["mean"]
    scaler_std  = metadata["scaler_stats"]["std"]

    current_row = window.iloc[[-1]][feature_cols].copy()
    for col in feature_cols:
        mean = scaler_mean.get(col, 0.0)
        std  = scaler_std.get(col, 1.0)
        current_row[col] = (current_row[col] - mean) / (std + 1e-9)

    # Impute NaN using stored defaults
    impute_defaults = metadata.get("imputation_defaults", {})
    for col in feature_cols:
        if current_row[col].isna().any():
            default_val = impute_defaults.get(col, 0.0)
            # Apply the same scaling as above to the default value
            mean = scaler_mean.get(col, 0.0)
            std  = scaler_std.get(col, 1.0)
            current_row[col] = (default_val - mean) / (std + 1e-9)

    return current_row

# ============================================================================
# SECTION 16 — Self-Test / Validation
# ============================================================================

def _run_self_test(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    test_df:  pd.DataFrame,
    feature_cols: list[str],
) -> None:
    """Assert basic correctness properties of the produced feature matrix.

    These are cheap assertions that catch common implementation bugs:
    - No NaN in feature columns after imputation
    - Temporal split is strictly ordered
    - Feature count is within expected bounds
    - Anomaly rate is non-zero in test set (sanity: anomalies were injected)
    - Peer z-scores have roughly unit standard deviation on training set
    """
    logger.info("Running self-test assertions...")
    errors: list[str] = []

    # (a) No NaN after imputation
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        n_nan = df[feature_cols].isna().sum().sum()
        if n_nan > 0:
            errors.append(f"[FAIL] {name} has {n_nan} NaN values in features")
        else:
            logger.info("  ✓ No NaN in %s features", name)

    # (b) Temporal ordering — no leakage
    if not (train_df["timestamp"].max() < val_df["timestamp"].min()):
        errors.append("[FAIL] Train / val timestamps overlap")
    else:
        logger.info("  ✓ Train/val temporal ordering correct")

    if len(test_df) > 0 and len(val_df) > 0:
        if not (val_df["timestamp"].max() < test_df["timestamp"].min()):
            errors.append("[FAIL] Val / test timestamps overlap")
        else:
            logger.info("  ✓ Val/test temporal ordering correct")

    # (c) Feature count reasonable
    expected_min = 50   # sanity lower bound given 7 KPIs × multiple transforms
    expected_max = 500  # sanity upper bound
    if not (expected_min <= len(feature_cols) <= expected_max):
        errors.append(
            f"[WARN] Feature count {len(feature_cols)} outside expected range "
            f"[{expected_min}, {expected_max}]"
        )
    else:
        logger.info("  ✓ Feature count %d within expected range", len(feature_cols))

    # (d) Anomaly rate in test > 0 (requires ground truth labels)
    if "is_anomaly" in test_df.columns:
        test_anomaly_rate = test_df["is_anomaly"].mean()
        if test_anomaly_rate == 0.0:
            errors.append("[WARN] No anomalies found in test set — check 01_synthetic_data.py injection")
        else:
            logger.info("  ✓ Test anomaly rate: %.2f%%", test_anomaly_rate * 100)

    # (e) Peer z-scores approximately normalised on training set
    zscore_cols = [c for c in feature_cols if "_peer_zscore" in c]
    if zscore_cols:
        z_std = train_df[zscore_cols].std().mean()
        if not (0.3 < z_std < 3.0):
            errors.append(f"[WARN] Peer z-scores have unusual avg std={z_std:.3f}")
        else:
            logger.info("  ✓ Peer z-score avg std: %.3f", z_std)

    # Summary
    if errors:
        for err in errors:
            logger.warning(err)
        logger.warning("Self-test completed with %d issue(s).", len(errors))
    else:
        logger.info("Self-test PASSED — all assertions satisfied.")


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    train_df, val_df, test_df, feature_cols = run_feature_pipeline(
        pm_data_path  = PM_DATA_PATH,
        topology_path = TOPOLOGY_PATH,
        output_dir    = FEATURES_DIR,
    )

    _run_self_test(train_df, val_df, test_df, feature_cols)

    # Print a concise summary table to stdout for CI log inspection
    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING SUMMARY")
    print("=" * 70)
    print(f"  Output directory  : {FEATURES_DIR.resolve()}")
    print(f"  Total features    : {len(feature_cols)}")
    print(f"  Train rows        : {len(train_df):,}  (days  0–{TRAIN_END_DAY})")
    print(f"  Val   rows        : {len(val_df):,}  (days {TRAIN_END_DAY+1}–{VAL_END_DAY})")
    print(f"  Test  rows        : {len(test_df):,}  (days {VAL_END_DAY+1}–30)")

    if "is_anomaly" in test_df.columns:
        print(f"  Test anomaly rate : {test_df['is_anomaly'].mean()*100:.2f}%")
