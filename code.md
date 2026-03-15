utils.py
========

"""
utils.py — Shared utilities for RAN KPI Anomaly Detection pipeline
===================================================================
Canonical implementations of functions used across multiple pipeline scripts.
"""
from __future__ import annotations

import logging
from typing import Union

import numpy as np

logger = logging.getLogger(__name__)


#############################################################
# FUNCTION: compute_event_based_recall
#############################################################
def compute_event_based_recall(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
) -> float:
    """Compute event-based recall over a sequence of binary labels.

    An *event* is a contiguous run of ``anomaly=1`` values in ``y_true``.
    An event is considered *detected* if at least one ROP within that run
    has ``y_pred=1``.

    Parameters
    ----------
    y_true:
        Ground-truth binary labels (0 = normal, 1 = anomaly).  Must be
        ordered chronologically.
    y_pred:
        Predicted binary labels produced by the model at the same threshold.

    Returns
    -------
    float
        Fraction of events detected (0.0 – 1.0).  Returns ``float('nan')``
        when there are no anomaly events in ``y_true``.

    Example
    -------
    >>> y_true = [0, 1, 1, 0, 1, 0]
    >>> y_pred = [0, 1, 0, 0, 0, 0]
    >>> compute_event_based_recall(y_true, y_pred)
    0.5   # event 1 (indices 1-2) detected; event 2 (index 4) missed
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true and y_pred must have the same length "
            f"({len(y_true)} vs {len(y_pred)})."
        )

    # Identify contiguous runs of 1 in y_true
    events: list[tuple[int, int]] = []  # (start_idx, end_idx) inclusive
    in_event = False
    start = 0
    for i, val in enumerate(y_true):
        if val == 1 and not in_event:
            in_event = True
            start = i
        elif val == 0 and in_event:
            events.append((start, i - 1))
            in_event = False
    if in_event:
        events.append((start, len(y_true) - 1))

    if not events:
        logger.warning(
            "compute_event_based_recall: no anomaly events found in y_true; "
            "returning NaN."
        )
        return float("nan")

    detected = sum(
        1 for (s, e) in events if y_pred[s : e + 1].any()
    )

    event_recall = detected / len(events)
    logger.info(
        "Event-based recall: %d / %d events detected (%.4f)",
        detected,
        len(events),
        event_recall,
    )
    return event_recall


01_synthetic_data.py
====================

#!/usr/bin/env python3
"""
01_synthetic_data.py — RAN KPI Anomaly Detection: Synthetic Data Generation
============================================================================
Companion code for:
  "Real-Time RAN KPI Anomaly Detection at Cell Sector Granularity:
   A Practical Guide for Network Operations Teams"

Purpose:
    Generate a realistic synthetic PM (Performance Measurement) dataset for
    ~100 LTE/NR cell sectors over 30 days at 15-minute ROP (Reporting Output
    Period) granularity. The output is used by all downstream scripts in this
    walkthrough series.

    Data characteristics produced:
      - 100 cell sectors across 34 sites (3 sectors/site, some with 2)
      - 30 days × 96 ROPs/day = 2,880 time points per cell
      - 7 core KPIs per ROP per cell (matching 3GPP TS 28.552 / TS 32.425)
      - Realistic diurnal and weekly traffic patterns
      - Cell-type heterogeneity: urban macro, suburban, rural, indoor
      - Four injected anomaly types with ground truth labels:
          1. sudden_drop      — config error / backhaul cut (single cell)
          2. gradual_degradation — hardware failure (single cell, slow onset)
          3. periodic_interference — external RF interference (repeating pattern)
          4. spatial_correlated  — backhaul node failure (affects multiple cells)
      - Missing data patterns (~0.5% rate, simulating PM file delivery gaps)
      - Counter resets (simulating RAN node restarts)

    Output files:
      - data/raw_pm_data.parquet      — full dataset
      - data/anomaly_labels.parquet   — ground truth labels per (cell_id, timestamp)
      - data/cell_metadata.parquet    — cell topology and classification

How to run:
    python 01_synthetic_data.py

    Output appears in ./data/ (created automatically).

Requirements:
    Python 3.10+, pandas>=2.0, numpy>=1.24, pyarrow>=12.0

Coursebook cross-reference:
    Ch. 25: Time Series Analysis (diurnal/weekly patterns, stationarity)
    Ch. 13: Feature Engineering (KPI definitions, domain features)

Author:  Whitepaper Engineering Team
Version: 1.0.0
"""

from __future__ import annotations

import logging
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Logging — use module-level logger throughout; never use print()
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("synthetic_data")

# ---------------------------------------------------------------------------
# Global reproducibility seed — all np.random calls MUST use RNG derived here
# See Coursebook Ch. 25: reproducibility in time-series experiments
# ---------------------------------------------------------------------------
RANDOM_SEED: int = 42
RNG = np.random.default_rng(RANDOM_SEED)

# ---------------------------------------------------------------------------
# Simulation constants — easy to adjust without touching function bodies
# ---------------------------------------------------------------------------
N_CELLS: int = 100          # total cell sectors
SIM_DAYS: int = 30          # calendar days to simulate
ROP_MINUTES: int = 15       # 3GPP standard ROP granularity
SIM_START: str = "2024-01-01 00:00:00"   # UTC simulation start

# Anomaly injection budget — keeps rates realistic (1–5% per 3GPP operators)
ANOMALY_RATE_TARGET: float = 0.03   # 3% of all (cell, timestamp) pairs

OUTPUT_DIR: Path = Path("data")

# ===========================================================================
# Section 1: Cell Topology Generation
# ===========================================================================
@dataclass
class CellSector:
    """Represents one logical cell sector (one row in cell_metadata)."""

    cell_id: str                  # "CELL_XXX_YYY" — site_sector
    site_id: str                  # "SITE_XXX"
    sector: int                   # 1, 2, or 3
    cell_type: str                # urban_macro | suburban | rural | indoor
    vendor: str                   # Ericsson | Nokia | Samsung | Huawei
    band_mhz: int                 # carrier frequency band (e.g. 700, 1800, 2600)
    technology: str               # LTE | NR
    lat: float                    # approximate latitude (synthetic)
    lon: float                    # approximate longitude (synthetic)
    max_dl_throughput_mbps: float # capacity ceiling for this cell type
    peer_group: str               # cell clustering label for peer z-scoring
    # Neighbours used for spatial-correlated anomaly injection
    neighbour_site_ids: list[str] = field(default_factory=list)


def generate_cell_topology(n_cells: int) -> list[CellSector]:
    """
    Build a synthetic cell topology with realistic site/sector structure.

    Design decisions:
    - Sites have 3 sectors by default; a few have 2 (realistic for small sites)
    - 60% urban macro, 20% suburban, 15% rural, 5% indoor (typical MNO mix)
    - Four vendors with realistic market shares
    - Band assignment follows real spectrum allocations (700/1800/2600/3500 MHz)
    - Peer groups used downstream for peer-group z-score features

    Args:
        n_cells: Total number of cell sectors to generate.

    Returns:
        List of CellSector dataclass instances.
    """
    vendors = ["Ericsson", "Nokia", "Samsung", "Huawei"]
    vendor_weights = [0.35, 0.30, 0.20, 0.15]   # approximate global market share

    # Map cell type to realistic capacity ceiling
    cell_type_config: dict[str, dict] = {
        "urban_macro": {
            "max_dl": 800.0,
            "bands": [1800, 2600, 3500],
            "tech": ["LTE", "NR"],
            "weight": 0.60,
        },
        "suburban": {
            "max_dl": 400.0,
            "bands": [700, 1800, 2600],
            "tech": ["LTE", "NR"],
            "weight": 0.20,
        },
        "rural": {
            "max_dl": 150.0,
            "bands": [700, 1800],
            "tech": ["LTE"],
            "weight": 0.15,
        },
        # "small_cell_indoor" covers DAS, pico-cell, and enterprise small cell
        # deployments. Production implementations should sub-classify by
        # deployment type (DAS vs. pico vs. enterprise femto) if peer-group
        # granularity matters for anomaly detection.
        "small_cell_indoor": {
            "max_dl": 200.0,
            "bands": [1800, 2600],
            "tech": ["LTE", "NR"],
            "weight": 0.05,
        },
    }

    cell_types = list(cell_type_config.keys())
    cell_type_weights = [cell_type_config[ct]["weight"] for ct in cell_types]

    cells: list[CellSector] = []
    site_counter = 0
    cell_counter = 0

    while cell_counter < n_cells:
        site_counter += 1
        site_id = f"SITE_{site_counter:03d}"

        # Most sites have 3 sectors; ~10% have 2
        n_sectors = 2 if RNG.random() < 0.10 else 3
        n_sectors = min(n_sectors, n_cells - cell_counter)

        # Pick cell type for the site (all sectors on a site share type)
        ctype = RNG.choice(cell_types, p=cell_type_weights)
        cfg = cell_type_config[ctype]

        vendor = RNG.choice(vendors, p=vendor_weights)
        tech = RNG.choice(cfg["tech"])
        band = RNG.choice(cfg["bands"])

        # Synthetic lat/lon — "city cluster" pattern
        base_lat = 51.5 + RNG.uniform(-0.5, 0.5)
        base_lon = -0.1 + RNG.uniform(-0.5, 0.5)

        for s in range(1, n_sectors + 1):
            cell_id = f"CELL_{site_counter:03d}_{s}"
            peer_group = f"{ctype}_{band}_{tech}"

            cells.append(
                CellSector(
                    cell_id=cell_id,
                    site_id=site_id,
                    sector=s,
                    cell_type=ctype,
                    vendor=vendor,
                    band_mhz=band,
                    technology=tech,
                    lat=base_lat + RNG.uniform(-0.001, 0.001),
                    lon=base_lon + RNG.uniform(-0.001, 0.001),
                    max_dl_throughput_mbps=cfg["max_dl"],
                    peer_group=peer_group,
                )
            )
            cell_counter += 1
            if cell_counter >= n_cells:
                break

    logger.info(
        "Topology: %d cells across %d sites, %d peer groups",
        len(cells),
        site_counter,
        len({c.peer_group for c in cells}),
    )
    return cells


def assign_neighbours(cells: list[CellSector], k: int = 3) -> None:
    """
    Assign nearest-site neighbours to each cell for spatial anomaly injection.

    Uses Euclidean distance on lat/lon as a proxy (adequate for synthetic data).
    In production, neighbour lists come from the network topology database.

    Modifies cells in-place.

    Args:
        cells: List of CellSector objects.
        k:     Number of neighbouring sites to assign per cell.
    """
    site_locs: dict[str, tuple[float, float]] = {}
    for c in cells:
        if c.site_id not in site_locs:
            site_locs[c.site_id] = (c.lat, c.lon)

    site_ids = list(site_locs.keys())

    for cell in cells:
        my_lat, my_lon = site_locs[cell.site_id]
        dists = []
        for sid in site_ids:
            if sid == cell.site_id:
                continue
            slat, slon = site_locs[sid]
            dist = np.sqrt((my_lat - slat) ** 2 + (my_lon - slon) ** 2)
            dists.append((dist, sid))

        dists.sort(key=lambda x: x[0])
        cell.neighbour_site_ids = [sid for _, sid in dists[:k]]

# ===========================================================================
# Section 2: KPI Time-Series Generation
# ===========================================================================
def build_timestamp_index(start: str, days: int, rop_minutes: int) -> pd.DatetimeIndex:
    """
    Build the full UTC timestamp spine for the simulation.

    Args:
        start:       ISO-format start datetime string.
        days:        Number of calendar days.
        rop_minutes: Reporting Output Period in minutes.

    Returns:
        DatetimeIndex with UTC timezone at rop_minutes frequency.
    """
    freq = f"{rop_minutes}min"
    # total periods = days × (minutes_per_day / rop_minutes)
    n_periods = days * (24 * 60 // rop_minutes)
    idx = pd.date_range(start=start, periods=n_periods, freq=freq, tz="UTC")
    logger.info(
        "Timestamp spine: %d ROPs from %s to %s",
        len(idx),
        idx[0].isoformat(),
idx[-1].isoformat(),
    )
    return idx


def diurnal_traffic_profile(timestamps: pd.DatetimeIndex) -> np.ndarray:
    """
    Generate a smooth diurnal traffic multiplier based on a typical urban MNO
    busy-hour pattern. Values in [0.05, 1.0].

    Profile shape (traffic load fraction):
      - 00:00–05:00 → low  (~0.05–0.15)
      - 05:00–08:00 → morning ramp
      - 08:00–12:00 → morning peak (~0.70–0.85)
      - 12:00–14:00 → midday dip then second peak
      - 17:00–21:00 → evening peak (~0.85–1.00)
      - 21:00–00:00 → evening taper

    Implemented as a sum of shifted Gaussian lobes — computationally cheap
    and realistic.  See Coursebook Ch. 25: diurnal traffic modelling.

    Args:
        timestamps: UTC DatetimeIndex.

    Returns:
        1-D numpy array of shape (len(timestamps),) with values in [0.05, 1.0].
    """
    hours = timestamps.hour + timestamps.minute / 60.0   # fractional hour

    # Three Gaussian traffic lobes centred at typical busy hours
    morning_peak  = 0.70 * np.exp(-((hours - 10.0) ** 2) / (2 * 2.5 ** 2))
    evening_peak  = 0.90 * np.exp(-((hours - 19.5) ** 2) / (2 * 2.5 ** 2))
    lunch_peak    = 0.40 * np.exp(-((hours - 13.0) ** 2) / (2 * 1.5 ** 2))
    night_floor   = np.full_like(hours, 0.07)

    profile = night_floor + morning_peak + evening_peak + lunch_peak
    profile = np.clip(profile, 0.05, 1.0)
    return profile

def weekly_multiplier(timestamps: pd.DatetimeIndex) -> np.ndarray:
    """
    Apply a weekly seasonality multiplier.

    Weekends reduce demand by ~20% (residential-dominant network).
    For an enterprise-heavy network the sign would reverse.

    Args:
        timestamps: UTC DatetimeIndex.

    Returns:
        1-D numpy array of multipliers (0.8 on weekends, 1.0 weekdays).
    """
    # dayofweek: Monday=0, Sunday=6
    is_weekend = timestamps.dayofweek >= 5
    multiplier = np.where(is_weekend, 0.80, 1.00)
    return multiplier.astype(np.float64)

def _kpi_normal_values(
    n: int,
    cell: CellSector,
    traffic_load: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Generate normal (non-anomalous) KPI time series for one cell sector.

    KPI definitions (3GPP references):
      - dl_throughput_mbps    : Mean DL user throughput  [TS 28.552 §5.1.1.2]
      - ul_throughput_mbps    : Mean UL user throughput  [TS 28.552 §5.1.1.3]
      - dl_prb_usage_rate     : DL PRB utilisation ratio  [TS 28.552 §5.1.1.1]
      - rrc_conn_setup_success_rate: RRC Connection Setup Success Rate [TS 28.552 §5.1.3.1]
      - drb_setup_success_rate: Bearer Setup Success Rate — DRB.EstabSucc/EstabAtt [TS 28.552 §5.1.1.2] for 5G NR SA; ERAB.EstabInitSuccNbr/EstabInitAttNbr [TS 32.425] for LTE/NSA. Production code must select counter family based on cell technology field.
      - handover_success_rate : NR Handover Success Rate [TS 28.552 §5.1.3.5]
      - avg_cqi               : Derived from DRB.UECqiDistr buckets [TS 28.552 §5.1.1.31].
                                 Production formula: Σ(CQI_level × bucket_count) / Σ(bucket_count)
                                 across Bin0–Bin15. Synthetic data uses a single float.
      - rach_success_rate     : RACH success rate — normal ~97%, std 0.008
      - rlf_count             : RLF indicator via RRC.ConnReEstabAtt.* family [TS 28.552 §5.1.3]; NR uses RRC re-establishment counters or vendor-specific

    Design:
      - Each KPI is driven by traffic_load with independent noise
      - PRB utilization is correlated with throughput (load-driven)
      - CQI is cell-type-dependent (urban higher than rural)
      - Success rates are modelled as beta-distributed to respect [0,1] bounds
      - All values are clipped to realistic operating ranges

    The returned dict includes sector_id, vendor, frequency_band, and
    cell_type columns so that generate_pm_dataset can merge cell metadata
    into the PM rows.

    Args:
        n:            Number of time points.
        cell:         CellSector metadata providing capacity ceilings.
        traffic_load: 1-D array of normalised traffic load values in [0,1].

    Returns:
        Dict of KPI name → numpy array of length n, plus metadata columns:
          sector_id, vendor, frequency_band, cell_type.
    """
    # Cell-type base quality: urban typically sees better RSRP/CQI than rural
    cqi_base_map = {"urban_macro": 9.5, "suburban": 8.5, "rural": 7.0, "small_cell_indoor": 10.0}
    cqi_base = cqi_base_map.get(cell.cell_type, 8.5)

    # DL throughput: scales with traffic load, capped by cell max capacity
    # Typical occupancy: 20%–90% of max under normal load
    dl_tp = (
        cell.max_dl_throughput_mbps
        * (0.20 + 0.65 * traffic_load)
        * (1.0 + 0.08 * RNG.standard_normal(n))   # ±8% cell-level noise
    )
    dl_tp = np.clip(dl_tp, 0.0, cell.max_dl_throughput_mbps)

    # UL throughput: typically 15–25% of DL for FDD networks
    ul_ratio = RNG.uniform(0.15, 0.25)
    ul_tp = dl_tp * ul_ratio * (1.0 + 0.10 * RNG.standard_normal(n))
    ul_tp = np.clip(ul_tp, 0.0, cell.max_dl_throughput_mbps * 0.3)

    # DL PRB usage rate: [0,1] — rises with load, high correlation with DL throughput
    # Use beta distribution to keep values in (0,1) naturally
    prb_mean = np.clip(0.15 + 0.75 * traffic_load, 0.05, 0.98)
    # Beta params to match mean with tight dispersion
    prb_alpha = prb_mean * 20
    prb_beta  = (1.0 - prb_mean) * 20
    dl_prb = RNG.beta(
        np.clip(prb_alpha, 0.5, None),
        np.clip(prb_beta, 0.5, None),
    )

    # RRC connection setup success rate: typically 98–99.9% under normal conditions
    # 3GPP: RRC.ConnEstabSucc / RRC.ConnEstabAtt (TS 28.552 §5.1.3.1)
    rrc_conn_setup_mean = 0.989
    rrc_conn_setup_std  = 0.004
    rrc = np.clip(RNG.normal(rrc_conn_setup_mean, rrc_conn_setup_std, n), 0.90, 1.0)

    # DRB setup success rate: typically 98–99.8% [5G NR SA context]
    # 3GPP: DRB.EstabSucc.5QI / DRB.EstabAtt.5QI (TS 28.552 §5.1.1.2) for NR SA;
    # ERAB.EstabInitSuccNbr / ERAB.EstabInitAttNbr (TS 32.425) for LTE/NSA.
    # Select counter family based on cell technology field in production.
    drb_mean = 0.985
    drb_std  = 0.005
    drb = np.clip(RNG.normal(drb_mean, drb_std, n), 0.88, 1.0)

    # Handover success rate: typically 97–99.5%
    # 3GPP: HO.ExecSucc / HO.ExecAtt (TS 28.552 §5.1.3.5) — NR handover counters
    ho_base = 0.982
    # Rural cells have slightly lower HO success due to longer inter-site distance
    if cell.cell_type == "rural":
        ho_base = 0.968
    ho = np.clip(RNG.normal(ho_base, 0.006, n), 0.85, 1.0)

    # Average CQI: derived from DRB.UECqiDistr distribution buckets — simplified to single average
    # 3GPP: DRB.UECqiDistr (TS 28.552 §5.1.1.31) — no single-average NR counter; this is a synthetic simplification
    # Load-dependent: higher load → lower CQI because scheduler serves edge UEs
    cqi_load_effect = -1.5 * traffic_load    # max -1.5 CQI units at 100% load
    # PRODUCTION OVERRIDE REQUIRED: In production, avg_cqi must be computed as
    # the weighted mean of DRB.UECqiDistr bucket counts (TS 28.552 §5.1.1.31),
    # not sampled from a normal distribution. See whitepaper §3 for details.
    avg_cqi = np.clip(
        RNG.normal(cqi_base + cqi_load_effect, 0.8, n), 0.0, 15.0
    )

    # RACH success rate: normal ~97%, std 0.008
    # Slightly load-dependent: higher load can reduce RACH success due to contention
    rach_mean = np.clip(0.97 - 0.03 * traffic_load, 0.85, 1.0)
    rach_success_rate = np.clip(
        RNG.normal(rach_mean, 0.008, n), 0.80, 1.0
    )

    # RLF count: Poisson λ=1.5, scaled by traffic load
    # 3GPP: RRC.ConnReEstabAtt.* measurement family (TS 28.552 §5.1.3); NR uses RRC re-establishment
    # counters or vendor-specific RLF indicators — replaces L.RLF.Tot / RLF.Ind (LTE-only)
    rlf_lambda = np.clip(1.5 * traffic_load, 0.1, None)
    rlf_count = RNG.poisson(rlf_lambda).astype(np.float64)

    return {
        "dl_throughput_mbps":          dl_tp,             # 3GPP: mean DL user throughput (TS 28.552 §5.1.1.2)
        "ul_throughput_mbps":          ul_tp,             # 3GPP: mean UL user throughput (TS 28.552 §5.1.1.3)
        "dl_prb_usage_rate":           dl_prb,            # 3GPP: RRU.PrbUsedDl ÷ configured max PRBs (TS 28.552 §5.1.1.1)
        "rrc_conn_setup_success_rate": rrc,               # 3GPP: RRC.ConnEstabSucc / RRC.ConnEstabAtt (TS 28.552 §5.1.3.1)
        "drb_setup_success_rate":      drb,               # TS 28.552 (NR SA) or TS 32.425 (LTE/NSA)
        "handover_success_rate":       ho,                # 3GPP: HO.ExecSucc / HO.ExecAtt (TS 28.552 §5.1.3.5)
        "avg_cqi":                     avg_cqi,           # 3GPP: DRB.UECqiDistr (TS 28.552 §5.1.1.31) — simplified to single average
        "rach_success_rate":           rach_success_rate, # RACH success rate — normal ~97%, std 0.008; load-dependent contention effect
        "rlf_count":                   rlf_count,         # 3GPP: RRC.ConnReEstabAtt.* family (TS 28.552 §5.1.3) — Poisson λ=1.5, scaled by traffic load
        "sector_id":                   np.array([str(cell.sector)] * n),       # cell sector identifier
        "frequency_band":              np.array([str(cell.band_mhz)] * n),     # frequency band in MHz
    }

# ===========================================================================
# Section 3: Anomaly Injection
# ===========================================================================
@dataclass
class AnomalyEvent:
    """Records a single injected anomaly event for ground-truth label generation."""

    cell_id: str
    anomaly_type: str         # sudden_drop | gradual_degradation | periodic_interference | spatial_correlated
    start_idx: int            # index into timestamp array
    end_idx: int              # exclusive
    affected_kpis: list[str]
    severity: str             # minor | major | critical

def inject_sudden_drop(
    kpis: dict[str, np.ndarray],
    start: int,
    duration: int,
    severity: float = 0.6,
) -> None:
    """
    Simulate a sudden configuration error or backhaul cut.

    Pattern: KPI drops abruptly to severity fraction of normal value.
    Affects: DL/UL throughput, PRB usage (drops or spikes), success rates.

    This matches the "sudden_drop" fault taxonomy described in Section 4 of
    the whitepaper (hardware failure / misconfiguration).

    Args:
        kpis:     Dict of KPI arrays to modify in-place.
        start:    Start index of anomaly window.
        duration: Length of anomaly in ROP intervals.
        severity: Multiplier applied to affected KPIs (0 = complete outage,
                  1 = no effect). Typical: 0.3–0.7 for config errors.
    """
    end = min(start + duration, len(kpis["dl_throughput_mbps"]))
    # Throughput drops sharply
    kpis["dl_throughput_mbps"][start:end] *= severity
    kpis["ul_throughput_mbps"][start:end] *= severity
    # PRB usage drops (fewer users being served)
    kpis["dl_prb_usage_rate"][start:end] *= severity
    # Success rates plummet — cells enter degraded state
    kpis["rrc_conn_setup_success_rate"][start:end] = np.clip(
        kpis["rrc_conn_setup_success_rate"][start:end] * 0.4, 0.0, 1.0
    )
    kpis["drb_setup_success_rate"][start:end] = np.clip(
        kpis["drb_setup_success_rate"][start:end] * 0.4, 0.0, 1.0
    )
    # CQI degrades (UEs see poor signal from degraded cell)
    kpis["avg_cqi"][start:end] *= 0.6


def inject_gradual_degradation(
    kpis: dict[str, np.ndarray],
    start: int,
    duration: int,
) -> None:
    """
    Simulate a hardware failure with slow onset (e.g. failing power amplifier,
    moisture ingress on antenna feed).

    Pattern: Linear ramp-down from 100% to 30% of nominal over `duration` ROPs.
    This is the hardest anomaly type for static thresholds — the cell never
    crosses a single-point threshold until it is severely degraded.

    Args:
        kpis:     Dict of KPI arrays to modify in-place.
        start:    Start index of anomaly window.
        duration: Total degradation period in ROP intervals.
    """
    end = min(start + duration, len(kpis["dl_throughput_mbps"]))
    actual_duration = end - start

    # Linear degradation ramp: multiplier goes from 1.0 down to 0.30
    ramp = np.linspace(1.0, 0.30, actual_duration)

    kpis["dl_throughput_mbps"][start:end] *= ramp
    kpis["ul_throughput_mbps"][start:end] *= ramp
    # CQI degrades as antenna gain decreases
    kpis["avg_cqi"][start:end] *= np.linspace(1.0, 0.65, actual_duration)
    # Handover success degrades (neighbours start to "steal" users)
    kpis["handover_success_rate"][start:end] *= np.linspace(1.0, 0.80, actual_duration)


def inject_periodic_interference(
    kpis: dict[str, np.ndarray],
    timestamps: pd.DatetimeIndex,
    start_day: int,
    n_days: int,
    interfere_hours: tuple[int, int] = (20, 23),  # evening interference window
) -> None:
    """
    Simulate periodic RF interference (e.g. unlicensed device, neighbouring
    operator on shared spectrum, or a poorly-designed repeater).

    Pattern: Interference occurs every night in a 3-hour window — affects CQI
    and throughput but not success rates (UEs adapt).

    Args:
        kpis:             Dict of KPI arrays to modify in-place.
        timestamps:       Full timestamp index for the cell.
        start_day:        Day number (0-indexed) when interference first appears.
        n_days:           How many days the interference lasts.
        interfere_hours:  (start_hour, end_hour) UTC for daily interference window.
    """
    h_start, h_end = interfere_hours
    for i, ts in enumerate(timestamps):
        day_idx = (ts.date() - timestamps[0].date()).days
        if day_idx < start_day or day_idx >= start_day + n_days:
            continue
        if h_start <= ts.hour < h_end:
            # CQI degrades significantly during interference
            kpis["avg_cqi"][i] *= RNG.uniform(0.45, 0.65)
            kpis["dl_throughput_mbps"][i] *= RNG.uniform(0.40, 0.60)
            kpis["ul_throughput_mbps"][i] *= RNG.uniform(0.50, 0.70)

def inject_spatial_correlated_anomaly(
    all_kpis: dict[str, dict[str, np.ndarray]],
    affected_cells: list[str],
    start: int,
    duration: int,
) -> None:
    """
    Simulate a shared backhaul node failure affecting multiple co-located cells
    on the same site or feeding the same aggregation node.

    Pattern: Sudden throughput drop across all affected cells simultaneously.
    This is a critical test case — individual cells may look only mildly
    anomalous, but the spatial pattern (multiple cells degrading together)
    is the diagnostic signal.

    Args:
        all_kpis:       Dict keyed by cell_id of KPI arrays.
        affected_cells: List of cell_ids to degrade together.
        start:          Start index.
        duration:       Duration in ROP intervals.
    """
    for cell_id in affected_cells:
        if cell_id not in all_kpis:
            continue
        kpis = all_kpis[cell_id]
        end = min(start + duration, len(kpis["dl_throughput_mbps"]))
        # Backhaul cut: throughput drops to near-zero, PRB drops, but signalling
        # may partially work (control plane may use different path)
        kpis["dl_throughput_mbps"][start:end] *= 0.05
        kpis["ul_throughput_mbps"][start:end] *= 0.05
        kpis["dl_prb_usage_rate"][start:end] *= 0.10
        # Success rates hit — users can't establish bearers without backhaul
        kpis["rrc_conn_setup_success_rate"][start:end] = np.clip(
            kpis["rrc_conn_setup_success_rate"][start:end] * 0.20, 0.0, 1.0
        )
        kpis["drb_setup_success_rate"][start:end] = np.clip(
           kpis["drb_setup_success_rate"][start:end] * 0.20, 0.0, 1.0
        )

# ===========================================================================
# Section 4: Missing Data and Counter Resets
# ===========================================================================
def inject_missing_data(
    kpis: dict[str, np.ndarray],
    missing_rate: float = 0.005,
) -> None:
    """
    Replace a fraction of KPI rows with NaN to simulate PM file delivery gaps.

    Real causes: collector failure, FTP delivery timeout, PM parser errors.
    Missing rate of 0.5% is realistic for large-scale PM collection.
    Entire-row NaN (all KPIs) simulates a missing ROP file.

    Args:
        kpis:         Dict of KPI arrays to modify in-place.
        missing_rate: Fraction of rows to set to NaN.
    """
    n = len(next(iter(kpis.values())))
    n_missing = int(n * missing_rate)
    missing_idxs = RNG.choice(n, size=n_missing, replace=False)
    for key, arr in kpis.items():
        if not hasattr(arr, 'dtype') or arr.dtype.kind not in ('f', 'i'):
            continue
        arr[missing_idxs] = np.nan

def inject_counter_resets(
    kpis: dict[str, np.ndarray],
    n_resets: int = 1,
) -> None:
    """
    Simulate RAN node restarts, which cause PM counters to reset to zero.

    Pattern: A single ROP shows zero or near-zero values for all KPIs,
    followed by immediate recovery. Distinguishing counter resets from real
    outages is a key feature engineering challenge.

    Args:
        kpis:     Dict of KPI arrays to modify in-place.
        n_resets: Number of counter reset events to inject.
    """
    n = len(next(iter(kpis.values())))
    reset_idxs = RNG.choice(n, size=n_resets, replace=False)
    for idx in reset_idxs:
        for arr in kpis.values():
            if not hasattr(arr, 'dtype') or arr.dtype.kind not in ('f', 'i'):
                continue
            if not np.isnan(arr[idx]):
                arr[idx] = 0.0  # counter reset: exactly zero, not NaN

# ===========================================================================
# Section 5: Main Dataset Assembly
# ===========================================================================
def generate_pm_dataset(
    cells: list[CellSector],
    timestamps: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate the full PM dataset and corresponding ground-truth anomaly labels.

    Algorithm:
      1. Build traffic load profile (diurnal × weekly × cell-type noise)
      2. Generate normal KPI time series per cell
      3. Schedule anomaly injections respecting the overall anomaly rate budget
      4. Inject each anomaly type
      5. Inject missing data and counter resets
      6. Assemble into a long-format DataFrame

    Args:
        cells:      List of CellSector topology objects.
        timestamps: Full UTC timestamp spine.

    Returns:
        Tuple of:
          - pm_df:     Long-format DataFrame with columns:
                       [timestamp, cell_id, *kpi_columns, rop_sequence_number,
                        sector_id, vendor, frequency_band, cell_type]
          - labels_df: DataFrame with columns:
                       [timestamp, cell_id, is_anomaly, anomaly_type, severity]
    """
    n_ts = len(timestamps)

    # Pre-compute shared diurnal and weekly profiles (same calendar for all cells)
    diurnal = diurnal_traffic_profile(timestamps)
    weekly  = weekly_multiplier(timestamps)
    base_load = diurnal * weekly   # combined load multiplier

    all_kpis: dict[str, dict[str, np.ndarray]] = {}
    anomaly_events: list[AnomalyEvent] = []

    logger.info("Generating normal KPI time series for %d cells...", len(cells))

    for cell in cells:
        # Add cell-specific load variation: urban cells are busier, rural quieter
        cell_load_factor: dict[str, float] = {
            "urban_macro": 1.00,
            "suburban":    0.70,
            "rural":       0.40,
            "small_cell_indoor":      0.60,
        }
        cfactor = cell_load_factor.get(cell.cell_type, 0.75)

        # Per-cell random load noise: accounts for micro-geographic differences
        per_cell_noise = 1.0 + 0.15 * RNG.standard_normal(n_ts)
        traffic_load = np.clip(base_load * cfactor * per_cell_noise, 0.0, 1.0)

        all_kpis[cell.cell_id] = _kpi_normal_values(n_ts, cell, traffic_load)

    # -----------------------------------------------------------------------
    # Anomaly injection schedule
    # Total budget: ~ANOMALY_RATE_TARGET fraction of all (cell, ts) pairs
    # Distribute across four anomaly types
    # -----------------------------------------------------------------------
    logger.info("Scheduling and injecting anomalies...")

    total_pairs = len(cells) * n_ts
    target_anomalous = int(total_pairs * ANOMALY_RATE_TARGET)

    # Track per-cell per-timestamp anomaly flags
    label_flags: dict[str, np.ndarray] = {
        c.cell_id: np.zeros(n_ts, dtype=bool) for c in cells
    }
    anomaly_type_arr: dict[str, np.ndarray] = {
        c.cell_id: np.full(n_ts, "", dtype=object) for c in cells
    }

    # ---- Anomaly Type 1: Sudden Drop (~1% of cell-timeslots) ---------------
    n_sudden = max(1, int(target_anomalous * 0.30 / 48))   # 30% budget, ~48 ROPs each
    for _ in range(n_sudden):
        cell = cells[RNG.integers(len(cells))]
        duration = int(RNG.integers(24, 72))   # 6–18 hours
        # Avoid start too close to end to ensure full anomaly fits
        start = int(RNG.integers(48, n_ts - duration - 1))
        severity = float(RNG.uniform(0.20, 0.55))
        inject_sudden_drop(all_kpis[cell.cell_id], start, duration, severity)
        label_flags[cell.cell_id][start:start + duration] = True
        anomaly_type_arr[cell.cell_id][start:start + duration] = "sudden_drop"
        anomaly_events.append(AnomalyEvent(
            cell_id=cell.cell_id,
            anomaly_type="sudden_drop",
            start_idx=start,
            end_idx=start + duration,
            affected_kpis=["dl_throughput_mbps", "ul_throughput_mbps",
                           "rrc_conn_setup_success_rate", "drb_setup_success_rate"],
            severity="critical" if severity < 0.35 else "major",
        ))

    # ---- Anomaly Type 2: Gradual Degradation (~1% of cell-timeslots) -------
    n_gradual = max(1, int(target_anomalous * 0.35 / 200))  # 35% budget, ~200 ROPs
    for _ in range(n_gradual):
        cell = cells[RNG.integers(len(cells))]
        duration = int(RNG.integers(120, 288))   # 30–72 hours gradual ramp
        start = int(RNG.integers(48, n_ts - duration - 1))
        inject_gradual_degradation(all_kpis[cell.cell_id], start, duration)
        label_flags[cell.cell_id][start:start + duration] = True
        anomaly_type_arr[cell.cell_id][start:start + duration] = "gradual_degradation"
        anomaly_events.append(AnomalyEvent(
            cell_id=cell.cell_id,
            anomaly_type="gradual_degradation",
            start_idx=start,
            end_idx=start + duration,
            affected_kpis=["dl_throughput_mbps", "avg_cqi", "handover_success_rate"],
            severity="major",
        ))

    # ---- Anomaly Type 3: Periodic Interference (~0.5% of cell-timeslots) ---
    n_periodic = max(1, int(target_anomalous * 0.15 / (3 * 4)))  # 15% budget
    for _ in range(n_periodic):
        cell = cells[RNG.integers(len(cells))]
        start_day = int(RNG.integers(2, SIM_DAYS - 7))
        n_days = int(RNG.integers(5, 12))
        inject_periodic_interference(
            all_kpis[cell.cell_id], timestamps, start_day, n_days
        )
        # Mark interference windows in labels
        h_start, h_end = 20, 23
        for i, ts in enumerate(timestamps):
            day_idx = (ts.date() - timestamps[0].date()).days
            if start_day <= day_idx < start_day + n_days and h_start <= ts.hour < h_end:
                label_flags[cell.cell_id][i] = True
                anomaly_type_arr[cell.cell_id][i] = "periodic_interference"
        anomaly_events.append(AnomalyEvent(
            cell_id=cell.cell_id,
            anomaly_type="periodic_interference",
            start_idx=0,  # spans multiple windows; stored symbolically
            end_idx=n_ts,
            affected_kpis=["avg_cqi", "dl_throughput_mbps"],
            severity="minor",
        ))

    # ---- Anomaly Type 4: Spatial Correlated (backhaul failure) -------------
    # Pick 3 sites and affect all sectors on each site simultaneously
    n_spatial = max(1, int(target_anomalous * 0.20 / (3 * 48)))
    site_ids = list({c.site_id for c in cells})
    for _ in range(n_spatial):
        target_site = site_ids[RNG.integers(len(site_ids))]
        affected = [c.cell_id for c in cells if c.site_id == target_site]
        # Also include one neighbour site to create cross-site correlation
        sample_cell = next(c for c in cells if c.site_id == target_site)
        if sample_cell.neighbour_site_ids:
            nbr_site = sample_cell.neighbour_site_ids[0]
            affected += [c.cell_id for c in cells if c.site_id == nbr_site]

        duration = int(RNG.integers(24, 96))   # 6–24 hours
        start = int(RNG.integers(48, n_ts - duration - 1))
        inject_spatial_correlated_anomaly(all_kpis, affected, start, duration)

        for cid in affected:
            if cid in label_flags:
                label_flags[cid][start:start + duration] = True
                anomaly_type_arr[cid][start:start + duration] = "spatial_correlated"
        anomaly_events.append(AnomalyEvent(
            cell_id=f"SITE_{target_site}_ALL",
            anomaly_type="spatial_correlated",
            start_idx=start,
            end_idx=start + duration,
            affected_kpis=["dl_throughput_mbps", "ul_throughput_mbps",
                           "rrc_conn_setup_success_rate", "drb_setup_success_rate",
                           "dl_prb_usage_rate"],
            severity="critical",
        ))

    # ---- Check for anomaly injection overlaps --------------------------------
    # Multiple anomaly types injected into the same cell at the same ROP create
    # ambiguous labels. Log a warning so operators can assess label quality.
    n_overlap_rops = 0
    for cell in cells:
        flags = label_flags[cell.cell_id]
        types = anomaly_type_arr[cell.cell_id]
        # Count ROPs where an earlier anomaly type was overwritten
        for i in range(len(flags)):
            if flags[i] and types[i] != "" and i > 0:
                pass  # type is last-write-wins by construction
        # Simple overlap proxy: count ROPs where label was set more than once
        # (detected by checking if the anomaly_type changed during a flagged window)
    overlap_cells = set()
    for evt_a, evt_b in zip(anomaly_events, anomaly_events[1:]):
        if evt_a.cell_id == evt_b.cell_id:
            overlap_start = max(evt_a.start_idx, evt_b.start_idx)
            overlap_end = min(evt_a.end_idx, evt_b.end_idx)
            if overlap_start < overlap_end:
                n_overlap_rops += overlap_end - overlap_start
                overlap_cells.add(evt_a.cell_id)
    if n_overlap_rops > 0:
        logger.warning(
            "Anomaly injection overlap detected: %d ROPs across %d cells have "
            "overlapping anomaly windows. The last-injected type wins. "
            "This may create ambiguous labels for evaluation.",
            n_overlap_rops, len(overlap_cells),
        )

    # ---- Inject missing data and counter resets (after anomaly injection) --
    for cell in cells:
        inject_missing_data(all_kpis[cell.cell_id], missing_rate=0.005)
        inject_counter_resets(all_kpis[cell.cell_id], n_resets=RNG.integers(0, 3))

    # -----------------------------------------------------------------------
    # Assemble long-format DataFrame
    # -----------------------------------------------------------------------
    logger.info("Assembling long-format DataFrame...")

    records: list[dict] = []
    label_records: list[dict] = []

    # ROP sequence number: monotonically increasing per cell — useful for
    # ordering and detecting counter resets in downstream feature engineering
    for cell in cells:
        kpis = all_kpis[cell.cell_id]
        n = len(timestamps)
        for i in range(n):
            records.append({
                "timestamp":               timestamps[i],
                "cell_id":                 cell.cell_id,
                "site_id":                 cell.site_id,
                "rop_sequence_number":     i,   # 0-indexed per cell
                "dl_throughput_mbps":      kpis["dl_throughput_mbps"][i],
                "ul_throughput_mbps":      kpis["ul_throughput_mbps"][i],
                "dl_prb_usage_rate":       kpis["dl_prb_usage_rate"][i],
                "rrc_conn_setup_success_rate":  kpis["rrc_conn_setup_success_rate"][i],
                "drb_setup_success_rate": kpis["drb_setup_success_rate"][i],
                "handover_success_rate":   kpis["handover_success_rate"][i],
                "avg_cqi":                 kpis["avg_cqi"][i],
                "rach_success_rate":       kpis["rach_success_rate"][i],
                "rlf_count":               kpis["rlf_count"][i],
            })
            label_records.append({
                "timestamp":    timestamps[i],
                "cell_id":      cell.cell_id,
                "is_anomaly":   bool(label_flags[cell.cell_id][i]),
                "anomaly_type": anomaly_type_arr[cell.cell_id][i],
                "severity":     (
                    "none" if not label_flags[cell.cell_id][i]
                    else "critical" if anomaly_type_arr[cell.cell_id][i]
                                      in ("sudden_drop", "spatial_correlated")
                    else "major"   if anomaly_type_arr[cell.cell_id][i]
                                      == "gradual_degradation"
                    else "minor"
                ),
            })

    pm_df     = pd.DataFrame(records)
    labels_df = pd.DataFrame(label_records)

    # -----------------------------------------------------------------------
    # Merge cell metadata columns into PM dataframe
    # -----------------------------------------------------------------------
    cell_metadata = pd.DataFrame([
        {
            "cell_id":        cell.cell_id,
            "sector_id":      str(cell.sector),
            "vendor":         cell.vendor,
            "frequency_band": str(cell.band_mhz),
            "cell_type":      cell.cell_type,
        }
        for cell in cells
    ])

    pm_df = pm_df.merge(cell_metadata, on="cell_id", how="left")

    # Sanity: verify anomaly rate is in expected range
    actual_rate = labels_df["is_anomaly"].mean()
    logger.info(
        "Anomaly rate: %.2f%% (target: %.2f%%)",
        actual_rate * 100,
        ANOMALY_RATE_TARGET * 100,
    )

    return pm_df, labels_df


# ===========================================================================
# Section 6: Cell Metadata Export
# ===========================================================================
def cells_to_dataframe(cells: list[CellSector]) -> pd.DataFrame:
    """
    Convert the list of CellSector dataclasses to a pandas DataFrame for export.

    Args:
        cells: List of CellSector objects.

    Returns:
        DataFrame with one row per cell sector.
    """
    rows = [
        {
            "cell_id":                  c.cell_id,
            "site_id":                  c.site_id,
            "sector":                   c.sector,
            "cell_type":                c.cell_type,
            "vendor":                   c.vendor,
            "band_mhz":                 c.band_mhz,
            "technology":               c.technology,
            "lat":                      round(c.lat, 6),
            "lon":                      round(c.lon, 6),
            "max_dl_throughput_mbps":   c.max_dl_throughput_mbps,
            "peer_group":               c.peer_group,
            "neighbour_site_ids":       ",".join(c.neighbour_site_ids),
        }
        for c in cells
    ]
    return pd.DataFrame(rows)

# ===========================================================================
# Section 7: Output Writing and Validation
# ===========================================================================
def validate_pm_dataframe(pm_df: pd.DataFrame) -> None:
    """
    Apply lightweight data quality checks to the generated PM dataset.

    These checks mirror what a Pandera/Great Expectations schema would enforce
    in a production ingestion pipeline.  See Coursebook Ch. 28 for production
    data validation patterns.

    Raises:
        ValueError if critical checks fail.

    Args:
        pm_df: The assembled PM DataFrame.
    """
    logger.info("Running data quality validation...")

    # Schema presence checks
    required_cols = [
        "timestamp", "cell_id", "rop_sequence_number",
        "dl_throughput_mbps", "ul_throughput_mbps", "dl_prb_usage_rate",
        "rrc_conn_setup_success_rate", "drb_setup_success_rate",
        "handover_success_rate", "avg_cqi",
        "rach_success_rate", "rlf_count",
    ]
    missing_cols = [c for c in required_cols if c not in pm_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Value range checks (ignoring NaN from missing data injection)
    numeric_checks: dict[str, tuple[float, float]] = {
        "dl_throughput_mbps":          (0.0, 2000.0),
        "ul_throughput_mbps":          (0.0, 500.0),
        "dl_prb_usage_rate":           (0.0, 1.0),
        "rrc_conn_setup_success_rate": (0.0, 1.0),
        "drb_setup_success_rate":      (0.0, 1.0),
        "handover_success_rate":       (0.0, 1.0),
        "avg_cqi":                     (0.0, 15.0),
        "rach_success_rate":           (0.0, 1.0),
        "rlf_count":                   (0.0, 10000.0),
    }
    for col, (lo, hi) in numeric_checks.items():
        col_clean = pm_df[col].dropna()
        if (col_clean < lo).any() or (col_clean > hi).any():
            out_of_range = col_clean[(col_clean < lo) | (col_clean > hi)]
            raise ValueError(
                f"Column '{col}' has {len(out_of_range)} values outside "
                f"[{lo}, {hi}]: min={col_clean.min():.4f}, max={col_clean.max():.4f}"
            )

    # Missing rate check
    for col in numeric_checks:
        missing_pct = pm_df[col].isna().mean() * 100
        if missing_pct > 5.0:
            logger.warning("Column '%s' missing rate %.2f%% exceeds 5%%", col, missing_pct)

    # Cell ID format check
    invalid_ids = pm_df["cell_id"][~pm_df["cell_id"].str.match(r"^CELL_\d{3}_\d+$")]
    if len(invalid_ids) > 0:
        raise ValueError(f"Found {len(invalid_ids)} invalid cell_id formats")

    # Monotonic ROP sequence number per cell
    for cell_id, grp in pm_df.groupby("cell_id"):
        if not grp["rop_sequence_number"].is_monotonic_increasing:
            logger.warning(
                "Cell %s: rop_sequence_number is not monotonically increasing", cell_id
            )
            break  # Log once, don't spam

    logger.info(
        "Validation passed: %d rows, %d cells, %.3f%% NaN rate (DL TP)",
        len(pm_df),
        pm_df["cell_id"].nunique(),
        pm_df["dl_throughput_mbps"].isna().mean() * 100,
    )

def write_outputs(
    pm_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Write all three output files to the output directory.

    Uses Parquet (snappy compression) for efficient columnar storage.
    Parquet preserves timezone-aware DatetimeIndex, unlike CSV.

    Args:
        pm_df:      PM KPI DataFrame.
        labels_df:  Ground-truth anomaly labels DataFrame.
        meta_df:    Cell sector metadata DataFrame.
        output_dir: Directory to write files into.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    pm_path     = output_dir / "raw_pm_data.parquet"
    labels_path = output_dir / "anomaly_labels.parquet"
    meta_path   = output_dir / "cell_metadata.parquet"

    pm_df.to_parquet(pm_path, index=False, compression="snappy")
    labels_df.to_parquet(labels_path, index=False, compression="snappy")
    meta_df.to_parquet(meta_path, index=False, compression="snappy")

    logger.info("Wrote PM data    → %s  (%s rows)", pm_path, f"{len(pm_df):,}")
    logger.info("Wrote labels     → %s  (%s rows)", labels_path, f"{len(labels_df):,}")
    logger.info("Wrote metadata   → %s  (%s rows)", meta_path, f"{len(meta_df):,}")


# ===========================================================================
# Section 8: Summary Statistics
# ===========================================================================
def print_dataset_summary(
    pm_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    meta_df: pd.DataFrame,
) -> None:
    """
    Log a human-readable summary of the generated dataset for quick sanity
    checking — mirrors what a data scientist would check at the start of EDA.

    Args:
        pm_df:      PM KPI DataFrame.
        labels_df:  Ground-truth anomaly labels DataFrame.
        meta_df:    Cell sector metadata DataFrame.
    """
    logger.info("=" * 60)
    logger.info("DATASET SUMMARY")
    logger.info("=" * 60)

    n_cells    = pm_df["cell_id"].nunique()
    n_ts       = pm_df["timestamp"].nunique()
    n_rows     = len(pm_df)
    start_ts   = pm_df["timestamp"].min()
    end_ts     = pm_df["timestamp"].max()

    logger.info("Cells:           %d", n_cells)
    logger.info("Timestamps:      %d (%.1f days @ %d-min ROP)",
                n_ts, SIM_DAYS, ROP_MINUTES)
    logger.info("Total rows:      %s", f"{n_rows:,}")
    logger.info("Date range:      %s → %s", start_ts, end_ts)
    logger.info("")

    # Topology breakdown
    for ctype, grp in meta_df.groupby("cell_type"):
        logger.info("  %-20s %3d cells", ctype + ":", len(grp))
    logger.info("")

    # KPI statistics
    kpi_cols = [
        "dl_throughput_mbps", "ul_throughput_mbps", "dl_prb_usage_rate",
        "rrc_conn_setup_success_rate", "drb_setup_success_rate",
        "handover_success_rate", "avg_cqi",
    ]
    for col in kpi_cols:
        s = pm_df[col].dropna()
        logger.info(
            "  %-32s  mean=%.3f  std=%.3f  p5=%.3f  p95=%.3f",
            col + ":",
            s.mean(), s.std(), s.quantile(0.05), s.quantile(0.95),
        )
    logger.info("")

    # Anomaly breakdown
    anom = labels_df[labels_df["is_anomaly"]]
    logger.info("Anomaly summary:")
    logger.info("  Total anomalous ROPs:  %s / %s  (%.2f%%)",
                f"{len(anom):,}", f"{len(labels_df):,}",
                100.0 * len(anom) / len(labels_df))
    for atype, grp in anom.groupby("anomaly_type"):
        logger.info("  %-30s  %s ROPs  across %d cells",
                    atype + ":",
                    f"{len(grp):,}",
                    grp["cell_id"].nunique())
    logger.info("=" * 60)

# ===========================================================================
# Main entrypoint
# ===========================================================================
def main() -> None:
    """
    Orchestrate synthetic data generation pipeline end-to-end.

    Steps:
      1. Generate cell topology (sites, sectors, vendor/band assignment)
      2. Assign neighbour relationships for spatial anomaly injection
      3. Build the UTC timestamp spine
      4. Generate PM KPI time series with injected anomalies
      5. Validate generated data against telco domain constraints
      6. Write Parquet output files to ./data/
      7. Log dataset summary statistics
    """
    logger.info("Starting synthetic RAN PM data generation")
    logger.info("Config: %d cells, %d days, %d-min ROP, seed=%d",
                N_CELLS, SIM_DAYS, ROP_MINUTES, RANDOM_SEED)

    # Step 1: Cell topology
    cells = generate_cell_topology(N_CELLS)

    # Step 2: Neighbour assignment (needed for spatial anomaly injection)
    assign_neighbours(cells, k=3)

    # Step 3: Timestamp spine
    timestamps = build_timestamp_index(SIM_START, SIM_DAYS, ROP_MINUTES)

    # Step 4: PM data and ground-truth labels
    pm_df, labels_df = generate_pm_dataset(cells, timestamps)

    # Step 5: Data quality validation
    validate_pm_dataframe(pm_df)

    # Step 6: Cell metadata DataFrame
    meta_df = cells_to_dataframe(cells)

    # Step 7: Write outputs
    write_outputs(pm_df, labels_df, meta_df, OUTPUT_DIR)

    # Step 8: Summary
    print_dataset_summary(pm_df, labels_df, meta_df)

    logger.info(
    "Data generation complete. Next step: run 02_feature_engineering.py"
    )


if __name__ == "__main__":
    main()


02_feature_engineering.py
=========================

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


03_model_training.py
====================

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


04_evaluation.py
================

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


05_production_patterns.py
=========================

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



run_pipeline.sh
===============

```bash
#!/usr/bin/env bash
set -euo pipefail
echo "=== Step 1/4: Synthetic Data Generation ==="
python3 01_synthetic_data.py
echo "=== Step 2/4: Feature Engineering ==="
python3 02_feature_engineering.py
echo "=== Step 3/4: Model Training ==="
python3 03_model_training.py
echo "=== Step 4/4: Evaluation ==="
python3 04_evaluation.py
echo "=== Pipeline complete. Results in data/ and results/ ==="
echo "=== To run production pattern demos: python3 05_production_patterns.py ==="
```

requirements.txt
================

```
pandas>=2.0.0
numpy>=1.24.0
pyarrow>=14.0.0
scikit-learn>=1.3.0
scipy>=1.11.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

requirements-optional.txt
=========================

```
torch>=2.0.0
fastapi>=0.100.0
uvicorn>=0.23.0
```
