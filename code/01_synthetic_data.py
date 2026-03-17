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
