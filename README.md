# Real-Time RAN KPI Anomaly Detection at Cell Sector Granularity
Replace static threshold alerting with ML-based anomaly detection — from zero labels to full production rollout.

**Chirag Shinde** — chirag.m.shinde@gmail.com

**Static thresholds cost a 10,000-cell operator ~A$7.45M/year in false alarm triage.** This repository provides a complete, vendor-neutral architecture and working code for ML-based RAN anomaly detection that reduces false alarms by 70%, cutting that cost to ~A$2.2M/year.

📄 [Read the whitepaper](whitepaper.md) · 💻 [Companion code](https://github.com/cs-cmyk/ran-kpi-anomaly-detection/code) · 📚 [ML Coursebook](https://github.com/cs-cmyk/full-stack-ml-coursebook)

---

## What's in this repo

| File | Description |
|------|-------------|
| [`whitepaper.md`](whitepaper.md) | Full whitepaper (~18,000 words) covering business case, architecture, implementation, and production deployment |
| [`code.md`](code.md) | All companion code in a single navigable document |
| [`all-diagrams.md`](all-diagrams.md) | Mermaid architecture and data flow diagrams |
| [`transcript.md`](transcript.md) | Conference talk walkthrough transcript (~18 min) |
| [`01_synthetic_data.py`](01_synthetic_data.py) | Generates realistic synthetic RAN PM counter data with injected anomalies |
| [`02_feature_engineering.py`](02_feature_engineering.py) | Temporal features, peer-group z-scores, rolling statistics, train/val/test splits |
| [`03_model_training.py`](03_model_training.py) | Isolation Forest → Random Forest → LSTM Autoencoder cascade training |
| [`04_evaluation.py`](04_evaluation.py) | Per-tier and cascade metrics, confusion matrices, bootstrap confidence intervals |
| [`05_production_patterns.py`](05_production_patterns.py) | BentoML serving, Flink-compatible features, drift detection, Prometheus metrics |
| [`utils.py`](utils.py) | Shared utilities (event-based recall, logging, constants) |
| [`run_pipeline.sh`](run_pipeline.sh) | Runs the full 01→04 pipeline end-to-end |

## Quick start

```bash
git clone https://github.com/cs-cmyk/ran-kpi-anomaly-detection.git
cd ran-kpi-anomaly-detection

pip install -r requirements.txt
# Optional (LSTM tier + REST serving): pip install -r requirements-optional.txt

./run_pipeline.sh
```

The pipeline generates synthetic data, engineers features, trains models, and produces evaluation results in `data/` and `results/`. No real network data or GPU required.

## Requirements

**Core** (CPU only, ~2 min pipeline runtime):

- Python 3.10+
- pandas, numpy, pyarrow, scikit-learn, scipy, matplotlib, seaborn

**Optional** (LSTM tier + production serving patterns):

- PyTorch 2.0+
- FastAPI, uvicorn

See [`requirements.txt`](requirements.txt) and [`requirements-optional.txt`](requirements-optional.txt).

## Architecture

The system ingests PM counters via O1 file transfer or E2SM-KPM streaming, engineers ~150 domain-specific features per cell sector per reporting period, and scores each cell using a three-tier model cascade:

| Phase | Model | Labels needed | F1 range | Use case |
|-------|-------|--------------|----------|----------|
| 1 | Isolation Forest | None | 0.55–0.65 | Cold start — deploy immediately |
| 2 | Random Forest | 50–500 events | 0.85–0.90 | Trust building — highest interpretability via SHAP |
| 3 | LSTM Autoencoder | 500+ events | 0.78–0.85 | Gradual degradation — temporal pattern detection |

Each phase runs in shadow mode before promotion. Label gates control transitions — no manual cutover.

## Key differentiators

- **Per-cell peer-group z-scores.** A CBD macro is never compared to a rural small cell. Features are normalised within topologically similar peer groups.
- **Phased labelling pipeline.** Starts with zero labels (Isolation Forest), bootstraps a labelled dataset from Phase 1 flags, and gates Phase 2/3 promotion on label volume thresholds.
- **Topology-change-aware retraining.** Event-driven triggers fire on cell additions, parameter changes, and neighbour-list updates — not just calendar schedules.
- **~150 engineered features per cell.** Rolling statistics, day-over-day/week-over-week deltas, rate-of-change, temporal encodings, missing data flags, and cross-KPI ratios.

## Who this is for

| Role | Start with | Key sections |
|------|-----------|-------------|
| CTO / VP Engineering | Executive Summary → §1 Business Case | §10 Limitations, §11 Implementation Roadmap |
| NOC Manager | §1 Business Case → §8 Evaluation | §6 System Design, §9 Production Considerations |
| RF Engineer | §3 Data Requirements → §5 Proposed Approach | §6 System Design, §7 Implementation Walkthrough |
| Data Scientist / ML Engineer | §5 Proposed Approach → §7 Implementation Walkthrough | §3 Data, §8 Evaluation, companion code |

## Coursebook connection

This whitepaper is part of the [Full-Stack ML Coursebook](https://github.com/cs-cmyk/full-stack-ml-coursebook) series. Prerequisite chapters: Ch. 1 (Linear Algebra Foundations), Ch. 6 (Probability and Bayesian Thinking), Ch. 14 (Decision Trees and Random Forests), Ch. 18 (Recurrent Neural Networks), Ch. 19 (Time-Series Analysis and Forecasting), and Ch. 22 (MLOps and Production ML).

## Licence

This work is licensed under [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International](LICENSE). You may share and adapt the material with attribution, but not for commercial purposes. Adaptations must be distributed under the same licence.

---

## Citation

If referencing this work, please cite as:

> Shinde, C. *Real-Time RAN KPI Anomaly Detection at Cell Sector Granularity.* 2025. [https://github.com/cs-cmyk/ran-kpi-anomaly-detection]
