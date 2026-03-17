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
