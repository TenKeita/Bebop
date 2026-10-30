"""Aggregation of a committee's binary forecasts (pure functions, unit-tested in tests/test_offline.py).

Default = WEIGHTED MEDIAN: the 2026 bot analyses found median aggregation preferred over means/extremising among
winning bots, and the Q2-2025 winner gave its best model extra weight. The weight lets the strongest model (by the
FutureEval leaderboard) anchor the result while the others can still pull it when they agree.
Alternative = trimmed mean of log-odds (unweighted), kept for comparison on resolved questions.
Both results are capped to [clip_low, clip_high] (capping correlated with winning, r = +0.48).
"""
from __future__ import annotations

import math
from collections.abc import Sequence

EPS = 1e-4


def logit(p: float) -> float:
    p = min(max(p, EPS), 1 - EPS)
    return math.log(p / (1 - p))


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _clip(p: float, clip: tuple[float, float]) -> float:
    return min(max(p, clip[0]), clip[1])


def weighted_median(values: Sequence[float], weights: Sequence[float]) -> float:
    """Weighted median; when the cumulative weight hits exactly half, the two neighbouring values are averaged."""
    if not values or len(values) != len(weights):
        raise ValueError("values and weights must be non-empty and the same length")
    pairs = sorted(zip(values, weights))
    total = sum(w for _, w in pairs)
    if total <= 0:
        raise ValueError("weights must sum to a positive number")
    acc = 0.0
    for i, (v, w) in enumerate(pairs):
        acc += w
        if math.isclose(acc, total / 2) and i + 1 < len(pairs):
            return (v + pairs[i + 1][0]) / 2
        if acc > total / 2:
            return v
    return pairs[-1][0]


def aggregate_binary(probabilities: Sequence[float], weights: Sequence[float] | None = None,
                     method: str = "weighted_median", extremize: float = 1.0,
                     clip: tuple[float, float] = (0.02, 0.98), trim: bool = True) -> float:
    """Combine one question's committee forecasts into a single probability."""
    if not probabilities:
        raise ValueError("no probabilities to aggregate")
    if method == "weighted_median":
        w = list(weights) if weights is not None else [1.0] * len(probabilities)
        p = weighted_median([float(x) for x in probabilities], w)
        if extremize != 1.0:
            p = sigmoid(extremize * logit(p))
        return _clip(p, clip)
    if method == "logodds_mean":
        xs = sorted(logit(float(p)) for p in probabilities)
        if trim and len(xs) >= 5:
            xs = xs[1:-1]
        return _clip(sigmoid(extremize * sum(xs) / len(xs)), clip)
    raise ValueError(f"unknown aggregation method: {method}")
