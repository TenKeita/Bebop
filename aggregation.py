"""Aggregation of a committee's binary forecasts (pure functions, unit-tested in tests/test_offline.py).

Why log-odds: under a log score, the geometric mean of odds is the standard aggregator for probability forecasts
from several sources (it treats 1% vs 2% as the big difference it is). A trimmed mean (drop the highest and lowest
when there are >= 5 runs) protects against one confused run. Optional extremisation (alpha > 1) pushes the result
away from 50%; it stays OFF (alpha = 1) until resolved questions show the committee is under-confident.
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


def aggregate_binary(probabilities: Sequence[float], extremize: float = 1.0,
                     clip: tuple[float, float] = (0.02, 0.98), trim: bool = True) -> float:
    """Trimmed mean of log-odds, times `extremize`, back to a probability, clipped to `clip`."""
    if not probabilities:
        raise ValueError("no probabilities to aggregate")
    xs = sorted(logit(float(p)) for p in probabilities)
    if trim and len(xs) >= 5:
        xs = xs[1:-1]
    p = sigmoid(extremize * sum(xs) / len(xs))
    return min(max(p, clip[0]), clip[1])
