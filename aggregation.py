"""Aggregation and calibration of the committee's forecasts (pure functions, unit-tested in tests/test_offline.py).

Every choice below was tested OUT OF SAMPLE on Metaculus's public Q1-2025 tournament file (45 bots, 424 questions;
parameters fitted on the first half of the season, scored on the second half against the real field —
LLM_Ventures/research/audit_2026_11_11/m5_design_space.py and m6_numeric_agg.py, 2026-11-12):

  BINARY
  * Weighted MEAN OF LOG-ODDS is the default. With the five models that looked best in the first half, the second half
    scored: mean of log-odds 610, plain median 362, median with the "best" model weighted 3 only 34 — the model that
    looks best is partly luck, so no single model gets more than a mild extra weight.
  * A downward SHIFT in log-odds. 30% of questions resolved Yes while the average bot forecast was 40%; all 45 bots
    were biased upwards, and the least biased bot won. A shift of -0.3 lifted the committee from 34 to 505 on the
    second half (-0.6 gave 810 for early-2025 models; -0.3 is the cautious value for today's models).
  * Clipping to [2%, 98%] made no difference in either direction; kept as cheap insurance.

  NUMBERS (and dates)
  * MEAN of the runs' CDFs (a mixture) instead of the template's pointwise median: mean log score -4.61 vs -4.73.
  * Mixed with a FLAT distribution over the question's range: bots' 80% ranges held the truth only 58% of the time.
    15% flat weight: -4.59, and the worst single question improved from -8.5 to -7.1.

  MULTIPLE CHOICE: mixing with a uniform forecast did not help -> left to the template.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

EPS = 1e-4
MIN_STEP = 6e-5            # Metaculus needs every CDF step >= 5e-5
MAX_STEP = 0.19            # ... and no bucket above 0.2


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


def combine_binary(probabilities: Sequence[float], weights: Sequence[float] | None = None,
                   method: str = "logodds_mean") -> float:
    """The committee's raw probability (before any calibration)."""
    if not probabilities:
        raise ValueError("no probabilities to aggregate")
    w = [1.0] * len(probabilities) if weights is None else [float(x) for x in weights]
    if len(w) != len(probabilities):
        raise ValueError("values and weights must be the same length")
    if sum(w) <= 0:
        raise ValueError("weights must sum to a positive number")
    if method == "logodds_mean":
        return sigmoid(sum(wi * logit(float(p)) for wi, p in zip(w, probabilities)) / sum(w))
    if method == "weighted_median":
        return weighted_median([float(x) for x in probabilities], w)
    raise ValueError(f"unknown aggregation method: {method}")


def calibrate_binary(p: float, shift: float = 0.0, slope: float = 1.0) -> float:
    """logit(p') = slope * logit(p) + shift   (shift < 0 lowers every forecast; slope > 1 extremises)."""
    return sigmoid(slope * logit(p) + shift)


def blend_logodds(p: float, q: float, weight_q: float) -> float:
    """Log-odds blend of the committee (p) with an outside estimate (q), e.g. the market-data prior."""
    weight_q = min(max(weight_q, 0.0), 1.0)
    return sigmoid((1 - weight_q) * logit(p) + weight_q * logit(q))


def aggregate_binary(probabilities: Sequence[float], weights: Sequence[float] | None = None,
                     method: str = "logodds_mean", shift: float = 0.0, slope: float = 1.0,
                     clip: tuple[float, float] = (0.02, 0.98)) -> float:
    """Committee forecasts -> one calibrated, clipped probability."""
    return _clip(calibrate_binary(combine_binary(probabilities, weights, method), shift, slope), clip)


# ---------------------------------------------------------------------------------------------- numbers and dates
def mean_cdf(cdfs: Sequence[Sequence[float]], weights: Sequence[float] | None = None) -> np.ndarray:
    """Weighted mean of CDFs given on the same grid = the mixture of the runs' distributions."""
    a = np.asarray(cdfs, float)
    if a.ndim != 2 or a.shape[0] == 0:
        raise ValueError("need at least one CDF, all on the same grid")
    w = np.ones(a.shape[0]) if weights is None else np.asarray(weights, float)
    if len(w) != a.shape[0] or w.sum() <= 0:
        raise ValueError("bad weights")
    return (a * w[:, None]).sum(axis=0) / w.sum()


def fit_bounds(cdf: Sequence[float], lower_open: bool, upper_open: bool) -> np.ndarray:
    """Move the mass a CDF puts outside a CLOSED bound back inside (rescaling), leaving open bounds untouched."""
    c = np.clip(np.asarray(cdf, float), 0.0, 1.0)
    lo = 0.0 if lower_open else c[0]
    hi = 1.0 if upper_open else c[-1]
    if hi - lo <= 1e-9:                                   # everything outside a closed bound: no usable shape
        return np.linspace(0.0, 1.0, len(c))
    return (c - lo) / (hi - lo)


def mix_flat(cdf: Sequence[float], weight: float, lower_open: bool, upper_open: bool, tail: float = 0.05) -> np.ndarray:
    """(1 - weight) * cdf + weight * flat, where `flat` spreads its mass evenly over the range and leaves `tail`
    outside each OPEN bound. The result always rises by at least MIN_STEP per grid point."""
    c = np.asarray(cdf, float); n = len(c)
    weight = min(max(weight, 0.0), 1.0)
    a = tail if lower_open else 0.0
    b = 1.0 - tail if upper_open else 1.0
    out = (1 - weight) * c + weight * (a + (b - a) * np.linspace(0.0, 1.0, n))
    out = np.maximum.accumulate(np.clip(out, 0.0, 1.0))
    ramp = MIN_STEP * np.arange(n)                         # guarantee strictly increasing steps
    scale = 1.0 - ramp[-1]
    out = out * scale + ramp
    if not lower_open:
        out = out - out[0]
    if not upper_open:
        out = out / out[-1]
    return out


def cap_steps(cdf: Sequence[float], max_step: float = MAX_STEP) -> np.ndarray:
    """Spread any bucket above `max_step` over its neighbours (Metaculus rejects taller ones)."""
    c = np.asarray(cdf, float)
    pmf = np.diff(c)
    for _ in range(50):
        over = pmf > max_step
        if not over.any():
            break
        excess = (pmf[over] - max_step).sum()
        pmf[over] = max_step
        room = ~over
        if not room.any():
            break
        pmf[room] += excess * (pmf[room] + 1e-12) / (pmf[room] + 1e-12).sum()
    return np.concatenate([[c[0]], c[0] + np.cumsum(pmf)])


def aggregate_cdf(cdfs: Sequence[Sequence[float]], weights: Sequence[float] | None, lower_open: bool, upper_open: bool,
                  flat_weight: float = 0.15, tail: float = 0.05, prior_cdf: Sequence[float] | None = None,
                  prior_weight: float = 0.0) -> np.ndarray:
    """Runs' CDFs -> mixture -> optional blend with an outside prior -> flat mixing -> a CDF Metaculus accepts."""
    c = mean_cdf(cdfs, weights)
    if prior_cdf is not None and prior_weight > 0:
        q = fit_bounds(prior_cdf, lower_open, upper_open)
        if len(q) != len(c):
            raise ValueError("prior CDF on a different grid")
        w = min(max(prior_weight, 0.0), 1.0)
        c = (1 - w) * c + w * q
    return cap_steps(mix_flat(c, flat_weight, lower_open, upper_open, tail))
