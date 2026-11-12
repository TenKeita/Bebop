"""Market-data prior for price questions (pure numpy + requests; no LLM here).

About 14% of FutureEval questions (38% of the number questions) ask for a traded price, yield, spread or a return over
a window. Language models answer these with ranges that are too narrow and too high; markets are close to a random
walk over a few weeks. This module turns "instrument + window" into a full distribution by resampling the instrument's
own recent daily moves (block bootstrap, scaled to current volatility, zero drift), so the committee can start from it.

    spec = Spec(kind="diff_return_pct", a="yahoo:NVDA", b="yahoo:AAPL", start=date(2026, 6, 1), end=date(2026, 6, 12))
    sims = simulate(spec, fetch("yahoo:NVDA"), fetch("yahoo:AAPL"), asof=date.today())
    percentiles(sims), prob(sims, ">", 0.0), cdf_on_grid(sims, lower, upper, n=201)

Data: FRED (fredgraph.csv, no key) and Yahoo's public chart endpoint, both with a plain descriptive User-Agent. If a
source refuses or the data look wrong, the caller gets an exception and the bot simply forecasts without the prior.
Validated offline on past tournament questions: research/audit_2026_11_11/m7_market_prior_backtest.py (LLM_Ventures).
"""
from __future__ import annotations

import csv
import datetime as dt
import io
from dataclasses import dataclass

import numpy as np
import requests

UA = {"User-Agent": "committee-bot/1.0 (forecasting research; github.com/TenKeita/Bebop)"}
KINDS = ("level", "return_pct", "max", "min", "diff_return_pct", "ratio")
LOOKBACK, BLOCK, N_SIMS = 756, 5, 20000
VOL_RATIO_BOUNDS, VOL_PERSISTENCE = (0.6, 2.0), 0.97
MEAN_REVERT_SOURCES = ("VIX", "VIXCLS", "BAML", "OAS")          # log-level series pulled back to their recent mean


@dataclass
class Series:
    name: str
    dates: list            # list[dt.date], ascending
    close: np.ndarray
    high: np.ndarray | None = None
    low: np.ndarray | None = None

    def upto(self, asof: dt.date) -> "Series":
        """History known on `asof`: every bar dated on or before it."""
        n = int(np.searchsorted(np.array([d.toordinal() for d in self.dates]), asof.toordinal(), side="right"))
        cut = lambda x: None if x is None else x[:n]
        return Series(self.name, self.dates[:n], self.close[:n], cut(self.high), cut(self.low))


@dataclass
class Spec:
    kind: str                     # one of KINDS
    a: str                        # "yahoo:NVDA" or "fred:SP500"
    end: dt.date                  # last day of the window (its close is used)
    b: str | None = None          # second instrument for diff_return_pct / ratio
    start: dt.date | None = None  # first day of the window; returns are measured from the close BEFORE it
    intraday: bool = False        # max / min use daily highs / lows
    additive: bool = False        # yields and spreads: resample arithmetic changes, not log returns

    def validate(self) -> "Spec":
        if self.kind not in KINDS:
            raise ValueError(f"unknown kind {self.kind!r}")
        if self.kind in ("diff_return_pct", "ratio") and not self.b:
            raise ValueError(f"{self.kind} needs a second instrument")
        if self.kind in ("return_pct", "diff_return_pct", "max", "min") and self.start is None:
            raise ValueError(f"{self.kind} needs a window start")
        if self.start and self.start > self.end:
            raise ValueError("window start after its end")
        return self


# ------------------------------------------------------------------ data
def fetch(source: str, timeout: float = 30.0) -> Series:
    kind, _, sym = source.partition(":")
    if kind == "fred":
        r = requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv", params={"id": sym}, headers=UA, timeout=timeout)
        r.raise_for_status()
        dates, vals = [], []
        for row in list(csv.reader(io.StringIO(r.text)))[1:]:
            try:
                v = float(row[1])
            except (ValueError, IndexError):
                continue
            dates.append(dt.date.fromisoformat(row[0])); vals.append(v)
        s = Series(source, dates, np.array(vals, float))
    elif kind == "yahoo":
        r = requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{requests.utils.quote(sym)}",
                         params={"range": "5y", "interval": "1d"}, headers=UA, timeout=timeout)
        r.raise_for_status()
        res = r.json()["chart"]["result"][0]; q = res["indicators"]["quote"][0]
        rows = [(dt.datetime.fromtimestamp(t, dt.timezone.utc).date(), c, h, l)
                for t, c, h, l in zip(res["timestamp"], q["close"], q["high"], q["low"]) if c is not None]
        s = Series(source, [x[0] for x in rows], np.array([x[1] for x in rows], float),
                   np.array([x[2] if x[2] is not None else x[1] for x in rows], float),
                   np.array([x[3] if x[3] is not None else x[1] for x in rows], float))
    else:
        raise ValueError(f"unknown data source {source!r} (use yahoo:SYMBOL or fred:SERIES)")
    if len(s.dates) < 260:
        raise ValueError(f"{source}: only {len(s.dates)} daily observations")
    return s


# ------------------------------------------------------------------ simulation
def business_days(after: dt.date, until: dt.date) -> list:
    out, d = [], after + dt.timedelta(days=1)
    while d <= until:
        if d.weekday() < 5:
            out.append(d)
        d += dt.timedelta(days=1)
    return out


def _moves(s: Series, additive: bool) -> np.ndarray:
    x = s.close if additive else np.log(s.close)
    return np.diff(x)


def _vol_scale(moves: np.ndarray, horizon: int) -> float:
    """Recent-to-long-run volatility ratio, faded towards 1 over the horizon (volatility mean-reverts)."""
    if len(moves) < 60 or moves.std() == 0:
        return 1.0
    ratio = float(np.clip(moves[-21:].std() / moves.std(), *VOL_RATIO_BOUNDS))
    k = np.arange(1, max(horizon, 1) + 1)
    return float(np.sqrt(np.mean(1.0 + (ratio ** 2 - 1.0) * VOL_PERSISTENCE ** k)))


def _pull(s: Series, additive: bool) -> tuple[float, float]:
    """(kappa, mu) of a daily pull towards the recent mean, for the few series that clearly mean-revert."""
    if not any(tag in s.name.upper() for tag in MEAN_REVERT_SOURCES):
        return 0.0, 0.0
    x = s.close[-LOOKBACK:] if additive else np.log(s.close[-LOOKBACK:])
    d, lag = np.diff(x), x[:-1] - x[:-1].mean()
    kappa = float(np.clip(-(d * lag).sum() / (lag ** 2).sum(), 0.0, 0.08))
    return kappa, float(x.mean())


def simulate(spec: Spec, sa: Series, sb: Series | None = None, asof: dt.date | None = None, n: int = N_SIMS,
             seed: int = 7) -> np.ndarray:
    """n draws of the quantity described by `spec`, given the history known on `asof` (default: today)."""
    spec.validate()
    asof = asof or dt.date.today()
    sa = sa.upto(asof)
    if sb is not None:
        sb = sb.upto(asof)
        common = sorted(set(sa.dates) & set(sb.dates))
        ia = [i for i, d in enumerate(sa.dates) if d in set(common)]; ib = [i for i, d in enumerate(sb.dates) if d in set(common)]
        pick = lambda s, ix: Series(s.name, [s.dates[i] for i in ix], s.close[ix], None if s.high is None else s.high[ix],
                                    None if s.low is None else s.low[ix])
        sa, sb = pick(sa, ia), pick(sb, ib)
    if len(sa.dates) < 260:
        raise ValueError("not enough history before the forecast date")
    last = sa.dates[-1]
    if (asof - last).days > 7:
        raise ValueError(f"{sa.name}: data end {last}, too old for a forecast on {asof}")
    future = business_days(last, spec.end)
    H = len(future)
    rng = np.random.default_rng(seed)

    def paths(s: Series):
        mv = _moves(s, spec.additive)[-LOOKBACK:]
        mv = (mv - mv.mean()) * _vol_scale(mv, H)                      # zero drift, current volatility
        return mv

    ma = paths(sa); mb = paths(sb) if sb is not None else None
    m = len(ma) if mb is None else min(len(ma), len(mb))
    ma = ma[-m:]; mb = None if mb is None else mb[-m:]
    if H > 0:
        nb = -(-H // BLOCK)
        starts = rng.integers(0, m - BLOCK + 1, size=(n, nb))
        idx = (starts[:, :, None] + np.arange(BLOCK)[None, None, :]).reshape(n, -1)[:, :H]
    else:
        idx = np.zeros((n, 0), dtype=int)

    def walk(s: Series, mv: np.ndarray) -> np.ndarray:
        """Simulated closes, shape (n, H)."""
        x0 = s.close[-1] if spec.additive else np.log(s.close[-1])
        kappa, mu = _pull(s, spec.additive)
        steps = mv[idx]
        if kappa == 0.0 or H == 0:
            x = x0 + np.cumsum(steps, axis=1)
        else:
            x = np.empty((n, H)); cur = np.full(n, x0)
            for t in range(H):
                cur = cur + kappa * (mu - cur) + steps[:, t]; x[:, t] = cur
        return x if spec.additive else np.exp(x)

    A = walk(sa, ma); B = walk(sb, mb) if sb is not None else None

    def base_and_window(s: Series, sim: np.ndarray):
        """(close before the window start, per-draw closes inside the window incl. the realised part, realised highs/lows)."""
        hist_in = [i for i, d in enumerate(s.dates) if spec.start and d >= spec.start]
        if spec.start is None:
            base = np.full(n, s.close[-1]); fut_from = 0
        elif spec.start <= last:                                        # the window has started: its base is realised
            before = [i for i, d in enumerate(s.dates) if d < spec.start]
            base = np.full(n, s.close[before[-1]]); fut_from = 0
        else:                                                           # base = simulated close of the last day before start
            k = max([j for j, d in enumerate(future) if d < spec.start], default=-1)
            base = sim[:, k] if k >= 0 else np.full(n, s.close[-1]); fut_from = k + 1
        return base, sim[:, fut_from:], hist_in

    if spec.kind == "level":
        return A[:, -1] if H else np.full(n, sa.close[-1])
    if spec.kind == "ratio":
        return (A[:, -1] / B[:, -1]) if H else np.full(n, sa.close[-1] / sb.close[-1])
    end_a = A[:, -1] if H else np.full(n, sa.close[-1])
    base_a, win_a, hist_a = base_and_window(sa, A)
    if spec.kind == "return_pct":
        return 100.0 * (end_a / base_a - 1.0)
    if spec.kind == "diff_return_pct":
        end_b = B[:, -1] if H else np.full(n, sb.close[-1]); base_b, _, _ = base_and_window(sb, B)
        return 100.0 * (end_a / base_a - 1.0) - 100.0 * (end_b / base_b - 1.0)
    hi = spec.kind == "max"
    ext = win_a
    if spec.intraday and (sa.high is not None) and ext.shape[1]:
        ratio = ((sa.high if hi else sa.low) / sa.close)[-m:]            # resampled with the same days as the moves
        k0 = A.shape[1] - ext.shape[1]
        ext = ext * ratio[idx[:, k0:]]
    sim_ext = (ext.max(axis=1) if hi else ext.min(axis=1)) if ext.shape[1] else None
    if hist_a:
        src = (sa.high if (spec.intraday and sa.high is not None) else sa.close) if hi else \
              (sa.low if (spec.intraday and sa.low is not None) else sa.close)
        real = float(src[hist_a].max() if hi else src[hist_a].min())
        sim_ext = np.full(n, real) if sim_ext is None else (np.maximum(sim_ext, real) if hi else np.minimum(sim_ext, real))
    if sim_ext is None:
        raise ValueError("the window contains no trading day")
    return sim_ext


# ------------------------------------------------------------------ summaries
def percentiles(sims: np.ndarray, ps=(5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95)) -> dict:
    return {int(p): float(np.percentile(sims, p)) for p in ps}


def prob(sims: np.ndarray, op: str, threshold: float, threshold2: float | None = None) -> float:
    if op == ">":
        p = (sims > threshold).mean()
    elif op == ">=":
        p = (sims >= threshold).mean()
    elif op == "<":
        p = (sims < threshold).mean()
    elif op == "<=":
        p = (sims <= threshold).mean()
    elif op == "between":
        p = ((sims > threshold) & (sims < threshold2)).mean()
    else:
        raise ValueError(f"unknown comparison {op!r}")
    return float(p)


def cdf_on_grid(sims: np.ndarray, grid_values) -> np.ndarray:
    """P(quantity <= x) at each nominal value of the question's CDF grid."""
    s = np.sort(sims)
    return np.searchsorted(s, np.asarray(grid_values, float), side="right") / len(s)


def describe(spec: Spec, sims: np.ndarray, unit: str = "") -> str:
    p = percentiles(sims)
    return (f"Quantitative baseline from market data ({spec.a}" + (f" vs {spec.b}" if spec.b else "") +
            f", zero-drift resampling of its own recent daily moves at current volatility, {len(sims):,} paths): "
            f"median {p[50]:.4g}{unit}; 50% interval {np.percentile(sims, 25):.4g} to {np.percentile(sims, 75):.4g}; "
            f"90% interval {p[5]:.4g} to {p[95]:.4g}. Markets are close to a random walk over a few weeks: move away "
            f"from this baseline only for specific, dated information that the price cannot already contain.")
