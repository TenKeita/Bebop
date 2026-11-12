"""From a question to a market_prior.Spec.

Two parsers, in this order:
  1. RULES for the recurring price families of the tournament (about 80 questions over the last four seasons): the
     biweekly return gaps (Nvidia vs Apple/Microsoft, gold/crude/Nasdaq-100 futures vs S&P 500 futures), the biweekly
     VIX high, the biweekly high-yield spread, and "Will the S&P 500 go up in <month>?". Deterministic and tested.
  2. A cheap LANGUAGE MODEL for any other price question: it must answer with a small JSON object, which is validated
     here and then checked against real data before it is used.
A wrong parse is the main risk, so the caller also checks the simulated values against the question's own range, and
model-parsed priors get a lower weight than rule-parsed ones.
"""
from __future__ import annotations

import calendar
import datetime as dt
import json
import re
from dataclasses import dataclass

import numpy as np

import market_prior as MP

HINT = re.compile(
    r"(s&p|nasdaq|dow jones|russell 2000|stock price|share price|shares? of|close (above|below|at)|closing (price|value)|"
    r"market cap|bitcoin|btc|ethereum|price of|oil price|brent|wti|crude|gold|silver|natural gas|copper|treasury yield|"
    r"10-year|2-year|option-adjusted|high yield|exchange rate|usd/|eur/|/usd|dollar index|vix|nikkei|ftse|dax|stoxx|"
    r"hang seng|returns? exceed|percentage change in|futures)", re.I)
MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_abbr) if m}
MONTHS.update({m.lower(): i for i, m in enumerate(calendar.month_name) if m})
OPS = (">", ">=", "<", "<=", "between")


@dataclass
class MarketPrior:
    spec: MP.Spec
    sims: np.ndarray
    source: str                       # "rule" or "llm"
    weight: float                     # share of the final forecast given to this prior
    op: str | None = None             # binary questions: Yes when the quantity <op> threshold
    threshold: float | None = None
    threshold2: float | None = None
    scale: float = 1.0                # the question's unit = the quantity x scale

    @property
    def values(self) -> np.ndarray:
        return self.sims * self.scale

    def p_yes(self) -> float | None:
        if self.op is None or self.threshold is None:
            return None
        return MP.prob(self.values, self.op, self.threshold, self.threshold2)

    def text(self, unit: str = "") -> str:
        s = MP.describe(self.spec, self.values, unit)
        p = self.p_yes()
        if p is not None:
            s += f" By this baseline, the probability of Yes is {100 * p:.0f}%."
        return s


# ------------------------------------------------------------------ rules for the recurring families
_PAIR = [
    (r"nvidia'?s? stock price returns? exceed apple", "yahoo:NVDA", "yahoo:AAPL"),
    (r"nvidia'?s? stock price returns? exceed microsoft", "yahoo:NVDA", "yahoo:MSFT"),
    (r"gold futures.{0,25}returns? exceed s&p 500 futures", "yahoo:GC=F", "yahoo:ES=F"),
    (r"crude oil futures.{0,25}returns? exceed s&p 500 futures", "yahoo:CL=F", "yahoo:ES=F"),
    (r"nasdaq-?100 futures.{0,25}returns? exceed s&p 500 futures", "yahoo:NQ=F", "yahoo:ES=F"),
]
_PERIOD = re.compile(r"\((\w{3,9})\.? (\d{1,2})(?:,? \d{4})? ?[-–] ?(\w{3,9})\.? (\d{1,2})(?:,? \d{4})?\)")


def _period(title: str, resolve: dt.date) -> tuple[dt.date, dt.date] | None:
    """'(Dec 22 - Jan 2)' -> dates, the year taken from the scheduled resolution date."""
    m = _PERIOD.search(title)
    if not m or m.group(1).lower() not in MONTHS or m.group(3).lower() not in MONTHS:
        return None
    m1, d1, m2, d2 = MONTHS[m.group(1).lower()], int(m.group(2)), MONTHS[m.group(3).lower()], int(m.group(4))
    for y2 in (resolve.year, resolve.year - 1, resolve.year + 1):          # the end lies within ~10 days of resolution
        try:
            end = dt.date(y2, m2, d2); start = dt.date(y2 if m1 <= m2 else y2 - 1, m1, d1)
        except ValueError:
            continue
        if abs((end - resolve).days) <= 10 and dt.timedelta(0) <= end - start <= dt.timedelta(days=45):
            return start, end
    return None


def rule_spec(title: str, resolve: dt.date) -> tuple[MP.Spec, str | None, float | None] | None:
    """(spec, op, threshold) for a recognised family, else None. op/threshold are set for yes/no families only."""
    t = title.lower()
    per = _period(title, resolve)
    if per:
        start, end = per
        for pat, a, b in _PAIR:
            if re.search(pat, t):
                return MP.Spec("diff_return_pct", a, end, b=b, start=start), None, None
        if re.search(r"maximum intraday value of the vix", t):
            return MP.Spec("max", "yahoo:^VIX", end, start=start, intraday=True), None, None
        if re.search(r"ending value of the ice bofa us high yield option-adjusted spread", t):
            return MP.Spec("level", "fred:BAMLH0A0HYM2", end), None, None
    m = re.search(r"will the s&p 500(?: index)? go up in (\w+)(?: (\d{4}))?\??", t)
    if m and m.group(1) in MONTHS:
        month = MONTHS[m.group(1)]; year = int(m.group(2)) if m.group(2) else (resolve - dt.timedelta(days=5)).year
        first = dt.date(year, month, 1); last = dt.date(year, month, calendar.monthrange(year, month)[1])
        return MP.Spec("return_pct", "yahoo:^GSPC", last, start=first), ">", 0.0
    return None


# ------------------------------------------------------------------ language-model parser
LLM_PROMPT = """You convert a forecasting question into a data request. Reply with ONE JSON object and nothing else.

If the question's outcome is NOT simply a traded price, index level, yield, spread, exchange rate, or a return / high /
low of one over a window, reply {{"market_question": false}}.

Otherwise reply:
{{"market_question": true,
  "kind": "level" | "return_pct" | "max" | "min" | "diff_return_pct" | "ratio",
  "a": "yahoo:<Yahoo Finance symbol>" or "fred:<FRED series id>",
  "b": the second instrument in the same form, or null   (needed for diff_return_pct and ratio),
  "start": "YYYY-MM-DD" or null   (first day of the window; returns are measured from the close before it),
  "end": "YYYY-MM-DD"             (last day whose close counts),
  "intraday": true if the high/low must use intraday extremes, else false,
  "additive": true for yields and spreads quoted in percent, else false,
  "scale": number by which the quantity must be multiplied to be in the question's unit (usually 1),
  "op": ">" | ">=" | "<" | "<=" | "between" | null   (yes/no questions only: Yes when quantity <op> threshold),
  "threshold": number or null, "threshold2": number or null}}

Meaning of kind: level = closing value on `end`; return_pct = percent change from the close before `start` to the
close of `end`; max / min = highest / lowest value from `start` to `end`; diff_return_pct = return_pct of a minus
return_pct of b; ratio = a divided by b on `end`.
"Stays above X until D" is kind "min" with op ">"; "reaches X before D" is kind "max" with op ">=".
Examples of symbols: yahoo:^GSPC, yahoo:^VIX, yahoo:NVDA, yahoo:BTC-USD, yahoo:GC=F, yahoo:CL=F, yahoo:EURUSD=X,
fred:DGS10, fred:BAMLH0A0HYM2. If you are not sure of the symbol or of the dates, reply {{"market_question": false}}.

Today is {today}.
Question: {title}
Resolution criteria: {criteria}
Fine print: {fine_print}
Question type: {qtype}. {range_text}
"""


def parse_llm_json(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return d if isinstance(d, dict) else None


def spec_from_json(d: dict) -> tuple[MP.Spec, str | None, float | None, float | None, float] | None:
    """Validated (spec, op, threshold, threshold2, scale) from the model's JSON, else None."""
    if not d or d.get("market_question") is not True:
        return None
    try:
        date = lambda v: dt.date.fromisoformat(str(v)[:10]) if v else None
        for key in ("a", "b"):
            v = d.get(key)
            if v is not None and not re.fullmatch(r"(yahoo|fred):[A-Za-z0-9\^=\.\-_]{1,24}", str(v)):
                return None
        spec = MP.Spec(kind=str(d["kind"]), a=str(d["a"]), end=date(d["end"]), b=d.get("b") or None, start=date(d.get("start")),
                       intraday=bool(d.get("intraday")), additive=bool(d.get("additive"))).validate()
        op = d.get("op") or None
        if op is not None and op not in OPS:
            return None
        thr = None if d.get("threshold") is None else float(d["threshold"])
        thr2 = None if d.get("threshold2") is None else float(d["threshold2"])
        if op is not None and thr is None or op == "between" and thr2 is None:
            return None
        scale = 1.0 if d.get("scale") is None else float(d["scale"])
        if not (1e-9 < abs(scale) < 1e12):
            return None
        return spec, op, thr, thr2, scale
    except (KeyError, TypeError, ValueError):
        return None


def plausible(values: np.ndarray, lower: float | None, upper: float | None) -> bool:
    """A parse is rejected when the simulated values sit far outside the question's own range (wrong unit/instrument)."""
    if lower is None or upper is None or not np.isfinite(values).all():
        return bool(np.isfinite(values).all())
    span = upper - lower
    inside = ((values > lower - span) & (values < upper + span)).mean()
    return bool(inside >= 0.9)
