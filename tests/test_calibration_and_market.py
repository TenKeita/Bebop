"""Offline tests for the 2026-11-12 changes: number aggregation, the market-data prior and its parsers, and the
discrete / date question paths (no network, no tokens).  Run:  .venv/Scripts/python -m pytest -q tests"""
from __future__ import annotations

import asyncio
import datetime as dt
import json
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from forecasting_tools import DateQuestion, GeneralLlm, NumericQuestion  # noqa: E402
from forecasting_tools.data_models.questions import DiscreteQuestion  # noqa: E402

import forecast_log  # noqa: E402
import market_prior as MP  # noqa: E402
import market_question as MQ  # noqa: E402
import template_bot  # noqa: E402
from aggregation import (MAX_STEP, MIN_STEP, aggregate_cdf, blend_logodds, cap_steps, fit_bounds, logit, mean_cdf,  # noqa: E402
                         mix_flat, sigmoid)

import fakes  # noqa: E402
from fakes import EXPECTED_BINARY, MODELS, bq, fake_structure_output, make_bot, make_fake_invoke, synthetic_series  # noqa: E402

TODAY = dt.datetime.now(dt.timezone.utc).date()


@pytest.fixture
def offline(monkeypatch):
    for k in ("ASKNEWS_CLIENT_ID", "ASKNEWS_SECRET", "ASKNEWS_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    fake, calls = make_fake_invoke()
    monkeypatch.setattr(GeneralLlm, "invoke", fake)
    monkeypatch.setattr(template_bot, "structure_output", fake_structure_output)
    monkeypatch.setitem(fakes.SETTINGS, "market_json", None)
    monkeypatch.setitem(fakes.SETTINGS, "numeric_value", lambda p, i: p + 5 * i)
    return calls


@pytest.fixture
def market(monkeypatch):
    """market_prior.fetch answers from synthetic series; records what was asked for."""
    asked: list[str] = []

    def fake_fetch(source, timeout=30.0):
        asked.append(source)
        return synthetic_series(source, seed=sum(map(ord, source)) % 97, start=100.0 if "VIX" not in source else 18.0)
    monkeypatch.setattr(MP, "fetch", fake_fetch)
    return asked


# ---------------------------------------------------------------- number aggregation
def test_mean_cdf_is_the_mixture():
    x = np.linspace(0, 1, 201)
    narrow_low, narrow_high = np.clip((x - 0.2) * 10, 0, 1), np.clip((x - 0.7) * 10, 0, 1)
    m = mean_cdf([narrow_low, narrow_high])
    assert m[100] == pytest.approx(0.5)                          # half the mass sits below the middle...
    assert np.median(np.vstack([narrow_low, narrow_high]), axis=0)[100] == pytest.approx(0.5)
    assert mean_cdf([narrow_low, narrow_high], [3, 1])[100] == pytest.approx(0.75)
    with pytest.raises(ValueError):
        mean_cdf([])


@pytest.mark.parametrize("lower_open, upper_open", [(True, True), (False, True), (True, False), (False, False)])
def test_mix_flat_makes_a_valid_wider_cdf(lower_open, upper_open):
    x = np.linspace(0, 1, 201)
    spike = np.where(x < 0.5, 0.0, 1.0)                           # everything in one bucket: invalid for Metaculus
    c = cap_steps(mix_flat(spike, 0.15, lower_open, upper_open, tail=0.05))
    steps = np.diff(c)
    assert len(c) == 201 and steps.min() >= MIN_STEP * 0.99 and steps.max() <= MAX_STEP + 1e-9
    assert (c[0] == pytest.approx(0.0, abs=1e-12)) == (not lower_open) and (c[-1] == pytest.approx(1.0, abs=1e-12)) == (not upper_open)
    if lower_open:
        assert 0.005 < c[0] < 0.02                                 # 15% x 5% of flat mass below an open bound
    if upper_open:
        assert 0.005 < 1 - c[-1] < 0.02
    wide = mix_flat(np.clip((x - 0.45) * 10, 0, 1), 0.15, True, True)
    assert wide[40] > 0.02 and wide[160] < 0.98                    # mass now exists away from the model's range


def test_fit_bounds_and_prior_blend():
    x = np.linspace(0, 1, 201)
    prior = 0.1 + 0.8 * x                                          # 10% below and 10% above the range
    assert fit_bounds(prior, True, True)[0] == pytest.approx(0.1)
    closed = fit_bounds(prior, False, False)
    assert closed[0] == pytest.approx(0.0) and closed[-1] == pytest.approx(1.0)
    llm = np.clip((x - 0.6) * 5, 0, 1)                             # the committee's range: 0.6 .. 0.8
    out = aggregate_cdf([llm, llm], [2, 1], True, True, flat_weight=0.15, prior_cdf=np.clip((x - 0.2) * 5, 0, 1), prior_weight=0.8)
    med = float(np.interp(0.5, out, x))
    assert 0.25 < med < 0.45                                       # pulled most of the way to the prior (0.2 .. 0.4)
    assert np.diff(out).min() >= MIN_STEP * 0.99
    with pytest.raises(ValueError):
        aggregate_cdf([llm], None, True, True, prior_cdf=[0.0, 1.0], prior_weight=0.5)   # different grid


def test_blend_logodds():
    assert blend_logodds(0.5, 0.9, 0.0) == pytest.approx(0.5)
    assert blend_logodds(0.5, 0.9, 1.0) == pytest.approx(0.9)
    assert blend_logodds(0.3, 0.7, 0.5) == pytest.approx(0.5)
    assert blend_logodds(0.4, 0.9, 0.8) == pytest.approx(sigmoid(0.2 * logit(0.4) + 0.8 * logit(0.9)))


# ---------------------------------------------------------------- market-data engine
def test_engine_level_and_return_are_centred_and_scale_with_time():
    s = synthetic_series("yahoo:TEST", seed=3)
    near = MP.simulate(MP.Spec("level", "yahoo:TEST", TODAY + dt.timedelta(days=7)), s, asof=TODAY)
    far = MP.simulate(MP.Spec("level", "yahoo:TEST", TODAY + dt.timedelta(days=56)), s, asof=TODAY)
    assert np.median(near) == pytest.approx(s.close[-1], rel=0.02)          # zero drift
    assert near.std() < far.std() < 4 * near.std()                          # spread grows roughly with sqrt(time)
    r = MP.simulate(MP.Spec("return_pct", "yahoo:TEST", TODAY + dt.timedelta(days=30), start=TODAY + dt.timedelta(days=10)), s, asof=TODAY)
    assert abs(np.median(r)) < 1.0 and 1.0 < r.std() < 8.0                   # ~14 trading days at ~1% a day


def test_engine_uses_what_is_already_realised():
    s = synthetic_series("yahoo:TEST", seed=4)
    start = s.dates[-6]                                                      # the window began 5 trading days ago
    base = s.close[-7]
    r = MP.simulate(MP.Spec("return_pct", "yahoo:TEST", TODAY, start=start), s, asof=TODAY)
    assert np.allclose(r, 100 * (s.close[-1] / base - 1))                    # nothing left to simulate
    hi = MP.simulate(MP.Spec("max", "yahoo:TEST", TODAY + dt.timedelta(days=10), start=start, intraday=True), s, asof=TODAY)
    assert hi.min() >= s.high[-6:].max() - 1e-9                              # never below the high already seen
    lo = MP.simulate(MP.Spec("min", "yahoo:TEST", TODAY + dt.timedelta(days=10), start=start), s, asof=TODAY)
    assert lo.max() <= s.close[-6:].min() + 1e-9


def test_engine_pairs_and_summaries():
    a, b = synthetic_series("yahoo:A", seed=5), synthetic_series("yahoo:B", seed=6, start=50.0)
    win = dict(start=TODAY + dt.timedelta(days=5), end=TODAY + dt.timedelta(days=19))
    d = MP.simulate(MP.Spec("diff_return_pct", "yahoo:A", b="yahoo:B", **win), a, b, asof=TODAY)
    assert abs(np.median(d)) < 1.5 and d.std() > 1.0
    ratio = MP.simulate(MP.Spec("ratio", "yahoo:A", TODAY + dt.timedelta(days=10), b="yahoo:B"), a, b, asof=TODAY)
    assert np.median(ratio) == pytest.approx(a.close[-1] / b.close[-1], rel=0.03)
    assert MP.prob(d, ">", -1e9) == 1.0 and MP.prob(d, "<", -1e9) == 0.0
    assert MP.prob(d, "between", np.percentile(d, 25), np.percentile(d, 75)) == pytest.approx(0.5, abs=0.01)
    grid = np.linspace(-20, 20, 201); c = MP.cdf_on_grid(d, grid)
    assert c[0] <= 0.01 and c[-1] >= 0.99 and (np.diff(c) >= 0).all()
    assert set(MP.percentiles(d)) == {5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95}
    assert "Quantitative baseline" in MP.describe(MP.Spec("diff_return_pct", "yahoo:A", b="yahoo:B", **win), d, " pp")


def test_engine_refuses_bad_input():
    s = synthetic_series("yahoo:TEST", seed=7)
    with pytest.raises(ValueError):
        MP.Spec("nonsense", "yahoo:TEST", TODAY).validate()
    with pytest.raises(ValueError):
        MP.Spec("ratio", "yahoo:TEST", TODAY).validate()                      # needs a second instrument
    with pytest.raises(ValueError):
        MP.Spec("return_pct", "yahoo:TEST", TODAY).validate()                 # needs a window start
    with pytest.raises(ValueError):
        MP.simulate(MP.Spec("level", "yahoo:TEST", TODAY + dt.timedelta(days=5)), synthetic_series("yahoo:TEST", days=100), asof=TODAY)
    stale = synthetic_series("yahoo:TEST", end=TODAY - dt.timedelta(days=30))
    with pytest.raises(ValueError):
        MP.simulate(MP.Spec("level", "yahoo:TEST", TODAY + dt.timedelta(days=5)), stale, asof=TODAY)
    with pytest.raises(ValueError):
        MP.fetch("ftp:whatever")
    assert MP.business_days(dt.date(2026, 11, 13), dt.date(2026, 11, 17)) == [dt.date(2026, 11, 16), dt.date(2026, 11, 17)]
    assert s.upto(s.dates[10]).dates[-1] == s.dates[10]


# ---------------------------------------------------------------- price-question parsers
REAL_TITLES = [   # titles of real tournament questions (2025-26) with their scheduled resolution date
    ("How much will Nvidia's stock price returns exceed Apple's in these biweekly periods of Q4 2025? (Dec 22 - Jan 2)",
     dt.date(2026, 1, 3), "diff_return_pct", "yahoo:NVDA", "yahoo:AAPL", dt.date(2025, 12, 22), dt.date(2026, 1, 2)),
    ("How much will Nvidia's stock price returns exceed Microsoft's in the following biweekly periods of Q3 2025? (Sep 29 - Oct 10)",
     dt.date(2025, 10, 11), "diff_return_pct", "yahoo:NVDA", "yahoo:MSFT", dt.date(2025, 9, 29), dt.date(2025, 10, 10)),
    ("How much will Gold Futures total price returns exceed S&P 500 Futures in these biweekly periods of Q2 2026? (Jun 1 - Jun 12)",
     dt.date(2026, 6, 12), "diff_return_pct", "yahoo:GC=F", "yahoo:ES=F", dt.date(2026, 6, 1), dt.date(2026, 6, 12)),
    ("How much will Crude Oil Futures total price returns exceed S&P 500 Futures's in these biweekly periods of Q4 2025? (Dec 22 - Jan 2)",
     dt.date(2026, 1, 3), "diff_return_pct", "yahoo:CL=F", "yahoo:ES=F", dt.date(2025, 12, 22), dt.date(2026, 1, 2)),
    ("How much will Nasdaq-100 Futures total price returns exceed S&P 500 Futures in these biweekly periods of Q1 2026? (Mar 16 - Mar 27)",
     dt.date(2026, 3, 28), "diff_return_pct", "yahoo:NQ=F", "yahoo:ES=F", dt.date(2026, 3, 16), dt.date(2026, 3, 27)),
    ("What will be the maximum intraday value of the VIX over these biweekly periods in Q2 2026? (Jun 1 - Jun 12)",
     dt.date(2026, 6, 12), "max", "yahoo:^VIX", None, dt.date(2026, 6, 1), dt.date(2026, 6, 12)),
    ("What will be the ending value of the ICE BofA US High Yield Option-Adjusted Spread for these biweekly periods of Q4 2025? (Dec 22 - Jan 2)",
     dt.date(2026, 1, 3), "level", "fred:BAMLH0A0HYM2", None, None, dt.date(2026, 1, 2)),
]


@pytest.mark.parametrize("title, resolve, kind, a, b, start, end", REAL_TITLES)
def test_rules_recognise_the_recurring_families(title, resolve, kind, a, b, start, end):
    assert MQ.HINT.search(title)
    spec, op, thr = MQ.rule_spec(title, resolve)
    assert (spec.kind, spec.a, spec.b, spec.end) == (kind, a, b, end) and op is None and thr is None
    if start:
        assert spec.start == start
    spec.validate()


def test_rules_month_question_and_non_matches():
    spec, op, thr = MQ.rule_spec("Will the S&P 500 index go up in February 2025?", dt.date(2025, 3, 1))
    assert (spec.kind, spec.a, spec.start, spec.end, op, thr) == ("return_pct", "yahoo:^GSPC", dt.date(2025, 2, 1), dt.date(2025, 2, 28), ">", 0.0)
    assert MQ.rule_spec("Will the S&P 500 index go up in December?", dt.date(2027, 1, 1))[0].end == dt.date(2026, 12, 31)
    for t in ("Will X happen before 2026-12-31?", "What will be the highest stock price of SpaceX in August 2026?",
              "How much will Nvidia's stock price returns exceed Apple's this year?"):
        assert MQ.rule_spec(t, dt.date(2026, 9, 1)) is None
    assert not MQ.HINT.search("Will the USDA close the recall of Ready-to-Eat products before March 1?")


def test_llm_json_is_validated():
    good = dict(market_question=True, kind="max", a="yahoo:TSLA", b=None, start="2026-12-01", end="2026-12-31", intraday=True,
                additive=False, scale=1, op=">=", threshold=500, threshold2=None)
    spec, op, thr, thr2, scale = MQ.spec_from_json(MQ.parse_llm_json("Here you go:\n```json\n" + json.dumps(good) + "\n```"))
    assert (spec.kind, spec.a, spec.start, spec.end, spec.intraday, op, thr, scale) == ("max", "yahoo:TSLA", dt.date(2026, 12, 1), dt.date(2026, 12, 31), True, ">=", 500.0, 1.0)
    bad = [dict(good, market_question=False), dict(good, a="http://evil.example/x"), dict(good, kind="guess"), dict(good, op="=="),
           dict(good, op=">", threshold=None), dict(good, kind="ratio", b=None), dict(good, end="soon"), dict(good, start="2027-02-01"),
           dict(good, scale=0), dict(good, op="between", threshold=1, threshold2=None)]
    assert all(MQ.spec_from_json(d) is None for d in bad)
    assert MQ.parse_llm_json("no json here") is None and MQ.spec_from_json(None) is None
    assert MQ.plausible(np.array([1.0, 2.0, 3.0]), 0, 10) and not MQ.plausible(np.full(100, 5000.0), 0, 10)
    assert not MQ.plausible(np.array([1.0, np.nan]), 0, 10)


# ---------------------------------------------------------------- the bot with a market prior
def month_question(i=21):
    first = TODAY.replace(day=1); nxt = (first + dt.timedelta(days=32)).replace(day=1)
    title = f"Will the S&P 500 index go up in {first:%B} {first.year}?"
    q = bq(i, text=title, scheduled_resolution_time=dt.datetime(nxt.year, nxt.month, nxt.day, tzinfo=dt.timezone.utc))
    return q, MP.Spec("return_pct", "yahoo:^GSPC", nxt - dt.timedelta(days=1), start=first)


def test_binary_rule_prior_is_blended_and_shown(offline, market):
    q, spec = month_question()
    bot = make_bot(binary_shift=-0.3)
    report = asyncio.run(bot.forecast_question(q))
    assert market == ["yahoo:^GSPC"]
    pm = MP.prob(MP.simulate(spec, MP.fetch("yahoo:^GSPC"), asof=TODAY), ">", 0.0)
    llm = sigmoid(logit(EXPECTED_BINARY) - 0.3)                               # the shift applies to the committee only
    want = min(max(sigmoid(0.2 * logit(llm) + 0.8 * logit(min(max(pm, 0.01), 0.99))), 0.02), 0.98)
    assert report.prediction == pytest.approx(want, abs=1e-6)
    prompts = [p for m, p in offline if m in MODELS]
    assert prompts and all("Quantitative baseline (market data)" in p and "probability of Yes is" in p for p in prompts)
    assert not any("You convert a forecasting question" in p for _, p in offline)   # a rule matched: no parser call
    rec = bot.run_log[-1]
    assert rec["market_source"] == "rule" and rec["market_weight"] == 0.8 and rec["final"] == pytest.approx(want, abs=1e-6)
    assert [m["model"] for m in rec["members"]] and sorted(m["model"] for m in rec["members"]) == sorted(MODELS)


def test_binary_llm_parsed_prior_and_its_failures(offline, market, monkeypatch):
    end = TODAY + dt.timedelta(days=30)
    title = "Will Tesla's stock price close above $105 on the last day of the window?"
    spec_json = dict(market_question=True, kind="level", a="yahoo:TSLA", b=None, start=None, end=end.isoformat(), intraday=False,
                     additive=False, scale=1, op=">", threshold=105.0, threshold2=None)
    monkeypatch.setitem(fakes.SETTINGS, "market_json", json.dumps(spec_json))
    bot = make_bot()
    report = asyncio.run(bot.forecast_question(bq(22, text=title)))
    pm = MP.prob(MP.simulate(MP.Spec("level", "yahoo:TSLA", end), MP.fetch("yahoo:TSLA"), asof=TODAY), ">", 105.0)
    assert report.prediction == pytest.approx(min(max(sigmoid(0.5 * logit(EXPECTED_BINARY) + 0.5 * logit(min(max(pm, 0.01), 0.99))), 0.02), 0.98), abs=1e-6)
    assert bot.run_log[-1]["market_source"] == "llm" and bot.run_log[-1]["market_weight"] == 0.5
    assert sum("You convert a forecasting question" in p for _, p in offline) == 1
    # (a) the parser says "not a market question", (b) it answers rubbish, (c) the data source is down,
    # (d) the parser is switched off, (e) the whole feature is switched off -> the plain committee forecast every time
    for i, (reply, fetch_ok, kw) in enumerate([('{"market_question": false}', True, {}), ("I think it is TSLA", True, {}),
                                              (json.dumps(spec_json), False, {}), (json.dumps(spec_json), True, {"market_llm": False}),
                                              (json.dumps(spec_json), True, {"market_prior": False})]):
        monkeypatch.setitem(fakes.SETTINGS, "market_json", reply)
        if not fetch_ok:
            monkeypatch.setattr(MP, "fetch", lambda *a, **k: (_ for _ in ()).throw(OSError("source down")))
        b2 = make_bot(**kw)
        rep = asyncio.run(b2.forecast_question(bq(30 + i, text=title)))
        assert rep.prediction == pytest.approx(EXPECTED_BINARY), (i, rep.prediction)
        assert b2.run_log[-1]["market_p"] is None


def test_binary_without_threshold_gets_no_prior(offline, market, monkeypatch):
    spec_json = dict(market_question=True, kind="level", a="yahoo:TSLA", end=(TODAY + dt.timedelta(days=20)).isoformat(), op=None, threshold=None)
    monkeypatch.setitem(fakes.SETTINGS, "market_json", json.dumps(spec_json))
    rep = asyncio.run(make_bot().forecast_question(bq(40, text="Will Tesla's stock price surprise everyone?")))
    assert rep.prediction == pytest.approx(EXPECTED_BINARY)


def family_question(i=50):
    start = TODAY + dt.timedelta(days=(7 - TODAY.weekday()) % 7 + 7); end = start + dt.timedelta(days=11)
    title = f"How much will Nvidia's stock price returns exceed Apple's in these biweekly periods? ({start:%b} {start.day} - {end:%b} {end.day})"
    q = NumericQuestion(question_text=title, upper_bound=20.0, lower_bound=-20.0, open_upper_bound=True, open_lower_bound=True,
                        unit_of_measure="pp", resolution_criteria="", fine_print="", background_info="", page_url=f"https://example.org/q/{i}",
                        id_of_question=i, scheduled_resolution_time=dt.datetime(end.year, end.month, end.day, 20, tzinfo=dt.timezone.utc))
    return q, start, end


def test_numeric_family_prior_pulls_the_distribution(offline, market, monkeypatch):
    monkeypatch.setitem(fakes.SETTINGS, "numeric_value", lambda p, i: round(8 + (p - 50) / 25 + 0.2 * i, 3))   # the committee says ~ +8 pp
    q, start, end = family_question()
    bot = make_bot()
    report = asyncio.run(bot.forecast_question(q))
    assert sorted(market) == ["yahoo:AAPL", "yahoo:NVDA"]
    cdf = report.prediction.get_cdf(); ps = [c.percentile for c in cdf]; xs = [c.value for c in cdf]
    assert len(cdf) == 201 and all(b > a for a, b in zip(ps, ps[1:])) and 0 < ps[0] < 0.2 and 0.8 < ps[-1] < 1
    med = float(np.interp(0.5, ps, xs))
    assert -3.0 < med < 4.0                                        # between the prior (~0) and the committee (~8), nearer the prior
    rec = bot.run_log[-1]
    assert rec["market_source"] == "rule" and rec["type"] == "NumericQuestion" and len(rec["members"]) == len(MODELS)
    no_prior = asyncio.run(make_bot(market_prior=False).forecast_question(q)).prediction.get_cdf()
    assert float(np.interp(0.5, [c.percentile for c in no_prior], [c.value for c in no_prior])) > 7.0


def test_numeric_prior_rejected_when_far_outside_the_range(offline, market, monkeypatch):
    monkeypatch.setitem(fakes.SETTINGS, "numeric_value", lambda p, i: round(8 + (p - 50) / 25 + 0.2 * i, 3))
    spec_json = dict(market_question=True, kind="level", a="yahoo:NVDA", end=(TODAY + dt.timedelta(days=20)).isoformat(), scale=1000)
    monkeypatch.setitem(fakes.SETTINGS, "market_json", json.dumps(spec_json))                 # wrong unit: values ~100,000
    q = NumericQuestion(question_text="What will Nvidia's stock price be at the end of the window?", upper_bound=20.0, lower_bound=-20.0,
                        open_upper_bound=True, open_lower_bound=True, resolution_criteria="", fine_print="", background_info="",
                        page_url="https://example.org/q/60", id_of_question=60)
    bot = make_bot()
    asyncio.run(bot.forecast_question(q))
    assert bot.run_log[-1]["market_source"] is None


# ---------------------------------------------------------------- number-question paths without a prior
def test_plain_numeric_uses_the_mixture_and_flat_mass(offline):
    q = NumericQuestion(question_text="How many?", upper_bound=200.0, lower_bound=0.0, open_upper_bound=True, open_lower_bound=False,
                        unit_of_measure="units", resolution_criteria="", fine_print="", background_info="",
                        page_url="https://example.org/q/70", id_of_question=70)
    bot = make_bot()
    cdf = asyncio.run(bot.forecast_question(q)).prediction.get_cdf(); ps = [c.percentile for c in cdf]
    assert len(cdf) == 201 and ps[0] == pytest.approx(0.0, abs=1e-9) and ps[-1] < 1.0
    assert ps[190] < 0.999                                          # flat mass above the models' own ranges (they stop at ~115)
    template = asyncio.run(make_bot(numeric_flat_weight=0.0).forecast_question(q)).prediction.get_cdf()
    width = lambda c: float(np.interp(0.9, [x.percentile for x in c], [x.value for x in c]) - np.interp(0.1, [x.percentile for x in c], [x.value for x in c]))
    assert width(cdf) > width(template)                              # the flat component widens the 80% range
    assert bot.run_log[-1]["members"][0]["p50"] < bot.run_log[-1]["members"][-1]["p50"]   # per-model medians are logged


def test_discrete_question(offline, monkeypatch):
    monkeypatch.setitem(fakes.SETTINGS, "numeric_value", lambda p, i: round(p / 10 + 0.1 * i, 2))       # 0.5 .. 9.9
    q = DiscreteQuestion(question_text="How many of the 10 will pass?", upper_bound=10.5, lower_bound=-0.5, open_upper_bound=False,
                         open_lower_bound=False, cdf_size=12, resolution_criteria="", fine_print="", background_info="",
                         page_url="https://example.org/q/80", id_of_question=80)
    bot = make_bot()
    cdf = asyncio.run(bot.forecast_question(q)).prediction.get_cdf(); ps = [c.percentile for c in cdf]
    assert len(cdf) == 12 and ps[0] == pytest.approx(0.0, abs=1e-9) and ps[-1] == pytest.approx(1.0, abs=1e-9)
    assert all(b > a for a, b in zip(ps, ps[1:]))
    assert bot.run_log[-1]["type"] == "DiscreteQuestion"


def test_date_question(offline):
    q = DateQuestion(question_text="When will it happen?", upper_bound=dt.datetime(2028, 1, 1, tzinfo=dt.timezone.utc),
                     lower_bound=dt.datetime(2026, 12, 1, tzinfo=dt.timezone.utc), open_upper_bound=True, open_lower_bound=False,
                     resolution_criteria="", fine_print="", background_info="", page_url="https://example.org/q/90", id_of_question=90)
    bot = make_bot()
    cdf = asyncio.run(bot.forecast_question(q)).prediction.get_cdf(); ps = [c.percentile for c in cdf]
    assert len(cdf) == 201 and ps[0] == pytest.approx(0.0, abs=1e-9) and ps[-1] < 1.0 and all(b > a for a, b in zip(ps, ps[1:]))
    assert bot.run_log[-1]["type"] == "DateQuestion" and bot.run_log[-1]["market_source"] is None


# ---------------------------------------------------------------- forecast log and run summary
def test_forecast_log_and_summary(offline, tmp_path):
    bot = make_bot(binary_shift=-0.3)
    asyncio.run(bot.forecast_questions([bq(101), bq(102)], return_exceptions=True))
    n, mean = bot.binary_summary()
    assert n == 2 and mean == pytest.approx(sigmoid(logit(EXPECTED_BINARY) - 0.3))
    path = forecast_log.write(bot.run_log, str(tmp_path / "log"), "test_questions", "balanced", False)
    lines = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 2 and {x["id"] for x in lines} == {101, 102}
    assert all(x["tier"] == "balanced" and x["published"] is False and x["shift"] == -0.3 and len(x["members"]) == 5 for x in lines)
    assert forecast_log.write([], str(tmp_path / "log"), "x", "y", True) is None
    assert forecast_log.write(bot.run_log, str(tmp_path / "log" / "a\x00b"), "x", "y", True) is None      # never raises


def test_credit_guard_uses_the_lower_of_cap_and_balance(monkeypatch):
    import budget

    class R:
        def __init__(self, d): self.d = d
        def raise_for_status(self): pass
        def json(self): return self.d

    def fake_get(url, **k):
        return R({"data": {"limit_remaining": 100.0}}) if url == budget.KEY_URL else R({"data": {"total_credits": 10.0, "total_usage": 3.2}})
    monkeypatch.setattr(budget.requests, "get", fake_get)
    g = budget.CreditGuard("key", soft_floor_usd=15, hard_floor_usd=3)
    assert abs(g.remaining_usd() - 6.8) < 1e-9
    tier, why = g.decide("balanced", "minimal")
    assert tier == "minimal" and "6.80" in why


def test_clamp_percentiles_keeps_far_tails_inside_tolerance():
    from datetime import datetime, timezone
    from types import SimpleNamespace
    from forecasting_tools import Percentile
    from bot_helpers import clamp_percentiles
    q = SimpleNamespace(lower_bound=datetime(2022, 5, 29, tzinfo=timezone.utc), upper_bound=datetime(2027, 5, 28, tzinfo=timezone.utc))
    lb, ub = q.lower_bound.timestamp(), q.upper_bound.timestamp(); rng = ub - lb
    vals = [ub - 0.3 * rng, ub - 0.1 * rng, ub + 3 * rng, ub + 4 * rng]          # two tails far beyond the tolerance
    out = clamp_percentiles([Percentile(percentile=p, value=v) for p, v in zip((0.1, 0.5, 0.9, 0.95), vals)], q)
    assert all(lb - 2 * rng <= x.value <= ub + 2 * rng for x in out)
    assert all(b.value > a.value for a, b in zip(out, out[1:]))
    assert out[0].value == vals[0] and out[1].value == vals[1]                     # values inside are untouched
