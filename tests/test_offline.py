"""Offline tests: the whole forecasting pipeline with every LLM call mocked (no network, no tokens, no credits).

Run:  .venv/Scripts/python -m pytest -q tests
"""
from __future__ import annotations

import asyncio
import pathlib
import re
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from forecasting_tools import BinaryQuestion, GeneralLlm, MultipleChoiceQuestion, NumericQuestion  # noqa: E402
from forecasting_tools.data_models.binary_report import BinaryPrediction  # noqa: E402
from forecasting_tools.data_models.multiple_choice_report import PredictedOption, PredictedOptionList  # noqa: E402
from forecasting_tools.data_models.numeric_report import Percentile  # noqa: E402

import budget  # noqa: E402
import committee_bot  # noqa: E402
import template_bot  # noqa: E402
from aggregation import aggregate_binary, logit  # noqa: E402
from committee_bot import NO_RESEARCH, CommitteeBot, build_committee  # noqa: E402

from fakes import MODEL_P, MODELS, bq, fake_structure_output, make_bot, make_fake_invoke  # noqa: E402


@pytest.fixture
def offline(monkeypatch):
    for k in ("ASKNEWS_CLIENT_ID", "ASKNEWS_SECRET", "ASKNEWS_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    fake, calls = make_fake_invoke()
    monkeypatch.setattr(GeneralLlm, "invoke", fake)
    monkeypatch.setattr(template_bot, "structure_output", fake_structure_output)
    return calls


# ---------------------------------------------------------------- aggregation
def test_aggregate_binary_trimmed_logodds():
    # trimmed: drop 0.30 and 0.90 -> log-odds mean of 0.4, 0.5, 0.6 = 0 -> 0.5
    assert aggregate_binary([0.3, 0.4, 0.5, 0.6, 0.9]) == pytest.approx(0.5)
    # fewer than 5 runs: no trimming
    assert aggregate_binary([0.2, 0.8]) == pytest.approx(0.5)


def test_aggregate_binary_clip_and_extremize():
    assert aggregate_binary([0.999] * 5) == pytest.approx(0.98)
    assert aggregate_binary([0.001] * 5) == pytest.approx(0.02)
    p = aggregate_binary([0.6] * 3, extremize=2.0)
    assert p == pytest.approx(1 / (1 + pow(2.718281828459045, -2 * logit(0.6))), rel=1e-6)
    with pytest.raises(ValueError):
        aggregate_binary([])


# ---------------------------------------------------------------- event loops (defensive guard)
def test_plain_asyncio_semaphore_binds_to_loop():
    """Why the bot keeps a per-loop semaphore: in plain asyncio (no nest_asyncio), a semaphore that had waiters in one
    asyncio.run cannot have waiters in another. Run in a clean subprocess (pytest's asyncio plugin changes loop
    handling)."""
    import subprocess
    code = ("import asyncio\nsem = asyncio.Semaphore(1)\n"
            "async def two(s):\n"
            "    async def u():\n"
            "        async with s:\n"
            "            await asyncio.sleep(0.01)\n"
            "    await asyncio.gather(u(), u())\n"
            "asyncio.run(two(sem))\n"
            "try:\n    asyncio.run(two(sem)); print('NO ERROR')\n"
            "except RuntimeError as e:\n    print('ERROR', e)\n")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60).stdout
    assert "different event loop" in out, out


def test_bot_survives_two_event_loops(offline):
    bot = make_bot()
    for loop_run in range(2):                                   # like main.py: tournament, then MiniBench
        reports = asyncio.run(bot.forecast_questions([bq(10 + loop_run * 2), bq(11 + loop_run * 2)],
                                                     return_exceptions=True))
        assert all(not isinstance(r, BaseException) for r in reports), reports


# ---------------------------------------------------------------- binary end to end
def test_binary_committee_end_to_end(offline):
    bot = make_bot()
    report = asyncio.run(bot.forecast_question(bq()))
    assert sorted(bot.models_used) == sorted(MODELS)           # every committee member ran exactly once
    assert report.prediction == pytest.approx(0.5)              # trimmed log-odds mean (see above)
    research_prompts = [p for _, p in offline if "OUTSIDE VIEW" in p or "RECENT news" in p]
    assert len(research_prompts) == 2                           # outside + inside view
    forecast_prompts = [p for m, p in offline if m in MODELS]
    assert all("Base rate: 30%" in p and "News: nothing decisive" in p for p in forecast_prompts)


def test_research_failure_never_abstains(monkeypatch):
    for k in ("ASKNEWS_CLIENT_ID", "ASKNEWS_SECRET", "ASKNEWS_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    fake, calls = make_fake_invoke(research_fails=True)
    monkeypatch.setattr(GeneralLlm, "invoke", fake)
    monkeypatch.setattr(template_bot, "structure_output", fake_structure_output)
    bot = make_bot()
    report = asyncio.run(bot.forecast_question(bq(2)))
    assert report.prediction == pytest.approx(0.5)
    assert any(NO_RESEARCH in p for m, p in calls if m in MODELS)


def test_get_llm_outside_a_run_is_the_default(offline):
    bot = make_bot()
    assert bot.get_llm("default", "llm").model == MODELS[0]
    assert bot.get_llm("parser", "llm").model == "openrouter/fam1/parser"


# ---------------------------------------------------------------- multiple choice and numeric end to end
def test_multiple_choice_end_to_end(offline):
    bot = make_bot()
    q = MultipleChoiceQuestion(question_text="Which?", options=["Alpha", "Beta", "Gamma"], resolution_criteria="",
                               fine_print="", background_info="", page_url="https://example.org/q/3", id_of_question=3)
    report = asyncio.run(bot.forecast_question(q))
    probs = {o.option_name: o.probability for o in report.prediction.predicted_options}
    assert set(probs) == {"Alpha", "Beta", "Gamma"}
    assert sum(probs.values()) == pytest.approx(1.0, abs=1e-6)
    assert sorted(bot.models_used) == sorted(MODELS)


def test_numeric_end_to_end(offline):
    bot = make_bot()
    q = NumericQuestion(question_text="How many?", upper_bound=200.0, lower_bound=0.0, open_upper_bound=True,
                        open_lower_bound=False, unit_of_measure="units", resolution_criteria="", fine_print="",
                        background_info="", page_url="https://example.org/q/4", id_of_question=4)
    report = asyncio.run(bot.forecast_question(q))
    cdf = [p.percentile for p in report.prediction.get_cdf()]
    assert len(cdf) == 201
    assert all(b >= a for a, b in zip(cdf, cdf[1:]))           # monotone
    assert cdf[0] == pytest.approx(0.0, abs=1e-9)               # closed lower bound
    assert cdf[-1] < 1.0                                        # open upper bound keeps some mass above


# ---------------------------------------------------------------- budget guard and committee construction
class _Resp:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def test_credit_guard(monkeypatch):
    g = budget.CreditGuard("key", soft_floor_usd=15, hard_floor_usd=3)
    for remaining, expected in ((50.0, "economy"), (10.0, "minimal"), (2.0, None)):
        monkeypatch.setattr(budget.requests, "get", lambda *a, r=remaining, **k: _Resp({"data": {"limit_remaining": r}}))
        tier, why = g.decide("economy", "minimal")
        assert tier == expected, why
    monkeypatch.setattr(budget.requests, "get", lambda *a, **k: (_ for _ in ()).throw(OSError("offline")))
    assert g.decide("economy", "minimal")[0] == "economy"      # unknown -> keep the requested tier
    assert budget.CreditGuard(None, 15, 3).remaining_usd() is None


def test_build_committee_drops_missing_models():
    tier = "standard"
    names = committee_bot.C.TIERS[tier]
    available = {n.removeprefix("openrouter/") for n in names[1:]}
    c = build_committee(tier, available)
    assert len(c) == len(names)                                  # topped up to the full committee size
    assert [m.model for m in c][:len(names) - 1] == names[1:]    # own models first, in order
    assert all(m.model.removeprefix("openrouter/") in available | {n.removeprefix("openrouter/")
               for t in committee_bot.C.TIERS.values() for n in t} for m in c)
    only_minimal = {n.removeprefix("openrouter/") for n in committee_bot.C.TIERS["minimal"]}
    fallback = build_committee(tier, only_minimal)
    assert len(fallback) == len(names)
    assert all(m.model.removeprefix("openrouter/") in only_minimal for m in fallback)
    with pytest.raises(RuntimeError):
        build_committee(tier, set())


def test_two_event_loops_plain_python():
    """The real check for the semaphore fix: plain Python, two asyncio.run calls with 2 questions each."""
    import subprocess
    script = pathlib.Path(__file__).with_name("two_loops_check.py")
    out = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=300)
    assert out.stdout.strip().endswith("OK"), out.stdout[-2000:] + out.stderr[-2000:]


def test_forecasting_tools_reuses_one_loop():
    """forecasting_tools applies nest_asyncio: asyncio.run reuses one loop, so the template's class-level semaphore is
    safe today. If a future version drops that patch, this test fails and the per-loop guard becomes essential."""
    import subprocess
    code = chr(10).join([
        "import asyncio, forecasting_tools",
        "loops = []",
        "async def f():",
        "    loops.append(asyncio.get_running_loop())",
        "asyncio.run(f()); asyncio.run(f())",
        "print('SAME' if loops[0] is loops[1] else 'DIFFERENT')",
    ])
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120).stdout
    assert out.strip().endswith("SAME"), out
