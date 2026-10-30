"""Shared fakes for the offline tests and the plain-Python two-event-loop check (no network, no tokens)."""
from __future__ import annotations

import re

from forecasting_tools import BinaryQuestion, GeneralLlm
from forecasting_tools.data_models.binary_report import BinaryPrediction
from forecasting_tools.data_models.multiple_choice_report import PredictedOption, PredictedOptionList
from forecasting_tools.data_models.numeric_report import Percentile

from committee_bot import CommitteeBot, Member

MODEL_P = {"openrouter/fam1/m1": 0.30, "openrouter/fam2/m2": 0.40, "openrouter/fam3/m3": 0.50,
           "openrouter/fam4/m4": 0.60, "openrouter/fam5/m5": 0.90}
MODELS = list(MODEL_P)
WEIGHTS = {MODELS[0]: 3.0}                      # the anchor model weighs 3, the others 1 (like config.py tiers)


def make_fake_invoke(research_fails: bool = False, followup: str = "query one\nquery two"):
    calls: list[tuple[str, str]] = []

    async def fake_invoke(self, prompt, *args, **kwargs):
        calls.append((self.model, prompt))
        if "OUTSIDE VIEW" in prompt:
            if research_fails:
                raise RuntimeError("search down")
            return "Base rate: 30% (3 of 10 similar cases)."
        if "RECENT news" in prompt:
            if research_fails:
                raise RuntimeError("news down")
            return "News: nothing decisive yet."
        if "Name at most TWO specific facts" in prompt:
            return followup
        if prompt.startswith("Answer concisely with dated sources"):
            return "Found: a dated fact."
        i = MODELS.index(self.model) if self.model in MODELS else 0
        if "The options are" in prompt:
            a = 40 + 5 * i
            return f"Alpha: {a}%\nBeta: {90 - a}%\nGamma: 10%"
        if "Percentile 10:" in prompt:
            s = 5 * i
            return "\n".join(f"Percentile {p}: {p + s}" for p in (5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95))
        return f"(a)...(e)\nProbability: {round(MODEL_P.get(self.model, 0.5) * 100)}%"
    return fake_invoke, calls


async def fake_structure_output(text_to_structure, output_type, model=None, **kwargs):
    t = text_to_structure
    if output_type is BinaryPrediction:
        return BinaryPrediction(prediction_in_decimal=float(re.findall(r"Probability:\s*([\d.]+)%", t)[-1]) / 100)
    if output_type is PredictedOptionList:
        return PredictedOptionList(predicted_options=[PredictedOption(option_name=n, probability=float(p) / 100)
                                                      for n, p in re.findall(r"^(\w+):\s*([\d.]+)%", t, re.M)])
    if output_type == list[Percentile]:
        return [Percentile(percentile=int(a) / 100, value=float(b))
                for a, b in re.findall(r"Percentile (\d+):\s*([\d.]+)", t)]
    raise AssertionError(f"unexpected output type {output_type}")


def make_bot(**kwargs) -> CommitteeBot:
    parser = GeneralLlm(model="openrouter/fam1/parser")
    committee = [Member(GeneralLlm(model=m), WEIGHTS.get(m, 1.0)) for m in MODELS]
    return CommitteeBot(committee=committee, news_researcher="openrouter/x/news", outside_researcher="openrouter/x/outside",
                        followup_planner="openrouter/x/planner", followup_searcher="openrouter/x/search",
                        publish_reports_to_metaculus=False, skip_previously_forecasted_questions=False,
                        llms={"default": committee[0].llm, "summarizer": parser,
                              "researcher": GeneralLlm(model="openrouter/x/news"), "parser": parser}, **kwargs)


def bq(i=1) -> BinaryQuestion:
    return BinaryQuestion(question_text="Will X happen before 2026-12-31?", resolution_criteria="Resolves Yes if X.",
                          fine_print="", background_info="", page_url=f"https://example.org/q/{i}", id_of_question=i)
