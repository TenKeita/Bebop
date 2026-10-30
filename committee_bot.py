"""CommitteeBot — the LLM_Ventures Metaculus FutureEval bot (built 2026-10-30).

Subclass of Metaculus's official template (template_bot.FallTemplateBot2026, kept verbatim). It inherits every tested
question handler (binary, multiple choice, numeric, date, conditional; parsing; numeric CDF construction; comments;
submission) and changes only what the published 2025-26 bot analyses say matters
(research/A4_metaculus_futureeval.md, research/A4_bot_spec.md):

  1. A COMMITTEE of models across families instead of N runs of one model — model choice dominates, and diversity +
     aggregation helps. Each prediction run gets its own model through a per-task context variable (no race between
     the parallel runs of one question).
  2. Two research reports merged: an OUTSIDE view (reference classes, base rates) and an INSIDE view (latest news,
     current status). Never abstain: if both fail, forecast from the question text alone (a skipped question scores 0).
  3. Binary questions: the base rate is stated first, then the update; aggregation = trimmed mean of log-odds
     (+ optional extremisation), clipped to [2%, 98%] (aggregation.py).
  4. More forgiving: a question succeeds if >= 40% of the runs return (template: 50%).
Multiple-choice, numeric and date questions keep the template's prompts and aggregation (median CDF).
"""
from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import os
import urllib.request
from datetime import datetime

from forecasting_tools import (AskNewsSearcher, BinaryQuestion, GeneralLlm, MetaculusQuestion, ReasonedPrediction,
                               clean_indents)

import config as C
from aggregation import aggregate_binary
from template_bot import FallTemplateBot2026

logger = logging.getLogger(__name__)
_RUN_MODEL: contextvars.ContextVar[GeneralLlm | None] = contextvars.ContextVar("_RUN_MODEL", default=None)
NO_RESEARCH = ("(No research could be retrieved for this question. Forecast from the question text, its resolution "
               "criteria and general knowledge; lean towards base rates and the status quo.)")


def available_openrouter_models(timeout: float = 30) -> set[str] | None:
    """IDs of the models OpenRouter currently serves (public endpoint), or None if the list cannot be read."""
    try:
        req = urllib.request.Request("https://openrouter.ai/api/v1/models", headers={"User-Agent": "committee-bot"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return {m["id"] for m in json.load(r)["data"]}
    except Exception as e:                                       # noqa: BLE001
        logger.warning(f"could not read the OpenRouter model list ({e!r}); using the configured models unchecked")
        return None


def build_committee(tier: str, available: set[str] | None) -> list[GeneralLlm]:
    """The tier's models that still exist, topped up to the tier's committee size with available models from the
    cheaper tiers (a 1-model 'committee' would lose the aggregation benefit)."""
    def alive(names: list[str]) -> list[str]:
        return [n for n in names if available is None or n.removeprefix("openrouter/") in available]
    order = list(C.TIERS)                                       # premium -> ... -> minimal
    target = len(C.TIERS[tier])
    names = alive(C.TIERS[tier])
    for cheaper in order[order.index(tier) + 1:]:
        for n in alive(C.TIERS[cheaper]):
            if len(names) >= target:
                break
            names.append(n)
    if not names:
        raise RuntimeError("no configured committee model is available on OpenRouter")
    dropped = sorted(set(C.TIERS[tier]) - set(names))
    if dropped:
        logger.warning(f"tier '{tier}': models not available and dropped: {dropped}")
    return [GeneralLlm(model=n, timeout=C.LLM_TIMEOUT_S, allowed_tries=2) for n in names]


class CommitteeBot(FallTemplateBot2026):
    def __init__(self, *, committee: list[GeneralLlm], news_researcher: str, outside_researcher: str,
                 extremize: float = 1.0, clip: tuple[float, float] = (0.02, 0.98), **kwargs):
        if not committee:
            raise ValueError("empty committee")
        kwargs.setdefault("research_reports_per_question", 1)
        kwargs.setdefault("predictions_per_research_report", len(committee))
        kwargs.setdefault("required_successful_predictions", 0.4)
        kwargs.setdefault("enable_summarize_research", False)
        super().__init__(**kwargs)
        self.committee = committee
        self.news_researcher = news_researcher
        self.outside_researcher = outside_researcher
        self.extremize, self.clip = extremize, clip
        self.models_used: list[str] = []                       # which model ran each prediction (logs, tests)
        self._run_idx: dict[object, int] = {}
        self._lock: asyncio.Lock | None = None
        self._lock_loop: asyncio.AbstractEventLoop | None = None

    # ------------------------------------------------------------------ committee: one model per prediction run
    def get_llm(self, purpose: str = "default", guarantee_type=None):        # type: ignore[override]
        model = _RUN_MODEL.get()
        if purpose == "default" and model is not None:
            return model.model if guarantee_type == "string_name" else model
        return super().get_llm(purpose, guarantee_type)

    def _loop_lock(self) -> asyncio.Lock:
        loop = asyncio.get_running_loop()
        if self._lock is None or self._lock_loop is not loop:      # main.py runs one asyncio.run per tournament
            self._lock, self._lock_loop = asyncio.Lock(), loop
        return self._lock

    def _loop_semaphore(self) -> asyncio.Semaphore:
        """Per-event-loop version of the template's CLASS-level `_concurrency_limiter` — a defensive guard.
        main.py (like the template) calls asyncio.run once per tournament. Today forecasting_tools applies
        nest_asyncio on import, so asyncio.run reuses ONE loop and the template's semaphore is safe (checked
        2026-10-30). Without that patch, a semaphore that had waiters in a first loop raises "bound to a different
        event loop" in the next one; this guard keeps the bot correct either way
        (tests: test_plain_asyncio_semaphore_binds_to_loop, test_forecasting_tools_reuses_one_loop,
        test_two_event_loops_plain_python)."""
        loop = asyncio.get_running_loop()
        if getattr(self, "_sem_loop", None) is not loop:
            self._sem, self._sem_loop = asyncio.Semaphore(self._max_concurrent_questions), loop
        return self._sem

    async def _make_prediction(self, question: MetaculusQuestion, research: str):
        async with self._loop_lock():
            key = question.id_of_question or question.page_url or id(question)
            idx = self._run_idx.get(key, 0)
            self._run_idx[key] = idx + 1
        model = self.committee[idx % len(self.committee)]
        self.models_used.append(model.model)
        token = _RUN_MODEL.set(model)
        try:
            return await super()._make_prediction(question, research)
        finally:
            _RUN_MODEL.reset(token)

    # ------------------------------------------------------------------ research: outside view + inside view
    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._loop_semaphore():
            news, outside = await asyncio.gather(self._news_research(question), self._outside_view_research(question),
                                                 return_exceptions=True)
            parts = []
            for title, text in (("Outside view (reference classes and base rates)", outside),
                                ("Inside view (latest news and current status)", news)):
                if isinstance(text, str) and text.strip():
                    parts.append(f"## {title}\n{text.strip()}")
                else:
                    logger.warning(f"{title} research failed for {question.page_url}: {text!r}")
            research = "\n\n".join(parts) if parts else NO_RESEARCH
            logger.info(f"Research for URL {question.page_url}:\n{research}")
            return research

    @staticmethod
    def _question_block(question: MetaculusQuestion) -> str:
        return clean_indents(
            f"""
            Question:
            {question.question_text}

            This question's outcome will be determined by the specific criteria below:
            {question.resolution_criteria}

            {question.fine_print}

            Question closes: {question.close_time}. Scheduled resolution: {question.scheduled_resolution_time}.
            Today is {datetime.now().strftime("%Y-%m-%d")}.
            """
        )

    async def _news_research(self, question: MetaculusQuestion) -> str:
        if (os.getenv("ASKNEWS_CLIENT_ID") and os.getenv("ASKNEWS_SECRET")) or os.getenv("ASKNEWS_API_KEY"):
            try:
                return await AskNewsSearcher().call_preconfigured_version("asknews/news-summaries",
                                                                          question.question_text)
            except Exception as e:                               # noqa: BLE001
                logger.warning(f"AskNews failed, falling back to {self.news_researcher}: {e!r}")
        prompt = clean_indents(
            f"""
            You are an assistant to a superforecaster. Give a concise but detailed rundown of the most relevant
            RECENT news and the current status of the situation below, including whether the question would resolve
            Yes or No based on current information, and any scheduled events before the resolution date.
            Cite sources with dates. You do not produce forecasts yourself.

            {self._question_block(question)}
            """
        )
        return await GeneralLlm(model=self.news_researcher, timeout=C.LLM_TIMEOUT_S, allowed_tries=2).invoke(prompt)

    async def _outside_view_research(self, question: MetaculusQuestion) -> str:
        prompt = clean_indents(
            f"""
            You are a research assistant to a superforecaster. Do NOT forecast. Provide the OUTSIDE VIEW:
            1. Two to four reference classes of similar past events or situations, with their historical base rates
               (numbers and sources).
            2. How often things like this happen within a time window like the one left before the question resolves.
            3. Scheduled events, deadlines or data releases before the resolution date that could decide it.
            4. Any precise detail of the resolution criteria or fine print that makes Yes or No easier than it looks.
            Be concise (at most 350 words), factual, and cite sources.

            {self._question_block(question)}
            """
        )
        return await GeneralLlm(model=self.outside_researcher, timeout=C.LLM_TIMEOUT_S, allowed_tries=2).invoke(prompt)

    # ------------------------------------------------------------------ binary: base rate first, then update
    async def _run_forecast_on_binary(self, question: BinaryQuestion, research: str) -> ReasonedPrediction[float]:
        prompt = clean_indents(
            f"""
            You are a professional forecaster.

            Question:
            {question.question_text}

            Background:
            {question.background_info}

            This question's outcome will be determined by the specific criteria below. These criteria have not yet
            been satisfied:
            {question.resolution_criteria}

            {question.fine_print}

            Your research assistants say:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}. The question closes {question.close_time}.

            Before answering you write:
            (a) The time left until the outcome is known.
            (b) The base rate: what the reference classes in the outside view imply, as a probability.
            (c) The status quo outcome if nothing changed.
            (d) The strongest evidence for Yes and for No from the inside view, and how far each should move you from
                the base rate.
            (e) Exactly what must happen for the question to resolve Yes under the resolution criteria and fine print.

            Good forecasters start from the base rate, put extra weight on the status quo (the world changes slowly
            most of the time), update only as much as the evidence warrants, and avoid probabilities below 2% or above
            98% unless the outcome is effectively decided.
            {self._get_conditional_disclaimer_if_necessary(question)}

            The last thing you write is your final answer as: "Probability: ZZ%", 0-100
            """
        )
        return await self._binary_prompt_to_forecast(question, prompt)

    # ------------------------------------------------------------------ aggregation
    async def _aggregate_predictions(self, predictions, question):
        if isinstance(question, BinaryQuestion) and predictions and all(isinstance(p, float) for p in predictions):
            return aggregate_binary(predictions, extremize=self.extremize, clip=self.clip)
        return await super()._aggregate_predictions(predictions, question)
