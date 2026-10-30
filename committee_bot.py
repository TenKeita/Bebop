"""CommitteeBot — a Metaculus FutureEval forecasting bot (built 2026-10-30, revised the same day against the 2026 bot
analyses; see config.py for the evidence).

Subclass of Metaculus's official template (template_bot.FallTemplateBot2026, kept verbatim). It inherits every tested
question handler (binary, multiple choice, numeric, date, conditional; parsing; CDF construction; comments;
submission) and changes only what the published analyses say matters:

  1. A WEIGHTED COMMITTEE across model families, anchored on the strongest model on the FutureEval leaderboard
     (Claude Opus) — frontier models dominate; 3-7 diverse models help. Each prediction run gets its own model via a
     per-task context variable (no race between the parallel runs of one question).
  2. RESEARCH = outside view (reference classes, base rates) + inside view (latest news) + one AGENTIC FOLLOW-UP
     round (a cheap model names the 1-2 facts that would most change the forecast; they are searched). Never abstain:
     if everything fails, forecast from the question text alone (a skipped question scores 0).
  3. Binary: base rate first, then the update; aggregation = WEIGHTED MEDIAN (median preferred by winners; the best
     model weighs more), capped to [2%, 98%].
  4. Numeric / date: 11 percentiles (5th..95th) instead of the template's 6, so the tails are stated rather than
     extrapolated (several winners flagged the template's numeric handling as weak).
  5. More forgiving: a question succeeds if >= 40% of the runs return (template: 50%).
Multiple-choice questions keep the template's prompt and median aggregation.
"""
from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import os
import urllib.request
from dataclasses import dataclass
from datetime import datetime

from forecasting_tools import (AskNewsSearcher, BinaryQuestion, DateQuestion, GeneralLlm, MetaculusQuestion,
                               NumericDistribution, NumericQuestion, ReasonedPrediction, clean_indents)

import config as C
from aggregation import aggregate_binary
from template_bot import FallTemplateBot2026

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Member:
    llm: GeneralLlm
    weight: float = 1.0

    @property
    def model(self) -> str:
        return self.llm.model


_RUN_MEMBER: contextvars.ContextVar[Member | None] = contextvars.ContextVar("_RUN_MEMBER", default=None)
NO_RESEARCH = ("(No research could be retrieved for this question. Forecast from the question text, its resolution "
               "criteria and general knowledge; lean towards base rates and the status quo.)")
PERCENTILES = (5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95)


def available_openrouter_models(timeout: float = 30) -> set[str] | None:
    """IDs of the models OpenRouter currently serves (public endpoint), or None if the list cannot be read."""
    try:
        req = urllib.request.Request("https://openrouter.ai/api/v1/models", headers={"User-Agent": "committee-bot"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return {m["id"] for m in json.load(r)["data"]}
    except Exception as e:                                       # noqa: BLE001
        logger.warning(f"could not read the OpenRouter model list ({e!r}); using the configured models unchecked")
        return None


def make_llm(model: str, reasoning_effort: str | None = None) -> GeneralLlm:
    kwargs = {"reasoning_effort": reasoning_effort} if reasoning_effort else {}
    return GeneralLlm(model=model, timeout=C.LLM_TIMEOUT_S, allowed_tries=2, **kwargs)


def build_committee(tier: str, available: set[str] | None) -> list[Member]:
    """The tier's members whose models still exist, topped up to the tier's size with available members of the
    cheaper tiers (a 1-model 'committee' would lose the aggregation benefit)."""
    def alive(members: list[dict]) -> list[dict]:
        return [m for m in members if available is None or m["model"].removeprefix("openrouter/") in available]
    order = list(C.TIERS)                                       # premium -> ... -> minimal
    target = len(C.TIERS[tier])
    chosen = alive(C.TIERS[tier])
    for cheaper in order[order.index(tier) + 1:]:
        for m in alive(C.TIERS[cheaper]):
            if len(chosen) >= target:
                break
            chosen.append(m)
    if not chosen:
        raise RuntimeError("no configured committee model is available on OpenRouter")
    dropped = sorted({m["model"] for m in C.TIERS[tier]} - {m["model"] for m in chosen})
    if dropped:
        logger.warning(f"tier '{tier}': models not available and replaced: {dropped}")
    return [Member(make_llm(m["model"], m.get("reasoning_effort")), m["weight"]) for m in chosen]


class CommitteeBot(FallTemplateBot2026):
    def __init__(self, *, committee: list[Member], news_researcher: str, outside_researcher: str,
                 aggregator: str = "weighted_median", extremize: float = 1.0,
                 clip: tuple[float, float] = (0.02, 0.98), research_followup: bool = True,
                 followup_planner: str | None = None, followup_searcher: str | None = None, **kwargs):
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
        self.aggregator, self.extremize, self.clip = aggregator, extremize, clip
        self.research_followup = research_followup
        self.followup_planner = followup_planner or C.FOLLOWUP_PLANNER
        self.followup_searcher = followup_searcher or C.FOLLOWUP_SEARCHER
        self.models_used: list[str] = []                       # which model ran each prediction (logs, tests)
        self._run_idx: dict[object, int] = {}
        self._run_results: dict[object, list[tuple[float, object]]] = {}
        self._lock: asyncio.Lock | None = None
        self._lock_loop: asyncio.AbstractEventLoop | None = None

    # ------------------------------------------------------------------ committee: one member per prediction run
    @staticmethod
    def _key(question: MetaculusQuestion) -> object:
        return question.id_of_question or question.page_url or id(question)

    def get_llm(self, purpose: str = "default", guarantee_type=None):        # type: ignore[override]
        member = _RUN_MEMBER.get()
        if purpose == "default" and member is not None:
            return member.model if guarantee_type == "string_name" else member.llm
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
        key = self._key(question)
        async with self._loop_lock():
            idx = self._run_idx.get(key, 0)
            self._run_idx[key] = idx + 1
        member = self.committee[idx % len(self.committee)]
        self.models_used.append(member.model)
        token = _RUN_MEMBER.set(member)
        try:
            result = await super()._make_prediction(question, research)
        finally:
            _RUN_MEMBER.reset(token)
        self._run_results.setdefault(key, []).append((member.weight, result.prediction_value))
        return result

    # ------------------------------------------------------------------ research: outside + inside view + follow-up
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
            if self.research_followup and parts:
                try:
                    follow = await self._followup_research(question, "\n\n".join(parts))
                    if follow:
                        parts.append(f"## Follow-up searches\n{follow}")
                except Exception as e:                           # noqa: BLE001 — follow-up is optional
                    logger.warning(f"follow-up research failed for {question.page_url}: {e!r}")
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
        return await make_llm(self.news_researcher).invoke(prompt)

    async def _outside_view_research(self, question: MetaculusQuestion) -> str:
        prompt = clean_indents(
            f"""
            You are a research assistant to a superforecaster. Do NOT forecast. Provide the OUTSIDE VIEW:
            1. Two to four reference classes of similar past events or situations, with their historical base rates
               (numbers and sources). Mention any previously resolved forecasting questions that were similar.
            2. How often things like this happen within a time window like the one left before the question resolves.
            3. Scheduled events, deadlines or data releases before the resolution date that could decide it.
            4. Any precise detail of the resolution criteria or fine print that makes Yes or No easier than it looks.
            Be concise (at most 350 words), factual, and cite sources.

            {self._question_block(question)}
            """
        )
        return await make_llm(self.outside_researcher).invoke(prompt)

    async def _followup_research(self, question: MetaculusQuestion, research: str) -> str:
        plan_prompt = clean_indents(
            f"""
            You help a forecaster. Read the question and the research so far. Name at most TWO specific facts that are
            still unknown and would most change the forecast, each written as a precise web search query.
            Reply with one query per line and nothing else; reply NONE if the research already covers what matters.

            {self._question_block(question)}

            Research so far:
            {research[:6000]}
            """
        )
        plan = await make_llm(self.followup_planner).invoke(plan_prompt)
        queries = [q.strip(" -*•\t") for q in plan.splitlines() if q.strip() and q.strip().upper() != "NONE"][:2]
        if not queries:
            return ""
        searcher = make_llm(self.followup_searcher)
        answers = await asyncio.gather(*[searcher.invoke(
            f"Answer concisely with dated sources (today is {datetime.now():%Y-%m-%d}): {q}") for q in queries],
            return_exceptions=True)
        return "\n\n".join(f"Q: {q}\nA: {a.strip()}" for q, a in zip(queries, answers)
                           if isinstance(a, str) and a.strip())

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
            (d) The strongest evidence for Yes and for No from the inside view and follow-up searches, and how far
                each should move you from the base rate.
            (e) Exactly what must happen for the question to resolve Yes under the resolution criteria and fine print.

            Good forecasters start from the base rate, put extra weight on the status quo (the world changes slowly
            most of the time), update only as much as the evidence warrants, and avoid probabilities below 2% or above
            98% unless the outcome is effectively decided.
            {self._get_conditional_disclaimer_if_necessary(question)}

            The last thing you write is your final answer as: "Probability: ZZ%", 0-100
            """
        )
        return await self._binary_prompt_to_forecast(question, prompt)

    # ------------------------------------------------------------------ numeric / date: 11 percentiles incl. tails
    def _percentile_answer_block(self, kind: str) -> str:
        unit = "YYYY-MM-DD" if kind == "date" else "XX"
        first, last = ("(oldest date)", "(newest date)") if kind == "date" else ("(lowest value)", "(highest value)")
        lines = [f"Percentile {p}: {unit}" + (f" {first}" if p == PERCENTILES[0] else f" {last}" if p == PERCENTILES[-1]
                                               else "") for p in PERCENTILES]
        return "\n".join(lines)

    async def _run_forecast_on_numeric(self, question: NumericQuestion, research: str
                                       ) -> ReasonedPrediction[NumericDistribution]:
        upper_bound_message, lower_bound_message = self._create_upper_and_lower_bound_messages(question)
        prompt = clean_indents(
            f"""
            You are a professional forecaster.

            Question:
            {question.question_text}

            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}

            Units for answer: {question.unit_of_measure if question.unit_of_measure else "Not stated (please infer this)"}

            Your research assistants say:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            {lower_bound_message}
            {upper_bound_message}

            Formatting Instructions:
            - Please notice the units requested and give your answer in these units (e.g. whether you represent a
              number as 1,000,000 or 1 million).
            - Never use scientific notation.
            - Always start with a smaller number (more negative if negative) and then increase from there. Each
              percentile's value must be larger than the previous one.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The base rate / historical range of this quantity and its recent trend.
            (c) The outcome if nothing changed, and if the current trend continued.
            (d) The expectations of experts and markets.
            (e) Unexpected scenarios that result in a low and in a high outcome, and how likely they are.

            {self._get_conditional_disclaimer_if_necessary(question)}
            You remind yourself that good forecasters are humble and set wide 90/10 and 95/5 intervals to account for
            unknown unknowns; the 5th and 95th percentiles describe the tails, not the most likely range.

            The last thing you write is your final answer as:
            "
            {self._percentile_answer_block("numeric")}
            "
            """
        )
        return await self._numeric_prompt_to_forecast(question, prompt)

    async def _run_forecast_on_date(self, question: DateQuestion, research: str
                                    ) -> ReasonedPrediction[NumericDistribution]:
        upper_bound_message, lower_bound_message = self._create_upper_and_lower_bound_messages(question)
        prompt = clean_indents(
            f"""
            You are a professional forecaster.

            Question:
            {question.question_text}

            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}

            Your research assistants say:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            {lower_bound_message}
            {upper_bound_message}

            Formatting Instructions:
            - This is a date question, and as such, the answer must be expressed in terms of dates.
            - The dates must be written in the format of YYYY-MM-DD. If hours matter, please append the date with the
              hour in UTC and military time: YYYY-MM-DDTHH:MM:SSZ. No other formatting is allowed.
            - Always start with a lower date chronologically and then increase from there; each percentile's date
              must be later than the previous one.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The base rate: how long similar things have taken historically.
            (c) The outcome if nothing changed, and if the current trend continued.
            (d) The expectations of experts and markets.
            (e) Unexpected scenarios that result in an early and in a late outcome.

            {self._get_conditional_disclaimer_if_necessary(question)}
            You remind yourself that good forecasters are humble and set wide intervals to account for unknown
            unknowns; the 5th and 95th percentiles describe the tails.

            The last thing you write is your final answer as:
            "
            {self._percentile_answer_block("date")}
            "
            """
        )
        return await self._date_prompt_to_forecast(question, prompt)

    # ------------------------------------------------------------------ aggregation
    async def _aggregate_predictions(self, predictions, question):
        if isinstance(question, BinaryQuestion) and predictions and all(isinstance(p, float) for p in predictions):
            runs = self._run_results.get(self._key(question), [])
            if len(runs) == len(predictions) and sorted(v for _, v in runs) == sorted(predictions):
                values, weights = [v for _, v in runs], [w for w, _ in runs]
            else:                                                  # bookkeeping mismatch: fall back to equal weights
                logger.warning(f"weights unavailable for {question.page_url}; using equal weights")
                values, weights = list(predictions), [1.0] * len(predictions)
            return aggregate_binary(values, weights, method=self.aggregator, extremize=self.extremize, clip=self.clip)
        return await super()._aggregate_predictions(predictions, question)
