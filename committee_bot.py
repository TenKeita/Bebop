"""CommitteeBot — a Metaculus FutureEval forecasting bot (built 2026-10-30; revised 2026-11-12 on Metaculus's own
tournament data — see aggregation.py and config.py for the evidence behind each choice).

Subclass of Metaculus's official template (template_bot.FallTemplateBot2026, kept verbatim). It inherits every tested
question handler (binary, multiple choice, numeric, discrete, date, conditional; parsing; CDF construction; comments;
submission) and changes only what measurably matters:

  1. A COMMITTEE of five different strong models (Claude Opus with a mild extra weight, GPT, Gemini Pro, Gemini Flash,
     a fifth for diversity). Each prediction run gets its own model via a per-task context variable.
  2. RESEARCH = outside view (base rates) + inside view (latest news) + one agentic follow-up round, plus a
     QUANTITATIVE BASELINE from market data for price questions (market_prior.py). Never abstains.
  3. Yes/no: base rate first in the prompt; mean of log-odds across the committee; a calibration SHIFT down (every
     bot in past tournaments over-predicted Yes); blend with the market baseline when there is one; capped [2%, 98%].
  4. Numbers / dates: 11 percentiles per run; the runs' CDFs are AVERAGED (a mixture, not the template's pointwise
     median), blended with the market baseline when there is one, then mixed with 15% of a flat distribution (past
     bots' ranges held the truth only 58% of the time when they claimed 80%).
  5. A structured log of every forecast, with each model's own answer (forecast_log.py), to fit weights later.
  6. More forgiving: a question succeeds if >= 40% of the runs return (template: 50%).
Multiple-choice questions keep the template's prompt and aggregation (smoothing did not help on past data).
"""
from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import os
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
from forecasting_tools import (AskNewsSearcher, BinaryQuestion, DateQuestion, GeneralLlm, MetaculusQuestion,
                               MultipleChoiceQuestion, NumericDistribution, NumericQuestion, ReasonedPrediction,
                               clean_indents)
from forecasting_tools.data_models.numeric_report import Percentile

import config as C
import market_prior as MP
import market_question as MQ
from aggregation import aggregate_cdf, blend_logodds, calibrate_binary, combine_binary
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
                 aggregator: str = "logodds_mean", binary_shift: float = 0.0, binary_slope: float = 1.0,
                 clip: tuple[float, float] = (0.02, 0.98), numeric_flat_weight: float = 0.15, numeric_tail: float = 0.05,
                 market_prior: bool = True, market_weight_rule: float = 0.8, market_weight_llm: float = 0.5,
                 market_llm: bool = True, market_parser: str | None = None, research_followup: bool = True,
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
        self.aggregator, self.clip = aggregator, clip
        self.binary_shift, self.binary_slope = binary_shift, binary_slope
        self.numeric_flat_weight, self.numeric_tail = numeric_flat_weight, numeric_tail
        self.market_prior, self.market_llm = market_prior, market_llm
        self.market_weight_rule, self.market_weight_llm = market_weight_rule, market_weight_llm
        self.market_parser = market_parser or C.PARSER
        self._market: dict[object, MQ.MarketPrior] = {}        # question -> market-data prior (price questions only)
        self.run_log: list[dict] = []                           # one record per aggregated question (forecast_log.py)
        self.research_followup = research_followup
        self.followup_planner = followup_planner or C.FOLLOWUP_PLANNER
        self.followup_searcher = followup_searcher or C.FOLLOWUP_SEARCHER
        self.models_used: list[str] = []                       # which model ran each prediction (logs, tests)
        self._run_idx: dict[object, int] = {}
        self._run_results: dict[object, list[tuple[float, object, str]]] = {}
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
        self._run_results.setdefault(key, []).append((member.weight, result.prediction_value, member.model))
        return result

    # ------------------------------------------------------------------ research: outside + inside view + follow-up
    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._loop_semaphore():
            news, outside, prior = await asyncio.gather(self._news_research(question), self._outside_view_research(question),
                                                        self._market_prior(question), return_exceptions=True)
            parts = []
            if isinstance(prior, MQ.MarketPrior):
                unit = f" {question.unit_of_measure}" if getattr(question, "unit_of_measure", None) else ""
                parts.append(f"## Quantitative baseline (market data)\n{prior.text(unit)}")
            for title, text in (("Outside view (reference classes and base rates)", outside),
                                ("Inside view (latest news and current status)", news)):
                if isinstance(text, str) and text.strip():
                    parts.append(f"## {title}\n{text.strip()}")
                else:
                    logger.warning(f"{title} research failed for {question.page_url}: {text!r}")
            if self.research_followup and any(not x.startswith("## Quantitative baseline") for x in parts):
                try:
                    follow = await self._followup_research(question, "\n\n".join(parts))
                    if follow:
                        parts.append(f"## Follow-up searches\n{follow}")
                except Exception as e:                           # noqa: BLE001 — follow-up is optional
                    logger.warning(f"follow-up research failed for {question.page_url}: {e!r}")
            research = "\n\n".join(parts) if parts else NO_RESEARCH
            logger.info(f"Research for URL {question.page_url}:\n{research}")
            return research

    async def _market_prior(self, question: MetaculusQuestion) -> MQ.MarketPrior | None:
        """A market-data prior for price questions (see market_prior.py); None for everything else and on ANY problem."""
        if not self.market_prior or isinstance(question, (MultipleChoiceQuestion, DateQuestion)):
            return None
        title = question.question_text or ""
        if not MQ.HINT.search(f"{title} {question.resolution_criteria or ''}"):
            return None
        try:
            today = datetime.now(timezone.utc).date()
            resolve = (question.scheduled_resolution_time or question.close_time or datetime.now(timezone.utc)).date()
            rule = MQ.rule_spec(title, resolve)
            if rule:
                (spec, op, thr), thr2, scale, source, weight = rule, None, 1.0, "rule", self.market_weight_rule
            elif self.market_llm:
                is_num = isinstance(question, NumericQuestion)
                rng = (f"The answer must lie between {question.lower_bound} and {question.upper_bound} "
                       f"(unit: {question.unit_of_measure or 'not stated'})." if is_num else "")
                text = await make_llm(self.market_parser).invoke(MQ.LLM_PROMPT.format(
                    today=today.isoformat(), title=title, criteria=question.resolution_criteria or "",
                    fine_print=question.fine_print or "", qtype="number" if is_num else "yes/no", range_text=rng))
                parsed = MQ.spec_from_json(MQ.parse_llm_json(text))
                if not parsed:
                    return None
                (spec, op, thr, thr2, scale), source, weight = parsed, "llm", self.market_weight_llm
            else:
                return None
            if isinstance(question, BinaryQuestion) and (op is None or thr is None):
                return None                                       # a yes/no question needs its threshold
            if (spec.end - today).days > 370 or (today - spec.end).days > 10:
                return None
            sa = await asyncio.to_thread(MP.fetch, spec.a)
            sb = await asyncio.to_thread(MP.fetch, spec.b) if spec.b else None
            sims = await asyncio.to_thread(MP.simulate, spec, sa, sb, today)
            prior = MQ.MarketPrior(spec, sims, source, weight, op, thr, thr2, scale)
            if isinstance(question, NumericQuestion) and not MQ.plausible(prior.values, question.lower_bound, question.upper_bound):
                logger.warning(f"market prior rejected (values far outside the question's range) for {question.page_url}")
                return None
            self._market[self._key(question)] = prior
            logger.info(f"market prior ({source}, weight {weight}) for {question.page_url}: {spec}")
            return prior
        except Exception as e:                                   # noqa: BLE001 — the prior is optional
            logger.warning(f"market prior unavailable for {question.page_url}: {e!r}")
            return None

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
                news = await AskNewsSearcher().call_preconfigured_version("asknews/news-summaries",
                                                                          question.question_text)
                if "No articles were found" not in news:
                    return news
                logger.warning(f"AskNews found no articles, falling back to {self.news_researcher}")
            except Exception as e:                               # noqa: BLE001 — e.g. credits used up, rate limit
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
    def _weights_for(self, key: object, predictions: list) -> tuple[list[float], list[str]]:
        """Weight and model of each prediction, matched to the recorded runs (equal weights if that fails)."""
        runs = list(self._run_results.get(key, []))
        weights, models = [], []
        for p in predictions:
            hit = next((r for r in runs if r[1] is p or (isinstance(p, float) and isinstance(r[1], float) and r[1] == p)), None)
            if hit is None:
                logger.warning("weights unavailable for a question; using equal weights")
                return [1.0] * len(predictions), ["?"] * len(predictions)
            runs.remove(hit); weights.append(hit[0]); models.append(hit[2])
        return weights, models

    def _aggregate_binary(self, predictions: list[float], question: BinaryQuestion) -> float:
        key = self._key(question)
        weights, models = self._weights_for(key, predictions)
        raw = combine_binary(predictions, weights, self.aggregator)
        p = calibrate_binary(raw, self.binary_shift, self.binary_slope)
        prior = self._market.get(key)
        pm = prior.p_yes() if prior else None
        if pm is not None:
            p = blend_logodds(p, min(max(pm, 0.01), 0.99), prior.weight)
        final = min(max(p, self.clip[0]), self.clip[1])
        self.run_log.append(dict(
            id=question.id_of_question, url=question.page_url, type="binary", title=question.question_text,
            members=[dict(model=m, weight=w, p=float(v)) for m, w, v in zip(models, weights, predictions)],
            raw=raw, shift=self.binary_shift, market_p=pm, market_source=prior.source if prior else None,
            market_weight=prior.weight if pm is not None else None, final=final))
        return final

    def _aggregate_numeric(self, predictions: list[NumericDistribution], question) -> NumericDistribution:
        key = self._key(question)
        weights, models = self._weights_for(key, predictions)
        cdfs = [p.get_cdf() for p in predictions]
        xs = [pt.value for pt in cdfs[0]]
        if any(len(c) != len(xs) or any(a.value != b for a, b in zip(c, xs)) for c in cdfs):
            raise ValueError("the runs' CDFs are not on the same grid")
        prior = self._market.get(key) if isinstance(question, NumericQuestion) else None
        prior_cdf = MP.cdf_on_grid(prior.values, xs) if prior else None
        final = aggregate_cdf([[pt.percentile for pt in c] for c in cdfs], weights, question.open_lower_bound,
                              question.open_upper_bound, self.numeric_flat_weight, self.numeric_tail, prior_cdf,
                              prior.weight if prior else 0.0)
        dist = NumericDistribution.from_question(
            [Percentile(value=v, percentile=float(q)) for v, q in zip(xs, final)], question)
        out = dist.get_cdf()                                       # validates exactly as publishing will
        at = lambda c, q: float(np.interp(q, [pt.percentile for pt in c], [pt.value for pt in c]))
        self.run_log.append(dict(
            id=question.id_of_question, url=question.page_url, type=question.__class__.__name__,
            title=question.question_text,
            members=[dict(model=m, weight=w, p05=at(c, .05), p50=at(c, .5), p95=at(c, .95))
                     for m, w, c in zip(models, weights, cdfs)],
            flat_weight=self.numeric_flat_weight, market_source=prior.source if prior else None,
            market_weight=prior.weight if prior else None,
            market_p50=float(np.median(prior.values)) if prior else None,
            final=dict(p05=at(out, .05), p50=at(out, .5), p95=at(out, .95))))
        return dist

    async def _aggregate_predictions(self, predictions, question):
        try:
            if isinstance(question, BinaryQuestion) and predictions and all(isinstance(p, float) for p in predictions):
                return self._aggregate_binary(list(predictions), question)
            if (isinstance(question, (NumericQuestion, DateQuestion)) and predictions
                    and all(isinstance(p, NumericDistribution) for p in predictions)):
                return self._aggregate_numeric(list(predictions), question)
        except Exception as e:                                   # noqa: BLE001 — never lose a forecast to post-processing
            logger.warning(f"custom aggregation failed for {question.page_url} ({e!r}); using the template's")
        return await super()._aggregate_predictions(predictions, question)

    def binary_summary(self) -> tuple[int, float | None]:
        """(number of yes/no forecasts made, their mean) — compare the mean with config.TARGET_YES_RATE."""
        ps = [r["final"] for r in self.run_log if r["type"] == "binary"]
        return len(ps), (sum(ps) / len(ps) if ps else None)
