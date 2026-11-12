"""Bot configuration: committees, research, aggregation, calibration and budget settings (env vars override).

Evidence used:
  * Metaculus FutureEval model leaderboard (data 2026-09-04, same scaffold for every model): Claude Opus 5 High 15.41,
    Claude Fable 5 High 13.77, Claude Opus 4.8 High 12.97, Gemini 3.6 Flash 12.70, GPT-5.5 Instant 12.66, GPT-5.6 Sol
    High 12.45, o3 11.73, Claude Sonnet 5 11.18, Grok 4.5 High 10.24 (peer score vs GPT-4o = 0).
  * "AI Forecasting in 2026: What 11 Analyses Say": frontier models dominate; 3-7 diverse models; capping helps;
    explicit base rates; agentic search beats one-shot; template numeric handling weak; winners ~$1.40/question.
  * Metaculus's public Q1-2025 tournament file, tested out of sample on 2026-11-12 (see aggregation.py):
      - companions matter: the same anchor scored 1,738 with four good companions and 1,125 with four weak ones,
        and the "best" model of one half-season was rank 35 in the next -> strong, DIFFERENT companions, and only a
        mild extra weight (2) for the anchor; Grok (weakest on the leaderboard) and the duplicate Gemini Flash are out;
      - every bot over-predicted Yes -> BINARY_SHIFT; number ranges were too narrow -> NUMERIC_FLAT_WEIGHT.
  * market_prior.py backtest (2026-11-12): on price questions the market-data baseline beat the average bot on 10 of
    11 yes/no questions (+25 peer points each) and was well calibrated on 77 number questions -> MARKET_PRIOR_*.
Model IDs and prices are OpenRouter's (checked 2026-11-12; the bot re-checks the IDs at start-up and replaces missing
models with cheaper-tier ones). Approximate cost per question incl. research and parsing, ~12k input tokens per run:
  premium ~$1.10   standard ~$0.60   balanced ~$0.40   minimal ~$0.12
Metaculus gives ~$100 to start and raises it only after above-average MiniBench results, so 'balanced' is the
default; the credit guard drops to 'minimal' below the soft floor and stops below the hard floor.
"""
from __future__ import annotations

import os

OR = "openrouter/"


def M(model: str, weight: float = 1.0, effort: str | None = None) -> dict:
    """A committee member: OpenRouter model, aggregation weight, reasoning effort (None = model default)."""
    return {"model": OR + model, "weight": float(weight), "reasoning_effort": effort}


TIERS: dict[str, list[dict]] = {
    "premium": [M("anthropic/claude-opus-5.5", 2, "high"), M("anthropic/claude-opus-5.5", 1, "high"),
                M("openai/gpt-6.1-sol", 1, "high"), M("google/gemini-4-argon", 1), M("google/gemini-3.8-flash", 1)],
    "standard": [M("anthropic/claude-opus-5.5", 2, "high"), M("openai/gpt-6.1-sol", 1, "medium"),
                 M("google/gemini-4-argon", 1), M("google/gemini-3.8-flash", 1), M("anthropic/claude-sonnet-5.5", 1)],
    "balanced": [M("anthropic/claude-opus-5.5", 2, "medium"), M("openai/gpt-6.1-sol", 1),
                 M("google/gemini-4-argon", 1), M("google/gemini-3.8-flash", 1), M("openai/gpt-6.1-terra", 1)],
    "minimal": [M("google/gemini-3.8-flash", 1), M("google/gemini-3.8-flash", 1), M("openai/gpt-6.1-terra", 1),
                M("openai/gpt-6.1-terra", 1), M("deepseek/deepseek-v4.1-flash", 1)],
}
CHEAPEST_TIER = "minimal"
# Research: inside view (latest news) = AskNews if its credentials exist (participant access reopens 2027-01-01),
# else Perplexity Sonar; outside view (base rates) = a search model that reasons; then one agentic follow-up round.
NEWS_RESEARCHER = {"premium": OR + "perplexity/sonar-pro", "standard": OR + "perplexity/sonar",
                   "balanced": OR + "perplexity/sonar", "minimal": OR + "perplexity/sonar"}
OUTSIDE_VIEW_RESEARCHER = {"premium": OR + "perplexity/sonar-reasoning-pro", "standard": OR + "perplexity/sonar-reasoning-pro",
                           "balanced": OR + "perplexity/sonar-reasoning-pro", "minimal": OR + "perplexity/sonar"}
FOLLOWUP_PLANNER = OR + "google/gemini-3.8-flash"   # names the 1-2 facts that would most change the forecast
FOLLOWUP_SEARCHER = OR + "perplexity/sonar"
PARSER = OR + "openai/gpt-6.1-terra"               # turns free-text answers into structured forecasts (cheap, reliable)


def env_str(name: str, default: str) -> str:
    v = os.getenv(name)
    return v if v not in (None, "") else default


def env_float(name: str, default: float) -> float:
    try:
        v = os.getenv(name)
        return float(v) if v not in (None, "") else default
    except (TypeError, ValueError):
        return default


def env_bool(name: str, default: bool) -> bool:
    return env_str(name, "true" if default else "false").strip().lower() in ("true", "1", "yes", "on")


TIER_MAIN = env_str("BOT_TIER_MAIN", "balanced")           # main seasonal tournament
TIER_MINIBENCH = env_str("BOT_TIER_MINIBENCH", "balanced")  # MiniBench (decides credit top-ups) + test questions

# --- yes/no questions
AGGREGATOR = env_str("BOT_AGGREGATOR", "logodds_mean")      # or "weighted_median"
BINARY_SHIFT = env_float("BOT_BINARY_SHIFT", -0.30)         # log-odds; < 0 lowers every Yes forecast (see aggregation.py)
BINARY_SLOPE = env_float("BOT_BINARY_SLOPE", 1.0)           # > 1 extremises; 1.0 = off
CLIP_LOW, CLIP_HIGH = env_float("BOT_CLIP_LOW", 0.02), env_float("BOT_CLIP_HIGH", 0.98)
TARGET_YES_RATE = 0.32                                      # 30% (Q1 2025) and 33% (Spring 2026) resolved Yes; the
                                                            # run summary prints our mean forecast next to it
# --- number and date questions
NUMERIC_FLAT_WEIGHT = env_float("BOT_NUMERIC_FLAT_WEIGHT", 0.15)
NUMERIC_TAIL = env_float("BOT_NUMERIC_TAIL", 0.05)          # mass the flat part leaves outside each OPEN bound

# --- market-data prior for price questions (market_prior.py, market_question.py)
MARKET_PRIOR = env_bool("BOT_MARKET_PRIOR", True)
MARKET_PRIOR_WEIGHT_RULE = env_float("BOT_MARKET_PRIOR_WEIGHT_RULE", 0.80)   # recognised recurring families
MARKET_PRIOR_WEIGHT_LLM = env_float("BOT_MARKET_PRIOR_WEIGHT_LLM", 0.50)     # parsed by the language model
MARKET_PRIOR_LLM = env_bool("BOT_MARKET_PRIOR_LLM", True)                    # allow the model-based parser

RESEARCH_FOLLOWUP = env_bool("BOT_RESEARCH_FOLLOWUP", True)
SOFT_FLOOR_USD = env_float("BOT_SOFT_FLOOR_USD", 15.0)      # below: drop to the cheapest tier
HARD_FLOOR_USD = env_float("BOT_HARD_FLOOR_USD", 3.0)       # below: forecast nothing
LLM_TIMEOUT_S = env_float("BOT_LLM_TIMEOUT_S", 240.0)
FORECAST_LOG_DIR = env_str("BOT_FORECAST_LOG_DIR", "forecast_log")   # one JSON line per question (uploaded as an artifact)
