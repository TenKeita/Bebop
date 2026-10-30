"""Bot configuration: committees, research, aggregation and budget settings (env vars override).

Evidence used (2026-10-30):
  * Metaculus FutureEval model leaderboard (data 2026-09-04, same scaffold for every model): Claude Opus 5 High 15.41,
    Claude Fable 5 High 13.77, Claude Opus 4.8 High 12.97, Gemini 3.6 Flash 12.70, GPT-5.5 Instant 12.66, GPT-5.6 Sol
    High 12.45, o3 11.73, Claude Sonnet 5 11.18, Grok 4.5 High 10.24 (peer score vs GPT-4o = 0).
    -> anchor every committee on Claude Opus (5.5 = newest), add the strong-and-cheap Gemini Flash line, plus OpenAI
       and Grok for decorrelation.
  * "AI Forecasting in 2026: What 11 Analyses Say": frontier models dominate; 3-7 diverse models; median preferred over
    extremising; capping helps (r = +0.48); explicit base rates (40% of top-15 vs 7%); agentic iterative search beats
    one-shot; template numeric handling flagged as weak; winners spent ~$1.40/question vs $0.50 for the bottom half.
Model IDs are OpenRouter IDs (checked 2026-10-30; the bot re-checks at start-up and replaces missing models with
cheaper-tier ones). Approximate cost per question incl. research and parsing, at ~12k input tokens per run:
  premium ~$1.00   standard ~$0.55   balanced ~$0.35   minimal ~$0.12
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
    "premium": [M("anthropic/claude-opus-5.5", 2, "high"), M("anthropic/claude-opus-5.5", 2, "high"),
                M("openai/gpt-6.1-sol", 1, "high"), M("google/gemini-4-argon", 1), M("x-ai/grok-4.7", 1)],
    "standard": [M("anthropic/claude-opus-5.5", 3, "medium"), M("openai/gpt-6.1-sol", 1, "medium"),
                 M("google/gemini-4-argon", 1), M("google/gemini-3.8-flash", 1), M("x-ai/grok-4.7", 1)],
    "balanced": [M("anthropic/claude-opus-5.5", 3, "medium"), M("google/gemini-3.8-flash", 1),
                 M("google/gemini-3.8-flash", 1), M("openai/gpt-6.1-terra", 1), M("x-ai/grok-4.7", 1)],
    "minimal": [M("google/gemini-3.8-flash", 1), M("google/gemini-3.8-flash", 1), M("google/gemini-3.8-flash", 1),
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
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


TIER_MAIN = env_str("BOT_TIER_MAIN", "balanced")           # main seasonal tournament
TIER_MINIBENCH = env_str("BOT_TIER_MINIBENCH", "balanced")  # MiniBench (decides credit top-ups) + test questions
AGGREGATOR = env_str("BOT_AGGREGATOR", "weighted_median")   # or "logodds_mean" (trimmed, unweighted)
EXTREMIZE = env_float("BOT_EXTREMIZE", 1.0)                 # 1.0 = off (evidence: median beats extremising)
CLIP_LOW, CLIP_HIGH = env_float("BOT_CLIP_LOW", 0.02), env_float("BOT_CLIP_HIGH", 0.98)
RESEARCH_FOLLOWUP = env_str("BOT_RESEARCH_FOLLOWUP", "true").lower() == "true"
SOFT_FLOOR_USD = env_float("BOT_SOFT_FLOOR_USD", 15.0)      # below: drop to the cheapest tier
HARD_FLOOR_USD = env_float("BOT_HARD_FLOOR_USD", 3.0)       # below: forecast nothing
LLM_TIMEOUT_S = env_float("BOT_LLM_TIMEOUT_S", 240.0)
