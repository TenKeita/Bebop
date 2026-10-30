"""Bot configuration: model tiers, research models, aggregation and budget settings (env vars override).

Model IDs are OpenRouter IDs (checked against https://openrouter.ai/api/v1/models on 2026-10-30; the bot re-checks
at start-up and drops any model that has disappeared). Approximate cost per question at ~12k input / ~4k output tokens
per run, 5 runs + research + parsing:
  premium  ~$0.55   standard ~$0.40   economy ~$0.20   minimal ~$0.08
Metaculus sponsors ~$100 of OpenRouter credit per bot maker (selectively in Fall 2026), so 'economy' is the default
until the operator approves a budget. Main-tournament questions are worth far more than MiniBench ones ($50k pool /
~490 questions vs $1k / ~60), so a richer tier can be used for the main tournament only (TIER_MAIN vs TIER_MINIBENCH).
"""
from __future__ import annotations

import os

OR = "openrouter/"
TIERS: dict[str, list[str]] = {
    # 5 runs per question, rotated across model families (diversity + aggregation; model quality dominates).
    "premium": [OR + "anthropic/claude-opus-5.5", OR + "openai/gpt-6.1-sol", OR + "google/gemini-4-argon",
                OR + "anthropic/claude-opus-5.5", OR + "openai/gpt-6.1-sol"],
    "standard": [OR + "anthropic/claude-opus-5.5", OR + "openai/gpt-6.1-sol", OR + "google/gemini-4-argon",
                 OR + "anthropic/claude-sonnet-5.5", OR + "openai/gpt-6.1-terra"],
    "economy": [OR + "anthropic/claude-sonnet-5.5", OR + "openai/gpt-6.1-terra", OR + "google/gemini-3.8-flash",
                OR + "anthropic/claude-sonnet-5.5", OR + "openai/gpt-6.1-terra"],
    "minimal": [OR + "openai/gpt-6.1-terra", OR + "google/gemini-3.8-flash", OR + "deepseek/deepseek-v4.1-flash",
                OR + "openai/gpt-6.1-terra", OR + "google/gemini-3.8-flash"],
}
CHEAPEST_TIER = "minimal"
# Research: inside view (latest news) = AskNews if its sponsored credentials exist, else Perplexity Sonar;
# outside view (base rates / reference classes) = a search model that reasons.
NEWS_RESEARCHER = {"premium": OR + "perplexity/sonar-pro", "standard": OR + "perplexity/sonar",
                   "economy": OR + "perplexity/sonar", "minimal": OR + "perplexity/sonar"}
OUTSIDE_VIEW_RESEARCHER = {"premium": OR + "perplexity/sonar-reasoning-pro", "standard": OR + "perplexity/sonar-reasoning-pro",
                           "economy": OR + "perplexity/sonar", "minimal": OR + "perplexity/sonar"}
PARSER = OR + "openai/gpt-6.1-terra"          # turns free-text answers into structured forecasts (cheap, reliable)


def env_str(name: str, default: str) -> str:
    v = os.getenv(name)
    return v if v not in (None, "") else default


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


TIER_MAIN = env_str("BOT_TIER_MAIN", "economy")          # main seasonal tournament
TIER_MINIBENCH = env_str("BOT_TIER_MINIBENCH", "economy")  # MiniBench + test questions
EXTREMIZE = env_float("BOT_EXTREMIZE", 1.0)               # 1.0 = off until resolved data says otherwise
CLIP_LOW, CLIP_HIGH = env_float("BOT_CLIP_LOW", 0.02), env_float("BOT_CLIP_HIGH", 0.98)
SOFT_FLOOR_USD = env_float("BOT_SOFT_FLOOR_USD", 15.0)    # below: drop to the cheapest tier
HARD_FLOOR_USD = env_float("BOT_HARD_FLOOR_USD", 3.0)     # below: forecast nothing (never run the key dry mid-question)
LLM_TIMEOUT_S = env_float("BOT_LLM_TIMEOUT_S", 240.0)
