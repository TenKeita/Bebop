# CommitteeBot — a Metaculus FutureEval forecasting bot

An automated forecasting bot for Metaculus's FutureEval bot tournaments (cash prizes for the most accurate bots, ~$175k
a year; bots only, no human in the loop). Built 2026-10-30 on Metaculus's official
[template](https://github.com/Metaculus/metac-bot-template) and its
[forecasting-tools](https://github.com/Metaculus/forecasting-tools) library.

## What it does differently from the template
| | Template | CommitteeBot |
|---|---|---|
| Forecast runs | 5 runs of one model | **a weighted committee of 5 runs across model families**, anchored on the FutureEval-leaderboard leader (Claude Opus, weight 3) with Gemini Flash, GPT and Grok for diversity; reasoning effort set per model |
| Research | one news summary | **outside view** (reference classes, base rates, similar past questions) + **inside view** (latest news) + **one agentic follow-up round** (the 1-2 facts that would most change the forecast are searched); never abstains |
| Binary prompt | status quo emphasis | **base rate first, then the update**, with the exact resolution condition spelled out |
| Binary aggregation | median | **weighted median** (best model weighs more), capped to [2%, 98%]; no extremising |
| Numeric / date | 6 percentiles | **11 percentiles (5th-95th)** so the tails are stated, not extrapolated |
| Robustness | question fails if <50% of runs return | fails only if <40% return; missing models are replaced from cheaper tiers |
| Cost control | — | **credit guard:** reads the OpenRouter key's remaining credit; drops to a cheaper tier, or stops, below set floors |

Multiple-choice and conditional questions keep the template's tested prompts and aggregation. Design choices follow the
2026 bot analyses and the FutureEval model leaderboard (sources in `config.py`).

## Files
| Path | Contents |
|---|---|
| `main.py` | entry point (`--mode tournament / test_questions / metaculus_cup`, `--dry-run`) |
| `committee_bot.py` | the bot (subclass of the template class) |
| `template_bot.py` | the official template class, verbatim (replace it when Metaculus updates it each season) |
| `config.py` | model tiers, research models, floors (env vars override) |
| `aggregation.py`, `budget.py` | pure helpers |
| `tests/` | offline tests: the whole pipeline with every LLM call mocked (19 tests) |
| `.github/workflows/` | scheduled forecasting (every 20 min), manual smoke test, offline tests on every push |
| `RUNBOOK.md` | what the operator does: accounts, keys, GitHub, pausing, budget |

## Local development
```
uv sync --group dev
uv run python -m pytest -q tests                                 # offline, no keys needed
uv run python main.py --mode test_questions --dry-run            # needs .env with METACULUS_TOKEN + OPENROUTER_API_KEY
```
