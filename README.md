# CommitteeBot — a Metaculus FutureEval forecasting bot

An automated forecasting bot for Metaculus's FutureEval bot tournaments (cash prizes for the most accurate bots, ~$175k
a year; bots only, no human in the loop). Built 2026-10-30 on Metaculus's official
[template](https://github.com/Metaculus/metac-bot-template) and its
[forecasting-tools](https://github.com/Metaculus/forecasting-tools) library.

## What it does differently from the template
| | Template | CommitteeBot |
|---|---|---|
| Forecast runs | 5 runs of one model | **a committee of 5 runs across model families** (OpenAI, Anthropic, Google), one model per run |
| Research | one news summary | **two reports:** outside view (reference classes, base rates) + inside view (latest news); never abstains |
| Binary prompt | status quo emphasis | **base rate first, then the update**, with the exact resolution condition spelled out |
| Binary aggregation | median | **trimmed mean of log-odds**, clipped to [2%, 98%] (optional extremisation, off by default) |
| Robustness | question fails if <50% of runs return | fails only if <40% return; missing models are replaced from cheaper tiers |
| Cost control | — | **credit guard:** reads the OpenRouter key's remaining credit; drops to a cheaper tier, or stops, below set floors |

Multiple-choice, numeric, date and conditional questions keep the template's tested prompts and aggregation.

## Files
| Path | Contents |
|---|---|
| `main.py` | entry point (`--mode tournament / test_questions / metaculus_cup`, `--dry-run`) |
| `committee_bot.py` | the bot (subclass of the template class) |
| `template_bot.py` | the official template class, verbatim (replace it when Metaculus updates it each season) |
| `config.py` | model tiers, research models, floors (env vars override) |
| `aggregation.py`, `budget.py` | pure helpers |
| `tests/` | offline tests: the whole pipeline with every LLM call mocked (13 tests) |
| `.github/workflows/` | scheduled forecasting (every 20 min), manual smoke test, offline tests on every push |
| `RUNBOOK.md` | what the operator does: accounts, keys, GitHub, pausing, budget |

## Local development
```
uv sync --group dev
uv run python -m pytest -q tests                                 # offline, no keys needed
uv run python main.py --mode test_questions --dry-run            # needs .env with METACULUS_TOKEN + OPENROUTER_API_KEY
```
