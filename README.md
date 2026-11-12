# CommitteeBot — a Metaculus FutureEval forecasting bot

An automated forecasting bot for Metaculus's FutureEval bot tournaments (cash prizes for the most accurate bots, ~$175k
a year; bots only, no human in the loop). Built 2026-10-30 on Metaculus's official
[template](https://github.com/Metaculus/metac-bot-template) and its
[forecasting-tools](https://github.com/Metaculus/forecasting-tools) library.

## What it does differently from the template
| | Template | CommitteeBot |
|---|---|---|
| Forecast runs | 5 runs of one model | **5 different strong models** (Claude Opus with weight 2, GPT-6.1 Sol, Gemini 4 Argon, Gemini 3.8 Flash, GPT-6.1 Terra on the default tier); reasoning effort set per model |
| Research | one news summary | **outside view** (base rates) + **inside view** (latest news) + **one agentic follow-up round**; never abstains |
| Price questions | — | **market-data baseline** (`market_prior.py`): the instrument's own recent daily moves resampled at current volatility, zero drift. Recognised question families are parsed by rules, others by a cheap model with strict validation |
| Yes/no aggregation | median | **mean of log-odds**, then a **calibration shift of −0.3** (past bots all over-predicted Yes), then a blend with the market baseline (80% weight for rule-parsed, 50% for model-parsed); capped to [2%, 98%] |
| Numbers / dates | 6 percentiles, pointwise median | **11 percentiles**; the runs' CDFs are **averaged**, blended with the market baseline, and mixed with **15% of a flat distribution** |
| Record keeping | — | **forecast log**: one JSON line per question with each model's forecast (uploaded as a 90-day run artifact) |
| Robustness | question fails if <50% of runs return | fails only if <40% return; missing models are replaced from cheaper tiers; any post-processing error falls back to the template's aggregation |
| Cost control | — | **credit guard:** reads the OpenRouter key's remaining credit; drops to a cheaper tier, or stops, below set floors |

**Evidence (2026-11-12, out of sample on Metaculus's public Q1-2025 tournament file, 45 bots, 424 questions).** Fitted
on the first half of the season, scored on the second against the real field:
- yes/no: mean of log-odds 610 points vs plain median 362 vs "best model weighted 3" 34; adding the −0.3 shift took
  the committee from 34 to 505;
- numbers: averaging the CDFs scored −4.61 (mean log probability of the true bucket) vs −4.73 for the median, and
  −4.59 with the flat mix;
- market baseline: on 11 past yes/no price questions it beat the average bot on 10 (+25 peer points per question);
  on 77 recurring number questions its 80% range held the truth 82% of the time.
Scripts: `LLM_Ventures/research/audit_2026_11_11/m5_design_space.py`, `m6_numeric_agg.py`,
`m7_market_prior_backtest.py`.

Multiple-choice and conditional questions keep the template's prompts and aggregation.

## Files
| Path | Contents |
|---|---|
| `main.py` | entry point (`--mode tournament / test_questions / metaculus_cup`, `--dry-run`) |
| `committee_bot.py` | the bot (subclass of the template class) |
| `template_bot.py` | the official template class, verbatim (replace it when Metaculus updates it each season) |
| `config.py` | model tiers, calibration settings, research models, floors (env vars override) |
| `aggregation.py` | aggregation and calibration (pure functions) |
| `market_prior.py`, `market_question.py` | market-data baseline and the question parsers |
| `forecast_log.py`, `budget.py` | forecast log, credit guard |
| `tests/` | offline tests: the whole pipeline with every LLM call and data source mocked (52 tests) |
| `.github/workflows/` | scheduled forecasting, manual smoke test, offline tests on every push |
| `RUNBOOK.md` | what the operator does: accounts, keys, GitHub, pausing, budget |

## Local development
```
uv sync --group dev
uv run python -m pytest -q tests                                 # offline, no keys needed
uv run python main.py --mode test_questions --dry-run            # needs .env with METACULUS_TOKEN + OPENROUTER_API_KEY
```
