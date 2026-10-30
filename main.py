"""Entry point of the LLM_Ventures Metaculus bot (CommitteeBot).

    python main.py --mode tournament        # main seasonal tournament + MiniBench (what the schedule runs)
    python main.py --mode test_questions    # the bot-testing-area (smoke test; safe to repeat)
    python main.py --mode metaculus_cup
    python main.py ... --dry-run            # compute forecasts but DO NOT submit them

Needs METACULUS_TOKEN (the bot account's token) and OPENROUTER_API_KEY (the sponsored key) in the environment
(GitHub Actions secrets, or a local .env that is never committed). AskNews credentials are optional.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os

import dotenv

dotenv.load_dotenv()

from bot_helpers import (check_environment, print_run_summary_banner, print_startup_banner,  # noqa: E402
                         silence_noisy_dependencies)

silence_noisy_dependencies()

from forecasting_tools import GeneralLlm, MetaculusClient  # noqa: E402

import config as C  # noqa: E402
from budget import CreditGuard  # noqa: E402
from committee_bot import CommitteeBot, available_openrouter_models, build_committee  # noqa: E402

logger = logging.getLogger(__name__)
URLS = {"tournament": "https://www.metaculus.com/tournament/fall-futureeval-2026/",
        "metaculus_cup": "https://www.metaculus.com/tournament/metaculus-cup-fall-2026/",
        "test_questions": "https://www.metaculus.com/tournament/bot-testing-area/"}


def make_bot(tier: str, publish: bool, available: set[str] | None) -> CommitteeBot:
    committee = build_committee(tier, available)
    parser = GeneralLlm(model=C.PARSER, timeout=C.LLM_TIMEOUT_S, allowed_tries=3)
    logger.info(f"tier '{tier}': committee {[m.model for m in committee]}")
    return CommitteeBot(
        committee=committee,
        news_researcher=C.NEWS_RESEARCHER[tier],
        outside_researcher=C.OUTSIDE_VIEW_RESEARCHER[tier],
        extremize=C.EXTREMIZE,
        clip=(C.CLIP_LOW, C.CLIP_HIGH),
        publish_reports_to_metaculus=publish,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=True,
        extra_metadata_in_explanation=True,
        llms={"default": committee[0], "summarizer": parser,
              "researcher": GeneralLlm(model=C.NEWS_RESEARCHER[tier]), "parser": parser},
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ap = argparse.ArgumentParser(description="LLM_Ventures CommitteeBot")
    ap.add_argument("--mode", choices=["tournament", "metaculus_cup", "test_questions"], default="tournament")
    ap.add_argument("--dry-run", action="store_true", help="compute forecasts but do not submit them")
    a = ap.parse_args()

    check_environment(strict=True)
    publish = not a.dry_run
    print_startup_banner(a.mode, will_publish=publish)
    client = MetaculusClient()
    plan = {"tournament": [(client.CURRENT_AI_COMPETITION_ID, C.TIER_MAIN), (client.CURRENT_MINIBENCH_ID, C.TIER_MINIBENCH)],
            "metaculus_cup": [(client.CURRENT_METACULUS_CUP_ID, C.TIER_MINIBENCH)],
            "test_questions": [("bot-testing-area", C.TIER_MINIBENCH)]}[a.mode]
    guard = CreditGuard(os.getenv("OPENROUTER_API_KEY"), C.SOFT_FLOOR_USD, C.HARD_FLOOR_USD)
    available = available_openrouter_models()
    reports = []
    for tournament_id, tier in plan:
        tier_used, why = guard.decide(tier, C.CHEAPEST_TIER)
        logger.info(f"{tournament_id}: {why}")
        if tier_used is None:
            continue
        bot = make_bot(tier_used, publish, available)
        if a.mode != "tournament":
            bot.skip_previously_forecasted_questions = False
        reports += asyncio.run(bot.forecast_on_tournament(tournament_id, return_exceptions=True))
    if reports:
        CommitteeBot.log_report_summary(reports)
    print_run_summary_banner(reports, will_publish=publish, tournament_url=URLS.get(a.mode))


if __name__ == "__main__":
    main()
