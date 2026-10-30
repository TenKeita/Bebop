"""Plain-Python check (no pytest plugins): the bot answers 2+ questions in each of two consecutive asyncio.run calls,
exactly like main.py does (main tournament, then MiniBench). Prints OK or the failure."""
import asyncio
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from forecasting_tools import GeneralLlm  # noqa: E402

import template_bot  # noqa: E402
from fakes import bq, fake_structure_output, make_bot, make_fake_invoke  # noqa: E402

fake, _ = make_fake_invoke()
GeneralLlm.invoke = fake
template_bot.structure_output = fake_structure_output
bot = make_bot()
for run in range(2):
    reports = asyncio.run(bot.forecast_questions([bq(100 + 2 * run), bq(101 + 2 * run)], return_exceptions=True))
    bad = [r for r in reports if isinstance(r, BaseException)]
    if bad:
        print("FAIL", run, bad)
        sys.exit(1)
print("OK")
