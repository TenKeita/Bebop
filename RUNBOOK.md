# RUNBOOK — what the operator does (≈ 30–45 minutes once, then ~1 h/month)

The bot is built and tested offline. These steps need the operator's own accounts and secrets; nothing below has been
done yet.

## 1. Metaculus (≈ 10 min)
1. Log in to (or create) your Metaculus account.
2. Go to https://www.metaculus.com/futureeval/participate/ → create a **bot account** → generate its **token**. Keep it
   private. It goes into GitHub as a secret (step 3), never into chat or files.
3. Fill in the participation form https://forms.gle/aQdYMq9Pisrf1v7d8 (3 required questions — mandatory before the bot
   forecasts). Answer the **LLM credits** section to request the sponsored OpenRouter key (~$100, given selectively in
   Fall 2026) and AskNews access if offered.
4. Rules to respect:
   - one prize-eligible bot per person;
   - **no human intervention on live questions** (no manual edits or re-submissions);
   - the bot writes the required comment itself.

## 2. GitHub repository (≈ 10 min)
1. Create a GitHub account if you have none.
2. Create a **new, empty, public** repository (e.g. `metaculus-committee-bot`). Public is better: GitHub Actions minutes
   are unlimited for public repositories; a private one has 2,000 free minutes/month, which the 20-minute schedule can
   exceed. Secrets stay private either way.
3. Push this folder:
   ```
   git push -u https://github.com/<you>/<repo>.git main
   ```
   Git's credential manager opens a browser sign-in once.

## 3. Secrets and settings in the repository (≈ 5 min)
`Settings → Secrets and variables → Actions`:
- **Secrets:**
  - `METACULUS_TOKEN` (bot token);
  - `OPENROUTER_API_KEY` (the sponsored key, or your own);
  - optional AskNews access (Metaculus participant access; next application window 2027-01-01): `ASKNEWS_API_KEY`,
    or the older `ASKNEWS_CLIENT_ID` + `ASKNEWS_SECRET` pair. AskNews's free plan (25 credits/month ≈ 4 questions)
    is too small to be worth adding.
- **Variables (optional spending controls):**
  - `BOT_TIER_MAIN` and `BOT_TIER_MINIBENCH` (`premium` / `standard` / `balanced` / `minimal`; default `balanced`);
  - `BOT_SOFT_FLOOR_USD` (default 15) and `BOT_HARD_FLOOR_USD` (default 3).

Then open the **Actions** tab and enable workflows.

## 4. First runs (≈ 10 min)
1. `Actions → Test Bot (bot-testing-area) → Run workflow` with **dry_run = true**. Check that the log shows forecasts and
   no errors (costs a few cents of credit).
2. Run it again with **dry_run = false**. To see the forecasts, open Metaculus → Settings → My Forecasting Bots →
   switch to the bot account, then open the bot-testing-area questions.
3. **Go live:** add the repository variable `BOT_LIVE` = `true` (Settings → Secrets and variables → Actions →
   Variables). From then on `Forecast on new AI tournament questions` runs every 20 minutes (main tournament +
   MiniBench), skipping questions already forecast. Set it to `false` (or delete it) to stop forecasting without
   touching anything else.

## Good to know
- **Until both secrets exist**, the scheduled run skips itself and shows a warning: "Bot not configured yet". It is
  green, not red, so it does not send failure e-mails every 20 minutes.
- **Keep-alive.** GitHub switches off scheduled workflows in public repositories after 60 days without commits. The
  `Keep scheduled workflows alive` workflow makes an empty commit on the 1st of each month to prevent that.

## Pausing / stopping
`Actions → Forecast on new AI tournament questions → … → Disable workflow`. Re-enable the same way.

## Money
**Default: sponsored credits only.**
- The credit guard reads the key's remaining credit before every run:
  - below `BOT_SOFT_FLOOR_USD` → cheapest tier;
  - below `BOT_HARD_FLOOR_USD` → no forecasting.
- It never spends beyond the key.

**Approximate cost per question:**

| Tier | Cost |
|---|---|
| premium | ~$1.00 |
| standard | ~$0.55 |
| balanced (default) | ~$0.35 |
| minimal | ~$0.12 |

A season has ~500 main-tournament questions plus ~500 MiniBench ones.
- **~$100 of credit covers ~280 questions on `balanced`**, about half a season; below the soft floor the bot
  continues on `minimal`. Metaculus raises credits only after above-average MiniBench results.
- A full season on `balanced` costs ~$350; on `standard` ~$550. Anything beyond the sponsored credit needs your own
  OpenRouter credit — your decision.
- **Prizes:** paid by Metaculus to the bot's owner after each season. Declare them as income where you are taxed, and
  check that competition prizes are compatible with any employment or fellowship contract.

## Results
Monthly:
- the tournament leaderboard (main tournament + MiniBench);
- the bot's comments on resolved questions;
- the credit left.

The agent (Claude Code) reviews each season's resolved questions and proposes changes (e.g. switching extremisation on
if the committee proves under-confident). Changes go in between seasons or on MiniBench first.
