"""Credit guard: never spend beyond the sponsored OpenRouter credits unless the operator raises the floors.

Reads the remaining credit of the OpenRouter key before a run:
  * below HARD floor -> forecast nothing this run (log + exit cleanly),
  * below SOFT floor -> drop to the cheapest tier,
  * unknown (no limit info / API error) -> keep the requested tier (sponsored keys normally have a limit).
"""
from __future__ import annotations

import logging

import requests

logger = logging.getLogger(__name__)
KEY_URL = "https://openrouter.ai/api/v1/key"
CREDITS_URL = "https://openrouter.ai/api/v1/credits"


class CreditGuard:
    def __init__(self, api_key: str | None, soft_floor_usd: float, hard_floor_usd: float, timeout: float = 20):
        self.api_key = api_key
        self.soft_floor = float(soft_floor_usd)
        self.hard_floor = float(hard_floor_usd)
        self.timeout = timeout

    def remaining_usd(self) -> float | None:
        """Remaining credit in USD: the LOWER of the key's own limit and the account's balance, or None if neither can
        be read. (2026-11-13: a key capped at $100 on an account holding ~$10 reported $100 — the cap is not money.)"""
        if not self.api_key:
            return None
        headers = {"Authorization": f"Bearer {self.api_key}"}
        vals: list[float] = []
        try:
            r = requests.get(KEY_URL, headers=headers, timeout=self.timeout)
            r.raise_for_status()
            data = (r.json() or {}).get("data") or {}
            if data.get("limit_remaining") is not None:
                vals.append(float(data["limit_remaining"]))
        except Exception as e:                                   # noqa: BLE001 — a guard must never crash the bot
            logger.warning(f"key limit check failed: {e!r}")
        try:
            r = requests.get(CREDITS_URL, headers=headers, timeout=self.timeout)
            r.raise_for_status()
            c = (r.json() or {}).get("data") or {}
            if c.get("total_credits") is not None and c.get("total_usage") is not None:
                vals.append(float(c["total_credits"]) - float(c["total_usage"]))
        except Exception as e:                                   # noqa: BLE001
            logger.warning(f"account balance check failed: {e!r}")
        return min(vals) if vals else None

    def decide(self, requested_tier: str, cheapest_tier: str) -> tuple[str | None, str]:
        """(tier to use or None = do not forecast, human-readable reason)."""
        rem = self.remaining_usd()
        if rem is None:
            return requested_tier, "remaining credit unknown -> requested tier"
        if rem < self.hard_floor:
            return None, f"remaining credit ${rem:.2f} < hard floor ${self.hard_floor:.2f} -> no forecasting this run"
        if rem < self.soft_floor:
            return cheapest_tier, f"remaining credit ${rem:.2f} < soft floor ${self.soft_floor:.2f} -> '{cheapest_tier}'"
        return requested_tier, f"remaining credit ${rem:.2f} -> '{requested_tier}'"
