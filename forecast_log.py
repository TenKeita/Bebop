"""Structured record of every forecast: one JSON line per question with each model's own forecast, the weights, the
calibration applied and the market-data prior. The GitHub workflow uploads the folder as a run artifact (kept 90
days); `ops/bebop_collect_logs.py` on the operator's server downloads them. This is the data needed later to fit the
shift and the model weights (Metaculus hides resolutions from bot tokens, so nothing else records it).
The same forecasts are already printed in the public run log, so the artifact reveals nothing new."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


def write(records: list[dict], folder: str, mode: str, tier: str, published: bool) -> Path | None:
    """Append-safe: one new file per run. Never raises (a logging problem must not fail a forecasting run)."""
    if not records:
        return None
    try:
        now = datetime.now(timezone.utc)
        out = Path(folder); out.mkdir(parents=True, exist_ok=True)
        path = out / f"{now:%Y%m%dT%H%M%SZ}_{mode}.jsonl"
        with path.open("a", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(dict(r, ts=now.isoformat(), mode=mode, tier=tier, published=published), default=str) + "\n")
        logger.info(f"forecast log: {len(records)} record(s) -> {path}")
        return path
    except Exception as e:                                       # noqa: BLE001
        logger.warning(f"forecast log not written: {e!r}")
        return None
