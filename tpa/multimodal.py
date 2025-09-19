from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Iterable, List

from .llm import OllamaVision
from .logging import log
from .retrieve import Retrieved


_VIS_PROMPT = (
    "You are a UK planning officer reviewing a planning figure. Describe the key transport, "
    "design, massing, or environmental signals relevant to policy compliance. Be concise and "
    "cite potential issues or strengths."
)


def summarise_visuals(visual_chunks: Iterable[Retrieved], model_name: str = "gemma3:27b") -> List[dict]:
    if os.getenv("TPA_DISABLE_VISION") == "1":
        return []
    async def _run() -> List[dict]:
        client = OllamaVision(model=model_name)
        summaries: List[dict] = []
        for item in visual_chunks:
            asset_path = Path(item.chunk.metadata.get("asset", ""))
            if not asset_path.exists():
                continue
            try:
                summary = await client.analyse(_VIS_PROMPT, asset_path)
            except Exception as exc:  # pragma: no cover
                log(
                    "vision",
                    warning="analysis_failed",
                    chunk=item.chunk.id,
                    detail=str(exc),
                )
                summary = "MISSING VISION SUMMARY"
            summaries.append(
                {
                    "id": item.chunk.id,
                    "path": str(asset_path),
                    "summary": summary.strip() if summary else "MISSING VISION SUMMARY",
                }
            )
        return summaries

    try:
        return asyncio.run(_run())
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(_run())
        finally:
            loop.close()


__all__ = ["summarise_visuals"]
