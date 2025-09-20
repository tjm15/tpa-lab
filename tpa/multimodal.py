from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Iterable, List

from .llm import OllamaVision, GoogleGeminiClient
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
    # Allow environment override for which backend to use for vision summarisation.
    provider = os.getenv("TPA_LLM_PROVIDER", "ollama")

    async def _run() -> List[dict]:
        use_google = provider.lower() == "google"
        if use_google:
            try:
                # Allow alternative vision model override (defaults to same core model).
                model_override = os.getenv("TPA_GEMINI_VISION_MODEL") or os.getenv("TPA_VISION_MODEL") or "gemini-2.5-pro"
                vision_client: object = GoogleGeminiClient(model=model_override)
            except Exception as exc:  # pragma: no cover
                log("vision", warning="google_init_failed", detail=str(exc))
                vision_client = OllamaVision(model=model_name)
                use_google = False
        else:
            vision_client = OllamaVision(model=model_name)
        summaries: List[dict] = []
        for item in visual_chunks:
            asset_path = Path(item.chunk.metadata.get("asset", ""))
            if not asset_path.exists():
                continue
            try:
                if use_google and isinstance(vision_client, GoogleGeminiClient):  # type: ignore[arg-type]
                    summary = await vision_client.analyse_image(_VIS_PROMPT, asset_path)  # type: ignore[attr-defined]
                else:
                    summary = await vision_client.analyse(_VIS_PROMPT, asset_path)  # type: ignore[attr-defined]
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
