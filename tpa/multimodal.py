from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Iterable, List

from .llm import OllamaVision, GoogleGeminiClient
from .logging import log
from .retrieve import Retrieved


_VIS_PROMPT = (
    "You are a UK planning officer reviewing a {visual_type} extracted from the evidence base. "
    "Describe the key transport, design, massing, amenity, or environmental signals relevant to "
    "policy compliance. Highlight potential harms/benefits and reference any annotations or data."
)


def summarise_visuals(visual_chunks: Iterable[Retrieved], model_name: str | None = None) -> List[dict]:
    if os.getenv("TPA_DISABLE_VISION") == "1":
        return []
    # Allow environment override for which backend to use for vision summarisation.
    vision_model = model_name or os.getenv("TPA_VISION_MODEL") or "qwen3-vl:30b"
    provider = os.getenv("TPA_VISION_PROVIDER") or os.getenv("TPA_LLM_PROVIDER", "ollama")

    async def _run() -> List[dict]:
        use_google = provider.lower() == "google"
        if use_google:
            try:
                # Allow alternative vision model override (defaults to same core model).
                model_override = os.getenv("TPA_GEMINI_VISION_MODEL") or os.getenv("TPA_VISION_MODEL") or "gemini-2.5-pro"
                vision_client: object = GoogleGeminiClient(model=model_override)
            except Exception as exc:  # pragma: no cover
                log("vision", warning="google_init_failed", detail=str(exc))
                vision_client = OllamaVision(model=vision_model)
                use_google = False
        else:
            vision_client = OllamaVision(model=vision_model)
        summaries: List[dict] = []
        for item in visual_chunks:
            asset_path = Path(item.chunk.metadata.get("asset", ""))
            if not asset_path.exists():
                continue
            visual_kind = item.chunk.metadata.get("visual_type", "figure")
            prompt = _VIS_PROMPT.format(visual_type=visual_kind.replace("_", " "))
            try:
                if use_google and isinstance(vision_client, GoogleGeminiClient):  # type: ignore[arg-type]
                    summary = await vision_client.analyse_image(prompt, asset_path)  # type: ignore[attr-defined]
                else:
                    summary = await vision_client.analyse(prompt, asset_path)  # type: ignore[attr-defined]
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
                    "visual_type": visual_kind,
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
