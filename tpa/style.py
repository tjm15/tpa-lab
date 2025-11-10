from __future__ import annotations

from typing import Iterable, List

from .retrieve import Retrieved


def policy_style_seed(policies: Iterable[Retrieved], orientation: str = "pro_delivery") -> str:
    texts: List[str] = []
    for item in policies:
        text = (item.chunk.text or "").strip()
        if text:
            texts.append(text)
        if len(texts) >= 5:
            break
    tone = {
        "pro_delivery": (
            "Adopt a pro‑delivery stance where public benefits and housing delivery carry meaningful weight, "
            "but always hedge and respect statutory duties. Use phrases like 'on balance', 'would be acceptable "
            "subject to conditions', 'in accordance with policy', and 'material considerations'."
        ),
        "balanced": (
            "Maintain a neutral officer tone, carefully weighing harms and benefits and respecting statutory duties."
        ),
        "neutral": "Strictly neutral voice; avoid signalling a preference; cite policy tests precisely.",
    }.get(orientation, "pro‑delivery officer stance with caution.")

    seed = (
        "Calibrate style to echo local policy language. Reference hierarchy terms (Local Plan, London Plan, NPPF), "
        "and typical policy verbs (should, must, will not normally be permitted). Keep paragraphs concise and hedged.\n\n"
        f"Orientation: {tone}\n\n"
        "Sample policy fragments (do not quote verbatim, but echo style):\n- " + "\n- ".join(texts)
    )
    return seed


__all__ = ["policy_style_seed"]

