from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence

from .llm import BaseLLM


@dataclass
class Judgement:
    heading: str
    paragraph: str
    policy_compliant: bool
    style_score: float
    rationale: str


def _fallback_judge(heading: str, paragraph: str, orientation: str) -> Judgement:
    text = paragraph.lower()
    has_pol = "[pol:" in text
    hedged = any(w in text for w in ["on balance", "would", "subject to", "likely"])
    pro_delivery = any(w in text for w in ["benefit", "deliver", "public benefits"]) if orientation == "pro_delivery" else True
    score = (1.0 if has_pol else 0.0) + (0.5 if hedged else 0.0) + (0.5 if pro_delivery else 0.0)
    return Judgement(heading, paragraph, has_pol, min(score, 2.0), "rule-based")


async def _llm_judge_once(llm: BaseLLM, heading: str, paragraph: str, orientation: str) -> Judgement:
    prompt = (
        "You are a UK planning officer acting as a judge. Assess the paragraph for: (1) policy compliance signal, "
        "(2) officer style and hedging, (3) orientation consistency (pro-delivery if requested). Return JSON with keys: "
        "policy_compliant (bool), style_score (0..2), rationale (short)."
    )
    orientation_hint = f"Orientation: {orientation}."
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"{orientation_hint}\nHeading: {heading}\nParagraph: {paragraph}"},
    ]
    try:
        text = await llm.complete(messages)
    except Exception:
        return _fallback_judge(heading, paragraph, orientation)
    # Minimal robust parse: fall back if not JSON-like
    import json
    try:
        data = json.loads(text)
        return Judgement(
            heading,
            paragraph,
            bool(data.get("policy_compliant", False)),
            float(data.get("style_score", 1.0)),
            str(data.get("rationale", "")),
        )
    except Exception:
        return _fallback_judge(heading, paragraph, orientation)


def judge_paragraphs(
    llm: BaseLLM | None,
    heading: str,
    paragraphs: Sequence[str],
    orientation: str = "pro_delivery",
) -> List[Judgement]:
    if not paragraphs:
        return []
    if llm is None:
        return [_fallback_judge(heading, p, orientation) for p in paragraphs]
    try:
        return asyncio.run(_judge_all(llm, heading, paragraphs, orientation))
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(_judge_all(llm, heading, paragraphs, orientation))
        finally:
            loop.close()


async def _judge_all(llm: BaseLLM, heading: str, paragraphs: Sequence[str], orientation: str) -> List[Judgement]:
    results: List[Judgement] = []
    for p in paragraphs:
        results.append(await _llm_judge_once(llm, heading, p, orientation))
    return results


def select_by_judge(
    judgements: Sequence[Judgement],
    combine_mode: str = "judge_random",
    seed: int | None = None,
) -> str:
    valid = [j for j in judgements if j.policy_compliant]
    pool = valid or list(judgements)
    if not pool:
        return ""
    if seed is not None:
        random.seed(seed)
    if combine_mode == "best_of":
        return max(pool, key=lambda j: j.style_score).paragraph
    # judge_random: prefer better but keep diversity
    pool_sorted = sorted(pool, key=lambda j: j.style_score, reverse=True)
    top_k = max(1, min(3, len(pool_sorted)))
    choice = random.choice(pool_sorted[:top_k])
    return choice.paragraph


__all__ = ["Judgement", "judge_paragraphs", "select_by_judge"]

