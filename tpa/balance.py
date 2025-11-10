from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from .material_considerations import MaterialConsideration
from .llm import BaseLLM


def _bucket(topic: MaterialConsideration) -> str:
    if topic.recommendation == "unacceptable":
        return "harms"
    if topic.recommendation == "acceptable":
        return "benefits"
    return "neutral"


def _policy_cites(mc: MaterialConsideration) -> str:
    cites = []
    for finding in mc.findings:
        cites.extend(finding.supporting)
        cites.extend(finding.contradicting)
    return ", ".join(sorted(set(cites))) or "MISSING_CITES"


def _default_balance_narrative(summary: Dict[str, List[dict]]) -> str:
    harms = summary["harms"]
    benefits = summary["benefits"]
    text: List[str] = []
    if harms:
        harm_topics = ", ".join(item["topic"].title() for item in harms)
        text.append(
            f"Material harms remain in respect of {harm_topics}, with officers giving significant weight to statutory duties."
        )
    if benefits:
        benefit_topics = ", ".join(item["topic"].title() for item in benefits)
        text.append(
            f"Conversely, tangible public benefits arise from {benefit_topics}, which attract considerable positive weight."
        )
    if not text:
        text.append("Evidence to date is finely balanced, and no clear planning harm or benefit dominates.")
    text.append(
        "On balance, the recommendation turns on whether mitigation and enforceable conditions can realistically "
        "secure compliance with the highlighted policy tests."
    )
    return " ".join(text)


def _llm_balance_narrative(llm: BaseLLM | None, summary: Dict[str, List[dict]]) -> str | None:
    if llm is None:
        return None
    harms = summary["harms"]
    benefits = summary["benefits"]
    neutrals = summary["neutral"]
    payload = {
        "harms": harms,
        "benefits": benefits,
        "neutral": neutrals,
    }
    messages = [
        {
            "role": "system",
            "content": (
                "You are the case officer drafting the planning balance for a UK development management report. "
                "Write 2-3 paragraphs, hedged and evidence-led, weighing the listed harms/benefits/neutral points. "
                "Avoid numeric scoring; cite topics and policy duties qualitatively."
            ),
        },
        {
            "role": "user",
            "content": f"Material considerations summary:\n{payload}",
        },
    ]
    try:
        import asyncio

        return asyncio.run(llm.complete(messages))
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(llm.complete(messages))  # type: ignore[arg-type]
        finally:
            loop.close()
    except Exception:
        return None


def build_planning_balance(
    material_considerations: Sequence[MaterialConsideration],
    llm: BaseLLM | None = None,
) -> Tuple[str, Dict[str, List[dict]], str]:
    lines = ["# Planning Balance"]
    summary = {"harms": [], "benefits": [], "neutral": []}
    for mc in material_considerations:
        lines.append(f"## {mc.topic.title()}")
        lines.append(f"- Officer judgement: {mc.weight} weight, recommendation {mc.recommendation}")
        lines.append(f"- Key citations: {_policy_cites(mc)}")
        for finding in mc.findings:
            lines.append(f"  - {finding.conclusion}")
        bucket = _bucket(mc)
        summary[bucket].append(
            {
                "topic": mc.topic,
                "weight": mc.weight,
                "recommendation": mc.recommendation,
                "findings": [finding.conclusion for finding in mc.findings],
                "citations": _policy_cites(mc),
            }
        )
    narrative = _llm_balance_narrative(llm, summary) or _default_balance_narrative(summary)
    lines.append("\n## Officer Narrative\n")
    lines.append(narrative)
    return "\n".join(lines), summary, narrative


def detect_conflicts(material_considerations: Sequence[MaterialConsideration]) -> List[dict]:
    harms = [mc for mc in material_considerations if mc.recommendation == "unacceptable"]
    benefits = [mc for mc in material_considerations if mc.recommendation == "acceptable"]
    conflicts: List[dict] = []
    for harm in harms:
        for benefit in benefits:
            conflicts.append(
                {
                    "harm_topic": harm.topic,
                    "benefit_topic": benefit.topic,
                    "harm_weight": harm.weight,
                    "benefit_weight": benefit.weight,
                    "note": "Requires explicit Member judgement balancing conflicting considerations.",
                }
            )
    return conflicts


__all__ = ["build_planning_balance", "detect_conflicts"]
