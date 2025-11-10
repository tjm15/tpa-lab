from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import yaml

from .chunk_store import Chunk
from .claims import ClaimRecord, extract_claims, match_claims_to_policies
from .llm import BaseLLM
from .logging import log
from .retrieve import Retrieved
from .style import policy_style_seed
from .judge import judge_paragraphs, select_by_judge
from .quant import extract_quant_facts, facts_markdown
from .config import StyleConfig


@dataclass
class PromptTemplate:
    role: str
    headings: List[str]
    tone: str
    citation: str
    constraints: List[str]
    query_plan: List[str]

    @classmethod
    def from_yaml(cls, path: Path) -> "PromptTemplate":
        data = yaml.safe_load(path.read_text())
        return cls(
            role=data["role"],
            headings=data["structure"]["headings"],
            tone=data["style"]["tone"],
            citation=data["style"]["citation"],
            constraints=data.get("constraints", []),
            query_plan=data.get("query_plan", []),
        )

    def build_messages(self, summary: str, evidence_blocks: str) -> list[dict[str, str]]:
        plan = "\n".join(f"- {step}" for step in self.query_plan) if self.query_plan else ""
        user_content = (
            f"Section summary: {summary}\n\nEvidence:\n{evidence_blocks}\n\n"
            f"Constraints:\n" + "\n".join(self.constraints)
        )
        if plan:
            user_content += f"\n\nQuery plan:\n{plan}"
        return [
            {"role": "system", "content": f"{self.role}\nTone: {self.tone}. Citation style: {self.citation}."},
            {"role": "user", "content": user_content},
        ]


def _cite(chunk: Chunk) -> str:
    prefix = "POL" if chunk.kind == "policy" else "APP"
    if chunk.kind == "visual":
        prefix = "VIS"
    stem = Path(chunk.path).stem
    return f"[{prefix}:{stem}_p{chunk.page}]"


def _format_chunk(chunk: Chunk) -> str:
    return f"{_cite(chunk)} {chunk.text.strip()}"


def _build_evidence_lines(retrieved: Sequence[Retrieved], visual_summaries: Optional[List[Dict]]) -> List[str]:
    lines = []
    for item in retrieved:
        source_type = item.chunk.metadata.get("source_type", item.chunk.kind)
        lines.append(f"- {source_type}: score={item.score:.3f} {_format_chunk(item.chunk)}")
    if visual_summaries:
        for vis in visual_summaries:
            lines.append(f"- visual:{vis.get('visual_type', 'figure')} [VIS:{Path(vis['path']).stem}] {vis['summary']}")
    return lines


def _detect_policy_hierarchy(chunk: Chunk) -> str:
    text = (chunk.text or "").lower()
    path = Path(chunk.path).name.lower()
    if "nppf" in text or "nppf" in path:
        return "NPPF"
    if "london plan" in text:
        return "London Plan"
    if "spd" in text:
        return "SPD"
    if "local plan" in text or "lp" in text:
        return "Local Plan"
    return "Policy"


def _policy_outline(policies: Sequence[Retrieved]) -> str:
    if not policies:
        return "- Additional policy evidence required. [POL:MISSING]"
    lines = []
    for item in policies:
        hierarchy = _detect_policy_hierarchy(item.chunk)
        lines.append(f"- {hierarchy}: {item.chunk.text.strip()} {_cite(item.chunk)}")
    return "\n".join(lines)


def _cef_table_markdown(
    claims: Sequence[ClaimRecord],
    chunk_map: Dict[str, Chunk],
) -> str:
    if not claims:
        return "No explicit applicant claims detected. [POL:MISSING]"
    header = "| Claim | Evidence | Finding |\n| --- | --- | --- |\n"
    rows: List[str] = []
    for claim in claims:
        app_chunk = chunk_map.get(claim.source)
        claim_cite = _cite(app_chunk) if app_chunk else "[APP:MISSING]"
        evidence = f"{claim.text} {claim_cite}"
        if claim.matched_policies:
            cite_parts = []
            for pid in claim.matched_policies[:2]:
                policy_chunk = chunk_map.get(pid)
                cite_parts.append(_cite(policy_chunk) if policy_chunk else "[POL:MISSING]")
            policy_cites = ", ".join(cite_parts)
        else:
            policy_cites = "[POL:MISSING]"
        finding = (
            f"Policy alignment requires mitigation {policy_cites}"
            if claim.conflicts
            else f"Claim aligns where conditions secure delivery {policy_cites}"
        )
        rows.append(f"| {claim.text} | {evidence} | {finding} |")
    return header + "\n".join(rows)


def _contention_level(claims: Sequence[ClaimRecord], retrieved: Sequence[Retrieved]) -> str:
    conflict_score = sum(len(claim.conflicts) for claim in claims)
    objection_score = sum(1 for item in retrieved if item.source_type in {"objection", "consultee"})
    visual_weight = sum(1 for item in retrieved if item.chunk.kind == "visual")
    score = conflict_score + objection_score + max(visual_weight // 3, 0)
    if score >= 6:
        return "high"
    if score >= 3:
        return "medium"
    return "low"


def _render_claim_paragraphs(
    claims: Sequence[ClaimRecord],
    chunk_map: Dict[str, Chunk],
    fallback_policy: str,
) -> List[str]:
    paragraphs: List[str] = []
    for claim in claims:
        app_chunk = chunk_map.get(claim.source)
        app_cite = _cite(app_chunk) if app_chunk else "[APP:MISSING]"
        matched = [chunk_map.get(pid) for pid in claim.matched_policies if chunk_map.get(pid)]
        policy_cite = _cite(matched[0]) if matched else fallback_policy
        description = claim.text
        if claim.conflicts:
            paragraphs.append(
                f"The applicant states that {description} {app_cite}, but the cited policies impose"
                f" restrictive tests {policy_cite}. Additional mitigation is required to reconcile the"
                " conflict and a condition is advised."
            )
        else:
            paragraphs.append(
                f"The applicant's position that {description} {app_cite} generally tracks the"
                f" intent of {policy_cite}, provided detailed design controls secure delivery."
            )
    if not paragraphs:
        paragraphs.append(f"No explicit claims were parsed; policy compliance must therefore default to {fallback_policy}.")
    return paragraphs


def _render_consultations(retrieved: Sequence[Retrieved], fallback_policy: str) -> str:
    consultee_items = [item for item in retrieved if item.source_type == "consultee"]
    if not consultee_items:
        return f"No statutory consultee responses were retrieved; officers should confirm with specialists {fallback_policy}."
    lines = []
    for item in consultee_items:
        role = item.chunk.metadata.get("statutory_role") or "Consultee"
        lines.append(
            f"{role} raises {('holding objections' if 'object' in (item.chunk.text or '').lower() else 'comments')} "
            f"referencing {item.chunk.text.strip()} {_cite(item.chunk)} {fallback_policy}."
        )
    return "\n".join(lines)


def _render_objections(retrieved: Sequence[Retrieved], fallback_policy: str) -> str:
    objections = [item for item in retrieved if item.source_type in {"objection", "support"}]
    if not objections:
        return f"No third-party representations were parsed; monitoring should continue {fallback_policy}."
    lines = []
    for item in objections:
        tone = "support" if "support" in (item.chunk.text or "").lower() else "objection"
        lines.append(f"Public {tone}s cite {item.chunk.text.strip()} {_cite(item.chunk)} {fallback_policy}.")
    return "\n".join(lines)


def _render_visuals(visual_summaries: Optional[List[Dict]], fallback_policy: str) -> str:
    if not visual_summaries:
        return f"No visual material was interrogated for this section {fallback_policy}."
    lines = []
    for vis in visual_summaries:
        lines.append(f"- {vis['summary']} [VIS:{Path(vis['path']).stem}] {fallback_policy}")
    return "\n".join(lines)


def _render_risks(contention: str, fallback_policy: str) -> str:
    if contention == "high":
        base = "Evidence remains contested and further targeted retrieval plus conditions are required"
    elif contention == "medium":
        base = "Some technical matters remain to be proven at detailed design stage"
    else:
        base = "Residual risks are limited but delivery must be secured by enforceable conditions"
    return f"{base} {fallback_policy}."


def _render_balance(contention: str, fallback_policy: str, fallback_app: str) -> str:
    if contention == "high":
        base = (
            "Substantial harms identified in consultee evidence carry significant weight and outweigh the claimed"
            " benefits unless further mitigation is secured"
        )
    elif contention == "medium":
        base = (
            "Benefits and harms are closely balanced; acceptability relies on safeguards and adherence to the"
            " cited policy tests"
        )
    else:
        base = (
            "Benefits such as regeneration, access improvements, and housing delivery attract moderate weight and"
            " can outweigh limited harms"
        )
    return f"{base}. {fallback_app} {fallback_policy}"


def _assemble_markdown(
    section: str,
    policy_outline: str,
    cef_table: str,
    claims: Sequence[ClaimRecord],
    retrieved: Sequence[Retrieved],
    visual_summaries: Optional[List[Dict]],
    contention: str,
) -> str:
    chunk_map = {item.chunk.id: item.chunk for item in retrieved}
    policies = [item for item in retrieved if item.chunk.kind == "policy"]
    apps = [item for item in retrieved if item.chunk.kind == "app"]
    fallback_policy = _cite(policies[0].chunk) if policies else "[POL:MISSING]"
    fallback_app = _cite(apps[0].chunk) if apps else "[APP:MISSING]"
    claim_paragraphs = _render_claim_paragraphs(claims, chunk_map, fallback_policy)
    doc: List[str] = [f"## {section.title()}"]
    doc.append("### Policy Framework")
    doc.append(policy_outline)
    doc.append("### Claim-Evidence-Finding Table")
    doc.append(cef_table)
    heading = "### Impact Assessment" if contention == "high" else "### Assessment"
    doc.append(heading)
    doc.extend(claim_paragraphs)
    if contention == "high":
        doc.append("#### Public Benefits")
        doc.append(
            f"Public benefits identified by the applicant {fallback_app} carry weight but only if "
            f"strict compliance with {fallback_policy} is demonstrated in reserved matters."
        )
    doc.append("### Consultation Responses")
    doc.append(_render_consultations(retrieved, fallback_policy))
    doc.append("### Objections and Support")
    doc.append(_render_objections(retrieved, fallback_policy))
    doc.append("### Visual Evidence")
    doc.append(_render_visuals(visual_summaries, fallback_policy))
    doc.append("### Risks and Uncertainties")
    doc.append(_render_risks(contention, fallback_policy))
    doc.append("### Mini Planning Balance")
    doc.append(_render_balance(contention, fallback_policy, fallback_app))
    return "\n\n".join(doc)


def _format_stage(stage_name: str, messages: list[dict[str, str]]) -> str:
    lines = [f"### {stage_name}"]
    lines.extend(f"{m['role'].upper()}: {m['content']}" for m in messages)
    return "\n".join(lines)


def _execute_completion(llm: BaseLLM, messages: list[dict[str, str]]) -> str:
    try:
        return asyncio.run(llm.complete(messages))
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(llm.complete(messages))  # type: ignore[arg-type]
        finally:
            loop.close()


def _run_stage(
    stage_name: str,
    llm: BaseLLM | None,
    messages: list[dict[str, str]],
    prompt_segments: List[str],
    completion_segments: List[str],
) -> str:
    prompt_segments.append(_format_stage(stage_name, messages))
    if llm is None:
        return ""
    try:
        output = _execute_completion(llm, messages)
    except Exception as exc:  # pragma: no cover - LLM issues logged
        log("draft", warning="stage_failed", stage=stage_name, detail=str(exc))
        return ""
    if output:
        completion_segments.append(f"{stage_name}\n{output}")
    return output or ""


def _critic_and_repair(
    markdown: str,
    policies: Sequence[Retrieved],
    apps: Sequence[Retrieved],
) -> tuple[str, List[Dict], List[Dict]]:
    paragraphs = markdown.split("\n\n")
    fallback_policy = _cite(policies[0].chunk) if policies else "[POL:MISSING]"
    fallback_app = _cite(apps[0].chunk) if apps else "[APP:MISSING]"
    issues: List[Dict] = []
    for idx, para in enumerate(paragraphs):
        stripped = para.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "[POL:" not in para:
            paragraphs[idx] = f"{para} {fallback_policy}"
            issues.append({"paragraph": idx, "issue": "missing_policy_citation"})
        if ("applicant" in stripped.lower() or "[APP:" in stripped) and "[APP:" not in para:
            paragraphs[idx] = f"{paragraphs[idx]} {fallback_app}"
            issues.append({"paragraph": idx, "issue": "missing_app_citation"})
    repaired = "\n\n".join(paragraphs)
    repairs = [{"paragraph": issue["paragraph"], "action": issue["issue"]} for issue in issues]
    return repaired, issues, repairs


def compose_output(
    section: str,
    retrieved: Iterable[Retrieved],
    template: PromptTemplate,
    llm: BaseLLM | None,
    visual_summaries: Optional[List[Dict]] = None,
    style: StyleConfig | None = None,
) -> tuple[str, str, str, Dict]:
    retrieved_list = list(retrieved)
    visual_summaries = visual_summaries or []
    evidence_lines = _build_evidence_lines(retrieved_list, visual_summaries)
    policies = [item for item in retrieved_list if item.chunk.kind == "policy"]
    apps = [item for item in retrieved_list if item.chunk.kind == "app"]
    claims = extract_claims(apps)
    match_claims_to_policies(claims, policies)
    chunk_map = {item.chunk.id: item.chunk for item in retrieved_list}
    policy_outline_default = _policy_outline(policies)

    prompt_segments: List[str] = []
    completion_segments: List[str] = []

    policy_messages = [
        {"role": "system", "content": "You are a planning policy analyst distilling hierarchy and mandatory tests."},
        {
            "role": "user",
            "content": (
                "Summarise the hierarchy and key requirements from these policy excerpts. "
                "Use bullet points sorted by hierarchy level:\n"
                f"{policy_outline_default}"
            ),
        },
    ]
    policy_outline_model = _run_stage("Policy Framework Extraction", llm, policy_messages, prompt_segments, completion_segments)
    policy_outline_candidate = policy_outline_model.strip()
    policy_outline = policy_outline_candidate if policy_outline_candidate else policy_outline_default

    cef_default = _cef_table_markdown(claims, chunk_map)
    claims_list = "\n".join(f"- {claim.text}" for claim in claims) or "- No claims detected"
    cef_messages = [
        {"role": "system", "content": "You are a planning officer building a Claim-Evidence-Finding (CEF) table."},
        {
            "role": "user",
            "content": (
                f"Claims:\n{claims_list}\n\nExisting table:\n{cef_default}\n\n"
                "Return a Markdown table with Claim/Evidence/Finding columns, citing policies and applications."
            ),
        },
    ]
    cef_model = _run_stage("Claim-Evidence-Finding", llm, cef_messages, prompt_segments, completion_segments)
    cef_candidate = cef_model.strip()
    cef_table = cef_candidate if ("|" in cef_candidate and "[POL" in cef_candidate) else cef_default

    # Style seed from policy language and configured orientation
    orientation = (style.orientation if style else "pro_delivery")
    style_seed = policy_style_seed(policies, orientation=orientation)
    messages = template.build_messages(section, "\n".join(evidence_lines))
    # Prepend style seed to system content for drafting diversity
    if messages and messages[0]["role"] == "system":
        messages[0]["content"] += "\n\n" + style_seed

    # Multi‑draft generation with personas for stylistic diversity
    personas = [
        "Policy-led officer: emphasise policy tests and duties.",
        "Consultee-led officer: foreground consultee positions and conditions.",
        "Applicant-skeptical officer: scrutinise benefits and require mitigation.",
    ]
    if style and style.persona_variants:
        # Simple mapping of enum to descriptive persona blurbs
        mapping = {
            "policy_led": "Policy-led officer: emphasise policy tests and duties.",
            "consultee_led": "Consultee-led officer: foreground consultee positions and conditions.",
            "applicant_skeptical": "Applicant-skeptical officer: scrutinise benefits and require mitigation.",
            "balanced": "Balanced officer: hold even tone and avoid advocacy.",
        }
        personas = [mapping.get(p, p) for p in style.persona_variants]
    draft_variants: List[str] = []
    if llm is not None:
        for persona in personas:
            persona_msgs = list(messages)
            persona_msgs.append({"role": "system", "content": f"Persona: {persona}"})
            out = _run_stage("Draft Generation", llm, persona_msgs, prompt_segments, completion_segments)
            if out.strip():
                draft_variants.append(out.strip())
    else:
        # Dummy mode: single baseline
        out = _run_stage("Draft Generation", llm, messages, prompt_segments, completion_segments)
        if out.strip():
            draft_variants.append(out.strip())

    contention = _contention_level(claims, retrieved_list)
    base_markdown = _assemble_markdown(section, policy_outline, cef_table, claims, retrieved_list, visual_summaries, contention)
    # If we have drafts, selectively replace the Assessment and Mini Planning Balance sections using the judge
    chosen_assessment = None
    chosen_balance = None
    if draft_variants:
        def _extract_section(md: str, heading: str) -> str:
            parts = md.split(f"### {heading}")
            if len(parts) < 2:
                return ""
            tail = parts[1]
            seg = tail.split("\n### ")[0]
            return seg.strip()

        assessments = [_extract_section(d, "Assessment") for d in draft_variants]
        balances = [_extract_section(d, "Mini Planning Balance") for d in draft_variants]
        assessments = [a for a in assessments if a]
        balances = [b for b in balances if b]
        judged_a = judge_paragraphs(llm, "Assessment", assessments, orientation=orientation)
        judged_b = judge_paragraphs(llm, "Mini Planning Balance", balances, orientation=orientation)
        combine_mode = style.combine_mode if style else "judge_random"
        seed = style.seed if style else None
        chosen_assessment = select_by_judge(judged_a, combine_mode=combine_mode, seed=seed) if judged_a else None
        chosen_balance = select_by_judge(judged_b, combine_mode=combine_mode, seed=seed) if judged_b else None

    final_markdown = base_markdown
    if chosen_assessment:
        final_markdown = final_markdown.replace("### Assessment", f"### Assessment\n\n{chosen_assessment}")
    if chosen_balance:
        final_markdown = final_markdown.replace("### Mini Planning Balance", f"### Mini Planning Balance\n\n{chosen_balance}")

    # Quantitative facts section (strictly calculated)
    qfacts = extract_quant_facts(retrieved_list)
    qfacts_md = facts_markdown(qfacts)
    if "### Quantitative Matters" not in final_markdown:
        final_markdown += f"\n\n### Quantitative Matters\n{qfacts_md}\n"

    final_markdown, critic_issues, repairs = _critic_and_repair(final_markdown, policies, apps)

    diagnostics = {
        "policy_outline": policy_outline,
        "cef_table": cef_table,
        "contention": contention,
        "critic_issues": critic_issues,
        "repairs": repairs,
        "draft_variants": len(draft_variants),
    }

    prompt_text = "\n\n".join(prompt_segments)
    completion_text = "\n\n".join(completion_segments)
    return final_markdown, prompt_text, completion_text, diagnostics


def save_prompt(path: Path, prompt_text: str) -> None:
    path.write_text(prompt_text, encoding="utf-8")


__all__ = ["PromptTemplate", "compose_output", "save_prompt"]
