from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, TYPE_CHECKING

from .claims import extract_claims, match_claims_to_policies

from .chunk_store import Chunk
from .logging import log

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .retrieve import Retrieved

@dataclass
class ChunkRef:
    chunk_id: str
    cite_id: str
    kind: str
    page: int
    path: str
    text: str
    source_type: str


@dataclass
class Finding:
    claim: str
    supporting: List[str]
    contradicting: List[str]
    conclusion: str
    weight: str


@dataclass
class MaterialConsideration:
    topic: str
    policies: List[ChunkRef] = field(default_factory=list)
    statutory_consultees: List[str] = field(default_factory=list)
    applicant_evidence: List[ChunkRef] = field(default_factory=list)
    technical_evidence: List[ChunkRef] = field(default_factory=list)
    consultee_responses: List[ChunkRef] = field(default_factory=list)
    objections: List[ChunkRef] = field(default_factory=list)
    findings: List[Finding] = field(default_factory=list)
    weight: str = "neutral"
    recommendation: str = "requires_mitigation"


def _cite_id(chunk: Chunk) -> str:
    prefix = "POL" if chunk.kind == "policy" else "APP"
    if chunk.kind == "visual":
        prefix = "VIS"
    stem = Path(chunk.path).stem
    return f"{prefix}:{stem}_p{chunk.page}"


def _chunk_ref(chunk: Chunk) -> ChunkRef:
    return ChunkRef(
        chunk_id=chunk.id,
        cite_id=_cite_id(chunk),
        kind=chunk.kind,
        page=chunk.page,
        path=chunk.path,
        text=(chunk.text or "").strip(),
        source_type=chunk.metadata.get("source_type", chunk.kind),
    )


def detect_material_considerations(
    chunks: Iterable[Chunk],
    topic_map: Dict[str, Sequence[str]] | None = None,
) -> Dict[str, MaterialConsideration]:
    catalog: Dict[str, MaterialConsideration] = {}
    for chunk in chunks:
        topics: Sequence[str] = (
            topic_map.get(chunk.id, ["general"]) if topic_map is not None else chunk.metadata.get("topics") or ["general"]
        )
        source_type = chunk.metadata.get("source_type", chunk.kind)
        ref = _chunk_ref(chunk)
        for raw_topic in topics:
            topic = raw_topic.lower()
            mc = catalog.setdefault(topic, MaterialConsideration(topic=topic))
            if source_type == "policy":
                _append_unique(mc.policies, ref)
            elif source_type == "technical_report":
                _append_unique(mc.technical_evidence, ref)
            elif source_type == "consultee":
                _append_unique(mc.consultee_responses, ref)
                role = chunk.metadata.get("statutory_role")
                if role and role not in mc.statutory_consultees:
                    mc.statutory_consultees.append(role)
            elif source_type in {"objection", "support"}:
                _append_unique(mc.objections, ref)
            elif source_type == "visual":
                _append_unique(mc.applicant_evidence, ref)
            else:
                _append_unique(mc.applicant_evidence, ref)
    for mc in catalog.values():
        _derive_weight(mc)
    log("mc_detect", total=len(catalog))
    return catalog


def _append_unique(bucket: List[ChunkRef], ref: ChunkRef) -> None:
    if any(existing.chunk_id == ref.chunk_id for existing in bucket):
        return
    bucket.append(ref)


def _derive_weight(mc: MaterialConsideration) -> None:
    objection_count = len(mc.objections)
    support_count = len([ref for ref in mc.objections if "support" in ref.text.lower()])
    consultee_conflicts = len(mc.consultee_responses)
    baseline = objection_count + consultee_conflicts - support_count
    if baseline >= 4:
        mc.weight = "significant"
        mc.recommendation = "unacceptable"
    elif baseline >= 2:
        mc.weight = "moderate"
        mc.recommendation = "requires_mitigation"
    elif baseline <= -2:
        mc.weight = "moderate"
        mc.recommendation = "acceptable"
    else:
        mc.weight = "limited"
        mc.recommendation = "requires_mitigation"


def default_recipe_for(topic: str) -> Dict[str, Dict[str, float | int | List[str]]]:
    base = {
        "applicant_case": {"k": 4, "weight": 0.3},
        "policy": {"k": 6, "weight": 0.35},
        "technical_report": {"k": 3, "weight": 0.15},
        "consultee": {"k": 2, "weight": 0.15},
        "objection": {"k": 2, "weight": 0.05},
    }
    topic = topic.lower()
    if topic in {"heritage", "design"}:
        base["consultee"]["required"] = ["conservation"]
        base["policy"]["weight"] = 0.4
    if topic in {"transport"}:
        base["technical_report"]["k"] = 4
        base["consultee"]["required"] = ["highway", "tfl"]
    if topic in {"housing"}:
        base["applicant_case"]["k"] = 6
        base["policy"]["k"] = 8
    return base


def active_material_considerations(
    requested_topics: Sequence[str] | None,
    catalog: Dict[str, MaterialConsideration],
) -> List[MaterialConsideration]:
    if not catalog:
        return []
    active: List[MaterialConsideration] = []
    if requested_topics:
        for topic in requested_topics:
            key = topic.lower()
            mc = catalog.get(key)
            if mc:
                active.append(copy.deepcopy(mc))
    if not active:
        weighted = sorted(
            catalog.values(),
            key=lambda item: {"significant": 3, "moderate": 2, "limited": 1}.get(item.weight, 0),
            reverse=True,
        )
        if weighted:
            active.append(copy.deepcopy(weighted[0]))
    return active


def assess_material_consideration(
    mc: MaterialConsideration,
    retrieved: Sequence["Retrieved"],
) -> MaterialConsideration:
    assessed = copy.deepcopy(mc)
    chunk_map = {item.chunk.id: item.chunk for item in retrieved}
    for item in retrieved:
        ref = _chunk_ref(item.chunk)
        source_type = item.chunk.metadata.get("source_type", item.chunk.kind)
        if source_type == "policy":
            _append_unique(assessed.policies, ref)
        elif source_type == "technical_report":
            _append_unique(assessed.technical_evidence, ref)
        elif source_type == "consultee":
            _append_unique(assessed.consultee_responses, ref)
            role = item.chunk.metadata.get("statutory_role")
            if role and role not in assessed.statutory_consultees:
                assessed.statutory_consultees.append(role)
        elif source_type in {"objection", "support"}:
            _append_unique(assessed.objections, ref)
        else:
            _append_unique(assessed.applicant_evidence, ref)

    app_chunks = [item for item in retrieved if item.chunk.metadata.get("source_type", item.chunk.kind) == "applicant_case" or item.chunk.kind == "app"]
    policy_chunks = [item for item in retrieved if item.chunk.kind == "policy"]
    claims = extract_claims(app_chunks)
    match_claims_to_policies(claims, policy_chunks)

    findings: List[Finding] = []
    for claim in claims:
        app_chunk = chunk_map.get(claim.source)
        supporting = [_cite_id(app_chunk)] if app_chunk else ["APP:MISSING"]
        contradicting = [_cite_id(chunk_map[pid]) for pid in claim.conflicts if pid in chunk_map]
        policy_refs = [chunk_map.get(pid) for pid in claim.matched_policies if pid in chunk_map]
        conclusion = "Claim aligns with policy intent."
        weight = "limited"
        if contradicting:
            conclusion = "Claim conflicts with mandatory policy criteria."
            weight = "significant"
        elif policy_refs:
            conclusion = "Claim is supported where mitigation/conditions are secured."
            weight = "moderate"
        findings.append(
            Finding(
                claim=claim.text,
                supporting=supporting,
                contradicting=contradicting or ["POL:MISSING"],
                conclusion=conclusion,
                weight=weight,
            )
        )

    assessed.findings = findings
    if any(f.conclusion.startswith("Claim conflicts") for f in findings):
        assessed.weight = "significant"
        assessed.recommendation = "unacceptable"
    elif findings:
        assessed.weight = "moderate"
        assessed.recommendation = "requires_mitigation"
    return assessed


__all__ = [
    "ChunkRef",
    "Finding",
    "MaterialConsideration",
    "detect_material_considerations",
    "default_recipe_for",
    "active_material_considerations",
    "assess_material_consideration",
]
