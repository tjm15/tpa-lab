from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

from .retrieve import Retrieved
from .logging import log
from .multimodal import summarise_visuals


@dataclass
class ClaimRecord:
    claim_id: str
    text: str
    source: str
    matched_policies: List[str]
    conflicts: List[str]


def _extract_claims(app_chunks: Iterable[Retrieved]) -> List[ClaimRecord]:
    records: List[ClaimRecord] = []
    claim_counter = 0
    for item in app_chunks:
        text = item.chunk.text or ""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        for sentence in sentences:
            if len(sentence) < 30:
                continue
            if re.search(r"(applicant|proposes|will|seeks|states|deliver)", sentence, re.IGNORECASE):
                claim_counter += 1
                claim_id = f"CLAIM_{item.chunk.id.replace(':', '_')}_{claim_counter}"
                records.append(
                    ClaimRecord(
                        claim_id=claim_id,
                        text=sentence.strip(),
                        source=item.chunk.id,
                        matched_policies=[],
                        conflicts=[],
                    )
                )
    return records


def _match_policies(claims: List[ClaimRecord], policy_chunks: Iterable[Retrieved]) -> None:
    policy_index: Dict[str, Retrieved] = {pol.chunk.id: pol for pol in policy_chunks}
    for claim in claims:
        claim_words = set(re.findall(r"[a-zA-Z]+", claim.text.lower()))
        for policy in policy_index.values():
            policy_text = policy.chunk.text or ""
            policy_words = set(re.findall(r"[a-zA-Z]+", policy_text.lower()))
            overlap = claim_words & policy_words
            if len(overlap) >= 5:
                claim.matched_policies.append(policy.chunk.id)
                if re.search(r"must not|shall not|prohibited", policy_text, re.IGNORECASE):
                    claim.conflicts.append(policy.chunk.id)


def build_reasoning(section: str, retrieved: List[Retrieved], run_dir: Path) -> Path:
    app_items = [item for item in retrieved if item.chunk.kind == "app"]
    policy_items = [item for item in retrieved if item.chunk.kind == "policy"]
    visual_items = [item for item in retrieved if item.chunk.kind == "visual"]

    claims = _extract_claims(app_items)
    _match_policies(claims, policy_items)
    visual_summaries = summarise_visuals(visual_items) if visual_items else []

    reasoning = {
        "section": section,
        "claims": [
            {
                "id": claim.claim_id,
                "text": claim.text,
                "source": claim.source,
                "matched_policies": claim.matched_policies or ["MISSING_POLICY"],
                "conflicts": claim.conflicts,
            }
            for claim in claims
        ]
        or [
            {
                "id": "MISSING_CLAIMS",
                "text": "No extractable claims detected.",
                "source": "MISSING",
                "matched_policies": ["MISSING"],
                "conflicts": [],
            }
        ],
        "visual_summaries": visual_summaries,
        "retrieval_trace": [
            {
                "id": item.chunk.id,
                "kind": item.chunk.kind,
                "score": item.score,
                "path": item.chunk.path,
                "page": item.chunk.page,
            }
            for item in retrieved
        ],
    }

    reasoning_path = run_dir / "reasoning.json"
    reasoning_path.write_text(json.dumps(reasoning, indent=2), encoding="utf-8")
    log("reason", run=str(run_dir), claims=len(claims))
    return reasoning_path


__all__ = ["build_reasoning"]
