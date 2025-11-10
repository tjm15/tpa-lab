from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .retrieve import Retrieved


@dataclass
class ClaimRecord:
    claim_id: str
    text: str
    source: str
    matched_policies: List[str]
    conflicts: List[str]


def extract_claims(app_chunks: Iterable["Retrieved"]) -> List[ClaimRecord]:
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


def match_claims_to_policies(claims: List[ClaimRecord], policy_chunks: Iterable["Retrieved"]) -> None:
    for claim in claims:
        claim_words = set(re.findall(r"[a-zA-Z]+", claim.text.lower()))
        for policy in policy_chunks:
            policy_text = policy.chunk.text or ""
            policy_words = set(re.findall(r"[a-zA-Z]+", policy_text.lower()))
            overlap = claim_words & policy_words
            if len(overlap) >= 5:
                claim.matched_policies.append(policy.chunk.id)
                if re.search(r"must not|shall not|prohibited|refuse", policy_text, re.IGNORECASE):
                    claim.conflicts.append(policy.chunk.id)


__all__ = ["ClaimRecord", "extract_claims", "match_claims_to_policies"]
