from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List

from .retrieve import Retrieved
from .logging import log
from .multimodal import summarise_visuals
from .claims import ClaimRecord, extract_claims, match_claims_to_policies


def build_reasoning(
    section: str,
    retrieved: List[Retrieved],
    run_dir: Path,
    diagnostics: Dict | None = None,
    visual_summaries: List[dict] | None = None,
) -> Path:
    app_items = [item for item in retrieved if item.chunk.kind == "app"]
    policy_items = [item for item in retrieved if item.chunk.kind == "policy"]
    visual_items = [item for item in retrieved if item.chunk.kind == "visual"]

    claims = extract_claims(app_items)
    match_claims_to_policies(claims, policy_items)
    if visual_summaries is None:
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
    if diagnostics:
        reasoning["stage_diagnostics"] = diagnostics

    reasoning_path = run_dir / "reasoning.json"
    reasoning_path.write_text(json.dumps(reasoning, indent=2), encoding="utf-8")
    log("reason", run=str(run_dir), claims=len(claims))
    return reasoning_path


__all__ = ["build_reasoning"]
