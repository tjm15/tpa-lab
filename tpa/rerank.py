from __future__ import annotations

from typing import List

from .retrieve import Retrieved


def rerank(candidates: List[Retrieved], use_reranker: bool = False) -> List[Retrieved]:
    if not use_reranker:
        return candidates
    # Placeholder: real cross-encoder integration pending.
    return candidates


__all__ = ["rerank"]
