from __future__ import annotations

from typing import List

try:  # optional heavy dependency
    from sentence_transformers import CrossEncoder  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    CrossEncoder = None

from .retrieve import Retrieved
from .logging import log


_DEF_QUERY = "policy compliance"


def rerank(candidates: List[Retrieved], use_reranker: bool = False) -> List[Retrieved]:
    if not use_reranker or not candidates:
        return candidates
    if CrossEncoder is None:
        log("rerank", warning="cross_encoder_unavailable")
        return candidates

    try:
        model = CrossEncoder("BAAI/bge-reranker-large")
        pairs = [(_DEF_QUERY, cand.chunk.text or "") for cand in candidates]
        scores = model.predict(pairs)
        for cand, score in zip(candidates, scores):
            cand.rerank_score = float(score)
        reranked = sorted(candidates, key=lambda c: c.rerank_score or c.score, reverse=True)
        log("rerank", status="completed", count=len(reranked))
        return reranked
    except Exception as exc:  # pragma: no cover - robust fallback
        log("rerank", warning="failed", detail=str(exc))
        return candidates


__all__ = ["rerank"]
