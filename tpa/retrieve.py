from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

from .chunk_store import Chunk, read_chunks
from .config import RetrievalConfig
from .embed import encode_query, load_index
from .logging import log


@dataclass
class Retrieved:
    chunk: Chunk
    score: float
    rerank_score: float | None = None


def load_chunk_map(chunks_path: Path) -> Dict[str, Chunk]:
    return {chunk.id: chunk for chunk in read_chunks(chunks_path)}


def retrieve(
    query: str,
    chunks_path: Path,
    index_root: Path,
    retrieval_cfg: RetrievalConfig,
) -> List[Retrieved]:
    chunk_map = load_chunk_map(chunks_path)
    results: List[Retrieved] = []

    for kind, k in (("app", retrieval_cfg.k_app), ("policy", retrieval_cfg.k_policy)):
        kind_dir = index_root / kind
        if not kind_dir.exists():
            continue
        vectors, ids, mode = load_index(kind_dir)
        if not len(ids):
            continue
        top_k = min(k, len(ids))
        query_vec = encode_query(query, mode)
        scores = (vectors @ query_vec.T).reshape(-1)
        order = np.argsort(-scores)
        top_idxs = order[:top_k]
        # Adversarial (low-similarity) sampling for policy set.
        adv_idxs: List[int] = []
        if kind == "policy" and retrieval_cfg.k_adversarial > 0:
            adv_candidate = order[::-1][: min(retrieval_cfg.k_adversarial, len(order))]
            adv_idxs.extend(int(idx) for idx in adv_candidate if idx not in top_idxs)
        selected = list(dict.fromkeys(list(top_idxs) + adv_idxs))
        for idx in selected:
            chunk_id = ids[int(idx)]
            chunk = chunk_map.get(chunk_id)
            if not chunk:
                continue
            score = float(scores[int(idx)])
            weight = retrieval_cfg.mix_weights.get(chunk.kind, 1.0) if retrieval_cfg.mix_weights else 1.0
            adjusted_score = score * weight
            results.append(Retrieved(chunk=chunk, score=adjusted_score))

    dedup: Dict[str, Retrieved] = {}
    for item in results:
        existing = dedup.get(item.chunk.id)
        if existing is None or item.score > existing.score:
            dedup[item.chunk.id] = item

    ordered = sorted(dedup.values(), key=lambda r: r.score, reverse=True)
    max_ratio = 0.7
    adjusted = True
    while adjusted and len(ordered) > 1:
        adjusted = False
        total = len(ordered)
        app_count = sum(1 for item in ordered if item.chunk.kind == "app")
        policy_count = sum(1 for item in ordered if item.chunk.kind == "policy")
        for kind, count in (("app", app_count), ("policy", policy_count)):
            if count > max_ratio * total:
                for idx in range(len(ordered) - 1, -1, -1):
                    if ordered[idx].chunk.kind == kind:
                        ordered.pop(idx)
                        adjusted = True
                        break
                if adjusted:
                    break
    log("retrieve", total=len(ordered))
    return ordered


__all__ = ["retrieve", "Retrieved"]
