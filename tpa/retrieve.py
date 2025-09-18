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

    mode_per_kind: Dict[str, str] = {}
    for kind, k in (("app", retrieval_cfg.k_app), ("policy", retrieval_cfg.k_policy)):
        kind_dir = index_root / kind
        if not kind_dir.exists():
            continue
        vectors, ids, mode = load_index(kind_dir)
        mode_per_kind[kind] = mode
        if not len(ids):
            continue
        top_k = min(k, len(ids))
        query_vec = encode_query(query, mode)
        scores = (vectors @ query_vec.T).reshape(-1)
        order = np.argsort(-scores)[:top_k]
        for idx in order:
            chunk_id = ids[idx]
            chunk = chunk_map.get(chunk_id)
            if not chunk:
                continue
            results.append(Retrieved(chunk=chunk, score=float(scores[idx])))

    dedup: Dict[str, Retrieved] = {}
    for item in results:
        existing = dedup.get(item.chunk.id)
        if existing is None or item.score > existing.score:
            dedup[item.chunk.id] = item

    ordered = sorted(dedup.values(), key=lambda r: r.score, reverse=True)
    log("retrieve", total=len(ordered))
    return ordered


__all__ = ["retrieve", "Retrieved"]
