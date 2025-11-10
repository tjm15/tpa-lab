from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from .chunk_store import Chunk, read_chunks
from .config import RetrievalConfig
from .material_considerations import default_recipe_for
from .embed import encode_query, load_index
from .logging import log


@dataclass
class Retrieved:
    chunk: Chunk
    score: float
    rerank_score: float | None = None

    @property
    def source_type(self) -> str:
        return self.chunk.metadata.get("source_type", self.chunk.kind)


def load_chunk_map(chunks_path: Path) -> Dict[str, Chunk]:
    return {chunk.id: chunk for chunk in read_chunks(chunks_path)}


def retrieve(
    query: str,
    chunks_path: Path,
    index_root: Path,
    retrieval_cfg: RetrievalConfig,
    section: str | None = None,
    topics: Sequence[str] | None = None,
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
        candidate_cap = retrieval_cfg.max_candidates or len(ids)
        top_k = min(max(k, candidate_cap), len(ids))
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
            weight = retrieval_cfg.mix_weights.get(
                chunk.metadata.get("source_type", chunk.kind),  # type: ignore[arg-type]
                retrieval_cfg.mix_weights.get(chunk.kind, 1.0) if retrieval_cfg.mix_weights else 1.0,
            )
            adjusted_score = score * weight
            results.append(Retrieved(chunk=chunk, score=adjusted_score))

    dedup = _deduplicate(results)
    recipe = _resolve_recipe(retrieval_cfg, section, topics)
    ordered = _apply_recipe(dedup, recipe, retrieval_cfg)
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


def _deduplicate(items: List[Retrieved]) -> List[Retrieved]:
    dedup: Dict[str, Retrieved] = {}
    for item in items:
        existing = dedup.get(item.chunk.id)
        if existing is None or item.score > existing.score:
            dedup[item.chunk.id] = item
    return sorted(dedup.values(), key=lambda r: r.score, reverse=True)


def _resolve_recipe(
    cfg: RetrievalConfig, section: str | None, topics: Sequence[str] | None
) -> Dict[str, Dict[str, object]] | None:
    candidates: List[str] = []
    if section:
        candidates.append(section.lower())
    if topics:
        candidates.extend(topic.lower() for topic in topics)
    candidates.append("default")
    for key in candidates:
        if key and key in cfg.recipes:
            return cfg.recipes[key]
    if candidates:
        return default_recipe_for(candidates[0])
    return None


def _apply_recipe(
    items: List[Retrieved],
    recipe: Dict[str, Dict[str, object]] | None,
    cfg: RetrievalConfig,
) -> List[Retrieved]:
    if not recipe:
        return items
    buckets: Dict[str, List[Retrieved]] = defaultdict(list)
    for item in items:
        buckets[item.source_type].append(item)
    for bucket in buckets.values():
        bucket.sort(key=lambda i: i.score, reverse=True)
    selected: List[Retrieved] = []
    target = 0
    for source_type, spec in recipe.items():
        k = int(spec.get("k", 0))  # type: ignore[arg-type]
        target += max(k, 0)
        weight = float(spec.get("weight", 1.0))  # type: ignore[arg-type]
        required = [str(req).lower() for req in spec.get("required", [])]  # type: ignore[arg-type]
        bucket = buckets.get(source_type, [])
        if not bucket:
            log(
                "retrieve",
                warning="missing_source_type",
                source_type=source_type,
            )
            continue
        slice_items = bucket[:k] if k else bucket
        for cand in slice_items:
            cand.score *= weight
            selected.append(cand)
        if required:
            coverage = any(
                any(req in (cand.chunk.text or "").lower() for req in required) for cand in slice_items
            )
            if not coverage:
                for cand in bucket[k:]:
                    if any(req in (cand.chunk.text or "").lower() for req in required):
                        cand.score *= weight
                        selected.append(cand)
                        coverage = True
                        break
            if not coverage:
                log(
                    "retrieve",
                    warning="required_source_missing",
                    source_type=source_type,
                    required=required,
                )
    if target == 0:
        target = len(items)
    selected_ids = {sel.chunk.id for sel in selected}
    remainder = [item for item in items if item.chunk.id not in selected_ids]
    remainder.sort(key=lambda i: i.score, reverse=True)
    for cand in remainder:
        if len(selected) >= target or len(selected) >= cfg.max_candidates:
            break
        selected.append(cand)
        selected_ids.add(cand.chunk.id)
    return selected


__all__ = ["retrieve", "Retrieved"]
