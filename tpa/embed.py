from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import os

from .chunk_store import Chunk
from .logging import log

_HASH_DIM = 512


def _hash_encode(texts: List[str]) -> np.ndarray:
    mat = np.zeros((len(texts), _HASH_DIM), dtype="float32")
    for row, text in enumerate(texts):
        for token in text.lower().split():
            idx = hash(token) % _HASH_DIM
            mat[row, idx] += 1.0
        norm = np.linalg.norm(mat[row])
        if norm:
            mat[row] /= norm
    return mat


def _bge_encode(texts: List[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("BAAI/bge-large-en")
    embeddings = model.encode(texts, batch_size=16, normalize_embeddings=True)
    return np.asarray(embeddings, dtype="float32")


def _encode(texts: List[str]) -> Tuple[np.ndarray, str]:
    mode_override = os.getenv("TPA_EMBED_MODE")
    if mode_override == "hash":
        return _hash_encode(texts), "hash"
    try:
        return _bge_encode(texts), "bge"
    except Exception as exc:  # pragma: no cover - fallback for offline envs
        log("embed", warning="Falling back to hash encoder", detail=str(exc))
        return _hash_encode(texts), "hash"


def build_indexes(chunks: Iterable[Chunk], output_dir: Path) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    by_kind: Dict[str, List[Chunk]] = {}
    for chunk in chunks:
        by_kind.setdefault(chunk.kind, []).append(chunk)

    index_paths: Dict[str, Path] = {}
    for kind, items in by_kind.items():
        texts = [c.text for c in items]
        embeddings, mode = _encode(texts)
        kind_dir = output_dir / kind
        kind_dir.mkdir(parents=True, exist_ok=True)
        np.save(kind_dir / "vectors.npy", embeddings)
        (kind_dir / "ids.txt").write_text("\n".join(c.id for c in items), encoding="utf-8")
        meta = {"mode": mode}
        (kind_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
        index_paths[kind] = kind_dir
        log("embed", kind=kind, count=len(items), mode=mode)
    return index_paths


def load_index(kind_dir: Path) -> Tuple[np.ndarray, List[str], str]:
    vectors = np.load(kind_dir / "vectors.npy")
    ids = (kind_dir / "ids.txt").read_text(encoding="utf-8").splitlines()
    mode = json.loads((kind_dir / "meta.json").read_text(encoding="utf-8"))["mode"]
    return vectors, ids, mode


def encode_query(query: str, mode: str) -> np.ndarray:
    if mode == "bge":
        return _bge_encode([query])
    vector = _hash_encode([query])
    return vector


__all__ = ["build_indexes", "load_index", "encode_query"]
