from __future__ import annotations

import hashlib
from pathlib import Path
from typing import List

import fitz  # type: ignore

from .chunk_store import Chunk
from .config import Config
from .logging import log


def _infer_kind(path: Path) -> str:
    parts = {part.lower() for part in path.parts}
    if any("policy" in part for part in parts):
        return "policy"
    if any("plan" in part for part in parts):
        return "policy"
    return "app"


def ingest_pdfs(config: Config) -> List[Chunk]:
    chunks: List[Chunk] = []
    for input_dir in config.index.input_dirs:
        for pdf_path in sorted(input_dir.glob("**/*.pdf")):
            kind = _infer_kind(pdf_path)
            with fitz.open(pdf_path) as doc:
                for page_idx, page in enumerate(doc, start=1):
                    text = page.get_text("text").strip()
                    if not text:
                        continue
                    base = pdf_path.stem.replace(" ", "_")
                    chunk_id = f"{kind[:3].upper()}:{base}_p{page_idx}"
                    hash_value = hashlib.sha256(f"{pdf_path}:{page_idx}:{text}".encode("utf-8")).hexdigest()
                    chunks.append(
                        Chunk(
                            id=chunk_id,
                            kind=kind,
                            path=str(pdf_path.resolve()),
                            page=page_idx,
                            text=text,
                            hash=hash_value,
                        )
                    )
    log("ingest", total_chunks=len(chunks))
    return chunks


__all__ = ["ingest_pdfs"]
