from __future__ import annotations

from pathlib import Path
from typing import Iterable

import fitz  # type: ignore

from .retrieve import Retrieved


def export_pages(retrieved: Iterable[Retrieved], evidence_dir: Path) -> None:
    evidence_dir.mkdir(parents=True, exist_ok=True)
    seen: set[tuple[str, int]] = set()
    for item in retrieved:
        key = (item.chunk.path, item.chunk.page)
        if key in seen:
            continue
        seen.add(key)
        pdf_path = Path(item.chunk.path)
        with fitz.open(pdf_path) as doc:
            page = doc.load_page(item.chunk.page - 1)
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            out_path = evidence_dir / f"{item.chunk.id.replace(':', '_')}.png"
            pix.save(out_path)
