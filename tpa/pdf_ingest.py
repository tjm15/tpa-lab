from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, List

import fitz  # type: ignore

try:  # optional OCR dependencies
    import pytesseract  # type: ignore
    from PIL import Image  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pytesseract = None
    Image = None

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


def ingest_pdfs(config: Config, show_progress: bool = False) -> List[Chunk]:
    """Ingest PDFs into Chunk list.

    If show_progress is True and 'rich' is installed, a progress bar is shown.
    """
    chunks: List[Chunk] = []
    pdfs: List[Path] = []
    for input_dir in config.index.input_dirs:
        pdfs.extend(sorted(input_dir.glob("**/*.pdf")))

    iterator: Iterable[Path] = pdfs
    progress = None
    task_id = None
    if show_progress:
        try:  # optional dependency usage
            from rich.progress import Progress, BarColumn, TimeElapsedColumn, TaskProgressColumn

            progress = Progress(
                "[bold blue]ingest", BarColumn(), TaskProgressColumn(), TimeElapsedColumn()
            )
            progress.start()
            task_id = progress.add_task("pdfs", total=len(pdfs))
        except Exception:  # pragma: no cover - fallback silently
            progress = None

    for pdf_path in iterator:
        _ingest_single_pdf(pdf_path, config, chunks)
        if progress and task_id is not None:
            progress.advance(task_id)

    if progress:
        progress.stop()
    log("ingest", total_chunks=len(chunks), total_pdfs=len(pdfs))
    return chunks


def _ingest_single_pdf(pdf_path: Path, config: Config, chunks: List[Chunk]) -> None:
    kind = _infer_kind(pdf_path)
    with fitz.open(pdf_path) as doc:
        for page_idx, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            ocr_used = False
            if not text and config.index.ocr_fallback:
                if pytesseract and Image:
                    try:
                        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                        import io
                        image = Image.open(io.BytesIO(pix.tobytes("png")))
                        text = pytesseract.image_to_string(image).strip()
                        ocr_used = True
                    except Exception:  # pragma: no cover - noisy OCR failures
                        text = ""
                else:
                    log(
                        "ingest",
                        warning="ocr_fallback_requested_missing_dependency",
                        file=str(pdf_path),
                    )
            base = pdf_path.stem.replace(" ", "_")
            cross_refs = []
            for word in text.split():
                if word.upper().startswith("POL") or word.upper().startswith("LP"):
                    cross_refs.append(word)
            if text:
                chunk_id = f"{kind[:3].upper()}:{base}_p{page_idx}"
                hash_value = hashlib.sha256(
                    f"{pdf_path}:{page_idx}:{text}".encode("utf-8")
                ).hexdigest()
                chunks.append(
                    Chunk(
                        id=chunk_id,
                        kind=kind,
                        path=str(pdf_path.resolve()),
                        page=page_idx,
                        text=text,
                        hash=hash_value,
                        metadata={
                            "type": "text",
                            "cross_refs": cross_refs,
                            "ocr_used": ocr_used,
                        },
                    )
                )

            # Visual placeholder chunk to satisfy hybrid retrieval requirement.
            pix = page.get_pixmap(matrix=fitz.Matrix(1, 1))
            visual_dir = pdf_path.parent / "_visual_cache"
            visual_dir.mkdir(parents=True, exist_ok=True)
            visual_path = visual_dir / f"{base}_p{page_idx}.png"
            pix.save(visual_path)
            visual_id = f"VIS:{base}_p{page_idx}"
            hash_value = hashlib.sha256(str(visual_path).encode("utf-8")).hexdigest()
            chunks.append(
                Chunk(
                    id=visual_id,
                    kind="visual",
                    path=str(pdf_path.resolve()),
                    page=page_idx,
                    text=f"Visual content for {base} page {page_idx}",
                    hash=hash_value,
                    metadata={
                        "type": "visual",
                        "asset": str(visual_path.resolve()),
                        "cross_refs": cross_refs,
                    },
                )
            )


__all__ = ["ingest_pdfs"]
