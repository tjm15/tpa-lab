import json

import pytest

from tpa import embed
from tpa.chunk_store import Chunk


def _chunk(chunk_id: str, kind: str = "app", text: str = "sample text") -> Chunk:
    return Chunk(
        id=chunk_id,
        kind=kind,
        path=f"/tmp/{chunk_id}.pdf",
        page=1,
        text=text,
        hash="hash",
        metadata={"source_type": "applicant_case"},
    )


def test_build_indexes_hash_override(tmp_path, monkeypatch):
    monkeypatch.setenv("TPA_EMBED_MODE", "hash")
    chunks = [_chunk("APP:1"), _chunk("POL:1", kind="policy", text="policy text")]
    index_dir = tmp_path / "index"
    embed.build_indexes(chunks, index_dir)
    meta = json.loads((index_dir / "app" / "meta.json").read_text())
    assert meta["mode"] == "hash"


def test_qwen_failure_raises(monkeypatch, tmp_path):
    monkeypatch.delenv("TPA_EMBED_MODE", raising=False)
    monkeypatch.setattr(embed, "_qwen_encode", lambda texts: (_ for _ in ()).throw(RuntimeError("boom")))
    chunks = [_chunk("APP:fail")]
    with pytest.raises(RuntimeError):
        embed.build_indexes(chunks, tmp_path / "idx")
