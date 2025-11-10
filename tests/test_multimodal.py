from pathlib import Path

from tpa.chunk_store import Chunk
from tpa.retrieve import Retrieved
from tpa import multimodal


def _visual_chunk(path: Path) -> Chunk:
    return Chunk(
        id="VIS:1",
        kind="visual",
        path=str(path),
        page=1,
        text="Visual page",
        hash="vis",
        metadata={"source_type": "visual", "asset": str(path), "visual_type": "diagram"},
    )


def test_summarise_visuals_disabled(monkeypatch, tmp_path):
    monkeypatch.setenv("TPA_DISABLE_VISION", "1")
    chunk = _visual_chunk(tmp_path / "img.png")
    retrieved = [Retrieved(chunk=chunk, score=0.5)]
    assert multimodal.summarise_visuals(retrieved) == []


def test_summarise_visuals_with_stub(monkeypatch, tmp_path):
    monkeypatch.delenv("TPA_DISABLE_VISION", raising=False)
    asset = tmp_path / "img.png"
    asset.write_bytes(b"\x89PNG\r\n\x1a\n")  # minimal header; content unused by stub
    chunk = _visual_chunk(asset)
    retrieved = [Retrieved(chunk=chunk, score=0.9)]

    async def fake_analyse(self, prompt, image_path):
        return f"summary for {image_path.name}"

    monkeypatch.setattr(multimodal.OllamaVision, "analyse", fake_analyse, raising=False)
    summaries = multimodal.summarise_visuals(retrieved)
    assert summaries and summaries[0]["summary"].startswith("summary for")
