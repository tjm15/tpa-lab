from tpa.chunk_store import Chunk, write_chunks
from tpa.embed import build_indexes
from tpa.config import RetrievalConfig
from tpa.retrieve import retrieve


def _mk_chunk(
    chunk_id: str,
    kind: str,
    text: str,
    source_type: str,
) -> Chunk:
    return Chunk(
        id=chunk_id,
        kind=kind,
        path=f"/tmp/{chunk_id}.pdf",
        page=1,
        text=text,
        hash=chunk_id,
        metadata={"source_type": source_type},
    )


def test_retrieve_respects_recipe(tmp_path, monkeypatch):
    monkeypatch.setenv("TPA_EMBED_MODE", "hash")
    chunks = [
        _mk_chunk("APP:1", "app", "transport access improvement", "applicant_case"),
        _mk_chunk("TECH:1", "app", "technical transport statement", "technical_report"),
        _mk_chunk("POL:1", "policy", "policy T1 highway requirement", "policy"),
    ]
    chunks_path = tmp_path / "chunks.jsonl"
    index_dir = tmp_path / "index"
    write_chunks(chunks, chunks_path)
    build_indexes(chunks, index_dir)
    cfg = RetrievalConfig(
        k_app=2,
        k_policy=2,
        mix_weights={"applicant_case": 1.0, "technical_report": 1.0, "policy": 1.0},
        recipes={
            "transport": {
                "applicant_case": {"k": 1, "weight": 1.0},
                "technical_report": {"k": 1, "weight": 1.0},
                "policy": {"k": 1, "weight": 1.0},
            }
        },
    )
    results = retrieve(
        "transport policy compliance",
        chunks_path,
        index_dir,
        cfg,
        section="transport",
        topics=["transport"],
    )
    assert {item.source_type for item in results} == {"applicant_case", "technical_report", "policy"}


def test_retrieve_enforces_source_cap(tmp_path, monkeypatch):
    monkeypatch.setenv("TPA_EMBED_MODE", "hash")
    chunks = [
        _mk_chunk(f"APP:{idx}", "app", f"application text {idx}", "applicant_case") for idx in range(4)
    ] + [
        _mk_chunk("POL:X", "policy", "policy control text", "policy")
    ]
    chunks_path = tmp_path / "chunks_cap.jsonl"
    index_dir = tmp_path / "index_cap"
    write_chunks(chunks, chunks_path)
    build_indexes(chunks, index_dir)
    cfg = RetrievalConfig(k_app=4, k_policy=1)
    results = retrieve("transport compliance", chunks_path, index_dir, cfg)
    app_count = sum(1 for item in results if item.chunk.kind == "app")
    assert app_count <= int(0.7 * len(results)) or app_count == len(results) == 1
