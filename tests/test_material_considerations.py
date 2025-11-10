from tpa.chunk_store import Chunk
from tpa.material_considerations import (
    detect_material_considerations,
    active_material_considerations,
    assess_material_consideration,
)
from tpa.retrieve import Retrieved


def _chunk(chunk_id: str, kind: str, text: str, source_type: str, metadata_extra=None) -> Chunk:
    metadata = {"source_type": source_type}
    if metadata_extra:
        metadata.update(metadata_extra)
    return Chunk(
        id=chunk_id,
        kind=kind,
        path=f"/tmp/{chunk_id}.pdf",
        page=1,
        text=text,
        hash=chunk_id,
        metadata=metadata,
    )


def test_material_consideration_detection_and_assessment():
    chunks = [
        _chunk("APP:1", "app", "The applicant proposes a new access", "applicant_case"),
        _chunk("POL:1", "policy", "Policy T1 requires safe access", "policy"),
        _chunk(
            "CON:1",
            "app",
            "Highways officer raises a holding objection",
            "consultee",
            metadata_extra={"statutory_role": "Highways"},
        ),
    ]
    topic_map = {chunk.id: ["transport"] for chunk in chunks}
    catalog = detect_material_considerations(chunks, topic_map)
    assert "transport" in catalog
    active = active_material_considerations(["transport"], catalog)
    assert active, "Expected at least one active MC"
    retrieved = [
        Retrieved(chunk=chunks[0], score=0.9),
        Retrieved(chunk=chunks[1], score=0.8),
        Retrieved(chunk=chunks[2], score=0.5),
    ]
    assessed = assess_material_consideration(active[0], retrieved)
    assert assessed.findings, "Assessment should generate findings"
    assert "Highways" in assessed.statutory_consultees
