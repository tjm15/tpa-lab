from pathlib import Path

from tpa.author import PromptTemplate, compose_output
from tpa.chunk_store import Chunk
from tpa.retrieve import Retrieved


def _chunk(chunk_id: str, kind: str, text: str, source_type: str) -> Chunk:
    return Chunk(
        id=chunk_id,
        kind=kind,
        path=f"/tmp/{chunk_id}.pdf",
        page=1,
        text=text,
        hash=chunk_id,
        metadata={"source_type": source_type},
    )


def test_compose_output_generates_required_sections(tmp_path):
    app_chunk = _chunk("APP:1", "app", "The applicant proposes a new access strategy.", "applicant_case")
    policy_chunk = _chunk("POL:1", "policy", "Policy T1 requires safe and legible access.", "policy")
    retrieved = [
        Retrieved(chunk=app_chunk, score=0.9),
        Retrieved(chunk=policy_chunk, score=0.8),
    ]
    template = PromptTemplate.from_yaml(Path("tpa/prompts/transport.yaml"))
    markdown, prompt_text, completion_text, diagnostics = compose_output(
        "transport",
        retrieved,
        template,
        llm=None,
        visual_summaries=[{"id": "VIS:1", "path": "/tmp/vis.png", "summary": "CGI shows improved access", "visual_type": "diagram"}],
    )
    assert "Mini Planning Balance" in markdown
    assert "[POL:" in markdown
    assert "[APP:" in markdown
    assert "Claim-Evidence-Finding" in markdown
    assert "contention" in diagnostics
    assert prompt_text
    assert completion_text == ""
