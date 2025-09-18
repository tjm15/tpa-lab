from tpa.schemas import Chunk


def test_chunk_roundtrip():
    chunk = Chunk(chunk_id="doc_p001_c00", doc_id="doc", type="text")
    payload = chunk.model_dump_json()
    restored = Chunk.model_validate_json(payload)
    assert restored == chunk
