from __future__ import annotations

from pydantic import BaseModel


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    type: str

    # Provide backward compatible helpers
    def model_dump_json(self, **kwargs):  # pydantic v2 method already exists; kept explicit for clarity
        return super().model_dump_json(**kwargs)

    @classmethod
    def model_validate_json(cls, json_data: str):
        return super().model_validate_json(json_data)

__all__ = ["Chunk"]
