from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List


@dataclass
class Chunk:
    id: str
    kind: str
    path: str
    page: int
    text: str
    hash: str

    def to_json(self) -> str:
        return json.dumps(self.__dict__, ensure_ascii=False)


def write_chunks(chunks: List[Chunk], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(chunk.to_json())
            fh.write("\n")


def read_chunks(path: Path) -> Iterator[Chunk]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            payload = json.loads(line)
            yield Chunk(**payload)


__all__ = ["Chunk", "write_chunks", "read_chunks"]
