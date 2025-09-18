from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import hashlib
import json
import yaml


@dataclass
class IndexConfig:
    input_dirs: List[Path]
    ocr_fallback: bool = False


@dataclass
class RetrievalConfig:
    k_app: int = 6
    k_policy: int = 6
    k_adversarial: int = 0
    max_candidates: int = 60
    use_reranker: bool = False
    mix_weights: Dict[str, float] = None

    def __post_init__(self) -> None:
        if self.mix_weights is None:
            self.mix_weights = {"app": 0.5, "policy": 0.5}


@dataclass
class LLMConfig:
    provider: str = "ollama"
    model: str = "gpt-oss:20b"


@dataclass
class OutputConfig:
    cite_style: str = "inline_ids"
    save_zip_of_pages: bool = True


@dataclass
class Config:
    index: IndexConfig
    retrieval: RetrievalConfig
    llm: LLMConfig
    output: OutputConfig
    path: Path

    @property
    def hash(self) -> str:
        payload = {
            "index": {
                "input_dirs": [str(p) for p in self.index.input_dirs],
                "ocr_fallback": self.index.ocr_fallback,
            },
            "retrieval": {
                "k_app": self.retrieval.k_app,
                "k_policy": self.retrieval.k_policy,
                "k_adversarial": self.retrieval.k_adversarial,
                "max_candidates": self.retrieval.max_candidates,
                "use_reranker": self.retrieval.use_reranker,
                "mix_weights": self.retrieval.mix_weights,
            },
            "llm": {
                "provider": self.llm.provider,
                "model": self.llm.model,
            },
            "output": {
                "cite_style": self.output.cite_style,
                "save_zip_of_pages": self.output.save_zip_of_pages,
            },
        }
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
        return digest


def load_config(path: Path) -> Config:
    data = yaml.safe_load(path.read_text())
    index_cfg = IndexConfig(
        input_dirs=[Path(p) for p in data["index"]["input_dirs"]],
        ocr_fallback=data["index"].get("ocr_fallback", False),
    )
    retrieval_cfg = RetrievalConfig(**data.get("retrieval", {}))
    llm_cfg = LLMConfig(**data.get("llm", {}))
    output_cfg = OutputConfig(**data.get("output", {}))
    return Config(index=index_cfg, retrieval=retrieval_cfg, llm=llm_cfg, output=output_cfg, path=path)


__all__ = ["Config", "load_config"]
