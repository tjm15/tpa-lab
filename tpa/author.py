from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import yaml

from .chunk_store import Chunk
from .config import Config
from .llm import BaseLLM
from .logging import log
from .retrieve import Retrieved


@dataclass
class PromptTemplate:
    role: str
    headings: List[str]
    tone: str
    citation: str
    constraints: List[str]
    query_plan: List[str]

    @classmethod
    def from_yaml(cls, path: Path) -> "PromptTemplate":
        data = yaml.safe_load(path.read_text())
        return cls(
            role=data["role"],
            headings=data["structure"]["headings"],
            tone=data["style"]["tone"],
            citation=data["style"]["citation"],
            constraints=data.get("constraints", []),
            query_plan=data.get("query_plan", []),
        )

    def build_messages(self, summary: str, evidence_blocks: str) -> list[dict[str, str]]:
        user_content = f"Section summary: {summary}\n\nEvidence:\n{evidence_blocks}\n\nConstraints:\n" + "\n".join(self.constraints)
        return [
            {"role": "system", "content": f"{self.role}\nTone: {self.tone}. Citation style: {self.citation}."},
            {"role": "user", "content": user_content},
        ]


def _format_chunk(chunk: Chunk) -> str:
    prefix = "POL" if chunk.kind == "policy" else "APP"
    return f"[{prefix}:{Path(chunk.path).stem}_p{chunk.page}] {chunk.text.strip()}"


def compose_output(
    section: str,
    retrieved: Iterable[Retrieved],
    template: PromptTemplate,
    llm: BaseLLM | None,
    deterministic: bool = True,
) -> tuple[str, str]:
    evidence_lines = [
        f"- score={item.score:.3f}: {_format_chunk(item.chunk)}"
        for item in retrieved
    ]
    evidence_block = "\n".join(evidence_lines)

    policies = [item for item in retrieved if item.chunk.kind == "policy"]
    apps = [item for item in retrieved if item.chunk.kind == "app"]
    doc_intro = f"## {section.title()}\n"
    doc_intro += "### Claims\n"
    if apps:
        for item in apps:
            doc_intro += f"- {item.chunk.text.strip()} [APP:{Path(item.chunk.path).stem}_p{item.chunk.page}]\n"
    else:
        doc_intro += "- No explicit applicant claims retrieved.\n"
    doc_intro += "### Evidence\n"
    if policies:
        for item in policies:
            doc_intro += f"- {item.chunk.text.strip()} [POL:{Path(item.chunk.path).stem}_p{item.chunk.page}]\n"
    else:
        doc_intro += "- Additional policy evidence required.\n"
    doc_intro += "### Finding\n"
    if policies:
        doc_intro += (
            f"Policy extracts indicate key constraints. Primary reference "
            f"[POL:{Path(policies[0].chunk.path).stem}_p{policies[0].chunk.page}] must be satisfied.\n"
        )
    else:
        doc_intro += "Further policy evidence is needed to substantiate a finding.\n"
    doc_intro += "### Risks and Uncertainties\n- Evidence base incomplete; manual review required."
    if policies:
        doc_intro += f" [POL:{Path(policies[0].chunk.path).stem}_p{policies[0].chunk.page}]\n"
    else:
        doc_intro += "\n"
    doc_intro += "### Mini Planning Balance\n"
    if policies and apps:
        doc_intro += (
            f"Applicant benefits [APP:{Path(apps[0].chunk.path).stem}_p{apps[0].chunk.page}] "
            f"must be weighed against compliance with policy tests "
            f"[POL:{Path(policies[0].chunk.path).stem}_p{policies[0].chunk.page}]."
        )
    else:
        doc_intro += "Insufficient evidence for a planning balance; further retrieval needed."
    messages = template.build_messages(section, evidence_block)
    prompt_text = "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)
    return doc_intro, prompt_text


def save_prompt(path: Path, prompt_text: str) -> None:
    path.write_text(prompt_text, encoding="utf-8")


__all__ = ["PromptTemplate", "compose_output", "save_prompt"]
