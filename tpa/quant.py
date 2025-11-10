from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List

from pathlib import Path
from .retrieve import Retrieved
from .chunk_store import Chunk


@dataclass
class QuantFact:
    label: str
    value: str
    cite: str


def extract_quant_facts(retrieved: Iterable[Retrieved]) -> List[QuantFact]:
    facts: List[QuantFact] = []
    patterns = {
        "dwellings": re.compile(r"(\d{1,4})\s*(?:homes|dwellings|units|residential units)", re.I),
        "storeys": re.compile(r"(\d{1,2})\s*(?:storey|storeys)", re.I),
        "parking": re.compile(r"(\d{1,4})\s*(?:parking|car\s*spaces|spaces)", re.I),
        "sqm": re.compile(r"(\d{2,6})\s*(?:sqm|sq\.?\s*m|square metres)", re.I),
    }
    for item in retrieved:
        text = item.chunk.text or ""
        for label, rx in patterns.items():
            for m in rx.finditer(text):
                facts.append(QuantFact(label=label, value=m.group(1), cite=_cite_local(item.chunk)))
    # de‑dupe by (label, value)
    unique = {}
    for f in facts:
        unique[(f.label, f.value)] = f
    return list(unique.values())


def facts_markdown(facts: List[QuantFact]) -> str:
    if not facts:
        return "No quantitative statements could be verified from retrieved evidence."
    lines = ["- {}: {} {}".format(f.label.title(), f.value, f.cite) for f in facts]
    return "\n".join(lines)


def _cite_local(chunk: Chunk) -> str:
    prefix = "POL" if chunk.kind == "policy" else "APP"
    if chunk.kind == "visual":
        prefix = "VIS"
    stem = Path(chunk.path).stem
    return f"[{prefix}:{stem}_p{chunk.page}]"


__all__ = ["QuantFact", "extract_quant_facts", "facts_markdown"]
