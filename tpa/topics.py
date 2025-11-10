from __future__ import annotations

import asyncio
import re
from collections import Counter
from typing import Dict, Iterable, List, Sequence

from .chunk_store import Chunk
from .llm import BaseLLM
from .logging import log

_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "from",
    "this",
    "shall",
    "will",
    "would",
    "should",
    "into",
    "over",
    "under",
    "onto",
    "onto",
    "to",
    "of",
    "on",
    "in",
    "be",
    "is",
    "are",
    "a",
    "an",
    "by",
    "as",
    "or",
    "if",
    "at",
    "it",
    "its",
}


class TopicDetector:
    def __init__(self, llm: BaseLLM | None) -> None:
        self.llm = llm

    def suggest_topics(self, chunks: Sequence[Chunk], limit: int = 8) -> List[str]:
        texts = [c.text for c in chunks if c.text]
        if not texts:
            return ["general"]
        sample = texts[: min(len(texts), 40)]
        if self.llm:
            llm_topics = self._llm_topics(sample, limit)
            if llm_topics:
                return llm_topics[:limit]
        return self._fallback_topics(sample, limit)

    def build_topic_map(self, chunks: Sequence[Chunk], topics: Sequence[str]) -> Dict[str, List[str]]:
        normalized_topics = [topic.strip() for topic in topics if topic.strip()]
        topic_map: Dict[str, List[str]] = {}
        for chunk in chunks:
            text = (chunk.text or "").lower()
            assigned = []
            for topic in normalized_topics:
                if not topic:
                    continue
                token = topic.lower()
                if token in text:
                    assigned.append(topic)
            topic_map[chunk.id] = assigned or ["general"]
        return topic_map

    def rank_topics(self, chunk_ids: Iterable[str], topic_map: Dict[str, List[str]], limit: int = 5) -> List[str]:
        counter: Counter[str] = Counter()
        for chunk_id in chunk_ids:
            for topic in topic_map.get(chunk_id, []):
                counter[topic] += 1
        if not counter:
            return []
        return [topic for topic, _ in counter.most_common(limit)]

    def _llm_topics(self, sample: Sequence[str], limit: int) -> List[str]:
        prompt = (
            "You are an experienced UK planning officer. Based on the following excerpts, list up to "
            f"{limit} distinct material considerations (topics) that appear important. "
            "Return the list as comma-separated values without numbering."
        )
        joined = "\n---\n".join(sample)
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": joined},
        ]
        try:
            text = self._run_llm(messages)
        except Exception as exc:  # pragma: no cover - defensive
            log("topics", warning="llm_failed", detail=str(exc))
            return []
        candidates = re.split(r"[,\\n;]+", text)
        cleaned = []
        for cand in candidates:
            label = re.sub(r"[^a-zA-Z0-9\s\-]", "", cand).strip()
            if label and label.lower() not in {"", "n/a", "none"}:
                cleaned.append(label)
        seen = []
        for label in cleaned:
            lower = label.lower()
            if lower not in {s.lower() for s in seen}:
                seen.append(label)
        return seen

    def _run_llm(self, messages: List[dict]) -> str:
        if self.llm is None:
            return ""
        try:
            return asyncio.run(self.llm.complete(messages))
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self.llm.complete(messages))  # type: ignore[arg-type]
            finally:
                loop.close()

    def _fallback_topics(self, sample: Sequence[str], limit: int) -> List[str]:
        counter: Counter[str] = Counter()
        for text in sample:
            tokens = re.findall(r"[A-Za-z][A-Za-z\-]+", text.lower())
            phrases = []
            current: List[str] = []
            for token in tokens:
                if token in _STOPWORDS:
                    if current:
                        phrases.append(" ".join(current))
                        current = []
                else:
                    current.append(token)
            if current:
                phrases.append(" ".join(current))
            for phrase in phrases:
                if len(phrase) < 5:
                    continue
                counter[phrase] += 1
        if not counter:
            return ["general"]
        ranked = [phrase.title() for phrase, _ in counter.most_common(limit * 2)]
        deduped = []
        for phrase in ranked:
            lower = phrase.lower()
            if lower not in {p.lower() for p in deduped}:
                deduped.append(phrase)
            if len(deduped) >= limit:
                break
        return deduped or ["general"]


__all__ = ["TopicDetector"]
