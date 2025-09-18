from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

import httpx


class BaseLLM(ABC):
    @abstractmethod
    async def complete(self, messages: list[dict[str, str]]) -> str:
        ...


class DummyEcho(BaseLLM):
    async def complete(self, messages: list[dict[str, str]]) -> str:
        return "\n".join(message["content"] for message in messages if message["role"] == "user")


class OllamaLocal(BaseLLM):
    def __init__(self, model: str, base_url: str = "http://localhost:11434") -> None:
        self.model = model
        self.client = httpx.AsyncClient(base_url=base_url)

    async def complete(self, messages: list[dict[str, str]]) -> str:
        response = await self.client.post(
            "/api/chat",
            json={
                "model": self.model,
                "messages": messages,
            },
            timeout=None,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("message", {}).get("content", "")


def get_llm(provider: str, model: str) -> BaseLLM:
    if provider == "dummy":
        return DummyEcho()
    if provider == "ollama":
        return OllamaLocal(model=model)
    raise ValueError(f"Unsupported LLM provider: {provider}")


__all__ = ["BaseLLM", "DummyEcho", "OllamaLocal", "get_llm"]
