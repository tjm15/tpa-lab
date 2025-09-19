from __future__ import annotations

from abc import ABC, abstractmethod
import base64
from pathlib import Path
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


class OllamaVision:
    def __init__(self, model: str = "gemma3:27b", base_url: str = "http://localhost:11434") -> None:
        self.model = model
        self.client = httpx.AsyncClient(base_url=base_url)

    async def analyse(self, prompt: str, image_path: Path) -> str:
        image_bytes = image_path.read_bytes()
        encoded = base64.b64encode(image_bytes).decode("utf-8")
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image", "image": encoded},
                    ],
                }
            ],
        }
        response = await self.client.post("/api/chat", json=payload, timeout=None)
        response.raise_for_status()
        data = response.json()
        return data.get("message", {}).get("content", "")


def get_llm(provider: str, model: str) -> BaseLLM:
    if provider == "dummy":
        return DummyEcho()
    if provider == "ollama":
        return OllamaLocal(model=model)
    raise ValueError(f"Unsupported LLM provider: {provider}")


__all__ = ["BaseLLM", "DummyEcho", "OllamaLocal", "OllamaVision", "get_llm"]
