from __future__ import annotations

from abc import ABC, abstractmethod
import base64
import os
from pathlib import Path
from typing import Iterable, List

import httpx

# Optional Google GenAI imports (lazy fail with helpful error)
try:  # pragma: no cover - only executed when google-genai installed
    from google import genai  # type: ignore
    from google.genai import types  # type: ignore
except Exception:  # pragma: no cover - absence handled at runtime
    genai = None  # type: ignore
    types = None  # type: ignore

try:  # retry support (optional)
    from tenacity import retry, stop_after_attempt, wait_exponential
except Exception:  # pragma: no cover
    # Minimal shim if tenacity missing; executes function directly
    def retry(*args, **kwargs):  # type: ignore
        def deco(fn):
            return fn
        return deco
    def stop_after_attempt(n):  # type: ignore
        return None
    def wait_exponential(**kwargs):  # type: ignore
        return None


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


class GoogleGeminiClient(BaseLLM):
    """LLM client for Google Gemini 2.5 Pro (Developer API).

    Provides:
      - complete(): chat-style completion using aggregated messages
      - analyse_image(): vision description for a single image with prompt
        Future:
            - Streaming support: Google's API offers streamGenerateContent; we can expose an
                async generator yielding incremental text segments without changing external
                interface by adding optional `stream: bool` parameter returning full text when
                False, or an async iterator when True.
    """

    def __init__(self, model: str = "gemini-2.5-pro", temperature: float = 0.2, vertex: bool = False) -> None:
        if genai is None:
            raise ImportError("google-genai not installed. Install with `pip install google-genai`.")
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key and not vertex:
            raise RuntimeError("Missing GOOGLE_API_KEY (or GEMINI_API_KEY) for Google Gemini provider.")
        client_args = {}
        if vertex:
            # Vertex mode requires additional env vars (optional future use)
            project = os.getenv("GOOGLE_CLOUD_PROJECT")
            location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
            if not project:
                raise RuntimeError("Vertex mode requires GOOGLE_CLOUD_PROJECT env var.")
            client_args.update({"vertexai": True, "project": project, "location": location})
        else:
            client_args["api_key"] = api_key
        self._client = genai.Client(**client_args)
        self.model = model or "gemini-2.5-pro"
        self.temperature = temperature

    def _messages_to_contents(self, messages: List[dict[str, str]]) -> List[str]:
        system_parts = [m["content"] for m in messages if m.get("role") == "system"]
        user_parts = [m["content"] for m in messages if m.get("role") == "user"]
        # Merge system into first user block to approximate system role handling
        merged: List[str] = []
        sys_prefix = ("\n".join(system_parts).strip() + "\n\n") if system_parts else ""
        if user_parts:
            merged.append(sys_prefix + user_parts[0])
            for extra in user_parts[1:]:
                merged.append(extra)
        else:
            if sys_prefix:
                merged.append(sys_prefix)
        return merged or ["(empty prompt)"]

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))  # type: ignore
    async def complete(self, messages: List[dict[str, str]]) -> str:  # type: ignore[override]
        contents = self._messages_to_contents(messages)
        try:
            # google-genai async not yet GA; run in thread if necessary
            import asyncio
            loop = asyncio.get_running_loop()
            def _call():
                return self._client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=types.GenerateContentConfig(  # type: ignore
                        temperature=self.temperature,
                    ),
                )
            resp = await loop.run_in_executor(None, _call)
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"Gemini completion failed: {exc}") from exc
        # Prefer unified .text attribute, fallback to first candidate
        text = getattr(resp, "text", None)
        if not text and getattr(resp, "candidates", None):  # pragma: no cover
            try:
                text = resp.candidates[0].content.parts[0].text  # type: ignore
            except Exception:
                text = ""
        return (text or "").strip()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))  # type: ignore
    async def analyse_image(self, prompt: str, image_path: Path) -> str:
        try:
            import asyncio
            image_bytes = image_path.read_bytes()
            b64 = base64.b64encode(image_bytes).decode("utf-8")
            parts = [
                {"text": prompt},
                {"inline_data": {"mime_type": _guess_mime(image_path), "data": b64}},
            ]
            loop = asyncio.get_running_loop()
            def _call():
                return self._client.models.generate_content(
                    model=self.model,
                    contents=[{"parts": parts}],  # structured for inline data
                    config=types.GenerateContentConfig(temperature=self.temperature),  # type: ignore
                )
            resp = await loop.run_in_executor(None, _call)
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"Gemini image analysis failed: {exc}") from exc
        text = getattr(resp, "text", None)
        return (text or "").strip() or "MISSING VISION SUMMARY"


def _guess_mime(path: Path) -> str:
    ext = path.suffix.lower()
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }.get(ext, "image/png")


def get_llm(provider: str, model: str) -> BaseLLM:
    if provider == "dummy":
        return DummyEcho()
    if provider == "ollama":
        return OllamaLocal(model=model)
    if provider == "google":
        return GoogleGeminiClient(model=model)
    raise ValueError(f"Unsupported LLM provider: {provider}")

__all__ = [
    "BaseLLM",
    "DummyEcho",
    "OllamaLocal",
    "OllamaVision",
    "GoogleGeminiClient",
    "get_llm",
]
