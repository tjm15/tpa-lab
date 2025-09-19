from __future__ import annotations

from abc import ABC, abstractmethod
import base64
import os
from pathlib import Path
from typing import Iterable, List, Optional

import httpx

# Optional google-genai imports (lazy)
_GENAI_READY = False
try:  # pragma: no cover - import guarded
    from google import genai  # type: ignore
    from google.genai import types  # type: ignore
    _GENAI_READY = True
except Exception:  # pragma: no cover - missing dependency
    pass

try:  # pragma: no cover - retry is optional but installed in deps now
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
except Exception:  # pragma: no cover
    # Fallback no-op decorators
    def retry(*args, **kwargs):
        def _wrap(fn):
            return fn
        return _wrap

    def stop_after_attempt(*a, **k):
        return None

    def wait_exponential(*a, **k):
        return None

    def retry_if_exception_type(*a, **k):
        return lambda e: True


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
    """Google Gemini 2.5 Pro integration (Developer API or Vertex AI).

    Supports:
    - Text chat completion via generate_content
    - Simple image + text analysis via analyse_image

    Streaming roadmap:
        google-genai exposes a streaming variant (e.g. client.models.generate_content with stream=True
        or a dedicated stream method). To integrate streaming in this project, refactor BaseLLM to
        optionally return an async iterator of tokens/chunks. This class could then provide a
        `stream_complete(messages)` method yielding partial text deltas. For now we collect full
        responses to minimise interface churn.

    Notes:
    - Streaming not yet exposed; future extension could wrap client.models.generate_content_stream.
    - Messages are flattened into a single ordered list of strings (system first) because
      google-genai Python SDK accepts a list[str] for simple text prompts.
    """

    def __init__(
        self,
        model: str = "models/gemini-2.5-pro",
        use_vertex: bool | None = None,
        project: Optional[str] = None,
        location: Optional[str] = None,
        temperature: float = 0.2,
        top_p: float = 0.9,
        max_output_tokens: int = 2200,
    ) -> None:
        if not _GENAI_READY:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "google-genai not installed. Install with: pip install google-genai tenacity"
            )
        self.model = model or "models/gemini-2.5-pro"
        self.temperature = temperature
        self.top_p = top_p
        self.max_output_tokens = max_output_tokens

        env_vertex = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in ("1", "true", "yes")
        use_vertex = env_vertex if use_vertex is None else use_vertex
        if use_vertex:
            project = project or os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("PROJECT_ID")
            location = location or os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
            if not project:
                raise RuntimeError("GOOGLE_CLOUD_PROJECT required for Vertex AI mode")
            self.client = genai.Client(vertexai=True, project=project, location=location)
        else:
            api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise RuntimeError("Set GOOGLE_API_KEY (or GEMINI_API_KEY) for google provider")
            self.client = genai.Client(api_key=api_key)

    def _messages_to_contents(self, messages: List[dict[str, str]]) -> List[str]:
        parts: List[str] = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                parts.append(f"SYSTEM:\n{content}".strip())
            else:
                parts.append(content)
        return parts

    @retry(  # pragma: no cover - external service, retried on transient errors
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=1, max=8),
        retry=retry_if_exception_type(Exception),
    )
    async def complete(self, messages: list[dict[str, str]]) -> str:  # type: ignore[override]
        contents = self._messages_to_contents(messages)
        # google-genai client is synchronous; run in thread if needed.
        # Simplicity: call directly (SDK does network I/O) – acceptable in async context for now.
        try:
            resp = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_output_tokens=self.max_output_tokens,
                ),
            )
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"Gemini API error: {exc}") from exc
        text = getattr(resp, "text", None) or ""
        return (text or "").strip()

    @retry(  # pragma: no cover - network
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=1, max=8),
        retry=retry_if_exception_type(Exception),
    )
    async def analyse_image(self, prompt: str, image_path: Path, mime_type: Optional[str] = None) -> str:
        mime_type = mime_type or _guess_mime(image_path)
        img_bytes = image_path.read_bytes()
        inline = types.Part(
            inline_data=types.Blob(mime_type=mime_type, data=img_bytes)
        )
        parts = [prompt, inline]
        try:
            resp = self.client.models.generate_content(
                model=self.model,
                contents=parts,
                config=types.GenerateContentConfig(
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_output_tokens=self.max_output_tokens,
                ),
            )
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"Gemini vision API error: {exc}") from exc
        return (getattr(resp, "text", None) or "").strip() or "MISSING VISION SUMMARY"


def _guess_mime(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in {".png"}:  # fast path
        return "image/png"
    if ext in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if ext in {".webp"}:
        return "image/webp"
    if ext in {".gif"}:
        return "image/gif"
    return "application/octet-stream"


def get_llm(provider: str, model: str) -> BaseLLM:
    if provider == "dummy":
        return DummyEcho()
    if provider == "ollama":
        return OllamaLocal(model=model)
    if provider == "google":
        # Accept both raw model ids and short names; if user passes 'gemini-2.5-pro' prepend models/.
        if not model:
            model = "models/gemini-2.5-pro"
        elif not model.startswith("models/"):
            model = f"models/{model}" if model.startswith("gemini") else model
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
