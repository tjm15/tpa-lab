from __future__ import annotations

import os
import pytest

from tpa.llm import get_llm, GoogleGeminiClient


def test_factory_returns_google_client(monkeypatch):
    # Skip if key not set (we only test instantiation when available)
    key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not key:
        pytest.skip("GOOGLE_API_KEY not set; skipping Google client instantiation test")
    llm = get_llm("google", "gemini-2.5-pro")
    assert isinstance(llm, GoogleGeminiClient)


def test_factory_google_missing_key(monkeypatch):
    # Force missing key scenario
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        get_llm("google", "gemini-2.5-pro")
