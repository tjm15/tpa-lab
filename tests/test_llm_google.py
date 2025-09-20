from __future__ import annotations

import os

import pytest

from tpa.llm import get_llm, GoogleGeminiClient


def test_get_llm_google(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "DUMMY_KEY")
    llm = get_llm("google", "gemini-2.5-pro")
    assert isinstance(llm, GoogleGeminiClient)
    # Ensure complete() handles empty prompt gracefully (will raise due to dummy key) but we don't call network.
    # We won't invoke complete() to avoid network; just check attributes.
    assert llm.model == "gemini-2.5-pro"
