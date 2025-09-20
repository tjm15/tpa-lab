from __future__ import annotations

import os
import pytest

from tpa.llm import get_llm, GoogleGeminiClient


def test_get_llm_google_instantiates_with_dummy_key(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "DUMMY_KEY")
    llm = get_llm("google", "gemini-2.5-pro")
    assert isinstance(llm, GoogleGeminiClient)
    assert llm.model == "gemini-2.5-pro"


def test_get_llm_google_missing_key(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        get_llm("google", "gemini-2.5-pro")
