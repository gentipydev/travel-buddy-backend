"""classify_gemini_error contract: surface-aware 403 messaging.

The HTML-403 case must report the right cause for the active LLM surface
(Vertex IAM/enablement vs Developer-API datacenter-IP block) and keep a tail
of the raw error instead of swallowing it. Non-403 errors pass through.
"""
from __future__ import annotations

from app.llm import gemini_client
from app.llm.gemini_client import GEMINI_ACCESS_DENIED, classify_gemini_error


class _Err(Exception):
    def __init__(self, msg, code=None):
        super().__init__(msg)
        self.code = code


def test_html_403_developer_mode_points_to_vertex(monkeypatch):
    monkeypatch.setattr(gemini_client.settings, "gemini_use_vertex", False)
    out = classify_gemini_error(_Err("403 <html>Error 403 (Forbidden)</html>", code=403))
    assert out.startswith(GEMINI_ACCESS_DENIED)
    assert "datacenter IP" in out
    assert "GEMINI_USE_VERTEX=true" in out


def test_html_403_vertex_mode_points_to_iam(monkeypatch):
    monkeypatch.setattr(gemini_client.settings, "gemini_use_vertex", True)
    out = classify_gemini_error(_Err("403 <html>Error 403 (Forbidden)</html>", code=403))
    assert out.startswith(GEMINI_ACCESS_DENIED)
    assert "Vertex AI returned 403" in out
    assert "GOOGLE_CLOUD_PROJECT" in out
    # raw tail preserved, not swallowed
    assert "raw:" in out


def test_non_403_passes_through_unchanged():
    out = classify_gemini_error(_Err("500 internal error", code=500))
    assert out == "500 internal error"
    assert GEMINI_ACCESS_DENIED not in out
