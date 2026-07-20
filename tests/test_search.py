import httpx
import pytest

from agentic_search import config
from agentic_search.search import (
    RecordingBackend,
    SearchResult,
    TavilyBackend,
    make_search_backend,
)


class FakeResponse:
    def __init__(self, payload, status_error=None):
        self._payload = payload
        self._status_error = status_error

    def raise_for_status(self):
        if self._status_error is not None:
            raise self._status_error

    def json(self):
        return self._payload


def test_tavily_backend_parses_results(monkeypatch):
    seen = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        seen.update(url=url, headers=headers, json=json, timeout=timeout)
        return FakeResponse(
            {
                "results": [
                    {"title": "T1", "url": "https://a", "content": "正文A", "score": 0.9},
                    {"title": "", "url": "https://b", "content": "正文B"},
                ]
            }
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    results = TavilyBackend("key-123").search("北京天气", max_results=3)
    assert results == [
        SearchResult(title="T1", url="https://a", content="正文A", score=0.9),
        SearchResult(title="", url="https://b", content="正文B", score=0.0),
    ]
    assert seen["headers"]["Authorization"] == "Bearer key-123"
    assert seen["json"] == {"query": "北京天气", "max_results": 3}
    assert seen["timeout"] == config.WEB_SEARCH_TIMEOUT


def test_tavily_backend_raises_on_http_error(monkeypatch):
    err = httpx.HTTPStatusError("429", request=None, response=None)
    monkeypatch.setattr(httpx, "post", lambda *a, **kw: FakeResponse({}, status_error=err))
    with pytest.raises(httpx.HTTPStatusError):
        TavilyBackend("k").search("q", max_results=1)


def test_make_search_backend_none_without_key(monkeypatch):
    monkeypatch.setattr(config, "TAVILY_API_KEY", "")
    assert make_search_backend() is None


def test_make_search_backend_tavily_with_key(monkeypatch):
    monkeypatch.setattr(config, "TAVILY_API_KEY", "k")
    assert isinstance(make_search_backend(), TavilyBackend)


class StubBackend:
    def __init__(self, results):
        self._results = results

    def search(self, query, max_results):
        return self._results[:max_results]


def test_recording_backend_records_and_clears():
    results = [SearchResult("t", "https://a", "c")]
    rec = RecordingBackend(StubBackend(results))
    assert rec.search("q", max_results=5) == results
    assert rec.take_recorded() == results
    assert rec.take_recorded() == []  # 取走即清空
