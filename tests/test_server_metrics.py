from agentic_search.server import metrics


class _Inner:
    def __init__(self):
        self.last_route = "hybrid"
        self._recorded = ["doc"]

    def similarity_search(self, query, k):
        return ["hit"]

    def take_recorded(self):
        r, self._recorded = self._recorded, []
        return r


def _value(counter, **labels):
    return counter.labels(**labels)._value.get()


def test_record_cache_increments():
    before = _value(metrics.CACHE_LOOKUP_TOTAL, result="hit")
    metrics.record_cache(True)
    assert _value(metrics.CACHE_LOOKUP_TOTAL, result="hit") == before + 1


def test_metered_retriever_counts_route_and_calls():
    inner = _Inner()
    metered = metrics.MeteredRetriever(inner)
    calls_before = metrics.RETRIEVE_CALLS_TOTAL._value.get()
    route_before = _value(metrics.ROUTE_TOTAL, route="hybrid")
    assert metered.similarity_search("q", k=5) == ["hit"]
    assert metrics.RETRIEVE_CALLS_TOTAL._value.get() == calls_before + 1
    assert _value(metrics.ROUTE_TOTAL, route="hybrid") == route_before + 1
    assert metered.take_recorded() == ["doc"]
    assert metered.last_route == "hybrid"


def test_render_returns_prometheus_text():
    payload, content_type = metrics.render()
    assert b"agentic_search_request_latency_seconds" in payload
    assert "text/plain" in content_type


def test_observe_latency_increments_histogram():
    before = metrics.REQUEST_LATENCY.labels(endpoint="chat")._sum.get()
    metrics.observe_latency("chat", 0.5)
    assert metrics.REQUEST_LATENCY.labels(endpoint="chat")._sum.get() == before + 0.5


def test_observe_tokens_increments_histogram():
    before = metrics.TOKENS_PER_TURN._sum.get()
    metrics.observe_tokens(42)
    assert metrics.TOKENS_PER_TURN._sum.get() == before + 42


def test_record_route_ignores_falsy():
    before = _value(metrics.ROUTE_TOTAL, route="vector")
    metrics.record_route(None)
    metrics.record_route("")
    assert _value(metrics.ROUTE_TOTAL, route="vector") == before


def test_metered_search_backend_counts_ok_and_error():
    import pytest

    class Ok:
        def search(self, query, max_results):
            return ["r"]

    class Bad:
        def search(self, query, max_results):
            raise RuntimeError("boom")

    ok_before = _value(metrics.WEB_SEARCH_CALLS_TOTAL, result="ok")
    assert metrics.MeteredSearchBackend(Ok()).search("q", 3) == ["r"]
    assert _value(metrics.WEB_SEARCH_CALLS_TOTAL, result="ok") == ok_before + 1

    err_before = _value(metrics.WEB_SEARCH_CALLS_TOTAL, result="error")
    with pytest.raises(RuntimeError):
        metrics.MeteredSearchBackend(Bad()).search("q", 3)
    assert _value(metrics.WEB_SEARCH_CALLS_TOTAL, result="error") == err_before + 1
