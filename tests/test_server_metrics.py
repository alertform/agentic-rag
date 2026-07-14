from agentic_rag.server import metrics


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
    assert b"agentic_rag_request_latency_seconds" in payload
    assert "text/plain" in content_type
