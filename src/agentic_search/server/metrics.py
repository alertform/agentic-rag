"""Prometheus 指标 + 检索计量代理(边界埋点,核心模块不改)。"""
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)

REGISTRY = CollectorRegistry()

REQUEST_LATENCY = Histogram(
    "agentic_search_request_latency_seconds", "请求端到端延迟", ["endpoint"], registry=REGISTRY
)
ROUTE_TOTAL = Counter(
    "agentic_search_route_total", "检索通道路由计数", ["route"], registry=REGISTRY
)
CACHE_LOOKUP_TOTAL = Counter(
    "agentic_search_cache_lookup_total", "语义缓存查询结果计数", ["result"], registry=REGISTRY
)
RETRIEVE_CALLS_TOTAL = Counter(
    "agentic_search_retrieve_calls_total", "检索调用次数", registry=REGISTRY
)
WEB_SEARCH_CALLS_TOTAL = Counter(
    "agentic_search_web_search_calls_total", "Web 搜索调用次数", ["result"], registry=REGISTRY
)
TOKENS_PER_TURN = Histogram(
    "agentic_search_tokens_per_turn",
    "每轮生成 token 数(以流式 chunk 数近似)",
    registry=REGISTRY,
    buckets=(8, 16, 32, 64, 128, 256, 512, 1024),
)


def observe_latency(endpoint: str, seconds: float) -> None:
    REQUEST_LATENCY.labels(endpoint=endpoint).observe(seconds)


def record_cache(hit: bool) -> None:
    CACHE_LOOKUP_TOTAL.labels(result="hit" if hit else "miss").inc()


def record_route(route: str | None) -> None:
    if route:
        ROUTE_TOTAL.labels(route=route).inc()


def observe_tokens(n: int) -> None:
    TOKENS_PER_TURN.observe(n)


def record_web_search(ok: bool) -> None:
    WEB_SEARCH_CALLS_TOTAL.labels(result="ok" if ok else "error").inc()


def render() -> tuple[bytes, str]:
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST


class MeteredRetriever:
    """透明包装 HybridRetriever:每次检索记录 route 与调用计数。核心检索器零改动。"""

    def __init__(self, inner):
        self._inner = inner

    def similarity_search(self, query: str, k: int):
        hits = self._inner.similarity_search(query, k)
        RETRIEVE_CALLS_TOTAL.inc()
        record_route(self._inner.last_route)
        return hits

    def take_recorded(self):
        return self._inner.take_recorded()

    @property
    def last_route(self):
        return self._inner.last_route


class MeteredSearchBackend:
    """透明包装搜索后端:每次调用记录成功/失败计数。核心 search 模块零改动。"""

    def __init__(self, inner):
        self._inner = inner

    def search(self, query: str, max_results: int):
        try:
            results = self._inner.search(query, max_results)
        except Exception:
            record_web_search(False)
            raise
        record_web_search(True)
        return results
