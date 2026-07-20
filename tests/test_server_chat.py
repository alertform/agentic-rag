import json

from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.messages import AIMessageChunk

from agentic_search.server.app import create_app
from agentic_search.server.resources import RequestContext


class StubRetriever:
    def __init__(self, docs):
        self._recorded = []
        self._docs = docs
        self.last_route = "vector"

    def take_recorded(self):
        r, self._recorded = self._recorded, []
        return r


class StubGraph:
    def __init__(self, tokens, retriever=None, docs=None):
        self._tokens = tokens
        self._retriever = retriever
        self._docs = docs or []
        self.updated = []
        self.seen_config = None

    async def astream(self, inputs, config, stream_mode=None):
        self.seen_config = config
        if self._retriever is not None:
            self._retriever._recorded.extend(self._docs)  # 模拟工具检索命中记录
        for t in self._tokens:
            yield AIMessageChunk(content=t), {"langgraph_node": "agent"}

    def update_state(self, config, values):
        self.updated.append((config, values))


class StubCache:
    def __init__(self, hit=None):
        self._hit = hit
        self.stored = []

    def lookup(self, question, live_ids, allowed):
        return self._hit

    def store(self, **kwargs):
        self.stored.append(kwargs)


class FakeRegistry:
    def __init__(self, ctx_builder):
        self._ctx_builder = ctx_builder
        self.contexts = []

    def build_context(self, collection, role):
        ctx = self._ctx_builder(collection, role)
        self.contexts.append(ctx)
        return ctx


def _parse_events(text):
    events = []
    for block in text.strip().split("\n\n"):
        event, data = None, None
        for line in block.splitlines():
            if line.startswith("event:"):
                event = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data = line[len("data:"):].strip()
        if event:
            events.append((event, json.loads(data) if data else None))
    return events


def _doc():
    return Document(page_content="拿铁", metadata={"source": "a.md", "headers": "", "access": "public"})


def test_chat_streams_tokens_and_done_with_sources():
    retriever = StubRetriever([])
    graph = StubGraph(["星", "尘", "拿铁"], retriever=retriever, docs=[_doc()])
    cache = StubCache(hit=None)

    def build(collection, role):
        return RequestContext(graph, retriever, cache, {"id0"}, frozenset({"public"}))

    app = create_app(registry=FakeRegistry(build))
    with TestClient(app) as client:
        resp = client.post("/chat", json={"question": "招牌?", "thread_id": "t1"})
        assert resp.status_code == 200
        events = _parse_events(resp.text)
        tokens = [d["text"] for e, d in events if e == "token"]
        assert tokens == ["星", "尘", "拿铁"]
        done = [d for e, d in events if e == "done"][0]
        assert done["sources"] == ["a.md"]
        assert done["cache_hit"] is False
        assert done["route"] == "vector"
        assert done["request_id"]
    assert cache.stored  # 有命中块 + 有答案 → 写缓存


class StubWebRecorder:
    def __init__(self, results):
        self._results = results

    def take_recorded(self):
        r, self._results = self._results, []
        return r


def test_chat_web_answer_reports_web_sources_and_skips_cache():
    from agentic_search.search import SearchResult

    retriever = StubRetriever([])
    graph = StubGraph(["北京晴"], retriever=retriever, docs=[_doc()])
    cache = StubCache(hit=None)
    web = StubWebRecorder([SearchResult("天气", "https://w", "晴")])

    def build(collection, role):
        return RequestContext(
            graph, retriever, cache, {"id0"}, frozenset({"public"}), web_recorder=web
        )

    app = create_app(registry=FakeRegistry(build))
    with TestClient(app) as client:
        resp = client.post("/chat", json={"question": "北京天气?", "thread_id": "t1"})
        events = _parse_events(resp.text)
        done = [d for e, d in events if e == "done"][0]
        assert done["web_sources"] == ["https://w"]
        assert done["sources"] == ["a.md"]
    assert cache.stored == []  # 缓存写侧 gate:用过 web 的回答不缓存


def test_chat_cache_hit_short_circuits():
    from agentic_search.cache import CacheHit

    retriever = StubRetriever([])
    graph = StubGraph(["不应产生"], retriever=retriever)
    cache = StubCache(hit=CacheHit(question="招牌?", answer="拿铁", sources=["a.md"]))

    def build(collection, role):
        return RequestContext(graph, retriever, cache, set(), frozenset({"public"}))

    app = create_app(registry=FakeRegistry(build))
    with TestClient(app) as client:
        resp = client.post("/chat", json={"question": "招牌?", "thread_id": "t1"})
        events = _parse_events(resp.text)
        done = [d for e, d in events if e == "done"][0]
        assert done["cache_hit"] is True
        assert [d["text"] for e, d in events if e == "token"] == ["拿铁"]
    assert graph.updated  # 命中答案注入历史
    assert graph.seen_config is None  # 未走 astream


def test_chat_passes_thread_id_to_graph():
    retriever = StubRetriever([])
    graph = StubGraph(["x"], retriever=retriever)

    def build(collection, role):
        return RequestContext(graph, retriever, StubCache(), set(), frozenset({"public"}))

    app = create_app(registry=FakeRegistry(build))
    with TestClient(app) as client:
        client.post("/chat", json={"question": "hi", "thread_id": "sess-42"})
    assert graph.seen_config["configurable"]["thread_id"] == "sess-42"


def test_chat_rejects_unknown_role():
    def build(collection, role):
        return RequestContext(StubGraph([]), StubRetriever([]), StubCache(), set(), frozenset())

    app = create_app(registry=FakeRegistry(build))
    with TestClient(app) as client:
        resp = client.post("/chat", json={"question": "hi", "thread_id": "t", "role": "ceo"})
        assert resp.status_code == 422
