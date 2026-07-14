from langchain_core.documents import Document

from agentic_rag.server.resources import RequestContext, ResourceRegistry


class FakeStore:
    """鸭子类型向量库:满足 similarity_search / get。"""

    def __init__(self, docs):
        self._docs = docs

    def similarity_search(self, query, k, filter=None):
        return self._docs[:k]

    def get(self):
        return {
            "ids": [f"id{i}" for i in range(len(self._docs))],
            "documents": [d.page_content for d in self._docs],
            "metadatas": [d.metadata for d in self._docs],
        }


class FakeLLM:
    def bind_tools(self, tools):
        return self  # 不实际调用,build_graph 只需一个可 invoke 的对象;此处仅验证构造


def _docs():
    return [
        Document(page_content="星尘咖啡馆招牌是拿铁", metadata={"source": "a.md", "headers": "", "access": "public"}),
        Document(page_content="供应商 NX-42 供应咖啡豆", metadata={"source": "b.md", "headers": "", "access": "internal"}),
    ]


def _registry():
    docs = _docs()
    reg = ResourceRegistry(embeddings=object(), llm=FakeLLM())
    reg._stores["c"] = FakeStore(docs)  # 直接注入假 store,跳过 Chroma 构造
    return reg, docs


def test_build_retriever_instances_are_isolated():
    reg, _ = _registry()
    r1 = reg.build_retriever("c", frozenset({"public", "internal"}))
    r2 = reg.build_retriever("c", frozenset({"public", "internal"}))
    assert r1 is not r2
    # 一个记录检索命中,不影响另一个(竞态隔离的核心保证)
    r1.similarity_search("拿铁", k=2)
    assert r1.take_recorded() != []
    assert r2.take_recorded() == []


def test_prebuilt_index_cached():
    reg, _ = _registry()
    a = reg.prebuilt_index("c")
    b = reg.prebuilt_index("c")
    assert a is b  # 同一 collection 复用


def test_live_chunk_ids_from_store():
    reg, docs = _registry()
    assert reg.live_chunk_ids("c") == {f"id{i}" for i in range(len(docs))}


def test_live_chunk_ids_returns_frozenset_copy():
    reg, _ = _registry()
    ids = reg.live_chunk_ids("c")
    assert isinstance(ids, frozenset)  # immutable — cannot corrupt the process-wide cache


def test_visible_index_skips_caching_when_prebuilt_replaced_during_build(monkeypatch):
    # I1 guard: if invalidate replaces the prebuilt during the unlocked build_bm25_index,
    # the stale per-role index must NOT be cached.
    import agentic_rag.server.resources as res
    reg, _ = _registry()
    allowed = frozenset({"public"})  # fewer visible than all -> takes the build path
    real_build = res.build_bm25_index

    def racing_build(visible):
        idx = real_build(visible)
        reg._prebuilt["c"] = object()  # simulate invalidate+rebuild landing mid-build
        return idx

    monkeypatch.setattr(res, "build_bm25_index", racing_build)
    reg._visible_index("c", allowed)
    assert ("c", allowed) not in reg._role_index  # stale index not cached
