"""语义缓存测试:命中/失效/ACL 防泄漏。零 Ollama 依赖。"""
from langchain_core.embeddings import DeterministicFakeEmbedding

from agentic_search.cache import SemanticCache


def _cache(tmp_path):
    return SemanticCache(
        embeddings=DeterministicFakeEmbedding(size=64),
        persist_directory=str(tmp_path / "cache"),
    )


def _store_menu_answer(cache):
    cache.store(
        question="星尘拿铁多少钱?",
        answer="星尘拿铁售价 32 元。",
        sources=["menu.md"],
        chunk_ids=["chunk-menu-1"],
        access_levels=["public"],
    )


def test_same_question_hits(tmp_path):
    cache = _cache(tmp_path)
    _store_menu_answer(cache)
    hit = cache.lookup(
        "星尘拿铁多少钱?", live_chunk_ids={"chunk-menu-1"}, allowed_access={"public"}
    )
    assert hit is not None
    assert hit.answer == "星尘拿铁售价 32 元。"
    assert hit.sources == ["menu.md"]


def test_unrelated_question_misses(tmp_path):
    cache = _cache(tmp_path)
    _store_menu_answer(cache)
    hit = cache.lookup(
        "会员积分什么时候过期?", live_chunk_ids={"chunk-menu-1"}, allowed_access={"public"}
    )
    assert hit is None


def test_stale_chunk_invalidates_and_evicts(tmp_path):
    cache = _cache(tmp_path)
    _store_menu_answer(cache)
    # 源块 id 已不在主库(文档变更) → 失效且条目被清
    hit = cache.lookup(
        "星尘拿铁多少钱?", live_chunk_ids={"chunk-menu-NEW"}, allowed_access={"public"}
    )
    assert hit is None
    # 即使块 id 恢复,条目已被清除,仍不命中
    hit2 = cache.lookup(
        "星尘拿铁多少钱?", live_chunk_ids={"chunk-menu-1"}, allowed_access={"public"}
    )
    assert hit2 is None


def test_acl_blocks_but_keeps_entry(tmp_path):
    cache = _cache(tmp_path)
    cache.store(
        question="供应商 NX-42 供应什么?",
        answer="NX-42 供应埃塞俄比亚咖啡豆。",
        sources=["suppliers.pdf"],
        chunk_ids=["chunk-sup-1"],
        access_levels=["internal"],
    )
    # public 角色不可见 internal 派生的缓存 → 不命中但条目保留
    assert (
        cache.lookup(
            "供应商 NX-42 供应什么?",
            live_chunk_ids={"chunk-sup-1"},
            allowed_access={"public"},
        )
        is None
    )
    # staff 可见 → 命中
    hit = cache.lookup(
        "供应商 NX-42 供应什么?",
        live_chunk_ids={"chunk-sup-1"},
        allowed_access={"public", "internal"},
    )
    assert hit is not None
    assert "埃塞俄比亚" in hit.answer
