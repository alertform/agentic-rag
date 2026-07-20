"""FAQ 候选导出与缓存命中计数测试。零模型依赖。"""
from langchain_core.embeddings import DeterministicFakeEmbedding

from agentic_search.cache import SemanticCache
from agentic_search.faq import export_candidates


def _cache(tmp_path):
    return SemanticCache(
        embeddings=DeterministicFakeEmbedding(size=64),
        persist_directory=str(tmp_path / "cache"),
    )


def _store(cache, question, answer, access="public"):
    cache.store(
        question=question,
        answer=answer,
        sources=["menu.md"],
        chunk_ids=["chunk-1"],
        access_levels=[access],
    )


def test_hit_count_increments_only_on_served_hits(tmp_path):
    cache = _cache(tmp_path)
    _store(cache, "拿铁多少钱?", "32 元。", access="internal")

    # ACL 拦截的查询不计数
    cache.lookup("拿铁多少钱?", live_chunk_ids={"chunk-1"}, allowed_access={"public"})
    assert cache.entries()[0]["hit_count"] == 0

    # 正常命中两次 → 计数 2
    for _ in range(2):
        assert (
            cache.lookup(
                "拿铁多少钱?", live_chunk_ids={"chunk-1"}, allowed_access={"internal"}
            )
            is not None
        )
    assert cache.entries()[0]["hit_count"] == 2


def test_export_filters_by_min_hits(tmp_path):
    cache = _cache(tmp_path)
    _store(cache, "拿铁多少钱?", "32 元。")
    _store(cache, "周三营业吗?", "周三闭店维护。")
    for _ in range(2):
        cache.lookup("拿铁多少钱?", live_chunk_ids={"chunk-1"}, allowed_access={"public"})

    md = export_candidates(cache, min_hits=2)
    assert "## 拿铁多少钱?" in md
    assert "32 元。" in md
    assert "menu.md" in md
    assert "周三营业吗?" not in md, "命中不足的条目不应导出"
    assert "人工审核" in md


def test_export_empty_gives_hint(tmp_path):
    cache = _cache(tmp_path)
    md = export_candidates(cache, min_hits=2)
    assert "暂无" in md
