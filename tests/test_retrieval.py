"""混合检索测试:BM25 词面命中兜底向量盲区,RRF 融合。零 Ollama 依赖。"""
from langchain_core.documents import Document

from agentic_rag.retrieval import HybridRetriever


def _doc(text, source="corpus.md", headers="目录"):
    return Document(page_content=text, metadata={"source": source, "headers": headers})


CORPUS = [
    _doc("星尘拿铁是招牌饮品,售价 32 元。"),
    _doc("供应商编号 NX-42 提供埃塞俄比亚咖啡豆,月供 200 公斤。"),
    _doc("每周三闭店进行设备维护。"),
    _doc("星环会员年费 199 元,全场饮品 88 折。"),
]


class StubVectorStore:
    """可控向量检索结果的假库。"""

    def __init__(self, results):
        self._results = results

    def similarity_search(self, query, k):
        return self._results[:k]


def test_bm25_rescues_exact_term_missed_by_vector():
    # 向量端假装完全没召回 NX-42 那条(模拟专名/编号的向量盲区)
    vector_only = [CORPUS[0], CORPUS[2], CORPUS[3]]
    retriever = HybridRetriever(StubVectorStore(vector_only), CORPUS)
    hits = retriever.similarity_search("NX-42 供应商", k=3)
    assert any("NX-42" in d.page_content for d in hits), "BM25 应捞回词面精确命中"


def test_vector_results_still_present():
    vector_only = [CORPUS[3]]
    retriever = HybridRetriever(StubVectorStore(vector_only), CORPUS)
    hits = retriever.similarity_search("会员权益", k=4)
    assert any("星环会员" in d.page_content for d in hits)


def test_k_truncation_and_dedup():
    # 向量端与 BM25 端命中同一文档时应去重,且结果数不超过 k
    retriever = HybridRetriever(StubVectorStore(list(CORPUS)), CORPUS)
    hits = retriever.similarity_search("星尘拿铁 招牌", k=2)
    assert len(hits) == 2
    texts = [d.page_content for d in hits]
    assert len(set(texts)) == len(texts), "融合结果不应有重复文档"


def test_routing_rare_token_enables_bm25():
    # "NX-42" 只在一篇文档出现(稀有在库词)→ 应走混合,BM25 独有命中要进结果
    vector_only = [CORPUS[0], CORPUS[2], CORPUS[3]]  # 向量端不返回 NX-42 那篇
    retriever = HybridRetriever(StubVectorStore(vector_only), CORPUS)
    assert retriever.should_use_bm25("NX-42 供应商是哪家?") is True
    hits = retriever.similarity_search("NX-42 供应商是哪家?", k=4)
    assert any("NX-42" in d.page_content for d in hits)
    assert retriever.last_route == "hybrid"


def _dense_corpus():
    # 8 篇文档都含"咖啡豆烘焙"(df=8 > 阈值 3 → 常见);仅 1 篇含 "NX-42"(稀有)
    docs = [_doc(f"咖啡豆烘焙工艺记录第{i}篇,火候与风味细节各不相同。") for i in range(7)]
    docs.append(_doc("咖啡豆烘焙供应商 NX-42 的到货记录。"))
    return docs


def test_routing_common_tokens_use_vector_only():
    corpus = _dense_corpus()
    expected = corpus[2]
    retriever = HybridRetriever(StubVectorStore([expected]), corpus)
    q = "咖啡豆烘焙"  # 全部词元 df=8,超过阈值 max(3, 1) → 不触发 BM25
    assert retriever.should_use_bm25(q) is False
    hits = retriever.similarity_search(q, k=4)
    assert [d.page_content for d in hits] == [expected.page_content], "纯向量路由不应混入 BM25 噪声"
    assert retriever.last_route == "vector"


def test_routing_rare_token_in_dense_corpus_enables_bm25():
    corpus = _dense_corpus()
    retriever = HybridRetriever(StubVectorStore([corpus[0]]), corpus)
    assert retriever.should_use_bm25("NX-42 到货了吗") is True  # NX df=1 ≤ 3


def test_routing_unknown_token_does_not_trigger():
    retriever = HybridRetriever(StubVectorStore(list(CORPUS)), CORPUS)
    # 词元完全不在语料(df=0)→ BM25 无能为力,不触发
    assert retriever.should_use_bm25("ZZZZ-9999") is False


def test_bm25_index_carries_df_stats():
    from agentic_rag.retrieval import build_bm25_index

    index = build_bm25_index(CORPUS)
    assert index.doc_count == len(CORPUS)
    assert index.df.get("NX") == 1, "NX 只出现在一篇文档"
    # 同一文档内重复词元只计一次 df
    assert all(v <= len(CORPUS) for v in index.df.values())


def test_old_pickle_without_df_treated_stale(tmp_path):
    import pickle

    from agentic_rag.retrieval import BM25Index, build_bm25_index, load_bm25_index

    index = build_bm25_index(CORPUS)
    legacy = BM25Index.__new__(BM25Index)  # 模拟旧版对象:无 df 字段
    legacy.docs = index.docs
    legacy.bm25 = index.bm25
    legacy.digest = index.digest
    path = tmp_path / "legacy.pkl"
    with open(path, "wb") as f:
        pickle.dump(legacy, f)
    assert load_bm25_index(path, expected_digest=index.digest) is None


def test_bm25_index_roundtrip_and_digest(tmp_path):
    from agentic_rag.retrieval import build_bm25_index, load_bm25_index, save_bm25_index

    index = build_bm25_index(CORPUS)
    path = tmp_path / "bm25.pkl"
    save_bm25_index(index, path)

    loaded = load_bm25_index(path, expected_digest=index.digest)
    assert loaded is not None
    # 持久化索引与现场构建的检索结果一致
    r1 = HybridRetriever(StubVectorStore([]), CORPUS)
    r2 = HybridRetriever(StubVectorStore([]), [], prebuilt=loaded)
    q = "NX-42 供应商"
    assert [d.page_content for d in r1.similarity_search(q, k=3)] == [
        d.page_content for d in r2.similarity_search(q, k=3)
    ]


def test_bm25_index_stale_digest_returns_none(tmp_path):
    from agentic_rag.retrieval import build_bm25_index, load_bm25_index, save_bm25_index

    index = build_bm25_index(CORPUS)
    path = tmp_path / "bm25.pkl"
    save_bm25_index(index, path)
    assert load_bm25_index(path, expected_digest="不匹配的摘要") is None
    assert load_bm25_index(tmp_path / "不存在.pkl", expected_digest=index.digest) is None


def test_take_recorded_accumulates_and_clears():
    retriever = HybridRetriever(StubVectorStore(list(CORPUS)), CORPUS)
    retriever.similarity_search("星尘拿铁", k=2)
    retriever.similarity_search("会员年费", k=2)
    recorded = retriever.take_recorded()
    assert len(recorded) == 4, "两轮检索的命中应累计记录"
    assert retriever.take_recorded() == [], "取走后应清空"


def test_both_channels_hit_ranks_first():
    # 双通道都命中的文档经 RRF 融合应排最前
    vector_results = [CORPUS[0], CORPUS[2]]
    retriever = HybridRetriever(StubVectorStore(vector_results), CORPUS)
    hits = retriever.similarity_search("星尘拿铁 售价", k=3)
    assert "星尘拿铁" in hits[0].page_content
