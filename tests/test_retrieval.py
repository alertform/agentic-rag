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
