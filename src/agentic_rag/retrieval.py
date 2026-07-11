"""混合检索:jieba 分词 BM25 + 向量检索,RRF 融合。

向量检索擅长语义相近,但对编号、专名、精确词面(如 "NX-42")常有盲区;
BM25 恰好相反。RRF(Reciprocal Rank Fusion)按名次融合两路结果,无需调权。
"""
from langchain_core.documents import Document

_RRF_K = 60  # RRF 平滑常数,经验默认值


def _tokenize(text: str) -> list[str]:
    import jieba

    return [t for t in jieba.cut_for_search(text) if t.strip()]


def load_all_chunks(vector_store) -> list[Document]:
    """从 Chroma 取全量块(供 BM25 建索引)。"""
    data = vector_store.get()
    return [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(data["documents"], data["metadatas"])
    ]


class HybridRetriever:
    """鸭子类型兼容向量库接口(similarity_search),内部做 BM25+向量 RRF 融合。"""

    def __init__(self, vector_store, docs: list[Document], k_each: int = 20):
        from rank_bm25 import BM25Okapi

        self._vector_store = vector_store
        self._docs = docs
        self._k_each = k_each
        self._bm25 = BM25Okapi([_tokenize(d.page_content) for d in docs]) if docs else None

    def _bm25_search(self, query: str, k: int) -> list[Document]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self._docs[i] for i in ranked[:k] if scores[i] > 0]

    def similarity_search(self, query: str, k: int) -> list[Document]:
        vector_hits = self._vector_store.similarity_search(query, k=self._k_each)
        bm25_hits = self._bm25_search(query, k=self._k_each)

        fused: dict[str, tuple[float, Document]] = {}
        for hits in (vector_hits, bm25_hits):
            for rank, doc in enumerate(hits):
                key = doc.page_content
                score = 1.0 / (_RRF_K + rank + 1)
                prev_score = fused[key][0] if key in fused else 0.0
                fused[key] = (prev_score + score, doc)
        ranked = sorted(fused.values(), key=lambda pair: pair[0], reverse=True)
        return [doc for _, doc in ranked[:k]]
