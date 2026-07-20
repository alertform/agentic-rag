"""混合检索:jieba 分词 BM25 + 向量检索,RRF 融合。

向量检索擅长语义相近,但对编号、专名、精确词面(如 "NX-42")常有盲区;
BM25 恰好相反。RRF(Reciprocal Rank Fusion)按名次融合两路结果,无需调权。
"""
from langchain_core.documents import Document

from agentic_search import config

_RRF_K = 60  # RRF 平滑常数,经验默认值


import re

_WORD_RE = re.compile(r"\w")  # 字母/数字/汉字;纯标点词元会污染 BM25 与 df 路由判据


def _tokenize(text: str) -> list[str]:
    import jieba

    return [t for t in jieba.cut_for_search(text) if _WORD_RE.search(t)]


def load_all_chunks(vector_store) -> list[Document]:
    """从 Chroma 取全量块(供 BM25 建索引)。"""
    data = vector_store.get()
    return [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(data["documents"], data["metadatas"])
    ]


class BM25Index:
    """预构建的 BM25 索引:分词与建索引在 ingest 时一次完成,启动时直接加载。

    数万块语料下,每次 chat 启动全量 jieba 分词重建要分钟级——持久化后秒开。
    digest 是排序后块内容集合的 sha256,用于校验索引与向量库是否同步。
    df(词元 → 出现文档数)供查询特征路由判断词元稀有度。
    """

    def __init__(self, docs: list[Document], bm25, digest: str, df: dict, doc_count: int):
        self.docs = docs
        self.bm25 = bm25
        self.digest = digest
        self.df = df
        self.doc_count = doc_count


def corpus_digest(docs: list[Document]) -> str:
    """排序后块内容集合的摘要,用于校验 BM25 持久化索引与向量库同步。"""
    import hashlib

    hasher = hashlib.sha256()
    for content in sorted(d.page_content for d in docs):
        hasher.update(content.encode("utf-8"))
    return hasher.hexdigest()[:32]


def build_bm25_index(docs: list[Document]) -> BM25Index:
    from collections import Counter

    from rank_bm25 import BM25Okapi

    tokenized = [_tokenize(d.page_content) for d in docs]
    bm25 = BM25Okapi(tokenized) if docs else None
    df = Counter(tok for tokens in tokenized for tok in set(tokens))
    return BM25Index(docs, bm25, corpus_digest(docs), dict(df), len(docs))


def save_bm25_index(index: BM25Index, path) -> None:
    import pickle
    from pathlib import Path

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(index, f)


def load_bm25_index(path, expected_digest: str) -> BM25Index | None:
    """加载持久化索引;文件缺失或摘要与向量库不同步时返回 None(调用方重建)。"""
    import pickle
    from pathlib import Path

    if not Path(path).is_file():
        return None
    try:
        with open(path, "rb") as f:
            index = pickle.load(f)
    except Exception:
        return None
    if index.digest != expected_digest:
        return None
    if not hasattr(index, "df"):  # 旧版索引缺 df 统计,视为过期重建
        return None
    return index


class HybridRetriever:
    """鸭子类型兼容向量库接口(similarity_search),内部做 BM25+向量 RRF 融合。

    allowed_access 非 None 时启用 ACL 过滤:BM25 建索引前预过滤,
    向量通道透传 Chroma metadata filter——两条通道都过滤,不留旁路。
    """

    def __init__(
        self,
        vector_store,
        docs: list[Document],
        k_each: int = 20,
        allowed_access: set[str] | None = None,
        prebuilt: "BM25Index | None" = None,
    ):
        self._vector_store = vector_store
        self._allowed = allowed_access
        if prebuilt is not None:
            docs = prebuilt.docs
        if allowed_access is not None:
            filtered = [
                d for d in docs
                if d.metadata.get("access", "public") in allowed_access
            ]
        else:
            filtered = docs
        if prebuilt is not None and len(filtered) == len(prebuilt.docs):
            # 过滤未剔除任何块(如 manager 全可见)→ 直接复用持久化索引
            self._index = prebuilt
        else:
            # 受限角色需在可见子集上重建(数万块 + 受限角色场景可再做每角色索引)
            self._index = build_bm25_index(filtered)
        self._k_each = k_each
        self._recorded: list[Document] = []
        self.last_route: str | None = None

    def take_recorded(self) -> list[Document]:
        """取走并清空自上次调用以来的全部检索命中(供语义缓存记录来源块)。"""
        recorded, self._recorded = self._recorded, []
        return recorded

    def _bm25_search(self, query: str, k: int) -> list[Document]:
        if self._index.bm25 is None:
            return []
        scores = self._index.bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self._index.docs[i] for i in ranked[:k] if scores[i] > 0]

    def should_use_bm25(self, query: str) -> bool:
        """查询特征路由:查询含"稀有但在库"词元才启用 BM25 通道。

        依据(规模实验):稀有实体词上 BM25 是召回救星;术语密集语料的常见词
        查询上 BM25 反而注噪并拖延迟。阈值随语料规模自适应。
        """
        from math import ceil

        index = self._index
        if index.bm25 is None:
            return False
        threshold = max(
            config.ROUTE_RARE_DF_ABS, ceil(config.ROUTE_RARE_DF_RATIO * index.doc_count)
        )
        return any(0 < index.df.get(tok, 0) <= threshold for tok in _tokenize(query))

    def similarity_search(self, query: str, k: int) -> list[Document]:
        if self._allowed is None:
            vector_hits = self._vector_store.similarity_search(query, k=self._k_each)
        else:
            vector_hits = self._vector_store.similarity_search(
                query,
                k=self._k_each,
                filter={"access": {"$in": sorted(self._allowed)}},
            )
        if not self.should_use_bm25(query):
            self.last_route = "vector"
            hits = vector_hits[:k]
            self._recorded.extend(hits)
            return hits
        self.last_route = "hybrid"
        bm25_hits = self._bm25_search(query, k=self._k_each)

        fused: dict[str, tuple[float, Document]] = {}
        for hits in (vector_hits, bm25_hits):
            for rank, doc in enumerate(hits):
                key = doc.page_content
                score = 1.0 / (_RRF_K + rank + 1)
                prev_score = fused[key][0] if key in fused else 0.0
                fused[key] = (prev_score + score, doc)
        ranked = sorted(fused.values(), key=lambda pair: pair[0], reverse=True)
        hits = [doc for _, doc in ranked[:k]]
        self._recorded.extend(hits)
        return hits
