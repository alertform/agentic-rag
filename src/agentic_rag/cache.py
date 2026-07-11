"""QA 语义缓存:相似问题复用已验证答案。

三条安全边界(缺一不可):
1. 独立 collection,**不进检索池**——模型生成内容不能无审核地成为检索来源
2. 哈希失效:条目记录答案引用的 chunk_id 集,任一 id 已不在主库(源文档变更)即作废
3. ACL 防泄漏:条目记录源块 access 级别集,提问者可见范围不覆盖时不命中
"""
import uuid
from typing import NamedTuple

from agentic_rag import config


class CacheHit(NamedTuple):
    question: str
    answer: str
    sources: list[str]


class SemanticCache:
    def __init__(self, embeddings, persist_directory: str):
        from langchain_chroma import Chroma

        self._store = Chroma(
            collection_name=config.CACHE_COLLECTION,
            embedding_function=embeddings,
            persist_directory=persist_directory,
            collection_metadata={"hnsw:space": "cosine"},
        )

    def store(
        self,
        question: str,
        answer: str,
        sources: list[str],
        chunk_ids: list[str],
        access_levels: list[str],
    ) -> None:
        entry_id = uuid.uuid4().hex
        self._store.add_texts(
            [question],
            metadatas=[
                {
                    "entry_id": entry_id,
                    "answer": answer,
                    "sources": ",".join(sorted(set(sources))),
                    "chunk_ids": ",".join(sorted(set(chunk_ids))),
                    "access_levels": ",".join(sorted(set(access_levels))),
                    "hit_count": 0,
                }
            ],
            ids=[entry_id],
        )

    def entries(self) -> list[dict]:
        """全部缓存条目(供 FAQ 候选导出等观测用途)。"""
        data = self._store.get()
        return [
            {
                "question": text,
                "answer": meta["answer"],
                "sources": meta["sources"].split(",") if meta["sources"] else [],
                "hit_count": meta.get("hit_count", 0),
                "access_levels": meta["access_levels"].split(","),
            }
            for text, meta in zip(data["documents"], data["metadatas"])
        ]

    def lookup(
        self,
        question: str,
        live_chunk_ids: set[str],
        allowed_access: set[str],
    ) -> CacheHit | None:
        results = self._store.similarity_search_with_score(question, k=1)
        if not results:
            return None
        doc, distance = results[0]
        if distance > config.CACHE_DISTANCE_THRESHOLD:
            return None

        meta = doc.metadata
        referenced = set(meta["chunk_ids"].split(","))
        if not referenced <= live_chunk_ids:
            # 源文档已变更:条目永久作废,清除
            stale = self._store.get(where={"chunk_ids": meta["chunk_ids"]})
            if stale["ids"]:
                self._store.delete(ids=stale["ids"])
            return None

        levels = set(meta["access_levels"].split(","))
        if not levels <= allowed_access:
            # 权限不足:对该提问者不命中,但条目对高权限角色仍有效,保留
            return None

        # 命中计数(仅 metadata 更新,不重嵌入;FAQ 沉淀以此筛高频候选)
        if "entry_id" in meta:
            self._store._collection.update(
                ids=[meta["entry_id"]],
                metadatas=[{**meta, "hit_count": meta.get("hit_count", 0) + 1}],
            )

        return CacheHit(
            question=doc.page_content,
            answer=meta["answer"],
            sources=meta["sources"].split(",") if meta["sources"] else [],
        )
