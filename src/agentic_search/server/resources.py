"""进程级资源缓存 + 每请求轻量包装:并发安全的检索/图构造。

昂贵只读对象(embeddings/llm 客户端、每 collection 的 Chroma store 与 BM25 索引、
每 (collection, role) 可见子集索引)进程内缓存一次;每请求廉价构造独立的
HybridRetriever 包装 + retrieve 工具 + graph,使 _recorded/last_route 落在每请求
对象上——并发竞态结构性消失。
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

from langgraph.checkpoint.memory import MemorySaver

from agentic_search import config
from agentic_search.cache import SemanticCache
from agentic_search.graph import SYSTEM_PROMPT, SYSTEM_PROMPT_WEB, build_graph
from agentic_search.retrieval import (
    HybridRetriever,
    build_bm25_index,
    corpus_digest,
    load_all_chunks,
    load_bm25_index,
)
from agentic_search.search import RecordingBackend, make_search_backend
from agentic_search.server.metrics import MeteredRetriever, MeteredSearchBackend
from agentic_search.tools import make_retrieve_tool, make_web_search_tool


@dataclass
class RequestContext:
    graph: object
    retriever: MeteredRetriever
    cache: SemanticCache
    live_chunk_ids: frozenset[str]
    allowed_access: frozenset[str]
    web_recorder: RecordingBackend | None = None


class ResourceRegistry:
    def __init__(
        self,
        *,
        chroma_dir=None,
        embeddings=None,
        llm=None,
        checkpointer=None,
        role_index_cache_size: int = 8,
        search_backend=None,
    ):
        self._chroma_dir = Path(chroma_dir) if chroma_dir else config.CHROMA_DIR
        self._embeddings = embeddings
        self._llm = llm
        self._checkpointer = checkpointer or MemorySaver()
        # 注入优先(测试用 fake);未注入则走工厂,TAVILY_API_KEY 缺失时为 None → 纯 RAG
        self._search_backend = search_backend if search_backend is not None else make_search_backend()
        self._cache: SemanticCache | None = None
        self._stores: dict = {}
        self._prebuilt: dict = {}
        self._live_ids: dict = {}
        self._role_index: OrderedDict = OrderedDict()
        self._role_index_cap = role_index_cache_size
        # RLock: prebuilt_index()/live_chunk_ids() call store() while already
        # holding this lock (same thread) — a plain Lock would self-deadlock.
        self._lock = threading.RLock()

    @property
    def embeddings(self):
        if self._embeddings is None:
            from agentic_search.llm import make_embeddings

            self._embeddings = make_embeddings()
        return self._embeddings

    @property
    def llm(self):
        if self._llm is None:
            from agentic_search.llm import make_chat_llm

            self._llm = make_chat_llm()
        return self._llm

    @property
    def cache(self) -> SemanticCache:
        if self._cache is None:
            self._cache = SemanticCache(
                self.embeddings, persist_directory=str(self._chroma_dir)
            )
        return self._cache

    def store(self, collection: str):
        with self._lock:
            if collection not in self._stores:
                from langchain_chroma import Chroma

                self._stores[collection] = Chroma(
                    collection_name=collection,
                    embedding_function=self.embeddings,
                    persist_directory=str(self._chroma_dir),
                )
            return self._stores[collection]

    def _bm25_path(self, collection: str) -> Path:
        return self._chroma_dir / f"bm25_{collection}.pkl"

    def prebuilt_index(self, collection: str):
        with self._lock:
            if collection not in self._prebuilt:
                chunks = load_all_chunks(self.store(collection))
                idx = load_bm25_index(self._bm25_path(collection), corpus_digest(chunks))
                if idx is None:
                    idx = build_bm25_index(chunks)
                self._prebuilt[collection] = idx
            return self._prebuilt[collection]

    def live_chunk_ids(self, collection: str) -> frozenset[str]:
        with self._lock:
            if collection not in self._live_ids:
                self._live_ids[collection] = set(self.store(collection).get()["ids"])
            return frozenset(self._live_ids[collection])

    def _visible_index(self, collection: str, allowed: frozenset[str]):
        prebuilt = self.prebuilt_index(collection)
        visible = [
            d for d in prebuilt.docs if d.metadata.get("access", "public") in allowed
        ]
        if len(visible) == len(prebuilt.docs):
            return prebuilt  # 全可见:直接复用持久化索引,零重建
        key = (collection, allowed)
        with self._lock:
            if key in self._role_index:
                self._role_index.move_to_end(key)
                return self._role_index[key]
        idx = build_bm25_index(visible)
        with self._lock:
            # 仅当派生所依据的 prebuilt 仍是当前值时才缓存:invalidate 可能在无锁构建期间
            # 清除/替换了它,此时 idx 基于过期文档集,缓存会让受限角色被服务陈旧的 BM25 索引。
            if self._prebuilt.get(collection) is prebuilt:
                self._role_index[key] = idx
                self._role_index.move_to_end(key)
                while len(self._role_index) > self._role_index_cap:
                    self._role_index.popitem(last=False)
        return idx

    def build_retriever(self, collection: str, allowed: frozenset[str]) -> MeteredRetriever:
        prebuilt = self._visible_index(collection, allowed)
        inner = HybridRetriever(
            self.store(collection),
            prebuilt.docs,
            allowed_access=set(allowed),
            prebuilt=prebuilt,
        )
        return MeteredRetriever(inner)

    def build_context(self, collection: str, role: str) -> RequestContext:
        allowed = frozenset(config.ROLE_ACCESS[role])
        retriever = self.build_retriever(collection, allowed)
        tools = [make_retrieve_tool(retriever, k=config.TOP_K, verbose=False)]
        web_recorder = None
        if self._search_backend is not None:
            # 每请求独立 recorder:_recorded 落在请求对象上,并发竞态结构性消失
            web_recorder = RecordingBackend(MeteredSearchBackend(self._search_backend))
            tools.append(
                make_web_search_tool(
                    web_recorder, max_results=config.WEB_SEARCH_MAX_RESULTS, verbose=False
                )
            )
        system_prompt = SYSTEM_PROMPT_WEB if web_recorder is not None else SYSTEM_PROMPT
        graph = build_graph(
            self.llm.bind_tools(tools),
            tools,
            checkpointer=self._checkpointer,
            system_prompt=system_prompt,
        )
        return RequestContext(
            graph=graph,
            retriever=retriever,
            cache=self.cache,
            live_chunk_ids=self.live_chunk_ids(collection),
            allowed_access=allowed,
            web_recorder=web_recorder,
        )

    def ingest(self, collection: str, docs_dir=None, rebuild: bool = False) -> dict:
        from agentic_search import ingest as ingest_mod
        from agentic_search.retrieval import save_bm25_index

        directory = Path(docs_dir) if docs_dir else config.SAMPLE_DOCS_DIR
        if not directory.is_dir():
            raise FileNotFoundError(f"目录不存在: {directory}")
        # 精简镜像无 transcriber/captioner → 媒体文件跳过,仅 md/pdf
        chunks = ingest_mod.load_documents(directory)
        if not chunks:
            raise ValueError(f"{directory} 下无受支持文档 (md/pdf)")
        store = self.store(collection)
        if rebuild:
            store.reset_collection()
        added, removed = ingest_mod.sync_vector_store(store, chunks)
        save_bm25_index(build_bm25_index(chunks), self._bm25_path(collection))
        self.invalidate(collection)
        return {
            "collection": collection,
            "added": added,
            "removed": removed,
            "chunks": len(chunks),
        }

    def invalidate(self, collection: str) -> None:
        with self._lock:
            self._prebuilt.pop(collection, None)
            self._live_ids.pop(collection, None)
            for key in [k for k in self._role_index if k[0] == collection]:
                self._role_index.pop(key, None)

    def health(self, collection: str | None = None) -> dict:
        from agentic_search import preflight

        coll = collection or config.COLLECTION_NAME
        if config.BACKEND == "ollama":
            ollama = preflight.ollama_status()
            backend_ok = ollama["reachable"] and not ollama["missing_models"]
        else:
            ollama = {"backend": config.BACKEND, "reachable": None}
            backend_ok = True
        vs = preflight.vector_store_status(coll)
        ok = backend_ok and vs["count"] > 0
        return {
            "status": "ok" if ok else "degraded",
            "backend": config.BACKEND,
            "ollama": ollama,
            "vector_store": vs,
            "web_search": self._search_backend is not None,
        }
