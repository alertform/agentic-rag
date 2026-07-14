# FastAPI 服务化 + 可观测性 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 agentic-rag 从 CLI demo 升级为可服务化、可观测的准生产系统:FastAPI + SSE 流式服务层 → 容器化 → Prometheus/Grafana 可观测面板。

**Architecture:** server 是薄边界层,核心业务模块(`graph.py`/`retrieval.py`/`cache.py`/`llm.py`)一律不改。并发核心是进程级 `ResourceRegistry`(缓存昂贵只读对象)+ 每请求轻量包装(独立 `HybridRetriever`,使 `_recorded`/`last_route` 落在每请求对象上,竞态结构性消失)。埋点全在边界完成。

**Tech Stack:** FastAPI、uvicorn、sse-starlette、prometheus-client、structlog、Docker/compose、Prometheus、Grafana。既有栈:LangChain/LangGraph、Chroma、Ollama。

## Global Constraints

- Python `>=3.12,<3.13`(沿用现有 `requires-python`)。
- **核心业务模块不改**:`graph.py`、`retrieval.py`、`cache.py`、`llm.py` 零改动。
- **CLI 默认 `uv sync` 必须仍让 `python -m agentic_rag.ingest` 媒体可用**;精简 server 镜像不含媒体栈(`faster-whisper`/`av`/`pillow`)。机制:媒体入默认安装的 `media` dependency-group,镜像用 `--no-group media` 排除。
- server 依赖入 `[project.optional-dependencies]` 的 `server` extra(与既有 `vllm` extra 同风格)。
- Ollama 留宿主,容器经 `OLLAMA_BASE_URL=http://host.docker.internal:11434` 连。
- 单进程单 worker;内存态 `MemorySaver` checkpointer(对单进程正确)。
- 埋点只在 server 边界;不改核心模块。
- pytest 不依赖 Ollama(注入 stub 保持不变量)。
- 提交遵循 conventional commits;全局已禁用 attribution,提交信息不含署名。

---

## Task 1: 依赖与配置(server extra + media group + 服务配置)

**Files:**
- Modify: `pyproject.toml`
- Modify: `src/agentic_rag/config.py`
- Test: `tests/test_server_config.py`

**Interfaces:**
- Produces: `config.SERVER_HOST: str`、`config.SERVER_PORT: int`;`server` extra 可经 `uv sync --extra server` 安装;`media` dependency-group 默认安装、可 `--no-group media` 排除。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_server_config.py
from agentic_rag import config


def test_server_host_and_port_defaults():
    assert isinstance(config.SERVER_HOST, str) and config.SERVER_HOST
    assert isinstance(config.SERVER_PORT, int) and config.SERVER_PORT > 0
```

- [ ] **Step 2: 运行,确认失败**

Run: `uv run pytest tests/test_server_config.py -v`
Expected: FAIL — `AttributeError: module 'agentic_rag.config' has no attribute 'SERVER_HOST'`

- [ ] **Step 3: 改 `config.py`**——在文件末尾追加:

```python
# 服务层(FastAPI)绑定地址;容器内绑 0.0.0.0
SERVER_HOST = os.environ.get("AGENTIC_RAG_HOST", "0.0.0.0")
SERVER_PORT = int(os.environ.get("AGENTIC_RAG_PORT", "8080"))
```

- [ ] **Step 4: 改 `pyproject.toml`**——把媒体依赖从 `dependencies` 移入默认安装的 `media` group,新增 `server` extra。将 `[project] dependencies` 改为(移除 `faster-whisper`/`av`/`pillow`,保留 `pymupdf4llm`):

```toml
dependencies = [
    "langchain-core>=0.3",
    "langgraph>=0.2.60",
    "langchain-ollama>=0.3",
    "langchain-chroma>=0.2",
    "langchain-text-splitters>=0.3",
    "pymupdf4llm>=1.28.0",
    "rank-bm25>=0.2.2",
    "jieba>=0.42.1",
]

[project.optional-dependencies]
vllm = ["langchain-openai>=0.2"]
server = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.30",
    "sse-starlette>=2.1",
    "prometheus-client>=0.20",
    "structlog>=24.1",
    "httpx>=0.27",
]

[dependency-groups]
dev = ["pytest>=8.0"]
media = [
    "faster-whisper>=1.2.1",
    "av>=18.0.0",
    "pillow>=12.3.0",
]

[tool.uv]
default-groups = ["dev", "media"]
```

- [ ] **Step 5: 同步依赖**

Run: `uv sync --extra server`
Expected: 成功;`uv.lock` 更新。若无 `uv.lock` 先 `uv lock`。

- [ ] **Step 6: 运行测试,确认通过**

Run: `uv run pytest tests/test_server_config.py tests/ -v`
Expected: PASS(既有单测不受影响——媒体仍默认安装)

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock src/agentic_rag/config.py tests/test_server_config.py
git commit -m "chore: server extra + media 依赖分组 + 服务绑定配置"
```

---

## Task 2: 请求模型(schemas.py)

**Files:**
- Create: `src/agentic_rag/server/__init__.py`
- Create: `src/agentic_rag/server/schemas.py`
- Test: `tests/test_server_schemas.py`

**Interfaces:**
- Consumes: `config.ROLE_ACCESS`、`config.COLLECTION_NAME`。
- Produces: `ChatRequest(question, thread_id, role="manager", collection=COLLECTION_NAME)`、`IngestRequest(collection=COLLECTION_NAME, docs_dir=None, rebuild=False)`。非法 role / 空 question 抛 `ValidationError`。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_server_schemas.py
import pytest
from pydantic import ValidationError

from agentic_rag.server.schemas import ChatRequest, IngestRequest


def test_chat_request_defaults():
    req = ChatRequest(question="星尘咖啡馆的招牌是什么?", thread_id="t1")
    assert req.role == "manager"
    assert req.collection


def test_chat_request_rejects_unknown_role():
    with pytest.raises(ValidationError):
        ChatRequest(question="hi", thread_id="t1", role="ceo")


def test_chat_request_rejects_blank_question():
    with pytest.raises(ValidationError):
        ChatRequest(question="   ", thread_id="t1")


def test_ingest_request_defaults():
    req = IngestRequest()
    assert req.rebuild is False
    assert req.docs_dir is None
```

- [ ] **Step 2: 运行,确认失败**

Run: `uv run pytest tests/test_server_schemas.py -v`
Expected: FAIL — `ModuleNotFoundError: agentic_rag.server`

- [ ] **Step 3: 建包与模型**

```python
# src/agentic_rag/server/__init__.py
"""agentic-rag 服务层(FastAPI + SSE):薄边界层,核心业务模块不改。"""
```

```python
# src/agentic_rag/server/schemas.py
"""服务层请求模型:系统边界的输入校验(pydantic + 角色白名单)。"""
from pydantic import BaseModel, field_validator

from agentic_rag import config


class ChatRequest(BaseModel):
    question: str
    thread_id: str
    role: str = "manager"
    collection: str = config.COLLECTION_NAME

    @field_validator("role")
    @classmethod
    def _known_role(cls, v: str) -> str:
        if v not in config.ROLE_ACCESS:
            raise ValueError(f"未知角色 {v!r};可选: {sorted(config.ROLE_ACCESS)}")
        return v

    @field_validator("question")
    @classmethod
    def _nonempty_question(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("question 不能为空")
        return v


class IngestRequest(BaseModel):
    collection: str = config.COLLECTION_NAME
    docs_dir: str | None = None
    rebuild: bool = False
```

- [ ] **Step 4: 运行测试,确认通过**

Run: `uv run pytest tests/test_server_schemas.py -v`
Expected: PASS(4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/agentic_rag/server/__init__.py src/agentic_rag/server/schemas.py tests/test_server_schemas.py
git commit -m "feat: 服务层请求模型 (ChatRequest/IngestRequest) + 边界校验"
```

---

## Task 3: 结构化日志(logging_config.py)

**Files:**
- Create: `src/agentic_rag/server/logging_config.py`
- Test: `tests/test_server_logging.py`

**Interfaces:**
- Produces: `configure_logging(level=logging.INFO) -> None`;`logger`(structlog bound logger)。审计事件字段(role/collection/cache_hit/route/...)按 kwargs 传入即进 JSON。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_server_logging.py
from structlog.testing import capture_logs

from agentic_rag.server.logging_config import configure_logging, logger


def test_configure_logging_is_idempotent():
    configure_logging()
    configure_logging()  # 二次调用不应抛


def test_logger_emits_event_with_fields():
    with capture_logs() as logs:
        logger.info("chat_completed", role="staff", cache_hit=True, route="hybrid")
    assert logs[0]["event"] == "chat_completed"
    assert logs[0]["role"] == "staff"
    assert logs[0]["cache_hit"] is True
    assert logs[0]["route"] == "hybrid"
```

- [ ] **Step 2: 运行,确认失败**

Run: `uv run pytest tests/test_server_logging.py -v`
Expected: FAIL — `ModuleNotFoundError: agentic_rag.server.logging_config`

- [ ] **Step 3: 实现**

```python
# src/agentic_rag/server/logging_config.py
"""结构化日志:structlog JSON 渲染 + contextvars 合并(request_id 经 bound_contextvars 注入)。"""
import logging

import structlog


def configure_logging(level: int = logging.INFO) -> None:
    """配置 structlog 输出单行 JSON;merge_contextvars 让 request_id 等自动带入每条日志。"""
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(ensure_ascii=False),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


logger = structlog.get_logger("agentic_rag.server")
```

- [ ] **Step 4: 运行测试,确认通过**

Run: `uv run pytest tests/test_server_logging.py -v`
Expected: PASS(2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/agentic_rag/server/logging_config.py tests/test_server_logging.py
git commit -m "feat: 服务层结构化日志 (structlog JSON + contextvars)"
```

---

## Task 4: 指标与检索计量代理(metrics.py)

**Files:**
- Create: `src/agentic_rag/server/metrics.py`
- Test: `tests/test_server_metrics.py`

**Interfaces:**
- Produces:
  - `REGISTRY`(独立 `CollectorRegistry`,测试隔离)。
  - `observe_latency(endpoint: str, seconds: float)`、`record_cache(hit: bool)`、`record_route(route: str | None)`、`observe_tokens(n: int)`、`render() -> tuple[bytes, str]`。
  - `MeteredRetriever(inner)`:透明包装,`similarity_search` 每次 `RETRIEVE_CALLS_TOTAL.inc()` + `record_route(inner.last_route)`;`take_recorded()` / `last_route` 委托内层。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_server_metrics.py
from agentic_rag.server import metrics


class _Inner:
    def __init__(self):
        self.last_route = "hybrid"
        self._recorded = ["doc"]

    def similarity_search(self, query, k):
        return ["hit"]

    def take_recorded(self):
        r, self._recorded = self._recorded, []
        return r


def _value(counter, **labels):
    return counter.labels(**labels)._value.get()


def test_record_cache_increments():
    before = _value(metrics.CACHE_LOOKUP_TOTAL, result="hit")
    metrics.record_cache(True)
    assert _value(metrics.CACHE_LOOKUP_TOTAL, result="hit") == before + 1


def test_metered_retriever_counts_route_and_calls():
    inner = _Inner()
    metered = metrics.MeteredRetriever(inner)
    calls_before = metrics.RETRIEVE_CALLS_TOTAL._value.get()
    route_before = _value(metrics.ROUTE_TOTAL, route="hybrid")
    assert metered.similarity_search("q", k=5) == ["hit"]
    assert metrics.RETRIEVE_CALLS_TOTAL._value.get() == calls_before + 1
    assert _value(metrics.ROUTE_TOTAL, route="hybrid") == route_before + 1
    assert metered.take_recorded() == ["doc"]
    assert metered.last_route == "hybrid"


def test_render_returns_prometheus_text():
    payload, content_type = metrics.render()
    assert b"agentic_rag_request_latency_seconds" in payload
    assert "text/plain" in content_type
```

- [ ] **Step 2: 运行,确认失败**

Run: `uv run pytest tests/test_server_metrics.py -v`
Expected: FAIL — `ModuleNotFoundError: agentic_rag.server.metrics`

- [ ] **Step 3: 实现**

```python
# src/agentic_rag/server/metrics.py
"""Prometheus 指标 + 检索计量代理(边界埋点,核心模块不改)。"""
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)

REGISTRY = CollectorRegistry()

REQUEST_LATENCY = Histogram(
    "agentic_rag_request_latency_seconds", "请求端到端延迟", ["endpoint"], registry=REGISTRY
)
ROUTE_TOTAL = Counter(
    "agentic_rag_route_total", "检索通道路由计数", ["route"], registry=REGISTRY
)
CACHE_LOOKUP_TOTAL = Counter(
    "agentic_rag_cache_lookup_total", "语义缓存查询结果计数", ["result"], registry=REGISTRY
)
RETRIEVE_CALLS_TOTAL = Counter(
    "agentic_rag_retrieve_calls_total", "检索调用次数", registry=REGISTRY
)
TOKENS_PER_TURN = Histogram(
    "agentic_rag_tokens_per_turn",
    "每轮生成 token 数(以流式 chunk 数近似)",
    registry=REGISTRY,
    buckets=(8, 16, 32, 64, 128, 256, 512, 1024),
)


def observe_latency(endpoint: str, seconds: float) -> None:
    REQUEST_LATENCY.labels(endpoint=endpoint).observe(seconds)


def record_cache(hit: bool) -> None:
    CACHE_LOOKUP_TOTAL.labels(result="hit" if hit else "miss").inc()


def record_route(route: str | None) -> None:
    if route:
        ROUTE_TOTAL.labels(route=route).inc()


def observe_tokens(n: int) -> None:
    TOKENS_PER_TURN.observe(n)


def render() -> tuple[bytes, str]:
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST


class MeteredRetriever:
    """透明包装 HybridRetriever:每次检索记录 route 与调用计数。核心检索器零改动。"""

    def __init__(self, inner):
        self._inner = inner

    def similarity_search(self, query: str, k: int):
        hits = self._inner.similarity_search(query, k)
        RETRIEVE_CALLS_TOTAL.inc()
        record_route(self._inner.last_route)
        return hits

    def take_recorded(self):
        return self._inner.take_recorded()

    @property
    def last_route(self):
        return self._inner.last_route
```

- [ ] **Step 4: 运行测试,确认通过**

Run: `uv run pytest tests/test_server_metrics.py -v`
Expected: PASS(3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/agentic_rag/server/metrics.py tests/test_server_metrics.py
git commit -m "feat: 服务层 Prometheus 指标 + MeteredRetriever 计量代理"
```

---

## Task 5: preflight 非致命状态(供 /health)

**Files:**
- Modify: `src/agentic_rag/preflight.py`
- Test: `tests/test_preflight_status.py`

**Interfaces:**
- Produces: `ollama_status(require_generation=True) -> dict`(`{"reachable": bool, "missing_models": list[str]}`)、`vector_store_status(collection=None) -> dict`(`{"collection": str, "count": int, "exists": bool}`)。既有 `check_ollama`/`check_vector_store` 改为委托这两个函数再 `sys.exit`,CLI 行为不变。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_preflight_status.py
from agentic_rag import config, preflight


def test_ollama_status_unreachable(monkeypatch):
    monkeypatch.setattr(preflight, "_installed_models", lambda: None)
    status = preflight.ollama_status()
    assert status["reachable"] is False
    assert config.EMBEDDING_MODEL in status["missing_models"]


def test_ollama_status_missing_generation_model(monkeypatch):
    # 只装了嵌入模型;生成模型缺失应精确出现在 missing_models,嵌入模型不应出现
    monkeypatch.setattr(preflight, "_installed_models", lambda: [f"{config.EMBEDDING_MODEL}:latest"])
    status = preflight.ollama_status(require_generation=True)
    assert status["reachable"] is True
    assert status["missing_models"] == [config.GENERATION_MODEL]


def test_vector_store_status_shape():
    status = preflight.vector_store_status("does_not_exist_collection")
    assert status["exists"] is False
    assert status["count"] == 0
```

- [ ] **Step 2: 运行,确认失败**

Run: `uv run pytest tests/test_preflight_status.py -v`
Expected: FAIL — `AttributeError: module 'agentic_rag.preflight' has no attribute 'ollama_status'`

- [ ] **Step 3: 重构 preflight.py**——用状态函数替换 `check_ollama`/`check_vector_store` 内联逻辑(替换 `preflight.py:24-45`):

```python
def ollama_status(require_generation: bool = True) -> dict:
    """非致命就绪状态:服务是否可达、缺哪些模型。供 /health 使用(不 sys.exit)。"""
    models = _installed_models()
    required = [config.EMBEDDING_MODEL]
    if require_generation:
        required.append(config.GENERATION_MODEL)
    if models is None:
        return {"reachable": False, "missing_models": required}
    missing = [name for name in required if not _has_model(models, name)]
    return {"reachable": True, "missing_models": missing}


def check_ollama(require_generation: bool = True) -> None:
    status = ollama_status(require_generation)
    if not status["reachable"]:
        sys.exit(f"[preflight] 连不上 Ollama ({config.OLLAMA_BASE_URL})。先运行: ollama serve")
    for name in status["missing_models"]:
        sys.exit(f"[preflight] 缺少模型 {name}。先运行: ollama pull {name}")


def vector_store_status(collection: str | None = None) -> dict:
    """非致命向量库状态:目标 collection 是否存在、块数。"""
    import chromadb

    coll = collection or config.COLLECTION_NAME
    try:
        client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        count = client.get_collection(coll).count()
    except Exception:
        count = 0
    return {"collection": coll, "count": count, "exists": count > 0}


def check_vector_store() -> None:
    if vector_store_status()["count"] == 0:
        sys.exit("[preflight] 向量库为空。先运行: uv run python -m agentic_rag.ingest [md目录]")
```

- [ ] **Step 4: 运行测试,确认通过**

Run: `uv run pytest tests/test_preflight_status.py tests/ -v`
Expected: PASS(既有测试不受影响)

- [ ] **Step 5: Commit**

```bash
git add src/agentic_rag/preflight.py tests/test_preflight_status.py
git commit -m "refactor: preflight 暴露非致命状态函数 (供 /health),CLI 退出行为不变"
```

---

## Task 6: 资源注册表(resources.py — 并发核心)

**Files:**
- Create: `src/agentic_rag/server/resources.py`
- Test: `tests/test_server_resources.py`

**Interfaces:**
- Consumes: `HybridRetriever`、`build_bm25_index`、`corpus_digest`、`load_all_chunks`、`load_bm25_index`(retrieval);`build_graph`(graph);`SemanticCache`(cache);`make_retrieve_tool`(tools);`MeteredRetriever`(metrics);`config.ROLE_ACCESS`/`TOP_K`/`CHROMA_DIR`。
- Produces:
  - `RequestContext(graph, retriever, cache, live_chunk_ids: set[str], allowed_access: frozenset[str])`。
  - `ResourceRegistry(*, chroma_dir=None, embeddings=None, llm=None, checkpointer=None, role_index_cache_size=8)`,方法:`store(collection)`、`prebuilt_index(collection)`、`live_chunk_ids(collection)`、`build_retriever(collection, allowed: frozenset) -> MeteredRetriever`、`build_context(collection, role) -> RequestContext`、`ingest(collection, docs_dir=None, rebuild=False) -> dict`、`invalidate(collection)`、`health(collection=None) -> dict`。
  - 每请求 `build_retriever` 返回独立包装,`take_recorded`/`last_route` 互不共享(竞态消除)。

- [ ] **Step 1: 写失败测试**(用假 store + 真 BM25,不依赖 Ollama)

```python
# tests/test_server_resources.py
from langchain_core.documents import Document

from agentic_rag.server.resources import RequestContext, ResourceRegistry


class FakeStore:
    """鸭子类型向量库:满足 similarity_search / get。"""

    def __init__(self, docs):
        self._docs = docs

    def similarity_search(self, query, k, filter=None):
        return self._docs[:k]

    def get(self):
        return {"ids": [f"id{i}" for i in range(len(self._docs))]}


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
```

- [ ] **Step 2: 运行,确认失败**

Run: `uv run pytest tests/test_server_resources.py -v`
Expected: FAIL — `ModuleNotFoundError: agentic_rag.server.resources`

- [ ] **Step 3: 实现**

```python
# src/agentic_rag/server/resources.py
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

from agentic_rag import config
from agentic_rag.cache import SemanticCache
from agentic_rag.graph import build_graph
from agentic_rag.retrieval import (
    HybridRetriever,
    build_bm25_index,
    corpus_digest,
    load_all_chunks,
    load_bm25_index,
)
from agentic_rag.server.metrics import MeteredRetriever
from agentic_rag.tools import make_retrieve_tool


@dataclass
class RequestContext:
    graph: object
    retriever: MeteredRetriever
    cache: SemanticCache
    live_chunk_ids: set[str]
    allowed_access: frozenset[str]


class ResourceRegistry:
    def __init__(
        self,
        *,
        chroma_dir=None,
        embeddings=None,
        llm=None,
        checkpointer=None,
        role_index_cache_size: int = 8,
    ):
        self._chroma_dir = Path(chroma_dir) if chroma_dir else config.CHROMA_DIR
        self._embeddings = embeddings
        self._llm = llm
        self._checkpointer = checkpointer or MemorySaver()
        self._cache: SemanticCache | None = None
        self._stores: dict = {}
        self._prebuilt: dict = {}
        self._live_ids: dict = {}
        self._role_index: OrderedDict = OrderedDict()
        self._role_index_cap = role_index_cache_size
        self._lock = threading.Lock()

    @property
    def embeddings(self):
        if self._embeddings is None:
            from agentic_rag.llm import make_embeddings

            self._embeddings = make_embeddings()
        return self._embeddings

    @property
    def llm(self):
        if self._llm is None:
            from agentic_rag.llm import make_chat_llm

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

    def live_chunk_ids(self, collection: str) -> set[str]:
        with self._lock:
            if collection not in self._live_ids:
                self._live_ids[collection] = set(self.store(collection).get()["ids"])
            return self._live_ids[collection]

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
        retrieve = make_retrieve_tool(retriever, k=config.TOP_K, verbose=False)
        graph = build_graph(
            self.llm.bind_tools([retrieve]), [retrieve], checkpointer=self._checkpointer
        )
        return RequestContext(
            graph=graph,
            retriever=retriever,
            cache=self.cache,
            live_chunk_ids=self.live_chunk_ids(collection),
            allowed_access=allowed,
        )

    def ingest(self, collection: str, docs_dir=None, rebuild: bool = False) -> dict:
        from agentic_rag import ingest as ingest_mod
        from agentic_rag.retrieval import save_bm25_index

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
        from agentic_rag import preflight

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
        }
```

- [ ] **Step 4: 运行测试,确认通过**

Run: `uv run pytest tests/test_server_resources.py -v`
Expected: PASS(3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/agentic_rag/server/resources.py tests/test_server_resources.py
git commit -m "feat: ResourceRegistry 进程级资源缓存 + 每请求隔离检索器 (并发核心)"
```

---

## Task 7: 应用工厂 + 路由骨架(/health、/roles、/metrics)

**Files:**
- Create: `src/agentic_rag/server/app.py`
- Create: `src/agentic_rag/server/routes.py`
- Test: `tests/test_server_app.py`

**Interfaces:**
- Consumes: `ResourceRegistry`(app 默认构造)、`configure_logging`、`metrics.render`。
- Produces: `create_app(registry=None) -> FastAPI`;module 级 `app`。`router` 提供 `GET /health`、`GET /roles`、`GET /metrics`。测试注入自定义 registry(需具备 `health(collection=None) -> dict`)。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_server_app.py
from fastapi.testclient import TestClient

from agentic_rag.server.app import create_app


class FakeRegistry:
    def health(self, collection=None):
        return {"status": "ok", "backend": "ollama", "ollama": {}, "vector_store": {"count": 3}}


def _client():
    return TestClient(create_app(registry=FakeRegistry()))


def test_roles_endpoint():
    with _client() as client:
        resp = client.get("/roles")
        assert resp.status_code == 200
        assert set(resp.json()["roles"]) == {"public", "staff", "manager"}


def test_health_endpoint():
    with _client() as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


def test_metrics_endpoint():
    with _client() as client:
        resp = client.get("/metrics")
        assert resp.status_code == 200
        assert "agentic_rag_request_latency_seconds" in resp.text
```

- [ ] **Step 2: 运行,确认失败**

Run: `uv run pytest tests/test_server_app.py -v`
Expected: FAIL — `ModuleNotFoundError: agentic_rag.server.app`

- [ ] **Step 3: 实现 routes.py(骨架)**

```python
# src/agentic_rag/server/routes.py
"""服务层端点。核心业务逻辑在 ResourceRegistry;此处只做 HTTP 编排与埋点。"""
from fastapi import APIRouter, Request, Response

from agentic_rag import config
from agentic_rag.server import metrics

router = APIRouter()


def _registry(request: Request):
    return request.app.state.registry


@router.get("/roles")
async def roles():
    return {"roles": sorted(config.ROLE_ACCESS)}


@router.get("/health")
async def health(request: Request):
    return _registry(request).health()


@router.get("/metrics")
async def metrics_endpoint():
    payload, content_type = metrics.render()
    return Response(content=payload, media_type=content_type)
```

- [ ] **Step 4: 实现 app.py**

```python
# src/agentic_rag/server/app.py
"""FastAPI 应用工厂:lifespan 内构造 ResourceRegistry;module 级 app 供 uvicorn 加载。"""
from contextlib import asynccontextmanager

from fastapi import FastAPI

from agentic_rag.server.logging_config import configure_logging
from agentic_rag.server.resources import ResourceRegistry
from agentic_rag.server.routes import router


def create_app(registry=None) -> FastAPI:
    configure_logging()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.registry = registry or ResourceRegistry()
        yield

    application = FastAPI(title="agentic-rag", lifespan=lifespan)
    application.include_router(router)
    return application


app = create_app()
```

- [ ] **Step 5: 运行测试,确认通过**

Run: `uv run pytest tests/test_server_app.py -v`
Expected: PASS(3 passed)

- [ ] **Step 6: Commit**

```bash
git add src/agentic_rag/server/app.py src/agentic_rag/server/routes.py tests/test_server_app.py
git commit -m "feat: FastAPI 应用工厂 + /health /roles /metrics 端点"
```

---

## Task 8: /chat SSE 流式端点(含指标 + 审计日志)

**Files:**
- Modify: `src/agentic_rag/server/routes.py`
- Test: `tests/test_server_chat.py`

**Interfaces:**
- Consumes: `ResourceRegistry.build_context`、`RequestContext`、`ChatRequest`、`metrics.*`、`logger`、`chunk_id`;LangGraph `graph.astream(inputs, config, stream_mode="messages")` 异步产出 `(AIMessageChunk, metadata)`(与 CLI `stream_mode="messages"` 契约一致)。
- Produces: `POST /chat` 返回 `EventSourceResponse`。事件:`token`(`{"text": ...}`)与终止 `done`(`{"sources", "route", "cache_hit", "request_id"}`)。缓存命中短路为单 `token`+`done`。

- [ ] **Step 1: 写失败测试**(注入 stub graph/retriever/cache,不依赖 Ollama)

```python
# tests/test_server_chat.py
import json

from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.messages import AIMessageChunk

from agentic_rag.server.app import create_app
from agentic_rag.server.resources import RequestContext


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


def test_chat_cache_hit_short_circuits():
    from agentic_rag.cache import CacheHit

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
```

- [ ] **Step 2: 运行,确认失败**

Run: `uv run pytest tests/test_server_chat.py -v`
Expected: FAIL — 404(`/chat` 未定义)

- [ ] **Step 3: 在 routes.py 追加 /chat**——顶部补充导入,文件末尾追加端点:

```python
# routes.py 顶部导入区补充:
import json
import time
import uuid

import structlog
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from sse_starlette.sse import EventSourceResponse

from agentic_rag.ingest import chunk_id
from agentic_rag.server.logging_config import logger
from agentic_rag.server.schemas import ChatRequest
```

```python
# routes.py 末尾追加:
@router.post("/chat")
async def chat(request: Request, body: ChatRequest):
    registry = _registry(request)
    ctx = registry.build_context(body.collection, body.role)
    run_config = {
        "configurable": {"thread_id": body.thread_id},
        "recursion_limit": config.RECURSION_LIMIT,
    }
    allowed = set(ctx.allowed_access)
    hit = ctx.cache.lookup(body.question, ctx.live_chunk_ids, allowed)
    request_id = uuid.uuid4().hex[:12]

    async def event_stream():
        start = time.perf_counter()
        with structlog.contextvars.bound_contextvars(request_id=request_id):
            if hit is not None:
                metrics.record_cache(True)
                ctx.graph.update_state(
                    run_config,
                    {"messages": [HumanMessage(body.question), AIMessage(hit.answer)]},
                )
                yield {"event": "token", "data": json.dumps({"text": hit.answer}, ensure_ascii=False)}
                yield {
                    "event": "done",
                    "data": json.dumps(
                        {"sources": hit.sources, "route": None, "cache_hit": True, "request_id": request_id},
                        ensure_ascii=False,
                    ),
                }
                latency = time.perf_counter() - start
                metrics.observe_latency("chat", latency)
                logger.info(
                    "chat_completed", role=body.role, collection=body.collection,
                    cache_hit=True, route=None, sources=hit.sources, chunk_ids=[],
                    latency_ms=round(latency * 1000, 1),
                )
                return

            metrics.record_cache(False)
            ctx.retriever.take_recorded()  # 清空上一轮残留
            parts: list[str] = []
            n_tokens = 0
            async for chunk, meta in ctx.graph.astream(
                {"messages": [HumanMessage(body.question)]}, run_config, stream_mode="messages"
            ):
                if (
                    isinstance(chunk, AIMessageChunk)
                    and meta.get("langgraph_node") == "agent"
                    and chunk.content
                ):
                    text = str(chunk.content)
                    parts.append(text)
                    n_tokens += 1
                    yield {"event": "token", "data": json.dumps({"text": text}, ensure_ascii=False)}

            recorded = ctx.retriever.take_recorded()
            sources = sorted({d.metadata["source"] for d in recorded})
            route = ctx.retriever.last_route
            answer = "".join(parts).strip()
            if recorded and answer:
                ctx.cache.store(
                    question=body.question, answer=answer, sources=sources,
                    chunk_ids=[chunk_id(d) for d in recorded],
                    access_levels=[d.metadata.get("access", "public") for d in recorded],
                )
            metrics.observe_tokens(n_tokens)
            yield {
                "event": "done",
                "data": json.dumps(
                    {"sources": sources, "route": route, "cache_hit": False, "request_id": request_id},
                    ensure_ascii=False,
                ),
            }
            latency = time.perf_counter() - start
            metrics.observe_latency("chat", latency)
            logger.info(
                "chat_completed", role=body.role, collection=body.collection,
                cache_hit=False, route=route, sources=sources,
                chunk_ids=[chunk_id(d) for d in recorded],
                latency_ms=round(latency * 1000, 1),
            )

    return EventSourceResponse(event_stream())
```

- [ ] **Step 4: 运行测试,确认通过**

Run: `uv run pytest tests/test_server_chat.py -v`
Expected: PASS(4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/agentic_rag/server/routes.py tests/test_server_chat.py
git commit -m "feat: /chat SSE 流式端点 (缓存短路 + 指标 + 审计日志 + request_id)"
```

---

## Task 9: /ingest 端点(同步 + 进程级锁)

**Files:**
- Modify: `src/agentic_rag/server/routes.py`
- Test: `tests/test_server_ingest.py`

**Interfaces:**
- Consumes: `ResourceRegistry.ingest(collection, docs_dir, rebuild) -> dict`、`IngestRequest`。
- Produces: `POST /ingest` 同步触发增量索引(threadpool 执行);并发请求返回 409;成功返回 `{collection, added, removed, chunks}`。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_server_ingest.py
import asyncio

from fastapi.testclient import TestClient

from agentic_rag.server.app import create_app


class FakeRegistry:
    def __init__(self):
        self.calls = []

    def ingest(self, collection, docs_dir=None, rebuild=False):
        self.calls.append((collection, docs_dir, rebuild))
        return {"collection": collection, "added": 2, "removed": 0, "chunks": 5}


def test_ingest_returns_counts():
    reg = FakeRegistry()
    with TestClient(create_app(registry=reg)) as client:
        resp = client.post("/ingest", json={"collection": "c", "rebuild": True})
        assert resp.status_code == 200
        assert resp.json()["added"] == 2
        assert reg.calls == [("c", None, True)]


def test_ingest_rejects_concurrent(monkeypatch):
    from agentic_rag.server import routes

    async def _busy():
        # 手动占用锁,模拟一个进行中的 ingest
        await routes._ingest_lock.acquire()
        try:
            with TestClient(create_app(registry=FakeRegistry())) as client:
                resp = client.post("/ingest", json={})
                assert resp.status_code == 409
        finally:
            routes._ingest_lock.release()

    asyncio.run(_busy())
```

- [ ] **Step 2: 运行,确认失败**

Run: `uv run pytest tests/test_server_ingest.py -v`
Expected: FAIL — 404(`/ingest` 未定义)

- [ ] **Step 3: 在 routes.py 追加 /ingest**——顶部导入区补充,文件末尾追加:

```python
# routes.py 顶部导入区补充:
import asyncio

from fastapi import HTTPException
from starlette.concurrency import run_in_threadpool

from agentic_rag.server.schemas import IngestRequest

_ingest_lock = asyncio.Lock()
```

```python
# routes.py 末尾追加:
@router.post("/ingest")
async def ingest(request: Request, body: IngestRequest):
    if _ingest_lock.locked():
        raise HTTPException(status_code=409, detail="已有索引任务进行中")
    registry = _registry(request)
    async with _ingest_lock:
        return await run_in_threadpool(
            registry.ingest, body.collection, body.docs_dir, body.rebuild
        )
```

- [ ] **Step 4: 运行测试,确认通过**

Run: `uv run pytest tests/test_server_ingest.py -v`
Expected: PASS(2 passed)

- [ ] **Step 5: 全量回归**

Run: `uv run pytest -v`
Expected: PASS(所有单测,含既有测试)

- [ ] **Step 6: Commit**

```bash
git add src/agentic_rag/server/routes.py tests/test_server_ingest.py
git commit -m "feat: /ingest 同步端点 + 进程级锁 (并发 409)"
```

---

## Task 10: 多阶段 Dockerfile(精简镜像)

**Files:**
- Create: `Dockerfile`
- Create: `.dockerignore`

**Interfaces:**
- Consumes: `pyproject.toml`(`server` extra、`media` group)、`uv.lock`。
- Produces: 精简运行镜像(chat + md/pdf ingest,无媒体栈),`CMD` 起 `uvicorn agentic_rag.server.app:app`。

- [ ] **Step 1: 写 .dockerignore**

```
.venv
chroma_db
.media_cache
models
__pycache__
*.pyc
.git
.pytest_cache
docs
tests
```

- [ ] **Step 2: 写 Dockerfile**

```dockerfile
# syntax=docker/dockerfile:1
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS build
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
# 先只装依赖(缓存层):精简镜像排除媒体栈与 dev
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --extra server --no-group media --no-dev
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --extra server --no-group media --no-dev

FROM python:3.12-slim-bookworm AS runtime
WORKDIR /app
COPY --from=build /app /app
ENV PATH="/app/.venv/bin:$PATH" \
    AGENTIC_RAG_HOST=0.0.0.0 \
    AGENTIC_RAG_PORT=8080
EXPOSE 8080
CMD ["uvicorn", "agentic_rag.server.app:app", "--host", "0.0.0.0", "--port", "8080"]
```

- [ ] **Step 3: 确保 uv.lock 存在**

Run: `uv lock`
Expected: 生成/更新 `uv.lock`

- [ ] **Step 4: 构建镜像(验证)**

Run: `docker build -t agentic-rag:local .`
Expected: 构建成功;`docker run --rm agentic-rag:local python -c "import agentic_rag.server.app"` 无 ImportError(证明精简镜像可导入 server,且 `media.py`/`parsers.py` 惰性导入未拖入媒体栈)。

- [ ] **Step 5: Commit**

```bash
git add Dockerfile .dockerignore uv.lock
git commit -m "feat: 多阶段精简 Dockerfile (uv 构建, 无媒体栈)"
```

---

## Task 11: docker-compose + .env + Prometheus 配置

**Files:**
- Create: `docker-compose.yml`
- Create: `.env.example`
- Create: `docker/prometheus/prometheus.yml`

**Interfaces:**
- Consumes: Task 10 镜像。
- Produces: 三服务编排(agentic-rag + prometheus + grafana);Ollama 留宿主;Prometheus 抓 `agentic-rag:8080/metrics`。

- [ ] **Step 1: 写 .env.example**

```
AGENTIC_RAG_BACKEND=ollama
OLLAMA_BASE_URL=http://host.docker.internal:11434
GENERATION_MODEL=qwen3.5:9b
EMBEDDING_MODEL=bge-m3
AGENTIC_RAG_HOST=0.0.0.0
AGENTIC_RAG_PORT=8080
GRAFANA_PASSWORD=admin
```

- [ ] **Step 2: 写 docker/prometheus/prometheus.yml**

```yaml
global:
  scrape_interval: 5s
scrape_configs:
  - job_name: agentic-rag
    static_configs:
      - targets: ["agentic-rag:8080"]
```

- [ ] **Step 3: 写 docker-compose.yml**

```yaml
services:
  agentic-rag:
    build: .
    ports:
      - "8080:8080"
    environment:
      AGENTIC_RAG_BACKEND: ${AGENTIC_RAG_BACKEND:-ollama}
      OLLAMA_BASE_URL: ${OLLAMA_BASE_URL:-http://host.docker.internal:11434}
      GENERATION_MODEL: ${GENERATION_MODEL:-qwen3.5:9b}
      EMBEDDING_MODEL: ${EMBEDDING_MODEL:-bge-m3}
    volumes:
      - ./chroma_db:/app/chroma_db
    # Linux 需显式映射;Docker Desktop(Win/Mac)自动可用,保留无害
    extra_hosts:
      - "host.docker.internal:host-gateway"

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./docker/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml:ro

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      GF_SECURITY_ADMIN_PASSWORD: ${GRAFANA_PASSWORD:-admin}
      GF_AUTH_ANONYMOUS_ENABLED: "true"
    volumes:
      - ./docker/grafana/provisioning:/etc/grafana/provisioning:ro
    depends_on:
      - prometheus
```

> 备选(容器化 Ollama):如需把 Ollama 也纳入 compose,追加一个 `ollama: image: ollama/ollama` 服务 + 模型卷,并把 `OLLAMA_BASE_URL` 改为 `http://ollama:11434`;GPU 透传在 Windows 上较繁琐,故默认留宿主。

- [ ] **Step 4: 校验 compose 配置**

Run: `docker compose config`
Expected: 打印规范化配置,无语法错误(此时 grafana provisioning 目录尚空,Task 12 补齐)。

- [ ] **Step 5: Commit**

```bash
git add docker-compose.yml .env.example docker/prometheus/prometheus.yml
git commit -m "feat: docker-compose (app + prometheus + grafana, Ollama 留宿主) + .env 样例"
```

---

## Task 12: Grafana provisioning(数据源 + 面板)

**Files:**
- Create: `docker/grafana/provisioning/datasources/prometheus.yml`
- Create: `docker/grafana/provisioning/dashboards/dashboard.yml`
- Create: `docker/grafana/provisioning/dashboards/agentic-rag.json`
- Test: `tests/test_grafana_dashboard.py`

**Interfaces:**
- Produces: 开箱即用面板(QPS、p95 延迟、缓存命中率、路由分布),数据源指向 `http://prometheus:9090`。

- [ ] **Step 1: 写失败测试(dashboard JSON 结构自检)**

```python
# tests/test_grafana_dashboard.py
import json
from pathlib import Path

DASHBOARD = Path("docker/grafana/provisioning/dashboards/agentic-rag.json")


def test_dashboard_is_valid_json_with_panels():
    data = json.loads(DASHBOARD.read_text(encoding="utf-8"))
    titles = {p["title"] for p in data["panels"]}
    assert {"QPS", "p95 延迟", "缓存命中率", "路由分布"} <= titles
```

- [ ] **Step 2: 运行,确认失败**

Run: `uv run pytest tests/test_grafana_dashboard.py -v`
Expected: FAIL — `FileNotFoundError`

- [ ] **Step 3: 写数据源 provisioning**

```yaml
# docker/grafana/provisioning/datasources/prometheus.yml
apiVersion: 1
datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
```

- [ ] **Step 4: 写 dashboard provider**

```yaml
# docker/grafana/provisioning/dashboards/dashboard.yml
apiVersion: 1
providers:
  - name: agentic-rag
    type: file
    options:
      path: /etc/grafana/provisioning/dashboards
```

- [ ] **Step 5: 写 dashboard JSON**

```json
{
  "title": "agentic-rag 检索观测",
  "timezone": "browser",
  "schemaVersion": 39,
  "refresh": "5s",
  "time": {"from": "now-15m", "to": "now"},
  "panels": [
    {
      "id": 1, "title": "QPS", "type": "timeseries",
      "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
      "targets": [
        {"expr": "sum(rate(agentic_rag_request_latency_seconds_count[1m]))", "legendFormat": "qps"}
      ]
    },
    {
      "id": 2, "title": "p95 延迟", "type": "timeseries",
      "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
      "targets": [
        {"expr": "histogram_quantile(0.95, sum(rate(agentic_rag_request_latency_seconds_bucket[5m])) by (le))", "legendFormat": "p95"}
      ]
    },
    {
      "id": 3, "title": "缓存命中率", "type": "timeseries",
      "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8},
      "targets": [
        {"expr": "sum(rate(agentic_rag_cache_lookup_total{result=\"hit\"}[5m])) / clamp_min(sum(rate(agentic_rag_cache_lookup_total[5m])), 0.001)", "legendFormat": "hit ratio"}
      ]
    },
    {
      "id": 4, "title": "路由分布", "type": "timeseries",
      "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8},
      "targets": [
        {"expr": "sum by (route) (rate(agentic_rag_route_total[5m]))", "legendFormat": "{{route}}"}
      ]
    }
  ]
}
```

- [ ] **Step 6: 运行测试,确认通过**

Run: `uv run pytest tests/test_grafana_dashboard.py -v`
Expected: PASS(1 passed)

- [ ] **Step 7: 校验全栈 compose 配置**

Run: `docker compose config`
Expected: 无错误(provisioning 目录已就位)。

- [ ] **Step 8: Commit**

```bash
git add docker/grafana/provisioning tests/test_grafana_dashboard.py
git commit -m "feat: Grafana provisioning (Prometheus 数据源 + 检索观测面板)"
```

---

## Task 13: README 更新(路线标记 + 用法 + 决策记录)

**Files:**
- Modify: `README.md`

**Interfaces:**
- Produces: 「使用」补服务化用法;「工程决策记录」补显式演进项;「企业演进路线」标记 Phase 1-3 已完成;媒体分组说明。

- [ ] **Step 1: 「使用」节追加服务化用法**——在 CLI 用法块后追加:

```markdown
### 服务化(FastAPI + SSE)

```bash
uv sync --extra server                                   # 装服务依赖
uv run uvicorn agentic_rag.server.app:app --port 8080    # 起服务
# SSE 问答(role/collection/thread_id 透传给图):
curl -N -X POST localhost:8080/chat \
  -H 'Content-Type: application/json' \
  -d '{"question":"星尘咖啡馆的招牌是什么?","thread_id":"s1","role":"manager"}'
# 其它端点:GET /health、GET /roles、GET /metrics、POST /ingest
```

全栈(服务 + Prometheus + Grafana):`cp .env.example .env && docker compose up`。
Ollama 留宿主(compose 经 `host.docker.internal` 连);Grafana 面板见 `localhost:3000`,Prometheus 见 `localhost:9090`。
媒体依赖已归入默认安装的 `media` group(CLI `uv sync` 仍自带);精简 server 镜像以 `--no-group media` 排除。
```

- [ ] **Step 2: 「企业演进路线」节更新**——将 Phase 7 候选中已完成项标记,并追加已落地说明:

```markdown
## 企业演进路线

**已落地(Phase 1-3 服务化弧)**:FastAPI + SSE 流式服务化 ✅、多阶段容器化 + docker-compose ✅、Prometheus/Grafana 检索观测面板(QPS/p95/缓存命中率/路由分布)+ structlog 审计日志 ✅。设计与实现:`docs/superpowers/specs/2026-07-14-*` 与 `docs/superpowers/plans/2026-07-14-*`。

Phase 7 其余候选:多租户、答案级评估(LLM judge)、受限角色的每角色 BM25 索引、**并发压测**(locust QPS/p95 曲线,对比 Ollama 串行 vs vLLM continuous batching;后端切换点已就位于 `agentic_rag.llm`)。
```

- [ ] **Step 3: 「工程决策记录」节追加显式演进项**——追加:

```markdown
- **服务层加固延后(带标签的决策记录)**:当前为作品级单进程单 worker,内存态 `MemorySaver` 对单进程正确。以下按需补上,非静默省略:多 worker + 共享/持久 checkpointer(SqliteSaver/Redis);ingest 任务队列 + 状态轮询(大语料同步 ingest 会阻塞,demo 语料无需);Prometheus 多进程模式(`PROMETHEUS_MULTIPROC_DIR`,多 worker 才需);鉴权/限流;k8s manifests(本地 kind/minikube 跑通即可)。
- **可观测性边界埋点**:指标与审计日志只在 server 边界采集(缓存命中在调用点计量、route 经 `MeteredRetriever` 每检索记录),核心检索/缓存模块零改动——延续「不动业务代码」边界。
```

- [ ] **Step 4: 校验 markdown 无断链、渲染正常**

Run: `uv run pytest -v`
Expected: 全绿(README 改动不影响测试;此步做整体回归)。

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "docs: README 补服务化用法 + Phase 1-3 落地标记 + 演进决策记录"
```

---

## 自检(Self-Review)结论

- **Spec 覆盖**:§架构→Task 6;§Phase1(SSE/端点/DI/会话隔离/来源防伪)→Task 2,7,8,9;§Phase2(精简镜像/host-Ollama/compose)→Task 10,11;§Phase3(指标/structlog/Grafana)→Task 3,4,8,12;§显式演进项→Task 13;§媒体分组约束→Task 1,10;preflight 非致命状态(/health 依赖)→Task 5。全部有对应任务。
- **类型一致**:`MeteredRetriever`(Task 4)贯穿 resources(Task 6)与 routes(Task 8);`RequestContext` 字段(graph/retriever/cache/live_chunk_ids/allowed_access)在 Task 6 定义、Task 8 消费一致;`registry.ingest(collection, docs_dir, rebuild)` 签名 Task 6/9 一致;`ollama_status`/`vector_store_status` Task 5 定义、Task 6 `health()` 消费一致。
- **无占位符**:每步含实际代码/命令/预期输出。
- **已知验证点(实现时确认)**:LangGraph `graph.astream(stream_mode="messages")` 对"同步 node 内 `.invoke` + 流式回调"的 token 透出——与 CLI 同步 `stream()` 契约一致,astream 下 langgraph 以线程池驱动同步 node;若实测 token 不透出,回退方案为在 threadpool 内跑 `graph.stream` 并桥接为异步生成器(不改 `graph.py`)。此点在 Task 8 手动验收(真 Ollama)时确认。
