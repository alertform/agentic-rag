# 服务化 + 可观测性设计文档(Phase 1–3 一体化)

日期:2026-07-14
状态:设计已确认,待写实现计划
目标:把 agentic-rag 从"CLI demo"升级为**可服务化、可观测**的准生产系统——FastAPI + SSE 服务层 → 容器化 → 可观测性,三阶段作为一条连贯主线一次性设计。

## 背景与约束

- 本阶段目标是**有说服力的作品级(portfolio)工件**:真并发下正确(消除竞态)、能产出诚实的压测数字、Prometheus/Grafana 实时可看;但单进程单 worker、内存态 checkpointer、同步 ingest 即可满足。所有更重的加固(SqliteSaver、多 worker、ingest 任务队列、prometheus 多进程)以**带标签的"演进路线"决策记录**留档,不做静默省略——与既有 README「工程决策记录」(暂不上 reranker / 多租户延后)的风格一致。
- **核心不变量:server 是薄边界层**。`graph.py` / `retrieval.py` / `cache.py` / `llm.py` 一律不改,全部编排与埋点落在新增的 `server/` 文件里。CLI 路径(`chat.py`)保持原样可用。
- Phase 4(locust 压测)及之后不在本弧范围内;本设计只保证并发对象模型到那时无需返工。

## 已确认决策

| 决策点 | 结论 | 备注 |
|---|---|---|
| 并发对象模型 | **方案 C:进程级资源缓存 + 每请求轻量包装** | 见下「架构」节;A(共享检索器+锁)和 B(每请求全量重建)已否决 |
| 会话隔离 | 请求携带 `thread_id`,共享单个 `MemorySaver` | 内存态对单进程正确;持久化 checkpointer 为演进项 |
| 流式 | `app.astream(..., stream_mode="messages")` + `sse-starlette` | 复用 CLI 的 token 过滤逻辑(`agent` 节点的 `AIMessageChunk`) |
| ingest 端点 | **同步 + 进程级 asyncio 锁**(并发时 409),threadpool 执行 | demo 语料够用;后台任务队列为演进项 |
| Ollama 部署 | **留在宿主机**,容器经 `host.docker.internal` 连 | 绕开 Windows 上 GPU-in-Docker;贴合现有工作流 |
| 镜像 | **精简镜像**(chat + 文本 ingest);媒体栈可选 | 压测只用 chat;`faster-whisper/av/pillow` 经构建参数按需 |
| 依赖 | server 依赖入 **`server` 可选 extra** | 保持 CLI-only 安装精简,镜像与 `vllm` extra 同风格 |
| 埋点位置 | **边界埋点**,核心模块保持纯净 | cache 命中在调用点计量,route 在流式后读取 |
| 日志 | **structlog** JSON + `request_id` 中间件 | 每请求一条审计行 |
| worker | 单进程单 worker;多 worker + 共享 checkpointer 为演进项 | 内存态 `MemorySaver` 对单进程正确 |

## 架构:并发对象模型(承重决策)

**问题**:`HybridRetriever` 持有**每次调用的可变状态**——`self._recorded`(检索命中记录,`take_recorded()` 取走)与 `self.last_route`(`"vector"`/`"hybrid"`)。单线程 CLI 无碍;并发 HTTP 下,单个共享检索器会让两个请求互相冲掉对方的来源记录与路由标记。

**三种做法**:
- **方案 A(否决)单个共享检索器 + 锁**:串行化对话,Phase-4 QPS 曲线变成直线。
- **方案 B(否决)每请求全量重建**:正确但每次重载全量 chunks + 重建 BM25,徒增延迟。
- **方案 C(采用)进程级资源缓存 + 每请求轻量包装**。

**方案 C 细节**——`ResourceRegistry`(`server/resources.py`)在进程内缓存**昂贵、只读、线程安全**的对象:
- 一个 embeddings 客户端、一个 chat LLM 客户端(`make_embeddings()` / `make_chat_llm()`,均为无状态客户端);
- 每 collection 一个 `Chroma` store;
- 每 collection 一个预构建 `BM25Index`(经 `load_bm25_index` / `build_bm25_index`);
- 每 `(collection, role)` 可见 BM25 子集的 **LRU 缓存**(受限角色需在可见子集上重建,`manager` 全可见则直接复用 prebuilt)。

**每请求**廉价构造:一个新的 `HybridRetriever` 包装(复用缓存的 `prebuilt` 索引 + 共享 store——全可见分支零重建,受限角色命中 LRU)、一个新的 `retrieve` 工具、`llm.bind_tools([tool])`、`build_graph(bound, [tool], checkpointer=<共享 MemorySaver>)`。`_recorded` / `last_route` 现在落在每请求对象上 → **竞态结构性消失**,且热路径无昂贵重建。共享 `MemorySaver` 按 `thread_id` 分隔历史,跨请求持久(进程内)。

> `build_graph` 与 `bind_tools` 都是廉价装配操作;真正昂贵的是 chunk 加载与 BM25 构建,已被 registry 缓存。

## 项目结构(沿用「多小文件」风格)

```
src/agentic_rag/server/
├── __init__.py
├── app.py            # FastAPI 应用工厂、lifespan、中间件装配
├── routes.py         # /chat(SSE)、/ingest、/health、/roles
├── resources.py      # ResourceRegistry —— 上「架构」节的并发核心
├── schemas.py        # pydantic 请求/响应模型
├── metrics.py        # prometheus registry + 指标对象 + 辅助函数
└── logging_config.py # structlog + request_id 中间件
Dockerfile
docker-compose.yml
.env.example
docker/prometheus/prometheus.yml
docker/grafana/provisioning/
├── datasources/prometheus.yml
└── dashboards/{dashboard.yml, agentic-rag.json}
tests/test_server.py
```

`pyproject.toml` 新增 `[project.optional-dependencies]` 的 `server` extra:`fastapi`、`uvicorn[standard]`、`sse-starlette`、`prometheus-client`、`structlog`、`httpx`(测试用)。CLI-only 安装不受影响。

## Phase 1:FastAPI + SSE 服务层

### 应用工厂与依赖注入
`app.py` 提供 `create_app(registry=None)` 工厂:默认在 lifespan 内构造 `ResourceRegistry`;测试注入 stub。模块再暴露一个 `app = create_app()` 供 uvicorn 直接加载(`agentic_rag.server.app:app`),工厂本身供测试直接调用。**graph 的构造经可注入的构造点**,使 `/chat` 能用 stub graph(吐预置 token)测试,不依赖 Ollama——保持既有「pytest 不依赖 Ollama」不变量。

### 端点

| 端点 | 行为 |
|---|---|
| `POST /chat` | `EventSourceResponse`(sse-starlette)。`app.astream(..., stream_mode="messages")`,过滤 `agent` 节点的 `AIMessageChunk`(同 `chat.py:92-102`),逐 token yield `data:` 事件;末尾一条终止事件带结构化 `sources` + `route` + `cache_hit`。缓存命中短路为单条 ⚡ 事件秒回。 |
| `POST /ingest` | 同步触发增量索引(threadpool 执行,进程级 asyncio 锁,并发返回 409),返回 `{added, deleted, collection}`。 |
| `GET /health` | Ollama 可达(`preflight.check_ollama`,仅 ollama 后端)+ 向量库存在;返回 JSON 状态。 |
| `GET /roles` | 返回 `sorted(config.ROLE_ACCESS)`。 |
| `GET /metrics` | Prometheus 文本格式(Phase 3)。 |

### 请求模型(`schemas.py`)
`ChatRequest`:`question: str`、`role: str = "manager"`(校验属于 `ROLE_ACCESS`)、`collection: str = COLLECTION_NAME`、`thread_id: str`(会话隔离键)。系统边界做输入校验(pydantic + 角色白名单),失败 422/400。

### 来源与防伪
`sources` 取自 `retriever.take_recorded()` 的结构化 metadata(`d.metadata["source"]`),而非正则反解工具输出——沿用 commit `282a5c2` 已确立的防伪路径。缓存写入沿用 `chunk_id` + `access` 级别(同 `chat.py:111-119`)。

## Phase 2:容器化

### Dockerfile(多阶段)
- **build 阶段**:基于 `uv` 镜像,`uv sync`(默认含 `server` extra,不含媒体)装依赖到 `.venv`。
- **runtime 阶段**:slim base,拷贝 `.venv` + 源码,module 级 `app` 供 `CMD` 起 `uvicorn agentic_rag.server.app:app`。
- **媒体可选**:精简镜像不含媒体栈(`faster-whisper/av/pillow`);压测路径(chat)不需要。

> **约束(机制留待实现计划定,不在设计层过度指定)**:精简 server 镜像必须可在**不装媒体栈**的前提下构建;同时**默认 CLI 的 `uv sync` 必须仍让 `python -m agentic_rag.ingest` 媒体可用**。二者可同时满足的候选机制:媒体依赖入一个默认安装的 dependency-group,Dockerfile 以 `--no-group media` 排除;或多阶段只向 runtime 拷贝所需子集。选型时以"CLI 默认媒体不回退 + 镜像不含媒体"为验收判据。

### docker-compose.yml
三服务:`agentic-rag` + `prometheus` + `grafana`。
- `agentic-rag`:构建本仓库镜像;`environment` 经 `.env` 注入(`AGENTIC_RAG_BACKEND`、`OLLAMA_BASE_URL=http://host.docker.internal:11434` 等);挂载 `./chroma_db` 卷(持久化索引);暴露服务端口。
- Ollama **留宿主**:Docker Desktop(Windows/Mac)`host.docker.internal` 自动可用;Linux 需 `extra_hosts: ["host.docker.internal:host-gateway"]`(注释注明)。**备选**:注释掉的 `ollama` 容器服务 + 模型卷,作为另一 profile。
- `prometheus`:挂 `docker/prometheus/prometheus.yml`,抓 `agentic-rag:<port>/metrics`。
- `grafana`:provisioning 挂载 datasource + dashboard,开箱即面板。

### .env.example
列全:`AGENTIC_RAG_BACKEND`、`OLLAMA_BASE_URL`、`GENERATION_MODEL`、`EMBEDDING_MODEL`、服务 host/port、Grafana 初始口令等。

## Phase 3:可观测性

### 指标(`metrics.py`,`prometheus-client`)
| 指标 | 类型 | 标签 |
|---|---|---|
| 请求延迟 | `Histogram` | `endpoint` |
| 检索通道路由 | `Counter` `route_total` | `route=vector\|hybrid` |
| 缓存命中 | `Counter` `cache_lookup_total` | `result=hit\|miss` |
| 检索调用次数 | `Counter` `retrieve_calls_total` | — |
| 每轮 token 数 | `Histogram` `tokens_per_turn` | — |

**边界埋点**:cache 命中/未命中在 server 调用 `qa_cache.lookup` 处计量;route 在流式结束后读 `retriever.last_route`(每请求对象,无竞态);token 数由流式累计。核心模块(cache/retrieval)不改。

### 结构化日志(`logging_config.py`,structlog)
- `request_id` 经中间件生成 + contextvar 贯穿;JSON renderer 输出。
- 每请求一条审计行:`request_id, role, collection, chunk_ids, cache_hit, route, latency_ms`。对上「日志/监控/审计」可观测体系要求。

### Grafana 面板
provisioning 一个 datasource(Prometheus)+ 一块 dashboard:QPS、p95 延迟、缓存命中率时序、路由分布——即 README Phase 7 所述「检索观测面板」。

## 测试

不依赖 Ollama 的 pytest(`tests/test_server.py`,`fastapi.TestClient` + 注入 stub graph):
1. **SSE 帧**:stub graph 吐预置 token chunk,断言 `/chat` 事件流分帧正确、终止事件带 `sources/route/cache_hit`。
2. **缓存短路**:命中时单条 ⚡ 事件秒回,不进 graph。
3. **会话隔离**:不同 `thread_id` 历史互不串。
4. **`/health`、`/roles`**:状态/角色列表正确。
5. **registry 竞态**:并发构造的每请求包装不共享 `_recorded`(断言隔离)。

手动验收(有 Ollama 的机器):`uvicorn` 起服务 → curl SSE `/chat` 看流式;`docker compose up` 起全栈 → Grafana 看面板刷新。

## 显式演进项(带标签的决策记录,非静默省略)

各项在 README「工程决策记录」补一段"为何延后 / 如何补上":
- **多 worker + 共享/持久 checkpointer**(SqliteSaver 或 Redis):内存态 `MemorySaver` 仅对单进程正确;多 worker 需持久后端。
- **ingest 任务队列 + 状态轮询**:大语料下同步 ingest 会阻塞;demo 语料无需。
- **prometheus 多进程模式**(`PROMETHEUS_MULTIPROC_DIR`):多 worker 时才需要。
- **鉴权 / 限流**:JD 提到,但作品级 demo 暂不上。
- **k8s manifests**(Deployment + Service + ConfigMap):路线图的可选 Phase-2 stretch,本地 kind/minikube 跑通即可写"提供 K8s 部署清单"。

## 实现前研究步骤(写实现计划时执行)

按个人工作流「Research & Reuse」:实现前用 Context7 / 主文档核对 FastAPI + `sse-starlette` 的 `EventSourceResponse` 流式契约、`prometheus-client` 直方图/多进程用法、`structlog` JSON + contextvar 配置;GitHub code search 找 langgraph + FastAPI + SSE 的参考实现,能移植则移植。
