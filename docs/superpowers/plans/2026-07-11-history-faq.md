# 企业级演进 Phase 4:对话历史管理 + FAQ 沉淀

**Goal:** 解决长对话的上下文窗口截断(多轮"失忆");把高频问答沉淀为可审核的 FAQ 候选,经人工确认后走正常 ingest 入库。

**核心设计:**
- **历史裁剪在 LLM 视图层,不动 checkpointer 状态**:agent 节点构造 prompt 时,近 N 轮保留全部消息(含工具往返),更早轮次只保留"问题 + 最终回答"——检索块占历史体积的大头,答案已把它们蒸馏过,裁掉工具往返信息损失最小。裁剪必须成对移除 `AIMessage(tool_calls)` 与其 `ToolMessage`(只删一半会产生非法消息序列)
- **FAQ 沉淀 = 自动收集 + 人工把关 + 复用既有管道**:缓存条目记命中次数 → 导出高频候选为审核文档 → 人工确认后放入 `sample_docs/faq.md` → 正常 ingest(切块/哈希/ACL 全部复用,零新基建)。这是 Phase 3"模型内容不得无审核入库"边界的合规出口
- **多租户延后**:collection-per-tenant 是纯配置管道,单机 demo 无第二租户场景,做了也无从演示;README 记录模式即可

## Global Constraints

- 单测零模型依赖:历史裁剪用脚本化假模型断言"LLM 实际收到的消息";FAQ 导出用假缓存条目
- `NUM_CTX` 提到 16384(qwen3.5 GQA,8GB 显存预计可容;验收时 `ollama ps` 确认无 CPU 溢出,溢出则降 8192)

## Task 1: 对话历史裁剪(TDD)

**Files:** Modify `src/agentic_rag/graph.py`、`config.py`(`HISTORY_KEEP_TURNS = 4`、`NUM_CTX = 16384`)、`chat.py`(ChatOllama 传 num_ctx);Test `tests/test_graph_routing.py` 追加

**Produces:** `trim_for_llm(messages, keep_turns) -> list`:按 HumanMessage 划分轮次,最近 keep_turns 轮原样保留,更早轮次仅保留 Human + 最终 AI(丢弃 tool_calls AI 与 ToolMessage 对);`build_graph` 的 agent 节点改为对 `state["messages"]` 先裁剪再拼 SystemMessage

**Tests:** 轮数不足时原样;超出时旧轮工具对被剔除而问答保留;消息序列合法(无孤立 tool_calls/ToolMessage);用假模型验证 LLM 实际收到裁剪后视图而 checkpointer 状态完整

## Task 2: 缓存命中计数 + FAQ 候选导出(TDD)

**Files:** Modify `src/agentic_rag/cache.py`(store 时 `hit_count=0` + `entry_id`;lookup 命中时自增);Create `src/agentic_rag/faq.py`;Test `tests/test_cache.py`、`tests/test_faq.py`

**Produces:** `SemanticCache.entries() -> list[dict]`(全部条目含元数据);`faq.export_candidates(cache, min_hits=2) -> str`(markdown:审核说明头 + 每候选的问题/答案/来源/命中次数);`python -m agentic_rag.faq` 写 `faq_candidates.md`(加入 .gitignore,属工作文件)

**Tests:** 命中自增、未命中不增;min_hits 过滤;导出 markdown 含问题/答案/来源;空候选输出提示

## Task 3: 手动验收 + README

- 验收:① 6+ 轮长对话后追问第 1 轮内容,能答(问答对保留)且 `ollama ps` 无 CPU 溢出;② 同一问题命中缓存 2 次后 `python -m agentic_rag.faq` 导出候选;③ 把候选放入 `sample_docs/faq.md` → ingest(自动评估照跑)→ 清空缓存后新会话提问,直接从 faq.md 检索命中
- README:历史策略、FAQ 工作流三步图、多租户延后决策记录

## 完成定义

全部单测零模型依赖通过;三项验收通过;README 更新;逐 task 提交推送
