# 企业级演进 Phase 3:语义缓存 + 检索评估

**Goal:** 相似问题复用已验证答案(带哈希失效与 ACL 防泄漏);建立检索质量的量化评估。

**核心设计:**
- 缓存是独立 Chroma collection(`qa_cache`,cosine 空间),**不进检索池**——模型生成内容永远不能无审核地成为检索来源
- 失效复用 Phase 1 内容哈希:缓存条目记录其答案引用的 chunk_id 集,任一 id 已不在主库(=源文档变更/删除)即作废
- ACL 防泄漏:条目记录源块的 access 级别集,仅当提问者可见范围 ⊇ 该集合才命中
- **重排序刻意延后**:当前语料规模(15 块)排序不是瓶颈,且跨编码器需 torch 重依赖;待语料上规模、评估显示 top-k 排序拖后腿时另开计划

## Global Constraints

- 单测零 Ollama:缓存测试用 DeterministicFakeEmbedding + 临时 Chroma;评估指标测试用 stub 检索器
- 真实嵌入/生成只出现在 CLI 与手动验收

## Task 1: 语义缓存核心(TDD)

**Files:** Create `src/agentic_rag/cache.py`;Test `tests/test_cache.py`;config 增加 `CACHE_COLLECTION`、`CACHE_DISTANCE_THRESHOLD`

**Produces:** `SemanticCache(embeddings, persist_directory)`:
- `store(question, answer, sources, chunk_ids, access_levels)`(chunk_ids/levels/sources 存 csv metadata)
- `lookup(question, live_chunk_ids, allowed_access) -> CacheHit | None`:cosine 距离 ≤ 阈值 → 校验 chunk_ids ⊆ live_chunk_ids(失效则删条目返回 None)→ 校验 access_levels ⊆ allowed_access(不满足返回 None 但不删)
- `CacheHit`(answer/sources/question)

**Tests:** 相同问题命中;不相似问题不命中;源块 id 消失 → 失效且条目被清;access 超出可见范围 → 不命中(条目保留);阈值边界

## Task 2: chat 集成(TDD)

**Files:** Modify `retrieval.py`(检索命中记录)、`chat.py`;Test `tests/test_retrieval.py` 追加

**Produces:** `HybridRetriever.take_recorded() -> list[Document]`(累计返回并清空本轮检索命中,供缓存记 chunk_ids/access);chat 主循环:问题先查缓存 → 命中打印 `⚡ 缓存命中` + 答案 + 来源;未命中走图,流式累积答案文本,答毕且有引用时写缓存

**Tests:** take_recorded 累计/清空语义;有命中才可写缓存的门条件

## Task 3: 检索评估(TDD)

**Files:** Create `src/agentic_rag/evals.py`、`sample_evals.jsonl`;Test `tests/test_evals.py`

**Produces:** golden set(约 10 条:md/pdf/音频/视频来源 + ACL 案例,字段 question/expected_source/role);`hit_at_k(results, expected)`、`reciprocal_rank(results, expected)`;`python -m agentic_rag.evals`:对每条用对应角色的 HybridRetriever 检索,输出整体与分通道(纯向量 vs 混合)的 hit@5 / MRR 对比表

**Tests:** 指标数学正确(stub 结果);jsonl 加载与角色解析

## Task 4: 手动验收 + README

- 验收:① 同一问题问两遍,第二遍 `⚡ 缓存命中` 秒回;② 改 menu.md 重跑 ingest 后同一问题不再命中缓存(哈希失效);③ manager 问过的 internal 问题,public 角色问不会从缓存泄漏;④ `python -m agentic_rag.evals` 出分,混合检索 hit@5 ≥ 纯向量
- README:缓存机制与边界(不进检索池)、评估用法、"为什么暂不上重排序"决策记录

## 完成定义

全部单测零模型依赖通过;四项验收通过;README 更新;逐 task 提交推送
