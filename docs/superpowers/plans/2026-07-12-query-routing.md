# 企业级演进 Phase 6:查询特征路由

**Goal:** 把 Phase 5 规模实验的发现(BM25 价值随语料特征反转)变成机制:按查询词元的语料稀有度决定是否启用 BM25 通道——稀有词查询保留词面精确命中,常见词查询免除 BM25 噪声与延迟。

**判据(语料自适应,零硬编码模式):** 建索引时统计词元文档频率 df;查询词元中存在"稀有但在库"词元(`0 < df ≤ max(3, ⌈0.5%·N⌉)`)→ 混合检索,否则纯向量。
- 咖啡馆(N=16,阈值 3):"NX"(df≈2)触发 → NX-42 类查询仍走混合 ✓
- MDN(N=21838,阈值 110):"allSettled"(df≈十几)触发;"样式/查询"(df 数千)不触发 ✓
- 词元完全不在库(df=0)不触发——BM25 对它无能为力,开了也是白开

## Task 1: BM25Index 携带 df 统计(TDD)

**Files:** Modify `retrieval.py`;Test `tests/test_retrieval.py` 追加

**Produces:** `BM25Index` 增加 `df: dict[str,int]`、`doc_count: int`(build 时按文档去重词元统计);`load_bm25_index` 对缺 df 字段的旧版 pickle 返回 None(视为过期,调用方重建);`HybridRetriever` 内部统一持有 `BM25Index`(现场构建路径也经 `build_bm25_index`)

## Task 2: 路由逻辑(TDD)

**Files:** Modify `retrieval.py`、`config.py`(`ROUTE_RARE_DF_ABS = 3`、`ROUTE_RARE_DF_RATIO = 0.005`)

**Produces:** `HybridRetriever.should_use_bm25(query) -> bool`;`similarity_search` 按路由决定单/双通道;`last_route` 属性供测试与观测

**Tests:** 稀有词查询融合结果含 BM25 独有文档;纯常见词查询结果与纯向量一致(BM25 噪声不进入);阈值边界(df 恰等于阈值);df=0 词元不触发;空语料安全

## Task 3: 双语料复验 + 报告更新

- 重跑两套 ingest(重建含 df 的持久化索引)与 evals --timing
- 预期:咖啡馆 11 条不回归(NX-42 仍混合);MDN 15 条 MRR ≥ 纯向量基线 0.933(常见词查询噪声移除)且混合延迟趋近纯向量
- 基准报告追加"路由后"对照表;README 工程决策记录更新

## 完成定义

单测全绿零模型依赖;双语料评估达预期或如实记录差异;报告与 README 更新;提交推送
