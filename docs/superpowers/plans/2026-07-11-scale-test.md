# 企业级演进 Phase 5:规模实验(demo → 数万块真实语料)

**Goal:** 用真实规模语料(MDN 中文文档,数千文件/数万块)检验全链路,产出基准报告(索引吞吐、检索延迟 p50/p95、评估分数),让 reranker 等延后决策拿到数据。

**语料:** `mdn/translated-content` zh-CN 子集(sparse checkout web/javascript + web/css + web/html),CC-BY-SA,存 `F:\scale-corpus\`(不入库,.gitignore)。

**可预见的规模瓶颈(直接列为任务):**
- BM25 索引在 chat 启动时全量 jieba 分词重建——数万块时每次启动分钟级 → **索引持久化**(ingest 时构建落盘,chat/evals 按 chunk-id 摘要校验加载,失配才重建)
- 数万块一次性 `add_documents` → **分批嵌入 + 进度输出**
- 单一 collection 写死 → ingest/chat/evals 支持 `--collection`(顺带兑现"多租户 lite")

## Global Constraints

- 单测零模型依赖不变;规模语料不入 git;两套语料共存(咖啡馆 demo 照常可用)
- 基准数字全部实测并写入 `docs/benchmarks/`,不估算

## Task 1: BM25 索引持久化(TDD)

**Files:** Modify `retrieval.py`;Test `tests/test_retrieval.py` 追加

**Produces:** `build_bm25_index(docs) -> BM25Index`(含 docs/bm25/按排序 chunk-id 集的 sha 摘要);`save_bm25_index(index, path)` / `load_bm25_index(path, expected_digest) -> BM25Index | None`(pickle;摘要失配返回 None);`HybridRetriever(..., prebuilt: BM25Index | None)` 跳过重复分词构建;ingest 同步后构建落盘 `chroma_db/bm25_<collection>.pkl`,chat/evals 优先加载

**Tests:** roundtrip 等价;摘要失配返回 None;prebuilt 与现场构建检索结果一致

## Task 2: 分批嵌入 + 多 collection 参数化(TDD)

**Files:** Modify `ingest.py`(`EMBED_BATCH` 分批 + 每批进度打印;`--collection`)、`chat.py` / `evals.py`(`--collection`,evals 另加 `--golden`);config 增 `EMBED_BATCH = 256`

**Tests:** stub store 断言按批次调用与批大小;参数透传

## Task 3: 规模语料 golden set + 基准脚本

**Files:** Create `benchmarks/golden_mdn.jsonl`(15+ 条,基于语料实际内容与路径核对后编写);evals 增 `--timing`(每模式 p50/p95 查询延迟)

## Task 4: 跑规模实验 + 基准报告

- sparse clone 语料 → `ingest F:\scale-corpus\... --collection mdn_zh`(记录吞吐)→ evals --timing 双模式
- 报告 `docs/benchmarks/2026-07-11-scale-test.md`:语料规模、索引耗时、BM25 索引加载 vs 重建耗时、检索延迟、hit@5/MRR 双通道对比、**reranker 决策复盘**(分化则立项,不分化则续观)
- README:规模实验结论与两套语料用法

## 完成定义

单测全绿;规模语料完整索引;基准报告数字齐全并推送;咖啡馆 demo 回归不受影响
