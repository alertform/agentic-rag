# 规模实验报告:16 块 → 21,838 块(2026-07-11)

**语料**:MDN Web 文档中文版(`mdn/translated-content` files/zh-cn 的 web/javascript + web/css + web/html,CC-BY-SA)。1,921 个 markdown 文件,15MB 原文 → **21,838 块**(约为咖啡馆 demo 语料的 1,365 倍)。

**环境**:Windows 11 / RTX 4060 8GB / Ollama 0.31.2 / bge-m3 F16(嵌入)/ Chroma HNSW。

## 索引性能

| 阶段 | 耗时 | 备注 |
|---|---|---|
| 解析 + 切块(1,921 文件) | 1.3s | pymupdf4llm 未参与(纯 md) |
| 嵌入写库(14,926 新块) | 215.5s(**69.3 块/s**) | 批大小 128,含批间进度输出 |
| BM25 建索引 + 持久化 | 8.9s | jieba 分词 21,838 块 |
| BM25 持久化索引加载 | **0.20s**(21.3MB pickle) | 启动加速 **44x**,摘要校验防不同步 |

## 稳定性(实测故障与对策)

bge-m3 嵌入 runner 在 256 批持续负载下**两次崩溃**(分别在 2,560 / 6,912 块处,`connection refused` 到 runner 内部端口)。对策三件套后一轮跑通:批大小 256→128、失败退避重试(10s/20s/30s)、外层断点续跑(增量对账天然提供 checkpoint,重跑自动跳过已入库块)。

**教训**:嵌入服务要按"会崩"来设计管道;内容哈希增量对账让"断点续跑"零成本。

## 检索质量与延迟(15 条 golden set,k=5)

| 通道 | hit@5 | MRR | p50 | p95 |
|---|---|---|---|---|
| 纯向量 | 15/15(100%) | **0.933** | 213ms | 249ms |
| 混合(BM25+RRF) | 15/15(100%) | 0.922 | 272ms | 645ms |

## 结论

1. **规模可用性**:21.8k 块下 bge-m3 + Chroma HNSW 召回无压力(hit@5 100%),检索 p50 约 213ms——相对生成耗时(秒级)可忽略。
2. **reranker 判决:继续不立项**。瓶颈不在 top-k 排序(MRR ≈ 0.93,唯一非 rank-1 案例是双相关来源之争)。数据支撑的决策,非直觉。
3. **混合检索的价值依语料特征反转**:咖啡馆语料上 BM25 是稀有实体词(NX-42)的救星;MDN 这类**术语密集**语料上,高频词(Array/Promise 遍布全库)让 BM25 注入噪声候选,MRR 反降 0.011、p95 延迟 2.6x。
   → **下一步的正确杠杆是查询特征路由**(仅当查询含稀有词/精确代码词时启用 BM25 通道)或加权 RRF,而非 reranker。列入 Phase 6 候选。
4. **多语料隔离可用**:`--collection mdn_zh` 与咖啡馆 demo 共存,互不影响(咖啡馆回归:16 块,评估分数不变)。

## 复现

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/mdn/translated-content <dir>
cd <dir> && git sparse-checkout set files/zh-cn/web/javascript files/zh-cn/web/css files/zh-cn/web/html
uv run python -m agentic_rag.ingest <dir>/files/zh-cn --collection mdn_zh
uv run python -m agentic_rag.evals --collection mdn_zh --golden benchmarks/golden_mdn.jsonl --timing
uv run python -m agentic_rag.chat --collection mdn_zh
```
