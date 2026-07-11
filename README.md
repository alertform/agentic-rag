# agentic-rag

LangChain + LangGraph 的本地 agentic RAG。全链路离线:Ollama(qwen3.5:9b + bge-m3)+ Chroma,无需任何 API key。支持 Markdown / PDF / 音频 / 视频多模态语料、增量索引、BM25+向量混合检索、ACL 权限过滤、语义缓存、检索质量评估。

## 前置条件

1. 安装 [Ollama](https://ollama.com) 并启动:`ollama serve`
2. 拉取模型:`ollama pull qwen3.5:9b && ollama pull bge-m3`(约 6.6GB + 1.2GB;8GB 显存可完整驻留,换更大模型需相应显存)
3. 安装 [uv](https://docs.astral.sh/uv/)

## 使用

```bash
uv sync                                        # 首次:建环境(自动下载 Python 3.12)
uv run python -m agentic_rag.ingest            # 增量索引 sample_docs(或传任意文档目录)
uv run python -m agentic_rag.ingest --rebuild  # 全量重建向量库
uv run python -m agentic_rag.chat              # 开始问答,exit 退出
```

- 语料格式:`.md` 直读;`.pdf` 经 pymupdf4llm 归一化;`.wav/.mp3/.m4a` 经 faster-whisper 转写(时间窗分段,引用带时间戳);`.mp4/.mov` 音轨转写 + 关键帧经 qwen3.5 vision 描述
- 增量索引:块级内容哈希做稳定 ID,只嵌入新增/变更块、清理已消失块;重跑输出"新增 0, 删除 0"
- 权限过滤:语料目录放 `acl.json`(glob → access 级别),`chat --role public|staff|manager` 决定检索可见范围(向量与 BM25 双通道过滤)
- 试试问:"星尘咖啡馆的招牌饮品是什么?"、"供应商 NX-42 供应什么?"(PDF,编号词面命中靠 BM25)、"例会决定新店选址在哪?"(音频)、"SP-2026 评审的是什么新品?"(视频画面)
- 首次带媒体语料 ingest 需要 faster-whisper small 模型(约 460MB)。国内网络推荐从 ModelScope 直接下载到本地目录(实测比 hf-mirror 快若干量级),`media` 模块会优先使用:

  ```bash
  mkdir -p models/faster-whisper-small && cd models/faster-whisper-small
  for f in config.json tokenizer.json vocabulary.txt model.bin; do
    curl -sLO "https://modelscope.cn/models/gpustack/faster-whisper-small/resolve/master/$f"
  done
  ```

## 架构

- `parsers`:类型路由把多格式文档归一化为 markdown(文本是索引,原始文件是档案)
- `media`:音频 ASR 时间窗分段、视频关键帧 + VLM 描述——时间轴 locator 复用 headers 字段,引用格式零改动
- `acl`:acl.json glob 规则给块打 access 标,检索按角色可见范围过滤
- `ingest`:解析/转写 → 切块 → bge-m3 向量化 → Chroma 增量同步
- `retrieval`:jieba 分词 BM25 + 向量检索,RRF 融合(编号/专名词面命中兜底向量盲区)+ ACL 双通道过滤
- `graph`:手写 LangGraph StateGraph,agent 节点(qwen3.5:9b)⇄ ToolNode 循环,模型自主决定检索时机与 query
- `cache`:QA 语义缓存(独立 collection,cosine 阈值命中)——**不进检索池**(模型生成内容不能无审核地成为检索来源);失效复用块内容哈希(源文档一变即作废);带 ACL 检查防跨角色泄漏;命中计数供 FAQ 沉淀筛选
- `faq`:`python -m agentic_rag.faq` 导出高频问答候选 → 人工审核 → 放入 `sample_docs/faq.md` 重跑 ingest 入库(切块/哈希/ACL 全复用)——这是"模型内容不得无审核入库"边界的合规出口
- `evals`:golden set 检索评估,`uv run python -m agentic_rag.evals` 输出纯向量 vs 混合的 hit@k / MRR;**每次 ingest 默认语料目录后自动跑一次**
- `chat`:CLI 流式问答,实时展示检索过程,回答后汇总引用来源;缓存命中显示 ⚡ 并秒回;对话历史在 LLM 视图层裁剪(最近 `HISTORY_KEEP_TURNS` 轮完整,更早轮次仅留问答对),`num_ctx` 提升至 16K

## 规模实验(21,838 块实测)

MDN 中文文档(1,921 文件)实测:嵌入 69 块/s;检索 hit@5 双通道 100%,p50 213ms;BM25 持久化索引启动加载 0.20s(vs 现场重建 8.9s)。完整数字与结论见 `docs/benchmarks/2026-07-11-scale-test.md`。多语料共存:`ingest <目录> --collection <名>`,chat/evals 同参。

## 工程决策记录

- **暂不上重排序(reranker)**:当前语料规模下评估显示纯向量与混合检索均 hit@5=100%、MRR=1.0——排序不是瓶颈,跨编码器还需 ~2.5GB torch 依赖。待语料上规模、评估分数出现分化时再引入(评估集已就位,决策有数据依据)。
- **语义缓存与知识库严格分离**:自动回写模型答案到检索池会造成幻觉自我固化(错误答案被后续检索引用,越滚越"可信")。缓存只做同问复用,三重防线:独立 collection、哈希失效、ACL 检查;沉淀入库的唯一通道是 FAQ 人工审核工作流。
- **媒体解析缓存**:ASR/VLM 属采样式解析,同一文件重复解析文本会微变,导致块哈希变化、增量索引被无意义搅动且重复付模型开销——解析结果按文件内容哈希缓存(`.media_cache/`),文件不变不重解析。
- **历史裁剪在 LLM 视图层而非状态层**:checkpointer 保留完整历史(可审计、可回放),只在拼 prompt 时裁剪;旧轮次的检索块已被回答蒸馏,成对裁掉 tool_calls/ToolMessage 损失最小且保证消息序列合法。
- **多租户延后**:collection-per-tenant 是纯配置管道,单机 demo 无第二租户场景;需要时按"每租户独立 collection + 独立缓存 + 租户级 ACL 根规则"落地即可。

- **混合检索按语料特征取舍**:规模实验发现 BM25 的价值依语料反转——稀有实体词语料(编号/专名)上是召回救星,术语密集语料(MDN)上反而注入噪声(MRR -0.011、p95 延迟 2.6x)。下一步杠杆是查询特征路由(稀有词才开 BM25 通道)而非重排序。

## 企业演进路线

Phase 6 候选:查询特征路由/加权 RRF、多租户、答案级评估(LLM judge)、检索观测面板(命中率/延迟/缓存命中率时序)。

设计文档:`docs/superpowers/specs/2026-07-11-agentic-rag-demo-design.md`
实现计划:`docs/superpowers/plans/` 下按日期排列(基础版 → 多格式+增量+混合检索 → 音视频+ACL → 语义缓存+评估)

## 测试

```bash
uv run pytest -v   # 全部单测不依赖 Ollama
```
