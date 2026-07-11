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
- `cache`:QA 语义缓存(独立 collection,cosine 阈值命中)——**不进检索池**(模型生成内容不能无审核地成为检索来源);失效复用块内容哈希(源文档一变即作废);带 ACL 检查防跨角色泄漏
- `evals`:golden set 检索评估,`uv run python -m agentic_rag.evals` 输出纯向量 vs 混合的 hit@k / MRR
- `chat`:CLI 流式问答,实时展示检索过程,回答后汇总引用来源;缓存命中显示 ⚡ 并秒回

## 工程决策记录

- **暂不上重排序(reranker)**:当前语料 15 块,评估显示纯向量与混合检索均 hit@5=100%、MRR=1.0——排序不是瓶颈,跨编码器还需 ~2.5GB torch 依赖。待语料上规模、评估分数出现分化时再引入(评估集已就位,决策有数据依据)。
- **语义缓存与知识库严格分离**:自动回写模型答案到检索池会造成幻觉自我固化(错误答案被后续检索引用,越滚越"可信")。缓存只做同问复用,三重防线:独立 collection、哈希失效、ACL 检查。

## 企业演进路线

Phase 4 候选:重排序(评估分化后)、多租户 collection 隔离、对话历史管理(裁剪/滚动摘要)、FAQ 沉淀(自动收集 + 人工审核入库)。

设计文档:`docs/superpowers/specs/2026-07-11-agentic-rag-demo-design.md`
实现计划:`docs/superpowers/plans/` 下按日期排列(基础版 → 多格式+增量+混合检索 → 音视频+ACL → 语义缓存+评估)

## 测试

```bash
uv run pytest -v   # 全部单测不依赖 Ollama
```
