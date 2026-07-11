# agentic-rag

LangChain + LangGraph 的本地 agentic RAG。全链路离线:Ollama(qwen3.5:9b + bge-m3)+ Chroma,无需任何 API key。支持 Markdown / PDF 多格式语料、增量索引、BM25+向量混合检索。

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

- 语料格式:`.md` 直读;`.pdf` 经 pymupdf4llm 归一化为 markdown 后进同一切块管道
- 增量索引:块级内容哈希做稳定 ID,只嵌入新增/变更块、清理已消失块;重跑输出"新增 0, 删除 0"
- 试试问:"星尘咖啡馆的招牌饮品是什么?"或"供应商 NX-42 供应什么?"(后者在 PDF 语料里,编号类词面命中主要靠 BM25 通道)

## 架构

- `parsers`:类型路由把多格式文档归一化为 markdown(文本是索引,原始文件是档案)
- `ingest`:解析 → 标题切块(超长二次切分)→ bge-m3 向量化 → Chroma 增量同步
- `retrieval`:jieba 分词 BM25 + 向量检索,RRF 融合(编号/专名词面命中兜底向量盲区)
- `graph`:手写 LangGraph StateGraph,agent 节点(qwen3.5:9b)⇄ ToolNode 循环,模型自主决定检索时机与 query
- `chat`:CLI 流式问答,实时展示检索过程,回答后汇总引用来源

## 企业演进路线

Phase 2 规划:音视频接入(ASR 转写 + 关键帧 VLM 描述,metadata 带时间戳指回原始媒体)、权限过滤(metadata ACL)、重排序(reranker 精排)。

设计文档:`docs/superpowers/specs/2026-07-11-agentic-rag-demo-design.md`
实现计划:`docs/superpowers/plans/2026-07-11-agentic-rag-demo.md`、`docs/superpowers/plans/2026-07-11-enterprise-ingest.md`

## 测试

```bash
uv run pytest -v   # 全部单测不依赖 Ollama
```
