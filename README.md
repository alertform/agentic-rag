# agentic-rag

LangChain + LangGraph 的本地 agentic RAG 练手项目。全链路离线:Ollama(qwen3.5:9b + bge-m3)+ Chroma,无需任何 API key。

## 前置条件

1. 安装 [Ollama](https://ollama.com) 并启动:`ollama serve`
2. 拉取模型:`ollama pull qwen3.5:9b && ollama pull bge-m3`(约 6.6GB + 1.2GB;8GB 显存可完整驻留,换更大模型需相应显存)
3. 安装 [uv](https://docs.astral.sh/uv/)

## 使用

```bash
uv sync                                   # 首次:建环境(自动下载 Python 3.12)
uv run python -m agentic_rag.ingest       # 索引 sample_docs(或传任意 markdown 目录路径)
uv run python -m agentic_rag.chat         # 开始问答,exit 退出
```

试试问:"星尘咖啡馆的招牌饮品是什么?"——语料是虚构的,答对必然靠检索。

## 架构

- `ingest`:Markdown 按标题切块(超长二次切分)→ bge-m3 向量化 → 本地 Chroma
- `graph`:手写 LangGraph StateGraph,agent 节点(qwen3.5:9b)⇄ ToolNode 循环,模型自主决定检索时机与 query
- `chat`:CLI 流式问答,实时展示检索过程,回答后汇总引用来源

设计文档:`docs/superpowers/specs/2026-07-11-agentic-rag-demo-design.md`
实现计划:`docs/superpowers/plans/2026-07-11-agentic-rag-demo.md`

## 测试

```bash
uv run pytest -v   # 全部单测不依赖 Ollama
```
