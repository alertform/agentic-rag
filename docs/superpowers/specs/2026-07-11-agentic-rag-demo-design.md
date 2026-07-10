# Agentic RAG Demo 设计文档

日期:2026-07-11
状态:设计已确认,待写实现计划
目标:用 LangChain + LangGraph 搭一个本地全离线的 agentic RAG 练手项目

## 背景与约束

- 作者是 LangChain/LangGraph 新手,此项目以练手为首要目标,实用为次要目标。
- **实现将在另一台内存更大的机器上进行**(设计机内存不足以跑 Ollama)。本文档是跨机器交接的唯一依据,实现前先读完全文。
- 作者没有任何 LLM API key(Claude Code 走订阅认证,不可复用),因此全链路本地化:LLM 与 embedding 都走 Ollama。

## 已确认决策

| 决策点 | 结论 | 备注 |
|---|---|---|
| LLM 后端 | Ollama + `qwen3:8b` | 8B 级 tool-calling 可靠性最好;约 5GB |
| Embedding | Ollama + `bge-m3` | 中英双语检索效果好;约 1.2GB;与 LLM 共用运行时 |
| 向量库 | Chroma(本地持久化 `./chroma_db/`) | 零运维 |
| 语料 | 通用 Markdown 目录,默认 `./sample_docs` | 可指向 clone 的 OrbitOS 笔记库 |
| 交互形态 | CLI 循环问答 | 流式输出 + 引用展示 |
| 架构 | **方案 C:agentic RAG**,检索作为 tool,手写 StateGraph | 不用 `create_react_agent` 预制件,练手价值在自己搭 |
| Python | **3.12,uv 固定**(不用系统 3.14) | 避开新版本生态兼容坑 |

## 项目结构

```
agentic-rag-demo/
├── pyproject.toml               # uv 管理,requires-python = ">=3.12,<3.13"
├── README.md
├── sample_docs/                 # 自带中文示例 markdown(3-5 篇,保证开箱可复现)
├── src/agentic_rag/
│   ├── config.py                # 模型名、chunk 参数、路径等常量,集中一处
│   ├── ingest.py                # 离线索引管道(CLI 入口之一)
│   ├── tools.py                 # retrieve_docs 工具定义
│   ├── graph.py                 # StateGraph 组装
│   ├── preflight.py             # Ollama 服务/模型/向量库就绪检查
│   └── chat.py                  # CLI 问答入口
├── tests/
│   ├── test_ingest.py           # 切块逻辑单测(不依赖 Ollama)
│   └── test_graph_routing.py    # 图路由单测(FakeChatModel,不依赖 Ollama)
└── docs/superpowers/specs/      # 本文档
```

## 离线索引管道(ingest)

用法:`uv run python -m agentic_rag.ingest [md目录]`,缺省 `./sample_docs`。

流程:
1. 递归收集目录下所有 `.md` 文件
2. 切块:先 `MarkdownHeaderTextSplitter` 按标题层级拆分(保留标题路径进 metadata),超过 ~800 字符的块再用 `RecursiveCharacterTextSplitter` 二次切分,重叠 ~120 字符(15%)
3. `OllamaEmbeddings(model="bge-m3")` 向量化
4. 写入 Chroma,持久化到 `./chroma_db/`(重跑 ingest 先清空旧集合,保持幂等)

每个块的 metadata:`source`(相对文件路径)、`headers`(标题路径,如 `# 架构 > ## 检索层`)。这两个字段是 CLI 引用展示的数据来源。

## 在线 Agent 图(核心)

手写 `StateGraph`,结构:

```
[agent] ──有 tool_calls──→ [tools] ──→ 回到 [agent]
   │
   └──无 tool_calls──→ END(最终回答)
```

- **State**:`messages`,用 LangGraph 标准的 `add_messages` reducer 累积
- **agent 节点**:`ChatOllama(model="qwen3:8b")` 绑定 `retrieve_docs` 工具。模型自主决定是否检索、检索几轮、每轮 query 写什么
- **tools 节点**:`ToolNode` 执行 `retrieve_docs`
- **retrieve_docs 工具**:输入 `query: str` → Chroma 相似度检索 top-5 → 返回拼接文本,每块前缀 `[来源: <source> | <headers>]`;零命中时返回明确的"知识库中没有相关内容"字符串
- **多轮记忆**:`MemorySaver` checkpointer + 固定 thread_id,进程内多轮对话共享历史
- **防失控**:调用图时设 `recursion_limit=10`

**System prompt 硬约束**(应对 8B 模型偷懒不检索的已知风险):
- 涉及知识库内容的问题必须先调用 retrieve_docs,禁止凭记忆作答
- 检索不到就明确回答"资料里没有",禁止编造
- 回答末尾必须列出所用来源

若实测工具调用仍不稳定,调优项(不进首版):换 `qwen3:14b`;或在图里加"新用户问题后首轮强制走检索"的边。

## CLI(chat)

用法:`uv run python -m agentic_rag.chat`

- 循环读入用户问题,`exit`/`quit` 退出
- 流式打印回答 token
- 工具被调用时实时打印 `🔍 检索: <query>`,命中后打印来源文件名列表
- 回答结束后汇总打印本轮引用来源

## 错误处理(preflight)

`chat.py` 和 `ingest.py` 启动时统一跑 preflight 检查,任一失败即退出并给出可操作提示:

| 检查 | 失败提示 |
|---|---|
| Ollama 服务可达(`GET localhost:11434`) | 提示启动 `ollama serve` |
| `qwen3:8b` / `bge-m3` 已拉取 | 提示对应 `ollama pull` 命令 |
| (仅 chat)`./chroma_db/` 存在且集合非空 | 提示先跑 ingest |

## 测试

不依赖 Ollama 的 pytest 单测:
1. **test_ingest**:给定构造的 markdown 文本,断言切块数量、标题 metadata、overlap 行为正确
2. **test_graph_routing**:用 FakeChatModel 预置消息序列("发起 tool call"→"收到工具结果后给出最终答案"),断言图先进 tools 节点、再回 agent、最终到 END;以及"无 tool call 直接结束"的分支

手动验收(在有 Ollama 的机器上):
1. 问 sample_docs 里有答案的问题 → 应触发检索,回答正确且引用来源
2. 问库里没有的问题 → 应明说"资料里没有",不编造
3. 追问上一轮话题 → 应利用多轮记忆正确理解指代

## 后续演进(不进首版)

- 检索质量:BM25 混合检索、reranker
- 评估:RAGAS 指标 + 小评估集
- 模型:qwen3:14b / 强制首轮检索边
- 形态:Streamlit Web UI
