# Agentic RAG Demo 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现 spec(`docs/superpowers/specs/2026-07-11-agentic-rag-demo-design.md`)定义的本地全离线 agentic RAG:Markdown 索引管道 + 手写 LangGraph agent 图 + CLI 问答。

**Architecture:** 离线侧 `ingest` 把 Markdown 目录切块、经 Ollama bge-m3 向量化写入本地 Chroma;在线侧手写 StateGraph(agent 节点 ⇄ ToolNode 循环),qwen3:8b 自主决定调用 `retrieve_docs` 工具检索,CLI 流式输出并展示引用。

**Tech Stack:** Python 3.12 + uv、langgraph、langchain-ollama、langchain-chroma、langchain-text-splitters、pytest。

## Global Constraints

- Python **3.12**,由 uv 管理(`.python-version` 固定);不要用系统 Python 3.14
- **全本地**:只依赖 Ollama(`http://localhost:11434`),全程无任何 API key
- 模型名:生成 `qwen3:8b`,embedding `bge-m3`;向量库持久化在 `./chroma_db/`,collection 名 `agentic_rag`
- **禁止用 `create_react_agent` 预制件**(练手要求手写图;`ToolNode` 允许用)
- 所有 pytest 单测**不得依赖 Ollama**;需要 Ollama 的验证一律是"手动验证"步骤
- Task 1–6 在任何机器都能做(含跑测试);**Task 7–8 需要装好 Ollama 的机器**(`ollama pull qwen3:8b && ollama pull bge-m3`)
- 每个 Task 结束 commit 一次,commit message 用中文按仓库惯例:`<type>: <描述>`

---

### Task 1: 项目脚手架(uv 工程、config、示例语料)

**Files:**
- Create: `pyproject.toml`
- Create: `.python-version`
- Create: `.gitignore`
- Create: `src/agentic_rag/__init__.py`
- Create: `src/agentic_rag/config.py`
- Create: `sample_docs/menu.md`
- Create: `sample_docs/membership.md`
- Create: `sample_docs/operations.md`

**Interfaces:**
- Consumes: 无
- Produces: `agentic_rag.config` 模块,常量 `OLLAMA_BASE_URL: str`、`GENERATION_MODEL: str`、`EMBEDDING_MODEL: str`、`PROJECT_ROOT: Path`、`SAMPLE_DOCS_DIR: Path`、`CHROMA_DIR: Path`、`COLLECTION_NAME: str`、`CHUNK_SIZE: int`、`CHUNK_OVERLAP: int`、`TOP_K: int`、`RECURSION_LIMIT: int`;后续所有 Task 从这里取配置

- [ ] **Step 1: 写 `pyproject.toml`**

```toml
[project]
name = "agentic-rag-demo"
version = "0.1.0"
description = "LangChain + LangGraph 本地 agentic RAG 练手项目"
requires-python = ">=3.12,<3.13"
dependencies = [
    "langchain-core>=0.3",
    "langgraph>=0.2.60",
    "langchain-ollama>=0.3",
    "langchain-chroma>=0.2",
    "langchain-text-splitters>=0.3",
]

[dependency-groups]
dev = ["pytest>=8.0"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/agentic_rag"]
```

- [ ] **Step 2: 写 `.python-version`**

```
3.12
```

- [ ] **Step 3: 写 `.gitignore`**

```
.venv/
__pycache__/
*.pyc
.pytest_cache/
chroma_db/
```

- [ ] **Step 4: 写包骨架**

`src/agentic_rag/__init__.py`:

```python
```

(空文件即可)

`src/agentic_rag/config.py`:

```python
"""集中配置:模型、路径、切块与检索参数。"""
from pathlib import Path

OLLAMA_BASE_URL = "http://localhost:11434"
GENERATION_MODEL = "qwen3:8b"
EMBEDDING_MODEL = "bge-m3"

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SAMPLE_DOCS_DIR = PROJECT_ROOT / "sample_docs"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "agentic_rag"

CHUNK_SIZE = 800       # 二次切分块大小(字符)
CHUNK_OVERLAP = 120    # 块间重叠,约 15%
TOP_K = 5              # 检索返回块数
RECURSION_LIMIT = 10   # 图递归上限,防无限检索循环
```

- [ ] **Step 5: 写三篇虚构示例语料**(内容虚构是有意的:模型预训练里不可能有,答对必然靠检索)

`sample_docs/menu.md`:

```markdown
# 星尘咖啡馆菜单

## 招牌饮品

星尘拿铁是本店招牌,售价 32 元,采用秘制配方:双份浓缩、燕麦奶打底,顶部撒海盐焦糖碎。
月岩冷萃售价 28 元,冷泡 16 小时,带可可与柑橘尾韵。

## 烘焙点心

陨石可颂 18 元,内馅为黑芝麻流心。
环形山贝果 15 元,每日限量 40 个,售完即止。

## 季节限定

彗星气泡美式仅夏季供应,售价 26 元,加青柠与迷迭香。
```

`sample_docs/membership.md`:

```markdown
# 星尘咖啡馆会员制度

## 星环会员

星环会员年费 199 元,权益包括:全场饮品 88 折、每月赠饮一杯任意中杯饮品、生日当天免费升杯。

## 积分规则

消费 1 元累积 1 星尘点。500 星尘点可兑换任意饮品一杯,1000 星尘点可兑换烘焙点心套装。
积分有效期为自获得之日起 12 个月。

## 会员日

每月 8 号为会员日,星环会员当日消费双倍积分。
```

`sample_docs/operations.md`:

```markdown
# 星尘咖啡馆门店运营规范

## 营业时间

周一、周二、周四至周日:8:00 - 22:00。
每周三闭店进行设备维护,不对外营业。

## 特色活动

望远镜之夜:每月最后一个周六 20:00 起,店内天台开放天文观测,会员免费,非会员收费 30 元。

## 退换规则

饮品制作错误可当场免费重做;预付订单在取餐前 24 小时内可全额退款,超时不予退款。
```

- [ ] **Step 6: 验证环境**

Run: `uv sync && uv run python -c "from agentic_rag import config; print(config.GENERATION_MODEL)"`
Expected: 输出 `qwen3:8b`(uv 首次会自动下载 Python 3.12 并建 venv)

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "feat: 项目脚手架 (uv 工程 + config + 星尘咖啡馆示例语料)"
```

---

### Task 2: Markdown 切块逻辑(TDD)

**Files:**
- Create: `src/agentic_rag/ingest.py`(本 Task 只写 `split_markdown`)
- Test: `tests/test_ingest.py`

**Interfaces:**
- Consumes: `agentic_rag.config` 的 `CHUNK_SIZE`、`CHUNK_OVERLAP`
- Produces: `split_markdown(text: str, source: str) -> list[Document]`——每个 `Document` 的 `metadata` 恰好为 `{"source": <传入的 source>, "headers": <"h1 > h2 > h3" 形式的标题路径,顶层无标题时为 "">}`

- [ ] **Step 1: 写失败测试 `tests/test_ingest.py`**

```python
from agentic_rag.ingest import split_markdown


def test_split_keeps_source_and_header_path():
    text = "# 手册\n\n## 菜单\n\n拿铁 30 元。\n\n## 会员\n\n年费 200 元。"
    chunks = split_markdown(text, "manual.md")
    assert chunks, "应至少切出一个块"
    assert all(c.metadata["source"] == "manual.md" for c in chunks)
    menu = [c for c in chunks if "拿铁" in c.page_content]
    assert menu and menu[0].metadata["headers"] == "手册 > 菜单"


def test_long_section_resplit_with_overlap():
    body = "".join(f"第{i}句话内容。" for i in range(300))  # 远超 800 字符
    chunks = split_markdown(f"# 长文\n\n{body}", "long.md")
    assert len(chunks) >= 2, "超长小节应被二次切分"
    assert all(len(c.page_content) <= 800 for c in chunks)
    # 相邻块之间应有重叠:前块的尾部内容出现在后块里
    assert chunks[0].page_content[-60:] in chunks[1].page_content
    assert all(c.metadata["headers"] == "长文" for c in chunks)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_ingest.py -v`
Expected: FAIL,`ModuleNotFoundError: No module named 'agentic_rag.ingest'`

- [ ] **Step 3: 写实现 `src/agentic_rag/ingest.py`**

```python
"""离线索引管道:Markdown 目录 → 切块 → 向量化 → Chroma。"""
from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from agentic_rag import config

_HEADERS_TO_SPLIT_ON = [("#", "h1"), ("##", "h2"), ("###", "h3")]


def split_markdown(text: str, source: str) -> list[Document]:
    """按标题层级切块,超长小节按字符二次切分;metadata 带 source 和标题路径。"""
    header_splitter = MarkdownHeaderTextSplitter(_HEADERS_TO_SPLIT_ON, strip_headers=False)
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )
    chunks: list[Document] = []
    for section in header_splitter.split_text(text):
        headers = " > ".join(
            section.metadata[key]
            for key in ("h1", "h2", "h3")
            if section.metadata.get(key)
        )
        for piece in char_splitter.split_documents([section]):
            piece.metadata = {"source": source, "headers": headers}
            chunks.append(piece)
    return chunks
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_ingest.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add tests/test_ingest.py src/agentic_rag/ingest.py
git commit -m "feat: Markdown 标题切块 + 超长二次切分 (带 source/headers metadata)"
```

---

### Task 3: preflight 就绪检查(TDD)

**Files:**
- Create: `src/agentic_rag/preflight.py`
- Test: `tests/test_preflight.py`

**Interfaces:**
- Consumes: `agentic_rag.config`
- Produces: `check_ollama(require_generation: bool = True) -> None`(失败时 `sys.exit` 带可操作提示)、`check_vector_store() -> None`(向量库空/缺失时 `sys.exit`)、内部函数 `_installed_models() -> list[str] | None`(连不上 Ollama 返回 `None`;测试会 monkeypatch 它)

- [ ] **Step 1: 写失败测试 `tests/test_preflight.py`**

```python
import pytest

from agentic_rag import preflight


def test_ollama_down_exits_with_serve_hint(monkeypatch):
    monkeypatch.setattr(preflight, "_installed_models", lambda: None)
    with pytest.raises(SystemExit) as exc:
        preflight.check_ollama()
    assert "ollama serve" in str(exc.value)


def test_missing_model_exits_with_pull_hint(monkeypatch):
    monkeypatch.setattr(preflight, "_installed_models", lambda: ["qwen3:8b"])
    with pytest.raises(SystemExit) as exc:
        preflight.check_ollama()
    assert "ollama pull bge-m3" in str(exc.value)


def test_all_models_present_passes(monkeypatch):
    # Ollama 的 tags 接口常带 :latest 后缀,匹配逻辑要兼容
    monkeypatch.setattr(preflight, "_installed_models", lambda: ["qwen3:8b", "bge-m3:latest"])
    preflight.check_ollama()  # 不应抛出


def test_embedding_only_mode(monkeypatch):
    monkeypatch.setattr(preflight, "_installed_models", lambda: ["bge-m3:latest"])
    preflight.check_ollama(require_generation=False)  # ingest 场景:只要 embedding 模型
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_preflight.py -v`
Expected: FAIL,`ModuleNotFoundError: No module named 'agentic_rag.preflight'`

- [ ] **Step 3: 写实现 `src/agentic_rag/preflight.py`**

```python
"""启动前检查:Ollama 服务可达、模型已拉取、向量库非空。"""
import json
import sys
import urllib.error
import urllib.request

from agentic_rag import config


def _installed_models() -> list[str] | None:
    """返回 Ollama 已安装模型名列表;服务不可达时返回 None。"""
    try:
        with urllib.request.urlopen(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=3) as resp:
            data = json.load(resp)
        return [m["name"] for m in data.get("models", [])]
    except (urllib.error.URLError, OSError, ValueError):
        return None


def _has_model(models: list[str], name: str) -> bool:
    return any(m == name or m.startswith(f"{name}:") for m in models)


def check_ollama(require_generation: bool = True) -> None:
    models = _installed_models()
    if models is None:
        sys.exit(f"[preflight] 连不上 Ollama ({config.OLLAMA_BASE_URL})。先运行: ollama serve")
    required = [config.EMBEDDING_MODEL]
    if require_generation:
        required.append(config.GENERATION_MODEL)
    for name in required:
        if not _has_model(models, name):
            sys.exit(f"[preflight] 缺少模型 {name}。先运行: ollama pull {name}")


def check_vector_store() -> None:
    import chromadb

    try:
        client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        count = client.get_collection(config.COLLECTION_NAME).count()
    except Exception:
        count = 0
    if count == 0:
        sys.exit("[preflight] 向量库为空。先运行: uv run python -m agentic_rag.ingest [md目录]")
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_preflight.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add tests/test_preflight.py src/agentic_rag/preflight.py
git commit -m "feat: preflight 检查 (Ollama 服务/模型/向量库, 失败给可操作提示)"
```

---

### Task 4: ingest 管道收尾(目录加载 + 建库 + CLI 入口)

**Files:**
- Modify: `src/agentic_rag/ingest.py`(追加 `load_documents`、`build_vector_store`、`main`)
- Test: `tests/test_ingest.py`(追加 `test_load_documents_walks_tree`)

**Interfaces:**
- Consumes: Task 2 的 `split_markdown`、Task 3 的 `check_ollama`
- Produces: `load_documents(docs_dir: Path) -> list[Document]`(source 为相对 docs_dir 的路径);`build_vector_store(chunks: list[Document]) -> Chroma`(幂等,重跑先清空);`python -m agentic_rag.ingest [md目录]` 入口。Task 7 依赖同款 Chroma 打开方式(相同 collection_name / persist_directory / embedding)

- [ ] **Step 1: 追加失败测试到 `tests/test_ingest.py`**

```python
from agentic_rag.ingest import load_documents


def test_load_documents_walks_tree(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.md").write_text("# A\n\n内容甲。", encoding="utf-8")
    (tmp_path / "sub" / "b.md").write_text("# B\n\n内容乙。", encoding="utf-8")
    (tmp_path / "ignore.txt").write_text("非 markdown", encoding="utf-8")
    chunks = load_documents(tmp_path)
    assert {c.metadata["source"] for c in chunks} == {"a.md", "sub/b.md"}
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_ingest.py::test_load_documents_walks_tree -v`
Expected: FAIL,`ImportError: cannot import name 'load_documents'`

- [ ] **Step 3: 在 `src/agentic_rag/ingest.py` 追加实现**

文件顶部 import 区追加:

```python
import sys
from pathlib import Path
```

文件末尾追加:

```python
def load_documents(docs_dir: Path) -> list[Document]:
    """递归加载目录下所有 .md,返回切好的块。"""
    chunks: list[Document] = []
    for md_file in sorted(docs_dir.rglob("*.md")):
        source = md_file.relative_to(docs_dir).as_posix()
        chunks.extend(split_markdown(md_file.read_text(encoding="utf-8"), source))
    return chunks


def build_vector_store(chunks: list[Document]):
    """向量化写入本地 Chroma;先清空旧集合保证幂等。"""
    from langchain_chroma import Chroma
    from langchain_ollama import OllamaEmbeddings

    embeddings = OllamaEmbeddings(
        model=config.EMBEDDING_MODEL, base_url=config.OLLAMA_BASE_URL
    )
    store = Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(config.CHROMA_DIR),
    )
    store.reset_collection()
    store.add_documents(chunks)
    return store


def main() -> None:
    from agentic_rag.preflight import check_ollama

    docs_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else config.SAMPLE_DOCS_DIR
    if not docs_dir.is_dir():
        sys.exit(f"[ingest] 目录不存在: {docs_dir}")
    check_ollama(require_generation=False)
    chunks = load_documents(docs_dir)
    if not chunks:
        sys.exit(f"[ingest] {docs_dir} 下没有 .md 文件")
    build_vector_store(chunks)
    print(f"[ingest] 已索引 {len(chunks)} 个文档块 ← {docs_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 跑全部测试确认通过**

Run: `uv run pytest -v`
Expected: 7 passed(test_ingest 3 + test_preflight 4)

- [ ] **Step 5: 手动验证(仅限装好 Ollama 的机器;设计机跳过,在 Task 8 一并做)**

Run: `uv run python -m agentic_rag.ingest`
Expected: 输出形如 `[ingest] 已索引 N 个文档块 ← .../sample_docs`(N 约 9~12),生成 `chroma_db/` 目录

- [ ] **Step 6: Commit**

```bash
git add tests/test_ingest.py src/agentic_rag/ingest.py
git commit -m "feat: ingest 管道收尾 (目录递归加载 + Chroma 幂等建库 + CLI 入口)"
```

---

### Task 5: retrieve_docs 检索工具(TDD)

**Files:**
- Create: `src/agentic_rag/tools.py`
- Test: `tests/test_tools.py`

**Interfaces:**
- Consumes: 无(向量库以鸭子类型注入,任何有 `similarity_search(query, k)` 的对象都行)
- Produces: `NO_HIT_MESSAGE: str`;`format_chunks(docs: list[Document]) -> str`(每块前缀 `[来源: <source> | <headers>]`,块间 `\n\n---\n\n` 分隔);`make_retrieve_tool(vector_store, k: int, verbose: bool = False)` 返回名为 `retrieve_docs` 的 LangChain tool。Task 6/7 直接把返回值绑给 LLM 和 ToolNode;Task 7 的来源正则依赖这里的前缀格式

- [ ] **Step 1: 写失败测试 `tests/test_tools.py`**

```python
from langchain_core.documents import Document

from agentic_rag.tools import NO_HIT_MESSAGE, format_chunks, make_retrieve_tool


class StubStore:
    def __init__(self, docs):
        self._docs = docs

    def similarity_search(self, query, k):
        return self._docs[:k]


def _doc(text, source, headers):
    return Document(page_content=text, metadata={"source": source, "headers": headers})


def test_format_chunks_prefixes_source_and_headers():
    out = format_chunks([_doc("拿铁 32 元。", "menu.md", "菜单 > 招牌饮品")])
    assert out.startswith("[来源: menu.md | 菜单 > 招牌饮品]")
    assert "拿铁 32 元。" in out


def test_format_chunks_empty_returns_no_hit():
    assert format_chunks([]) == NO_HIT_MESSAGE


def test_retrieve_tool_invokes_store_and_formats():
    store = StubStore([_doc("年费 199 元。", "membership.md", "会员制度")])
    tool = make_retrieve_tool(store, k=5)
    assert tool.name == "retrieve_docs"
    out = tool.invoke({"query": "会员年费"})
    assert "[来源: membership.md | 会员制度]" in out
    assert "年费 199 元。" in out


def test_retrieve_tool_no_hit():
    tool = make_retrieve_tool(StubStore([]), k=5)
    assert tool.invoke({"query": "不存在的内容"}) == NO_HIT_MESSAGE
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_tools.py -v`
Expected: FAIL,`ModuleNotFoundError: No module named 'agentic_rag.tools'`

- [ ] **Step 3: 写实现 `src/agentic_rag/tools.py`**

```python
"""retrieve_docs 检索工具:agent 图里唯一的工具。"""
from langchain_core.documents import Document
from langchain_core.tools import tool

NO_HIT_MESSAGE = "知识库中没有相关内容。"


def format_chunks(docs: list[Document]) -> str:
    """把检索命中的块拼成带来源标注的文本;零命中返回 NO_HIT_MESSAGE。"""
    if not docs:
        return NO_HIT_MESSAGE
    parts = []
    for doc in docs:
        headers = doc.metadata.get("headers") or "-"
        parts.append(f"[来源: {doc.metadata['source']} | {headers}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def make_retrieve_tool(vector_store, k: int, verbose: bool = False):
    """工厂:把向量库封装成 retrieve_docs 工具。verbose 时向终端打印检索过程。"""

    @tool
    def retrieve_docs(query: str) -> str:
        """在本地知识库中检索与 query 相关的文档片段。涉及知识库内容的问题必须先调用本工具。"""
        if verbose:
            print(f"\n🔍 检索: {query}", flush=True)
        docs = vector_store.similarity_search(query, k=k)
        if verbose and docs:
            files = sorted({d.metadata["source"] for d in docs})
            print(f"   命中: {', '.join(files)}", flush=True)
        return format_chunks(docs)

    return retrieve_docs
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_tools.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add tests/test_tools.py src/agentic_rag/tools.py
git commit -m "feat: retrieve_docs 工具 (来源标注格式化 + 零命中显式返回)"
```

---

### Task 6: LangGraph agent 图(TDD)

**Files:**
- Create: `src/agentic_rag/graph.py`
- Test: `tests/test_graph_routing.py`

**Interfaces:**
- Consumes: Task 5 产出的 tool(测试里用假 tool 替代)
- Produces: `SYSTEM_PROMPT: str`;`AgentState`(TypedDict,`messages: Annotated[list[AnyMessage], add_messages]`);`build_graph(llm_with_tools, tools, checkpointer=None)` 返回编译后的图。**约定:传入的 `llm_with_tools` 必须已经 `bind_tools` 过**(生产码 `ChatOllama(...).bind_tools([tool])`,测试直接传脚本化假模型),图内部不再 bind

- [ ] **Step 1: 写失败测试 `tests/test_graph_routing.py`**

```python
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver

from agentic_rag.graph import build_graph


@tool
def retrieve_docs(query: str) -> str:
    """测试用假检索工具。"""
    return "[来源: menu.md | 菜单 > 招牌饮品]\n星尘拿铁 32 元。"


def _scripted(*messages):
    return GenericFakeChatModel(messages=iter(messages))


def test_tool_call_routes_to_tools_then_back_to_agent():
    llm = _scripted(
        AIMessage(
            content="",
            tool_calls=[{"name": "retrieve_docs", "args": {"query": "招牌饮品"}, "id": "call_1", "type": "tool_call"}],
        ),
        AIMessage(content="招牌是星尘拿铁,32 元。\n来源: menu.md"),
    )
    app = build_graph(llm, [retrieve_docs])
    result = app.invoke({"messages": [HumanMessage("招牌饮品是什么?")]})
    assert [m.type for m in result["messages"]] == ["human", "ai", "tool", "ai"]
    assert "星尘拿铁" in result["messages"][-1].content
    assert "星尘拿铁 32 元" in result["messages"][2].content  # 工具结果确实进了对话


def test_no_tool_call_goes_straight_to_end():
    app = build_graph(_scripted(AIMessage(content="你好!")), [retrieve_docs])
    result = app.invoke({"messages": [HumanMessage("你好")]})
    assert [m.type for m in result["messages"]] == ["human", "ai"]


def test_checkpointer_keeps_history_across_turns():
    llm = _scripted(AIMessage(content="第一轮回答"), AIMessage(content="第二轮回答"))
    app = build_graph(llm, [retrieve_docs], checkpointer=MemorySaver())
    cfg = {"configurable": {"thread_id": "t1"}}
    app.invoke({"messages": [HumanMessage("问题A")]}, cfg)
    result = app.invoke({"messages": [HumanMessage("问题B")]}, cfg)
    assert [m.type for m in result["messages"]] == ["human", "ai", "human", "ai"]
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_graph_routing.py -v`
Expected: FAIL,`ModuleNotFoundError: No module named 'agentic_rag.graph'`

- [ ] **Step 3: 写实现 `src/agentic_rag/graph.py`**

```python
"""手写 LangGraph agent 图:agent 节点 ⇄ ToolNode 循环。"""
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

SYSTEM_PROMPT = """你是一个基于本地知识库的问答助手。规则:
1. 凡是可能涉及知识库内容的问题,必须先调用 retrieve_docs 工具检索,禁止凭记忆直接回答。
2. 一次检索结果不理想时,可以换个说法再检索,但总共不要超过 3 次。
3. 只根据检索到的内容回答;检索不到就明确说"资料里没有找到",禁止编造。
4. 回答末尾用"来源:"列出实际引用的文件名。"""


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def build_graph(llm_with_tools, tools, checkpointer=None):
    """组装 agent 图。llm_with_tools 必须已 bind_tools;tools 给 ToolNode 执行。"""

    def agent(state: AgentState) -> dict:
        messages = [SystemMessage(SYSTEM_PROMPT), *state["messages"]]
        return {"messages": [llm_with_tools.invoke(messages)]}

    def route(state: AgentState) -> str:
        last = state["messages"][-1]
        return "tools" if getattr(last, "tool_calls", None) else END

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=checkpointer)
```

- [ ] **Step 4: 跑全部测试确认通过**

Run: `uv run pytest -v`
Expected: 14 passed

- [ ] **Step 5: Commit**

```bash
git add tests/test_graph_routing.py src/agentic_rag/graph.py
git commit -m "feat: 手写 StateGraph agent 图 (条件边路由 + ToolNode 循环 + checkpointer)"
```

---

### Task 7: CLI 问答入口

**Files:**
- Create: `src/agentic_rag/chat.py`

**Interfaces:**
- Consumes: `config`、`preflight.check_ollama` / `check_vector_store`、`make_retrieve_tool`、`build_graph`
- Produces: `python -m agentic_rag.chat` 交互入口(最终用户界面,无下游消费方)

- [ ] **Step 1: 写实现 `src/agentic_rag/chat.py`**

```python
"""CLI 交互问答:流式输出 + 检索过程展示 + 来源汇总。"""
import re

from langchain_chroma import Chroma
from langchain_core.messages import AIMessageChunk, HumanMessage, ToolMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.checkpoint.memory import MemorySaver

from agentic_rag import config, preflight
from agentic_rag.graph import build_graph
from agentic_rag.tools import make_retrieve_tool

_SOURCE_RE = re.compile(r"\[来源: ([^|\]]+?) \|")


def main() -> None:
    preflight.check_ollama()
    preflight.check_vector_store()

    embeddings = OllamaEmbeddings(
        model=config.EMBEDDING_MODEL, base_url=config.OLLAMA_BASE_URL
    )
    store = Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(config.CHROMA_DIR),
    )
    retrieve = make_retrieve_tool(store, k=config.TOP_K, verbose=True)
    # reasoning=False 关闭 qwen3 思考段;若所装 langchain-ollama 不支持该参数,删掉即可
    llm = ChatOllama(
        model=config.GENERATION_MODEL, base_url=config.OLLAMA_BASE_URL, reasoning=False
    )
    app = build_graph(llm.bind_tools([retrieve]), [retrieve], checkpointer=MemorySaver())
    run_config = {
        "configurable": {"thread_id": "cli"},
        "recursion_limit": config.RECURSION_LIMIT,
    }

    print("Agentic RAG demo — 输入问题开始对话,exit 退出")
    while True:
        try:
            question = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break

        sources: set[str] = set()
        print("助手: ", end="", flush=True)
        for chunk, meta in app.stream(
            {"messages": [HumanMessage(question)]}, run_config, stream_mode="messages"
        ):
            if isinstance(chunk, ToolMessage):
                sources |= set(_SOURCE_RE.findall(str(chunk.content)))
            elif (
                isinstance(chunk, AIMessageChunk)
                and meta.get("langgraph_node") == "agent"
                and chunk.content
            ):
                print(chunk.content, end="", flush=True)
        print()
        if sources:
            print(f"—— 本轮引用: {', '.join(sorted(sources))}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 静态验证(任何机器)**

Run: `uv run python -c "import agentic_rag.chat" && uv run pytest -v`
Expected: import 无报错,14 passed

- [ ] **Step 3: 冒烟验证(仅限 Ollama 机器;设计机跳过,在 Task 8 一并做)**

Run: `uv run python -m agentic_rag.ingest && uv run python -m agentic_rag.chat`,问"星尘咖啡馆的招牌饮品是什么?"
Expected: 打印 `🔍 检索: ...` 和命中文件,流式输出含"星尘拿铁""32"的回答,结尾有 `—— 本轮引用: menu.md`

- [ ] **Step 4: Commit**

```bash
git add src/agentic_rag/chat.py
git commit -m "feat: CLI 问答入口 (流式输出 + 检索过程展示 + 来源汇总)"
```

---

### Task 8: README 使用说明 + 手动验收(需 Ollama 机器)

**Files:**
- Modify: `README.md`(整体重写为下面内容)

**Interfaces:**
- Consumes: 前七个 Task 的全部成果
- Produces: 面向使用者的 README;spec 定义的三条验收结论

- [ ] **Step 1: 重写 `README.md`**(注意:下面用四反引号包裹,因为内容本身含代码围栏)

````markdown
# agentic-rag-demo

LangChain + LangGraph 的本地 agentic RAG 练手项目。全链路离线:Ollama(qwen3:8b + bge-m3)+ Chroma,无需任何 API key。

## 前置条件

1. 安装 [Ollama](https://ollama.com) 并启动:`ollama serve`
2. 拉取模型:`ollama pull qwen3:8b && ollama pull bge-m3`
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
- `graph`:手写 LangGraph StateGraph,agent 节点(qwen3:8b)⇄ ToolNode 循环,模型自主决定检索时机与 query
- `chat`:CLI 流式问答,实时展示检索过程,回答后汇总引用来源

设计文档:`docs/superpowers/specs/2026-07-11-agentic-rag-demo-design.md`
实现计划:`docs/superpowers/plans/2026-07-11-agentic-rag-demo.md`

## 测试

```bash
uv run pytest -v   # 全部单测不依赖 Ollama
```
````

- [ ] **Step 2: 手动验收——库内问题(spec 验收第 1 条)**

Run: `uv run python -m agentic_rag.chat`,问"星尘咖啡馆的招牌饮品是什么?多少钱?"
Expected: 触发检索;回答含"星尘拿铁"和"32 元";引用 menu.md

- [ ] **Step 3: 手动验收——库外问题(spec 验收第 2 条)**

同一会话继续问:"星尘咖啡馆的创始人是谁?"(语料中从未提及创始人)
Expected: 触发检索但明确回答"资料里没有找到"之类,不编造人名

- [ ] **Step 4: 手动验收——多轮追问(spec 验收第 3 条)**

同一会话继续问:"它的年费会员都有什么权益?"(靠上文指代"星尘咖啡馆")
Expected: 正确理解指代,检索后回答 88 折/每月赠饮/生日升杯,引用 membership.md

- [ ] **Step 5: 若验收发现模型偷懒不检索**

按 spec 调优项处理(任选其一,改后重跑 Step 2–4):把 `config.GENERATION_MODEL` 换成 `qwen3:14b`(需先 `ollama pull qwen3:14b`);此改动只动 config 一行。更大的改造(强制首轮检索边)不属于本计划,另开计划。

- [ ] **Step 6: Commit**

```bash
git add README.md
git commit -m "docs: README 使用说明 + 手动验收通过"
git push
```

---

## 完成定义

- `uv run pytest -v` 全绿(14 个用例,零 Ollama 依赖)
- Task 8 三条手动验收全部符合预期
- 所有 commit 已推送 `origin/main`
