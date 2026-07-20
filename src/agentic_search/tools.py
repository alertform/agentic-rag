"""agent 图的工具集:retrieve_docs(本地知识库)+ web_search(可选,Tavily)。"""
from langchain_core.documents import Document
from langchain_core.tools import tool

from agentic_search.search import SearchBackend, SearchResult

NO_HIT_MESSAGE = "知识库中没有相关内容。"
WEB_NO_HIT_MESSAGE = "Web 搜索没有找到相关结果。"


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


def format_results(results: list[SearchResult]) -> str:
    """把 web 搜索结果拼成带来源标注的文本;零结果返回 WEB_NO_HIT_MESSAGE。"""
    if not results:
        return WEB_NO_HIT_MESSAGE
    parts = []
    for r in results:
        parts.append(f"[来源: {r.title or r.url} | {r.url}]\n{r.content}")
    return "\n\n---\n\n".join(parts)


def make_web_search_tool(backend: SearchBackend, max_results: int, verbose: bool = False):
    """工厂:把搜索后端封装成 web_search 工具。

    后端异常(网络/限流/超时)不抛出,而是转成模型可见的错误文案——
    模型可据此改用本地检索或如实告知用户,agent 循环不中断。
    """

    @tool
    def web_search(query: str) -> str:
        """在互联网上搜索与 query 相关的最新公开信息。用于时效性问题或本地知识库没有的公共知识。"""
        if verbose:
            print(f"\n🌐 Web 搜索: {query}", flush=True)
        try:
            results = backend.search(query, max_results=max_results)
        except Exception as exc:  # noqa: BLE001 — 一切后端故障都降级为模型可见文案
            if verbose:
                print(f"   搜索失败: {exc}", flush=True)
            return f"Web 搜索失败({type(exc).__name__}),本次无法联网。请改用本地知识库检索或如实告知用户。"
        if verbose and results:
            print(f"   命中: {', '.join(r.url for r in results)}", flush=True)
        return format_results(results)

    return web_search
