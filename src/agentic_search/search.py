"""Web 搜索后端工厂:web_search 工具的唯一后端构造点(对齐 llm.py 工厂模式)。

下游只依赖 SearchBackend 协议;换搜索供应商只需新增一个实现类。
TAVILY_API_KEY 缺失时 make_search_backend 返回 None,调用方不绑定 web_search
工具,系统自动降级为纯本地 RAG——本地通道保持零 key 全离线可用。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from agentic_search import config

TAVILY_ENDPOINT = "https://api.tavily.com/search"


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    content: str
    score: float = 0.0


class SearchBackend(Protocol):
    def search(self, query: str, max_results: int) -> list[SearchResult]: ...


class TavilyBackend:
    """Tavily REST /search 直调(httpx)。返回已清洗的正文片段,专为 RAG 设计。"""

    def __init__(self, api_key: str, timeout: float | None = None):
        self._api_key = api_key
        self._timeout = timeout if timeout is not None else config.WEB_SEARCH_TIMEOUT

    def search(self, query: str, max_results: int) -> list[SearchResult]:
        import httpx

        resp = httpx.post(
            TAVILY_ENDPOINT,
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={"query": query, "max_results": max_results},
            timeout=self._timeout,
        )
        resp.raise_for_status()
        return [
            SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                content=item.get("content", ""),
                score=item.get("score", 0.0),
            )
            for item in resp.json().get("results", [])
        ]


class RecordingBackend:
    """透明包装:记录每轮实际返回的搜索结果(对齐 HybridRetriever.take_recorded)。

    用途:回答后汇总 web 引用 URL;语义缓存写侧 gate(用过 web 的回答不缓存)。
    """

    def __init__(self, inner: SearchBackend):
        self._inner = inner
        self._recorded: list[SearchResult] = []

    def search(self, query: str, max_results: int) -> list[SearchResult]:
        results = self._inner.search(query, max_results)
        self._recorded.extend(results)
        return results

    def take_recorded(self) -> list[SearchResult]:
        """取走并清空自上次调用以来的全部搜索结果。"""
        recorded, self._recorded = self._recorded, []
        return recorded


def make_search_backend() -> SearchBackend | None:
    """读 TAVILY_API_KEY 构造后端;未配置返回 None(调用方降级纯 RAG)。"""
    if not config.TAVILY_API_KEY:
        return None
    return TavilyBackend(config.TAVILY_API_KEY)
