from langchain_core.documents import Document

from agentic_search.search import SearchResult
from agentic_search.tools import (
    NO_HIT_MESSAGE,
    WEB_NO_HIT_MESSAGE,
    format_chunks,
    format_results,
    make_retrieve_tool,
    make_web_search_tool,
)


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


class StubSearchBackend:
    def __init__(self, results=None, error=None):
        self._results = results or []
        self._error = error

    def search(self, query, max_results):
        if self._error is not None:
            raise self._error
        return self._results[:max_results]


def test_format_results_prefixes_title_and_url():
    out = format_results([SearchResult("今日天气", "https://x/w", "北京晴 30 度")])
    assert out.startswith("[来源: 今日天气 | https://x/w]")
    assert "北京晴 30 度" in out


def test_format_results_falls_back_to_url_when_no_title():
    out = format_results([SearchResult("", "https://x/w", "正文")])
    assert out.startswith("[来源: https://x/w | https://x/w]")


def test_format_results_empty_returns_no_hit():
    assert format_results([]) == WEB_NO_HIT_MESSAGE


def test_web_search_tool_invokes_backend_and_formats():
    backend = StubSearchBackend([SearchResult("T", "https://a", "内容A")])
    tool = make_web_search_tool(backend, max_results=5)
    assert tool.name == "web_search"
    out = tool.invoke({"query": "任意"})
    assert "[来源: T | https://a]" in out
    assert "内容A" in out


def test_web_search_tool_error_returns_model_visible_message():
    tool = make_web_search_tool(StubSearchBackend(error=RuntimeError("boom")), max_results=5)
    out = tool.invoke({"query": "任意"})
    assert "Web 搜索失败" in out
    assert "RuntimeError" in out  # 错误类型可见,便于模型/日志判断,不泄漏内部细节


def test_web_search_tool_no_hit():
    tool = make_web_search_tool(StubSearchBackend([]), max_results=5)
    assert tool.invoke({"query": "任意"}) == WEB_NO_HIT_MESSAGE
