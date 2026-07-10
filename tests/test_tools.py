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
