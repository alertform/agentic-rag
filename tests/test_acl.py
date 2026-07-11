"""ACL 测试:glob 规则匹配、ingest 打标、检索双通道过滤。零模型依赖。"""
import json

from langchain_core.documents import Document

from agentic_rag.acl import access_for, load_acl
from agentic_rag.ingest import load_documents
from agentic_rag.retrieval import HybridRetriever


def test_first_matching_glob_wins(tmp_path):
    (tmp_path / "acl.json").write_text(
        json.dumps({"finance/*": "confidential", "*.pdf": "internal"}), encoding="utf-8"
    )
    rules = load_acl(tmp_path)
    assert access_for("finance/report.pdf", rules) == "confidential"
    assert access_for("suppliers.pdf", rules) == "internal"
    assert access_for("menu.md", rules) == "public"


def test_no_acl_file_all_public(tmp_path):
    assert load_acl(tmp_path) == []
    assert access_for("anything.md", []) == "public"


def test_acl_json_with_utf8_bom(tmp_path):
    # Windows PowerShell 写的 JSON 常带 BOM,必须兼容
    payload = json.dumps({"*.pdf": "internal"}).encode("utf-8")
    (tmp_path / "acl.json").write_bytes(b"\xef\xbb\xbf" + payload)
    rules = load_acl(tmp_path)
    assert access_for("suppliers.pdf", rules) == "internal"


def test_ingest_stamps_access(tmp_path):
    (tmp_path / "menu.md").write_text("# 菜单\n\n拿铁 32 元。", encoding="utf-8")
    (tmp_path / "salary.md").write_text("# 工资\n\n保密内容。", encoding="utf-8")
    (tmp_path / "acl.json").write_text(
        json.dumps({"salary.md": "confidential"}), encoding="utf-8"
    )
    chunks = load_documents(tmp_path)
    by_source = {c.metadata["source"]: c.metadata["access"] for c in chunks}
    assert by_source == {"menu.md": "public", "salary.md": "confidential"}


class StubStore:
    def __init__(self, results):
        self._results = results
        self.last_filter = "UNSET"

    def similarity_search(self, query, k, filter=None):
        self.last_filter = filter
        return self._results[:k]


def _doc(text, access):
    return Document(
        page_content=text, metadata={"source": "x.md", "headers": "", "access": access}
    )


def test_bm25_prefilters_by_access():
    # 至少 3 篇:2 篇语料会触发 BM25 的 IDF=0 退化(词项出现在恰好一半文档时)
    docs = [
        _doc("公开的拿铁价格 32 元。", "public"),
        _doc("每周三闭店进行设备维护。", "public"),
        _doc("内部的供应商 NX-42 名单。", "internal"),
    ]
    restricted = HybridRetriever(StubStore([]), docs, allowed_access={"public"})
    hits = restricted.similarity_search("NX-42 供应商", k=5)
    assert not any("NX-42" in d.page_content for d in hits), "public 角色不应看到 internal 块"

    full = HybridRetriever(StubStore([]), docs, allowed_access={"public", "internal"})
    hits2 = full.similarity_search("NX-42 供应商", k=5)
    assert any("NX-42" in d.page_content for d in hits2)


def test_vector_channel_receives_access_filter():
    store = StubStore([])
    r = HybridRetriever(store, [], allowed_access={"public", "internal"})
    r.similarity_search("任意问题", k=3)
    assert store.last_filter == {"access": {"$in": ["internal", "public"]}}


def test_no_acl_means_no_filter():
    store = StubStore([])
    r = HybridRetriever(store, [])
    r.similarity_search("任意问题", k=3)
    assert store.last_filter is None
