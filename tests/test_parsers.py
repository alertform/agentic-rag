import pytest

from agentic_search.ingest import load_documents
from agentic_search.parsers import UnsupportedFormatError, parse_file


def _make_pdf(path, text):
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    doc.save(str(path))
    doc.close()


def test_parse_md_passthrough(tmp_path):
    f = tmp_path / "a.md"
    f.write_text("# 标题\n\n正文内容。", encoding="utf-8")
    assert parse_file(f) == "# 标题\n\n正文内容。"


def test_parse_pdf_extracts_text(tmp_path):
    f = tmp_path / "b.pdf"
    _make_pdf(f, "Stardust supplier price list NX-42")
    assert "NX-42" in parse_file(f)


def test_unsupported_extension_raises(tmp_path):
    f = tmp_path / "c.txt"
    f.write_text("plain", encoding="utf-8")
    with pytest.raises(UnsupportedFormatError):
        parse_file(f)


def test_load_documents_collects_supported_formats(tmp_path):
    (tmp_path / "a.md").write_text("# A\n\n内容甲。", encoding="utf-8")
    _make_pdf(tmp_path / "b.pdf", "PDF content code NX-42")
    (tmp_path / "ignore.txt").write_text("非支持格式", encoding="utf-8")
    chunks = load_documents(tmp_path)
    assert {c.metadata["source"] for c in chunks} == {"a.md", "b.pdf"}
