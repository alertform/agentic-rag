from agentic_rag.ingest import load_documents, split_markdown


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
    # 相邻正文块之间应有重叠(标题行会被切成独立小块,不参与重叠,故只看正文块)
    body_chunks = [c for c in chunks if "句话内容" in c.page_content]
    assert len(body_chunks) >= 2
    assert body_chunks[0].page_content[-60:] in body_chunks[1].page_content
    assert all(c.metadata["headers"] == "长文" for c in chunks)


def test_load_documents_walks_tree(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.md").write_text("# A\n\n内容甲。", encoding="utf-8")
    (tmp_path / "sub" / "b.md").write_text("# B\n\n内容乙。", encoding="utf-8")
    (tmp_path / "ignore.txt").write_text("非 markdown", encoding="utf-8")
    chunks = load_documents(tmp_path)
    assert {c.metadata["source"] for c in chunks} == {"a.md", "sub/b.md"}
