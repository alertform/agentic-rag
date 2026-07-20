"""增量索引测试:内容哈希稳定 ID + upsert + 消失来源清理。零 Ollama 依赖。"""
from langchain_core.embeddings import DeterministicFakeEmbedding

from agentic_search import ingest


def _store(tmp_path):
    from langchain_chroma import Chroma

    return Chroma(
        collection_name="test_incremental",
        embedding_function=DeterministicFakeEmbedding(size=64),
        persist_directory=str(tmp_path / "chroma"),
    )


def _write_corpus(docs_dir, menu_price="32 元"):
    docs_dir.mkdir(exist_ok=True)
    (docs_dir / "menu.md").write_text(
        f"# 菜单\n\n## 饮品\n\n拿铁 {menu_price}。", encoding="utf-8"
    )
    (docs_dir / "rules.md").write_text(
        "# 规则\n\n## 退款\n\n24 小时内可退。", encoding="utf-8"
    )


def test_chunk_id_stable_and_content_sensitive(tmp_path):
    docs = tmp_path / "docs"
    _write_corpus(docs)
    chunks_a = ingest.load_documents(docs)
    chunks_b = ingest.load_documents(docs)
    assert [ingest.chunk_id(c) for c in chunks_a] == [ingest.chunk_id(c) for c in chunks_b]

    _write_corpus(docs, menu_price="35 元")
    chunks_c = ingest.load_documents(docs)
    ids_a = {ingest.chunk_id(c) for c in chunks_a}
    ids_c = {ingest.chunk_id(c) for c in chunks_c}
    assert ids_a != ids_c, "内容变化应产生新 id"


def test_incremental_ingest_is_idempotent(tmp_path):
    docs = tmp_path / "docs"
    _write_corpus(docs)
    store = _store(tmp_path)
    ingest.sync_vector_store(store, ingest.load_documents(docs))
    n1 = len(store.get()["ids"])
    ingest.sync_vector_store(store, ingest.load_documents(docs))
    n2 = len(store.get()["ids"])
    assert n1 == n2 > 0


def test_changed_file_replaces_old_chunks(tmp_path):
    docs = tmp_path / "docs"
    _write_corpus(docs)
    store = _store(tmp_path)
    ingest.sync_vector_store(store, ingest.load_documents(docs))

    _write_corpus(docs, menu_price="35 元")
    ingest.sync_vector_store(store, ingest.load_documents(docs))
    texts = store.get()["documents"]
    assert any("35 元" in t for t in texts)
    assert not any("32 元" in t for t in texts), "旧版本块应被清除"


def test_sync_batches_large_additions(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    for i in range(7):
        (docs / f"d{i}.md").write_text(f"# 文档{i}\n\n内容{i}。", encoding="utf-8")

    class RecordingStore:
        def __init__(self):
            self.batches = []

        def get(self):
            return {"ids": []}

        def add_documents(self, chunks, ids):
            self.batches.append(len(ids))

        def delete(self, ids):
            pass

    store = RecordingStore()
    ingest.sync_vector_store(store, ingest.load_documents(docs), batch_size=3)
    assert store.batches == [3, 3, 1], "7 个新块按批大小 3 应分为 3+3+1"


def test_deleted_file_chunks_removed(tmp_path):
    docs = tmp_path / "docs"
    _write_corpus(docs)
    store = _store(tmp_path)
    ingest.sync_vector_store(store, ingest.load_documents(docs))

    (docs / "rules.md").unlink()
    ingest.sync_vector_store(store, ingest.load_documents(docs))
    sources = {m["source"] for m in store.get()["metadatas"]}
    assert sources == {"menu.md"}
