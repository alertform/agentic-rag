"""离线索引管道:文档目录 → 解析归一化 → 切块 → 向量化 → Chroma(增量同步)。"""
import hashlib
import sys
from pathlib import Path

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


def load_documents(docs_dir: Path) -> list[Document]:
    """递归加载目录下所有受支持格式的文档(经解析层归一化为 markdown),返回切好的块。

    块 metadata 按目录 acl.json 打 access 标(无规则默认 public)。
    """
    from agentic_rag.acl import access_for, load_acl
    from agentic_rag.parsers import SUPPORTED_EXTENSIONS, parse_file

    rules = load_acl(docs_dir)
    chunks: list[Document] = []
    files = sorted(
        p for p in docs_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    for file in files:
        source = file.relative_to(docs_dir).as_posix()
        chunks.extend(split_markdown(parse_file(file), source))
    for chunk in chunks:
        chunk.metadata["access"] = access_for(chunk.metadata["source"], rules)
    return chunks


def chunk_id(doc: Document) -> str:
    """块的稳定 ID:source+headers+access+内容 的 sha256 前 32 位。

    access 参与哈希:ACL 规则变更时旧块被替换,metadata 才能跟着更新。
    """
    meta = doc.metadata
    key = (
        f"{meta['source']}|{meta['headers']}|"
        f"{meta.get('access', 'public')}|{doc.page_content}"
    )
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]


def sync_vector_store(store, chunks: list[Document]) -> tuple[int, int]:
    """增量对账:只嵌入新增/变更的块,删除库里已消失的块;返回 (新增数, 删除数)。"""
    desired = {chunk_id(c): c for c in chunks}
    existing = set(store.get()["ids"])
    new_ids = [i for i in desired if i not in existing]
    stale_ids = [i for i in existing if i not in desired]
    if new_ids:
        store.add_documents([desired[i] for i in new_ids], ids=new_ids)
    if stale_ids:
        store.delete(ids=stale_ids)
    return len(new_ids), len(stale_ids)


def build_vector_store(chunks: list[Document], rebuild: bool = False):
    """打开本地 Chroma 并同步块:默认增量;rebuild=True 时清空全量重建。"""
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
    if rebuild:
        store.reset_collection()
    added, removed = sync_vector_store(store, chunks)
    return store, added, removed


def main() -> None:
    from agentic_rag.preflight import check_ollama

    args = sys.argv[1:]
    rebuild = "--rebuild" in args
    positional = [a for a in args if not a.startswith("--")]
    docs_dir = Path(positional[0]) if positional else config.SAMPLE_DOCS_DIR
    if not docs_dir.is_dir():
        sys.exit(f"[ingest] 目录不存在: {docs_dir}")
    check_ollama(require_generation=False)
    chunks = load_documents(docs_dir)
    if not chunks:
        sys.exit(f"[ingest] {docs_dir} 下没有受支持格式的文档 (md/pdf)")
    _, added, removed = build_vector_store(chunks, rebuild=rebuild)
    mode = "全量重建" if rebuild else "增量同步"
    print(
        f"[ingest] {mode}完成: 共 {len(chunks)} 块 (新增 {added}, 删除 {removed}) ← {docs_dir}"
    )


if __name__ == "__main__":
    main()
