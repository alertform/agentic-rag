"""离线索引管道:Markdown 目录 → 切块 → 向量化 → Chroma。"""
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
    """递归加载目录下所有受支持格式的文档(经解析层归一化为 markdown),返回切好的块。"""
    from agentic_rag.parsers import SUPPORTED_EXTENSIONS, parse_file

    chunks: list[Document] = []
    files = sorted(
        p for p in docs_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    for file in files:
        source = file.relative_to(docs_dir).as_posix()
        chunks.extend(split_markdown(parse_file(file), source))
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
        sys.exit(f"[ingest] {docs_dir} 下没有受支持格式的文档 (md/pdf)")
    build_vector_store(chunks)
    print(f"[ingest] 已索引 {len(chunks)} 个文档块 ← {docs_dir}")


if __name__ == "__main__":
    main()
