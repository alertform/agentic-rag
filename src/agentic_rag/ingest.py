"""离线索引管道:Markdown 目录 → 切块 → 向量化 → Chroma。"""
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
