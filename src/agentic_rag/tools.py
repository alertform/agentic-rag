"""retrieve_docs 检索工具:agent 图里唯一的工具。"""
from langchain_core.documents import Document
from langchain_core.tools import tool

NO_HIT_MESSAGE = "知识库中没有相关内容。"


def format_chunks(docs: list[Document]) -> str:
    """把检索命中的块拼成带来源标注的文本;零命中返回 NO_HIT_MESSAGE。"""
    if not docs:
        return NO_HIT_MESSAGE
    parts = []
    for doc in docs:
        headers = doc.metadata.get("headers") or "-"
        parts.append(f"[来源: {doc.metadata['source']} | {headers}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def make_retrieve_tool(vector_store, k: int, verbose: bool = False):
    """工厂:把向量库封装成 retrieve_docs 工具。verbose 时向终端打印检索过程。"""

    @tool
    def retrieve_docs(query: str) -> str:
        """在本地知识库中检索与 query 相关的文档片段。涉及知识库内容的问题必须先调用本工具。"""
        if verbose:
            print(f"\n🔍 检索: {query}", flush=True)
        docs = vector_store.similarity_search(query, k=k)
        if verbose and docs:
            files = sorted({d.metadata["source"] for d in docs})
            print(f"   命中: {', '.join(files)}", flush=True)
        return format_chunks(docs)

    return retrieve_docs
