"""CLI 交互问答:流式输出 + 检索过程展示 + 来源汇总。"""
import re

from langchain_chroma import Chroma
from langchain_core.messages import AIMessageChunk, HumanMessage, ToolMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.checkpoint.memory import MemorySaver

from agentic_rag import config, preflight
from agentic_rag.graph import build_graph
from agentic_rag.retrieval import HybridRetriever, load_all_chunks
from agentic_rag.tools import make_retrieve_tool

_SOURCE_RE = re.compile(r"\[来源: ([^|\]]+?) \|")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Agentic RAG CLI 问答")
    parser.add_argument(
        "--role",
        default="manager",
        choices=sorted(config.ROLE_ACCESS),
        help="提问者角色,决定检索可见范围 (默认 manager 全可见)",
    )
    cli_args = parser.parse_args()
    allowed_access = config.ROLE_ACCESS[cli_args.role]

    preflight.check_ollama()
    preflight.check_vector_store()

    embeddings = OllamaEmbeddings(
        model=config.EMBEDDING_MODEL, base_url=config.OLLAMA_BASE_URL
    )
    store = Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(config.CHROMA_DIR),
    )
    retriever = HybridRetriever(
        store, load_all_chunks(store), allowed_access=allowed_access
    )
    retrieve = make_retrieve_tool(retriever, k=config.TOP_K, verbose=True)
    # reasoning=False 关闭 qwen3 思考段;若所装 langchain-ollama 不支持该参数,删掉即可
    llm = ChatOllama(
        model=config.GENERATION_MODEL, base_url=config.OLLAMA_BASE_URL, reasoning=False
    )
    app = build_graph(llm.bind_tools([retrieve]), [retrieve], checkpointer=MemorySaver())
    run_config = {
        "configurable": {"thread_id": "cli"},
        "recursion_limit": config.RECURSION_LIMIT,
    }

    print(f"Agentic RAG — 角色: {cli_args.role} | 输入问题开始对话,exit 退出")
    while True:
        try:
            question = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break

        sources: set[str] = set()
        print("助手: ", end="", flush=True)
        for chunk, meta in app.stream(
            {"messages": [HumanMessage(question)]}, run_config, stream_mode="messages"
        ):
            if isinstance(chunk, ToolMessage):
                sources |= set(_SOURCE_RE.findall(str(chunk.content)))
            elif (
                isinstance(chunk, AIMessageChunk)
                and meta.get("langgraph_node") == "agent"
                and chunk.content
            ):
                print(chunk.content, end="", flush=True)
        print()
        if sources:
            print(f"—— 本轮引用: {', '.join(sorted(sources))}")


if __name__ == "__main__":
    main()
