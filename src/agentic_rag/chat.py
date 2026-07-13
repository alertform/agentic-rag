"""CLI 交互问答:流式输出 + 检索过程展示 + 来源汇总 + 语义缓存。"""
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from agentic_rag import config, preflight
from agentic_rag.cache import SemanticCache
from agentic_rag.graph import build_graph
from agentic_rag.llm import make_chat_llm, make_embeddings
from agentic_rag.ingest import chunk_id
from agentic_rag.retrieval import (
    HybridRetriever,
    build_bm25_index,
    corpus_digest,
    load_all_chunks,
    load_bm25_index,
)
from agentic_rag.tools import make_retrieve_tool


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Agentic RAG CLI 问答")
    parser.add_argument(
        "--role",
        default="manager",
        choices=sorted(config.ROLE_ACCESS),
        help="提问者角色,决定检索可见范围 (默认 manager 全可见)",
    )
    parser.add_argument("--collection", default=config.COLLECTION_NAME)
    cli_args = parser.parse_args()
    allowed_access = config.ROLE_ACCESS[cli_args.role]

    if config.BACKEND == "ollama":
        preflight.check_ollama()
    preflight.check_vector_store()

    embeddings = make_embeddings()
    store = Chroma(
        collection_name=cli_args.collection,
        embedding_function=embeddings,
        persist_directory=str(config.CHROMA_DIR),
    )
    chunks = load_all_chunks(store)
    index_path = config.CHROMA_DIR / f"bm25_{cli_args.collection}.pkl"
    prebuilt = load_bm25_index(index_path, corpus_digest(chunks))
    if prebuilt is None:
        print("[chat] BM25 持久化索引缺失或过期,现场重建…", flush=True)
        prebuilt = build_bm25_index(chunks)
    retriever = HybridRetriever(
        store, chunks, allowed_access=allowed_access, prebuilt=prebuilt
    )
    retrieve = make_retrieve_tool(retriever, k=config.TOP_K, verbose=True)
    qa_cache = SemanticCache(embeddings, persist_directory=str(config.CHROMA_DIR))
    live_chunk_ids = set(store.get()["ids"])
    llm = make_chat_llm()
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

        hit = qa_cache.lookup(question, live_chunk_ids, allowed_access)
        if hit is not None:
            print(f"⚡ 缓存命中(相似问题: {hit.question})")
            print(f"助手: {hit.answer}")
            if hit.sources:
                print(f"—— 本轮引用(缓存): {', '.join(hit.sources)}")
            # 把缓存问答注入对话历史,保证后续追问的上下文连贯
            app.update_state(
                run_config,
                {"messages": [HumanMessage(question), AIMessage(hit.answer)]},
            )
            continue

        retriever.take_recorded()  # 清空上一轮残留
        answer_parts: list[str] = []
        print("助手: ", end="", flush=True)
        for chunk, meta in app.stream(
            {"messages": [HumanMessage(question)]}, run_config, stream_mode="messages"
        ):
            if (
                isinstance(chunk, AIMessageChunk)
                and meta.get("langgraph_node") == "agent"
                and chunk.content
            ):
                print(chunk.content, end="", flush=True)
                answer_parts.append(str(chunk.content))
        print()

        # 来源取自检索命中的结构化 metadata,而非正则反解工具输出文本——
        # 后者脆弱(格式契约),且语料正文写入字面量 "[来源: 伪造.md |" 即可伪造引用
        recorded = retriever.take_recorded()
        sources = sorted({d.metadata["source"] for d in recorded})
        if sources:
            print(f"—— 本轮引用: {', '.join(sources)}")

        answer_text = "".join(answer_parts).strip()
        if recorded and answer_text:
            qa_cache.store(
                question=question,
                answer=answer_text,
                sources=sources,
                chunk_ids=[chunk_id(d) for d in recorded],
                access_levels=[d.metadata.get("access", "public") for d in recorded],
            )


if __name__ == "__main__":
    main()
