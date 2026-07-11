"""检索质量评估:golden set 上的 hit@k 与 MRR,纯向量 vs 混合检索对比。

用法: uv run python -m agentic_rag.evals  (需 Ollama embedding 模型,不需要生成模型)
"""
import json
from pathlib import Path

from agentic_rag import config


def load_golden(path: Path) -> list[dict]:
    """逐行 JSONL,跳过空行;兼容 BOM。"""
    cases = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line:
            cases.append(json.loads(line))
    return cases


def hit_at_k(result_sources: list[str], expected: str, k: int) -> bool:
    return expected in result_sources[:k]


def reciprocal_rank(result_sources: list[str], expected: str) -> float:
    for rank, source in enumerate(result_sources, 1):
        if source == expected:
            return 1.0 / rank
    return 0.0


def sources_in_rank_order(docs) -> list[str]:
    """按命中名次去重提取 source 列表。"""
    seen: set[str] = set()
    ordered: list[str] = []
    for doc in docs:
        source = doc.metadata["source"]
        if source not in seen:
            seen.add(source)
            ordered.append(source)
    return ordered


def main() -> None:
    from langchain_chroma import Chroma
    from langchain_ollama import OllamaEmbeddings

    from agentic_rag.preflight import check_ollama
    from agentic_rag.retrieval import HybridRetriever, load_all_chunks

    check_ollama(require_generation=False)
    embeddings = OllamaEmbeddings(
        model=config.EMBEDDING_MODEL, base_url=config.OLLAMA_BASE_URL
    )
    store = Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(config.CHROMA_DIR),
    )
    chunks = load_all_chunks(store)
    golden = load_golden(config.PROJECT_ROOT / "sample_evals.jsonl")

    k = config.TOP_K
    agg = {"纯向量": [0, 0.0], "混合": [0, 0.0]}
    misses: list[str] = []
    for case in golden:
        allowed = config.ROLE_ACCESS[case.get("role", "manager")]
        for mode, bm25_docs in (("纯向量", []), ("混合", chunks)):
            retriever = HybridRetriever(store, bm25_docs, allowed_access=allowed)
            srcs = sources_in_rank_order(retriever.similarity_search(case["question"], k=k))
            hit = hit_at_k(srcs, case["expected_source"], k)
            agg[mode][0] += hit
            agg[mode][1] += reciprocal_rank(srcs, case["expected_source"])
            if mode == "混合" and not hit:
                misses.append(f"  MISS: {case['question']} (期望 {case['expected_source']}, 实得 {srcs})")

    n = len(golden)
    print(f"[evals] golden set {n} 条 | k={k}")
    for mode, (hits, rr_sum) in agg.items():
        print(f"  {mode}: hit@{k} = {hits}/{n} ({hits / n:.0%}), MRR = {rr_sum / n:.3f}")
    if misses:
        print("\n".join(misses))


if __name__ == "__main__":
    main()
