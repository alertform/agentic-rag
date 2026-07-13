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


def run_evals(store, chunks, golden_path: Path, timing: bool = False, prebuilt=None) -> None:
    """对给定向量库与块集跑 golden set,打印分通道 hit@k / MRR(可选延迟分位)。

    检索器按 (模式, 角色可见集) 缓存——数万块语料下 BM25 不能每条 case 重建。
    """
    import statistics
    import time

    from agentic_rag.retrieval import HybridRetriever

    golden = load_golden(golden_path)
    k = config.TOP_K
    retrievers: dict = {}

    def get_retriever(mode: str, allowed: frozenset):
        key = (mode, allowed)
        if key not in retrievers:
            if mode == "纯向量":
                retrievers[key] = HybridRetriever(store, [], allowed_access=set(allowed))
            else:
                retrievers[key] = HybridRetriever(
                    store, chunks, allowed_access=set(allowed), prebuilt=prebuilt
                )
        return retrievers[key]

    agg = {"纯向量": [0, 0.0], "混合": [0, 0.0]}
    latencies: dict[str, list[float]] = {"纯向量": [], "混合": []}
    misses: list[str] = []
    for case in golden:
        allowed = frozenset(config.ROLE_ACCESS[case.get("role", "manager")])
        for mode in ("纯向量", "混合"):
            retriever = get_retriever(mode, allowed)
            t0 = time.perf_counter()
            hits_docs = retriever.similarity_search(case["question"], k=k)
            latencies[mode].append(time.perf_counter() - t0)
            srcs = sources_in_rank_order(hits_docs)
            hit = hit_at_k(srcs, case["expected_source"], k)
            agg[mode][0] += hit
            agg[mode][1] += reciprocal_rank(srcs, case["expected_source"])
            if mode == "混合" and not hit:
                misses.append(f"  MISS: {case['question']} (期望 {case['expected_source']}, 实得 {srcs})")

    n = len(golden)
    print(f"[evals] golden set {n} 条 | k={k} | 语料 {len(chunks)} 块")
    for mode, (hits, rr_sum) in agg.items():
        line = f"  {mode}: hit@{k} = {hits}/{n} ({hits / n:.0%}), MRR = {rr_sum / n:.3f}"
        if timing and latencies[mode]:
            lat_ms = sorted(v * 1000 for v in latencies[mode])
            p50 = statistics.median(lat_ms)
            p95 = lat_ms[min(len(lat_ms) - 1, int(len(lat_ms) * 0.95))]
            line += f", 延迟 p50 = {p50:.0f}ms / p95 = {p95:.0f}ms"
        print(line)
    if misses:
        print("\n".join(misses))


def main() -> None:
    import argparse

    from langchain_chroma import Chroma

    from agentic_rag.llm import make_embeddings
    from agentic_rag.preflight import check_ollama
    from agentic_rag.retrieval import build_bm25_index, corpus_digest, load_all_chunks, load_bm25_index

    parser = argparse.ArgumentParser(description="检索质量评估")
    parser.add_argument("--collection", default=config.COLLECTION_NAME)
    parser.add_argument("--golden", default=str(config.PROJECT_ROOT / "sample_evals.jsonl"))
    parser.add_argument("--timing", action="store_true", help="输出每模式查询延迟 p50/p95")
    args = parser.parse_args()

    if config.BACKEND == "ollama":
        check_ollama(require_generation=False)
    embeddings = make_embeddings()
    store = Chroma(
        collection_name=args.collection,
        embedding_function=embeddings,
        persist_directory=str(config.CHROMA_DIR),
    )
    chunks = load_all_chunks(store)
    prebuilt = load_bm25_index(
        config.CHROMA_DIR / f"bm25_{args.collection}.pkl", corpus_digest(chunks)
    )
    if prebuilt is None:
        prebuilt = build_bm25_index(chunks)
    run_evals(store, chunks, Path(args.golden), timing=args.timing, prebuilt=prebuilt)


if __name__ == "__main__":
    main()
