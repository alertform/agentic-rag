"""FAQ 沉淀:从语义缓存导出高频问答候选,供人工审核后入库。

工作流(自动收集 → 人工把关 → 复用既有管道):
1. `python -m agentic_rag.faq` 导出命中达标的候选到 faq_candidates.md
2. 人工核对答案与来源、修正表述
3. 确认的问答放入 sample_docs/faq.md,重跑 ingest(切块/哈希/ACL 全部复用)

未经审核的候选不得直接入库——模型生成内容不能无审核地成为检索来源。
"""
from agentic_rag import config
from agentic_rag.cache import SemanticCache

REVIEW_HEADER = """# FAQ 候选(自动收集,须人工审核)

> 审核流程:核对答案与来源 → 修正表述 → 把确认的问答按"## 问题 + 正文"格式
> 放入 `sample_docs/faq.md` → 重跑 ingest。未经审核的候选不得直接入库。
"""


def export_candidates(cache: SemanticCache, min_hits: int = 2) -> str:
    """导出命中次数 ≥ min_hits 的缓存条目为审核用 markdown。"""
    candidates = [e for e in cache.entries() if e["hit_count"] >= min_hits]
    if not candidates:
        return f"(暂无命中次数 ≥ {min_hits} 的缓存条目;在 chat 中被重复问到的问题才会成为候选)"
    candidates.sort(key=lambda e: e["hit_count"], reverse=True)

    parts = [REVIEW_HEADER]
    for entry in candidates:
        parts.append(f"## {entry['question']}\n")
        parts.append(f"{entry['answer']}\n")
        parts.append(
            f"- 依据来源: {', '.join(entry['sources']) or '-'}\n"
            f"- 缓存命中次数: {entry['hit_count']}\n"
            f"- 涉及权限级别: {', '.join(entry['access_levels'])}\n"
        )
    return "\n".join(parts)


def main() -> None:
    from agentic_rag.llm import make_embeddings

    embeddings = make_embeddings()
    cache = SemanticCache(embeddings, persist_directory=str(config.CHROMA_DIR))
    out = config.PROJECT_ROOT / "faq_candidates.md"
    content = export_candidates(cache)
    out.write_text(content, encoding="utf-8")
    print(f"[faq] 候选已导出: {out}")
    print(content.splitlines()[0])


if __name__ == "__main__":
    main()
