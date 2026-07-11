"""评估工具测试:指标数学与 golden set 加载。零模型依赖。"""
import json

from langchain_core.documents import Document

from agentic_rag.evals import (
    hit_at_k,
    load_golden,
    reciprocal_rank,
    sources_in_rank_order,
)


def test_hit_at_k_respects_cutoff():
    srcs = ["a.md", "b.md", "c.md"]
    assert hit_at_k(srcs, "b.md", k=2)
    assert not hit_at_k(srcs, "c.md", k=2)
    assert not hit_at_k(srcs, "missing.md", k=5)


def test_reciprocal_rank():
    srcs = ["a.md", "b.md", "c.md"]
    assert reciprocal_rank(srcs, "a.md") == 1.0
    assert reciprocal_rank(srcs, "c.md") == 1.0 / 3
    assert reciprocal_rank(srcs, "missing.md") == 0.0


def test_sources_in_rank_order_dedupes():
    docs = [
        Document(page_content="x", metadata={"source": "a.md"}),
        Document(page_content="y", metadata={"source": "b.md"}),
        Document(page_content="z", metadata={"source": "a.md"}),
    ]
    assert sources_in_rank_order(docs) == ["a.md", "b.md"]


def test_load_golden_skips_blanks_and_bom(tmp_path):
    lines = [
        json.dumps({"question": "q1", "expected_source": "a.md", "role": "staff"}),
        "",
        json.dumps({"question": "q2", "expected_source": "b.md"}),
    ]
    path = tmp_path / "golden.jsonl"
    path.write_bytes(b"\xef\xbb\xbf" + "\n".join(lines).encode("utf-8"))
    cases = load_golden(path)
    assert len(cases) == 2
    assert cases[0]["role"] == "staff"
