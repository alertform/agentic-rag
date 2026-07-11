"""集中配置:模型、路径、切块与检索参数。"""
from pathlib import Path

OLLAMA_BASE_URL = "http://localhost:11434"
GENERATION_MODEL = "qwen3.5:9b"
EMBEDDING_MODEL = "bge-m3"

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SAMPLE_DOCS_DIR = PROJECT_ROOT / "sample_docs"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "agentic_rag"

# 角色 → 可见 access 级别(块的 access 由语料目录 acl.json 决定,默认 public)
ROLE_ACCESS = {
    "public": {"public"},
    "staff": {"public", "internal"},
    "manager": {"public", "internal", "confidential"},
}

CHUNK_SIZE = 800       # 二次切分块大小(字符)
CHUNK_OVERLAP = 120    # 块间重叠,约 15%
TOP_K = 5              # 检索返回块数
RECURSION_LIMIT = 10   # 图递归上限,防无限检索循环
