"""集中配置:模型、路径、切块与检索参数。"""
from pathlib import Path

OLLAMA_BASE_URL = "http://localhost:11434"
GENERATION_MODEL = "qwen3.5:9b"
EMBEDDING_MODEL = "bge-m3"

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SAMPLE_DOCS_DIR = PROJECT_ROOT / "sample_docs"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
MEDIA_CACHE_DIR = PROJECT_ROOT / ".media_cache"  # 媒体解析缓存(按文件内容哈希)
COLLECTION_NAME = "agentic_rag"

# 角色 → 可见 access 级别(块的 access 由语料目录 acl.json 决定,默认 public)
ROLE_ACCESS = {
    "public": {"public"},
    "staff": {"public", "internal"},
    "manager": {"public", "internal", "confidential"},
}

CACHE_COLLECTION = "qa_cache"      # 语义缓存独立 collection,不进检索池
CACHE_DISTANCE_THRESHOLD = 0.10    # cosine 距离阈值,超过视为不同问题

EMBED_BATCH = 256      # 嵌入写库批大小(数万块时避免单次超大请求并提供进度)

CHUNK_SIZE = 800       # 二次切分块大小(字符)
CHUNK_OVERLAP = 120    # 块间重叠,约 15%
TOP_K = 5              # 检索返回块数
RECURSION_LIMIT = 10   # 图递归上限,防无限检索循环
NUM_CTX = 8192         # Ollama 上下文窗口(4096 多轮易截断;16384 在 8GB 显存会溢出到 CPU)
HISTORY_KEEP_TURNS = 4 # LLM 视图保留完整消息的最近轮数,更早轮次仅留问答对
