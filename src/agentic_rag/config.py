"""集中配置:模型、路径、切块与检索参数。"""
import os
from pathlib import Path

# 推理后端:ollama(默认)| vllm。经 AGENTIC_RAG_BACKEND 环境变量切换,无需改代码。
# 所有 LLM/嵌入客户端统一经 agentic_rag.llm 工厂构造,是 Ollama ↔ vLLM 的唯一切换点。
BACKEND = os.environ.get("AGENTIC_RAG_BACKEND", "ollama")

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
GENERATION_MODEL = os.environ.get("GENERATION_MODEL", "qwen3.5:9b")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "bge-m3")

# vLLM 后端(BACKEND=vllm 时生效):OpenAI 兼容端点。生成与嵌入需各自独立的 vLLM 实例。
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_EMBED_BASE_URL = os.environ.get("VLLM_EMBED_BASE_URL", "http://localhost:8001/v1")
VLLM_API_KEY = os.environ.get("VLLM_API_KEY", "EMPTY")

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

EMBED_BATCH = 128      # 嵌入写库批大小(实测 256 批在持续负载下易压垮嵌入 runner)

# 查询特征路由:查询词元 df ≤ max(ABS, ⌈RATIO·N⌉) 视为稀有,触发 BM25 通道
ROUTE_RARE_DF_ABS = 3
ROUTE_RARE_DF_RATIO = 0.005

CHUNK_SIZE = 800       # 二次切分块大小(字符)
CHUNK_OVERLAP = 120    # 块间重叠,约 15%
TOP_K = 5              # 检索返回块数
RECURSION_LIMIT = 10   # 图递归上限,防无限检索循环
NUM_CTX = 8192         # Ollama 上下文窗口(4096 多轮易截断;16384 在 8GB 显存会溢出到 CPU)
HISTORY_KEEP_TURNS = 4 # LLM 视图保留完整消息的最近轮数,更早轮次仅留问答对
