"""启动前检查:Ollama 服务可达、模型已拉取、向量库非空。"""
import json
import sys
import urllib.error
import urllib.request

from agentic_rag import config


def _installed_models() -> list[str] | None:
    """返回 Ollama 已安装模型名列表;服务不可达时返回 None。"""
    try:
        with urllib.request.urlopen(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=3) as resp:
            data = json.load(resp)
        return [m["name"] for m in data.get("models", [])]
    except (urllib.error.URLError, OSError, ValueError):
        return None


def _has_model(models: list[str], name: str) -> bool:
    return any(m == name or m.startswith(f"{name}:") for m in models)


def check_ollama(require_generation: bool = True) -> None:
    models = _installed_models()
    if models is None:
        sys.exit(f"[preflight] 连不上 Ollama ({config.OLLAMA_BASE_URL})。先运行: ollama serve")
    required = [config.EMBEDDING_MODEL]
    if require_generation:
        required.append(config.GENERATION_MODEL)
    for name in required:
        if not _has_model(models, name):
            sys.exit(f"[preflight] 缺少模型 {name}。先运行: ollama pull {name}")


def check_vector_store() -> None:
    import chromadb

    try:
        client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        count = client.get_collection(config.COLLECTION_NAME).count()
    except Exception:
        count = 0
    if count == 0:
        sys.exit("[preflight] 向量库为空。先运行: uv run python -m agentic_rag.ingest [md目录]")
