"""启动前检查:Ollama 服务可达、模型已拉取、向量库非空。"""
import json
import sys
import urllib.error
import urllib.request

from agentic_search import config


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


def ollama_status(require_generation: bool = True) -> dict:
    """非致命就绪状态:服务是否可达、缺哪些模型。供 /health 使用(不 sys.exit)。"""
    models = _installed_models()
    required = [config.EMBEDDING_MODEL]
    if require_generation:
        required.append(config.GENERATION_MODEL)
    if models is None:
        return {"reachable": False, "missing_models": required}
    missing = [name for name in required if not _has_model(models, name)]
    return {"reachable": True, "missing_models": missing}


def check_ollama(require_generation: bool = True) -> None:
    status = ollama_status(require_generation)
    if not status["reachable"]:
        sys.exit(f"[preflight] 连不上 Ollama ({config.OLLAMA_BASE_URL})。先运行: ollama serve")
    for name in status["missing_models"]:
        sys.exit(f"[preflight] 缺少模型 {name}。先运行: ollama pull {name}")


def vector_store_status(collection: str | None = None) -> dict:
    """非致命向量库状态:目标 collection 是否存在、块数。"""
    import chromadb

    coll = collection or config.COLLECTION_NAME
    try:
        client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        count = client.get_collection(coll).count()
    except Exception:
        count = 0
    return {"collection": coll, "count": count, "exists": count > 0}


def check_vector_store() -> None:
    if vector_store_status()["count"] == 0:
        sys.exit("[preflight] 向量库为空。先运行: uv run python -m agentic_search.ingest [md目录]")
