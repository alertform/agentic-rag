from agentic_search import config, preflight


def test_ollama_status_unreachable(monkeypatch):
    monkeypatch.setattr(preflight, "_installed_models", lambda: None)
    status = preflight.ollama_status()
    assert status["reachable"] is False
    assert config.EMBEDDING_MODEL in status["missing_models"]


def test_ollama_status_missing_generation_model(monkeypatch):
    # 只装了嵌入模型;生成模型缺失应精确出现在 missing_models,嵌入模型不应出现
    monkeypatch.setattr(preflight, "_installed_models", lambda: [f"{config.EMBEDDING_MODEL}:latest"])
    status = preflight.ollama_status(require_generation=True)
    assert status["reachable"] is True
    assert status["missing_models"] == [config.GENERATION_MODEL]


def test_vector_store_status_shape():
    status = preflight.vector_store_status("does_not_exist_collection")
    assert status["exists"] is False
    assert status["count"] == 0
