import pytest

from agentic_rag import preflight


def test_ollama_down_exits_with_serve_hint(monkeypatch):
    monkeypatch.setattr(preflight, "_installed_models", lambda: None)
    with pytest.raises(SystemExit) as exc:
        preflight.check_ollama()
    assert "ollama serve" in str(exc.value)


def test_missing_model_exits_with_pull_hint(monkeypatch):
    monkeypatch.setattr(preflight, "_installed_models", lambda: ["qwen3:8b"])
    with pytest.raises(SystemExit) as exc:
        preflight.check_ollama()
    assert "ollama pull bge-m3" in str(exc.value)


def test_all_models_present_passes(monkeypatch):
    # Ollama 的 tags 接口常带 :latest 后缀,匹配逻辑要兼容
    monkeypatch.setattr(preflight, "_installed_models", lambda: ["qwen3:8b", "bge-m3:latest"])
    preflight.check_ollama()  # 不应抛出


def test_embedding_only_mode(monkeypatch):
    monkeypatch.setattr(preflight, "_installed_models", lambda: ["bge-m3:latest"])
    preflight.check_ollama(require_generation=False)  # ingest 场景:只要 embedding 模型
