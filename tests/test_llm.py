"""后端工厂测试:验证 Ollama ↔ vLLM 的唯一切换点接线正确。零模型依赖(构造客户端不连服务)。"""
import pytest

from agentic_search import config, llm


def test_make_chat_llm_ollama_wires_config(monkeypatch):
    monkeypatch.setattr(config, "BACKEND", "ollama")
    chat = llm.make_chat_llm()
    assert type(chat).__name__ == "ChatOllama"
    assert chat.model == config.GENERATION_MODEL
    assert chat.base_url == config.OLLAMA_BASE_URL
    assert chat.num_ctx == config.NUM_CTX


def test_make_embeddings_ollama_wires_config(monkeypatch):
    monkeypatch.setattr(config, "BACKEND", "ollama")
    emb = llm.make_embeddings()
    assert type(emb).__name__ == "OllamaEmbeddings"
    assert emb.model == config.EMBEDDING_MODEL
    assert emb.base_url == config.OLLAMA_BASE_URL


def test_unknown_backend_raises_with_hint(monkeypatch):
    monkeypatch.setattr(config, "BACKEND", "triton")
    for factory in (llm.make_chat_llm, llm.make_embeddings):
        with pytest.raises(ValueError) as exc:
            factory()
        assert "triton" in str(exc.value)
        assert "AGENTIC_SEARCH_BACKEND" in str(exc.value)


def test_make_chat_llm_vllm_returns_openai_client(monkeypatch):
    # vLLM 分支需 langchain-openai(可选依赖);未安装则跳过——切换点本身仍被上面的测试覆盖。
    pytest.importorskip("langchain_openai")
    monkeypatch.setattr(config, "BACKEND", "vllm")
    chat = llm.make_chat_llm()
    assert type(chat).__name__ == "ChatOpenAI"
    assert chat.model_name == config.GENERATION_MODEL


def test_make_embeddings_vllm_returns_openai_client(monkeypatch):
    pytest.importorskip("langchain_openai")
    monkeypatch.setattr(config, "BACKEND", "vllm")
    emb = llm.make_embeddings()
    assert type(emb).__name__ == "OpenAIEmbeddings"
    assert emb.model == config.EMBEDDING_MODEL
