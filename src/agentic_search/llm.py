"""LLM 后端工厂:生成/嵌入客户端的唯一构造点。

集中封装后端差异,使下游只依赖 LangChain 抽象类型(BaseChatModel / Embeddings),
把 Ollama ↔ vLLM 切换收敛到本文件一处。经 config.BACKEND(AGENTIC_SEARCH_BACKEND 环境变量)选择。
vLLM 迁移的完整步骤与坑见 README「推理后端」一节。
"""
from __future__ import annotations

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from agentic_search import config


def _unknown_backend() -> ValueError:
    return ValueError(
        f"未知 BACKEND={config.BACKEND!r};支持 'ollama' | 'vllm' "
        "(经 AGENTIC_SEARCH_BACKEND 环境变量设置)"
    )


def make_embeddings() -> Embeddings:
    """按当前后端构造嵌入客户端。返回 LangChain Embeddings 抽象,调用方不感知具体后端。"""
    if config.BACKEND == "ollama":
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(
            model=config.EMBEDDING_MODEL, base_url=config.OLLAMA_BASE_URL
        )
    if config.BACKEND == "vllm":
        # vLLM 嵌入走 OpenAI 兼容端点,需以 --task embed 单独启动一个 vLLM 实例。
        # check_embedding_ctx_length=False:关闭 OpenAIEmbeddings 的 tiktoken 长度校验(非 OpenAI 模型)。
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=config.EMBEDDING_MODEL,
            base_url=config.VLLM_EMBED_BASE_URL,
            api_key=config.VLLM_API_KEY,
            check_embedding_ctx_length=False,
        )
    raise _unknown_backend()


def make_chat_llm() -> BaseChatModel:
    """按当前后端构造对话模型(未 bind_tools)。返回 LangChain BaseChatModel 抽象。"""
    if config.BACKEND == "ollama":
        from langchain_ollama import ChatOllama

        # reasoning=False 关闭 qwen3 思考段;num_ctx 设定 Ollama 上下文窗口。
        return ChatOllama(
            model=config.GENERATION_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            reasoning=False,
            num_ctx=config.NUM_CTX,
        )
    if config.BACKEND == "vllm":
        # 迁移要点(详见 README):
        #  - num_ctx 无对应客户端参数,由 vLLM 启动 --max-model-len 决定;
        #  - 工具调用需 vLLM 启动 --enable-auto-tool-choice --tool-call-parser hermes(Qwen 系),
        #    这是承重点:整个 agent 图靠模型自主发 tool_call;
        #  - reasoning 关闭改经 extra_body 的 chat_template_kwargs 传递。
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=config.GENERATION_MODEL,
            base_url=config.VLLM_BASE_URL,
            api_key=config.VLLM_API_KEY,
            temperature=0,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
    raise _unknown_backend()
