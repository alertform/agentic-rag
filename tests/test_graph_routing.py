from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver

from agentic_search.graph import SYSTEM_PROMPT_WEB, build_graph


@tool
def retrieve_docs(query: str) -> str:
    """测试用假检索工具。"""
    return "[来源: menu.md | 菜单 > 招牌饮品]\n星尘拿铁 32 元。"


@tool
def web_search(query: str) -> str:
    """测试用假 Web 搜索工具。"""
    return "[来源: 天气网 | https://w]\n北京晴。"


def _scripted(*messages):
    return GenericFakeChatModel(messages=iter(messages))


def test_tool_call_routes_to_tools_then_back_to_agent():
    llm = _scripted(
        AIMessage(
            content="",
            tool_calls=[{"name": "retrieve_docs", "args": {"query": "招牌饮品"}, "id": "call_1", "type": "tool_call"}],
        ),
        AIMessage(content="招牌是星尘拿铁,32 元。\n来源: menu.md"),
    )
    app = build_graph(llm, [retrieve_docs])
    result = app.invoke({"messages": [HumanMessage("招牌饮品是什么?")]})
    assert [m.type for m in result["messages"]] == ["human", "ai", "tool", "ai"]
    assert "星尘拿铁" in result["messages"][-1].content
    assert "星尘拿铁 32 元" in result["messages"][2].content  # 工具结果确实进了对话


def test_no_tool_call_goes_straight_to_end():
    app = build_graph(_scripted(AIMessage(content="你好!")), [retrieve_docs])
    result = app.invoke({"messages": [HumanMessage("你好")]})
    assert [m.type for m in result["messages"]] == ["human", "ai"]


def test_web_search_tool_call_routes_through_toolnode():
    llm = _scripted(
        AIMessage(
            content="",
            tool_calls=[{"name": "web_search", "args": {"query": "北京天气"}, "id": "w1", "type": "tool_call"}],
        ),
        AIMessage(content="北京晴。\n来源: https://w"),
    )
    app = build_graph(llm, [retrieve_docs, web_search], system_prompt=SYSTEM_PROMPT_WEB)
    result = app.invoke({"messages": [HumanMessage("北京今天天气?")]})
    assert [m.type for m in result["messages"]] == ["human", "ai", "tool", "ai"]
    assert "北京晴" in result["messages"][2].content  # web 工具结果进了对话


def test_system_prompt_param_reaches_llm():
    captured: list[str] = []

    class SpyModel(GenericFakeChatModel):
        def invoke(self, input, *args, **kwargs):
            captured.append(input[0].content)
            return super().invoke(input, *args, **kwargs)

    llm = SpyModel(messages=iter([AIMessage(content="hi")]))
    app = build_graph(llm, [retrieve_docs], system_prompt="自定义PROMPT")
    app.invoke({"messages": [HumanMessage("你好")]})
    assert captured[0] == "自定义PROMPT"


def test_checkpointer_keeps_history_across_turns():
    llm = _scripted(AIMessage(content="第一轮回答"), AIMessage(content="第二轮回答"))
    app = build_graph(llm, [retrieve_docs], checkpointer=MemorySaver())
    cfg = {"configurable": {"thread_id": "t1"}}
    app.invoke({"messages": [HumanMessage("问题A")]}, cfg)
    result = app.invoke({"messages": [HumanMessage("问题B")]}, cfg)
    assert [m.type for m in result["messages"]] == ["human", "ai", "human", "ai"]


def _tool_turn(tag):
    from langchain_core.messages import ToolMessage

    return [
        HumanMessage(f"问题{tag}"),
        AIMessage(
            content="",
            tool_calls=[{"name": "retrieve_docs", "args": {"query": tag}, "id": f"c{tag}", "type": "tool_call"}],
        ),
        ToolMessage(content=f"块内容{tag}", tool_call_id=f"c{tag}"),
        AIMessage(content=f"回答{tag}"),
    ]


def test_trim_keeps_short_history_intact():
    from agentic_search.graph import trim_for_llm

    msgs = _tool_turn("A")
    assert trim_for_llm(msgs, keep_turns=4) == msgs


def test_trim_drops_tool_pairs_from_old_turns():
    from langchain_core.messages import ToolMessage

    from agentic_search.graph import trim_for_llm

    msgs = _tool_turn("A") + _tool_turn("B") + _tool_turn("C")
    trimmed = trim_for_llm(msgs, keep_turns=1)
    # 旧轮(A/B)只留问答对;最近一轮(C)完整保留
    assert [m.content for m in trimmed[:4]] == ["问题A", "回答A", "问题B", "回答B"]
    assert [m.type for m in trimmed[4:]] == ["human", "ai", "tool", "ai"]
    # 序列合法性:无孤立的 tool_calls AI 或 ToolMessage
    for i, m in enumerate(trimmed):
        if isinstance(m, ToolMessage):
            assert getattr(trimmed[i - 1], "tool_calls", None), "ToolMessage 前必须是 tool_calls AI"
        if getattr(m, "tool_calls", None):
            assert isinstance(trimmed[i + 1], ToolMessage), "tool_calls AI 后必须跟 ToolMessage"


def test_agent_sees_trimmed_view_but_state_keeps_all(monkeypatch):
    from agentic_search import config

    monkeypatch.setattr(config, "HISTORY_KEEP_TURNS", 1)
    captured: list[int] = []

    class SpyModel(GenericFakeChatModel):
        def invoke(self, input, *args, **kwargs):
            captured.append(len(input))
            return super().invoke(input, *args, **kwargs)

    llm = SpyModel(messages=iter(AIMessage(content=f"回答{i}") for i in range(4)))
    app = build_graph(llm, [retrieve_docs], checkpointer=MemorySaver())
    cfg = {"configurable": {"thread_id": "trim"}}
    for i in range(4):
        result = app.invoke({"messages": [HumanMessage(f"问题{i}")]}, cfg)

    assert len(result["messages"]) == 8, "checkpointer 状态应完整保留 4 轮 8 条"
    # 第 4 轮 LLM 收到: System(1) + 3 轮旧问答(6) + 本轮问题(1) = 8
    assert captured[-1] == 8
