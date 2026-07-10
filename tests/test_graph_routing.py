from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver

from agentic_rag.graph import build_graph


@tool
def retrieve_docs(query: str) -> str:
    """测试用假检索工具。"""
    return "[来源: menu.md | 菜单 > 招牌饮品]\n星尘拿铁 32 元。"


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


def test_checkpointer_keeps_history_across_turns():
    llm = _scripted(AIMessage(content="第一轮回答"), AIMessage(content="第二轮回答"))
    app = build_graph(llm, [retrieve_docs], checkpointer=MemorySaver())
    cfg = {"configurable": {"thread_id": "t1"}}
    app.invoke({"messages": [HumanMessage("问题A")]}, cfg)
    result = app.invoke({"messages": [HumanMessage("问题B")]}, cfg)
    assert [m.type for m in result["messages"]] == ["human", "ai", "human", "ai"]
