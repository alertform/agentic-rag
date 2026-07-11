"""手写 LangGraph agent 图:agent 节点 ⇄ ToolNode 循环 + 历史裁剪。"""
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from agentic_rag import config

SYSTEM_PROMPT = """你是一个基于本地知识库的问答助手。规则:
1. 除了纯寒暄,任何询问具体事实的问题(人、物、价格、数量、时间、规则等)都必须先调用 retrieve_docs 工具检索,禁止凭记忆或凭感觉直接回答;即使你觉得知识库里可能没有,也必须先检索确认。
2. 严禁在没有调用 retrieve_docs 的情况下声称"资料里没有找到"——说"没找到"之前必须至少真正检索过一次。
3. 一次检索结果不理想时,可以换个说法再检索,但总共不要超过 3 次。
4. 只根据检索到的内容回答;确实检索不到就明确说"资料里没有找到",禁止编造。
5. 多轮对话中,每个新的事实类问题都要针对该问题重新检索;不能仅凭此前轮次的检索结果就断定"资料里没有"。
6. 回答末尾用"来源:"列出实际引用的文件名。"""


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def trim_for_llm(messages: list[AnyMessage], keep_turns: int) -> list[AnyMessage]:
    """LLM 视图层的历史裁剪(不动 checkpointer 状态)。

    按 HumanMessage 划分轮次:最近 keep_turns 轮完整保留;更早轮次只留
    "问题 + 最终回答"——检索块占历史体积大头且已被回答蒸馏,裁掉损失最小。
    tool_calls AI 与 ToolMessage 必须成对移除,否则消息序列非法。
    """
    turns: list[list[AnyMessage]] = []
    current: list[AnyMessage] = []
    for msg in messages:
        if isinstance(msg, HumanMessage) and current:
            turns.append(current)
            current = []
        current.append(msg)
    if current:
        turns.append(current)
    if len(turns) <= keep_turns:
        return list(messages)

    trimmed: list[AnyMessage] = []
    for turn in turns[:-keep_turns]:
        for msg in turn:
            is_final_ai = isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None)
            if isinstance(msg, HumanMessage) or is_final_ai:
                trimmed.append(msg)
    for turn in turns[-keep_turns:]:
        trimmed.extend(turn)
    return trimmed


def build_graph(llm_with_tools, tools, checkpointer=None):
    """组装 agent 图。llm_with_tools 必须已 bind_tools;tools 给 ToolNode 执行。"""

    def agent(state: AgentState) -> dict:
        history = trim_for_llm(state["messages"], config.HISTORY_KEEP_TURNS)
        messages = [SystemMessage(SYSTEM_PROMPT), *history]
        return {"messages": [llm_with_tools.invoke(messages)]}

    def route(state: AgentState) -> str:
        last = state["messages"][-1]
        return "tools" if getattr(last, "tool_calls", None) else END

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=checkpointer)
