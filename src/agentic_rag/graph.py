"""手写 LangGraph agent 图:agent 节点 ⇄ ToolNode 循环。"""
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

SYSTEM_PROMPT = """你是一个基于本地知识库的问答助手。规则:
1. 凡是可能涉及知识库内容的问题,必须先调用 retrieve_docs 工具检索,禁止凭记忆直接回答。
2. 一次检索结果不理想时,可以换个说法再检索,但总共不要超过 3 次。
3. 只根据检索到的内容回答;检索不到就明确说"资料里没有找到",禁止编造。
4. 回答末尾用"来源:"列出实际引用的文件名。"""


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def build_graph(llm_with_tools, tools, checkpointer=None):
    """组装 agent 图。llm_with_tools 必须已 bind_tools;tools 给 ToolNode 执行。"""

    def agent(state: AgentState) -> dict:
        messages = [SystemMessage(SYSTEM_PROMPT), *state["messages"]]
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
