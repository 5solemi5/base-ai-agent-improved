"""v2 Supervisor — 실제 langgraph.StateGraph (Wave 4·5·6·7·8·10b·15 통합).

route → (chat 직행 | incident 파이프라인)
incident: plan(MCP Prompt 절차 반영) → gate(권위 게이트) → research(가드+병렬+재시도)
          → replan_check(부족 시 gap task 후 재조사) → analysis(needs_analysis=True일 때만
          실제 실행) → write → validate(반려 시 재작성) → finalize

모든 노드는 SupervisorDeps를 functools.partial로 주입받는 순수 함수다(Wave 13).
각 노드는 @traceable로 LangSmith에 계층 트리로 기록된다.

[개선 요약 — Mini-Project 과제 1번]
  1 성능   : route/plan/analyze의 동기 LLM 호출을 asyncio.to_thread로 — 이벤트 루프 블로킹 해소
  2 정확도 : 조사 대상은 agent/systems.py가 쿼리 기준으로 결정, 모니터링 대상이 아닌 시스템만
             물으면 unknown_system 노드에서 조사 없이 정직하게 안내(다른 시스템으로 바꿔치기 금지)
  3 RAG    : replan_check의 incident_type 선택을 키워드 규칙 → runbook 벡터 검색(agent/runbook_retriever.py)
  4 검증   : validate가 형식 + 근거(agent/grounding.py)를 함께 검사, 최대 시도 후 통과 시 경고 문구 첨부
  8 모듈화 : 노드 구현은 agent/nodes/ 패키지로 옮기고, 이 파일에는 '노드 연결(엣지)'만 남겼다
"""
from functools import partial

from langgraph.graph import END, StateGraph

from .deps import SupervisorDeps
from .nodes.analysis import analysis_node
from .nodes.planning import gate_node, plan_node, unknown_system_node
from .nodes.research import RETRIEVER, replan_check_node, research_node
from .nodes.routing import chat_answer_node, route_gate, route_node
from .nodes.state import AgentState
from .nodes.writing import (MAX_VALIDATION_ATTEMPTS, finalize_node, validate_node,
                            write_node)

# 예전 import 경로(from agent.graph import route_gate 등)를 쓰는 코드가 그대로 동작하게 다시 내보낸다
__all__ = ["build_graph", "AgentState", "route_gate", "RETRIEVER", "MAX_VALIDATION_ATTEMPTS"]


def _after_route(state: AgentState) -> str:
    return "chat_answer" if state["route"] == "chat" else "plan"


def _after_plan(state: AgentState) -> str:
    """[개선 2] 조사할 시스템이 하나도 없으면(모니터링 대상이 아닌 시스템만 물음) 안내 노드로."""
    has_research = any(t["agent"] == "research" for t in state.get("tasks", []))
    return "gate" if has_research else "unknown_system"


def _after_replan_check(state: AgentState) -> str:
    last = state["events"][-1]
    return "analysis" if last.get("is_sufficient") else "research"


def _after_validate(state: AgentState) -> str:
    return "finalize" if state.get("validation_feedback") is None else "write"


def build_graph(deps: SupervisorDeps):
    g = StateGraph(AgentState)
    node = lambda fn: partial(fn, deps=deps)   # Wave 13 — 명시적 의존성 주입
    g.add_node("route", node(route_node))
    g.add_node("chat_answer", node(chat_answer_node))
    g.add_node("plan", node(plan_node))
    g.add_node("unknown_system", node(unknown_system_node))
    g.add_node("gate", node(gate_node))
    g.add_node("research", node(research_node))
    g.add_node("replan_check", node(replan_check_node))
    g.add_node("analysis", node(analysis_node))
    g.add_node("write", node(write_node))
    g.add_node("validate", node(validate_node))
    g.add_node("finalize", node(finalize_node))

    g.set_entry_point("route")
    g.add_conditional_edges("route", _after_route,
                            {"chat_answer": "chat_answer", "plan": "plan"})
    g.add_edge("chat_answer", END)
    g.add_conditional_edges("plan", _after_plan,
                            {"gate": "gate", "unknown_system": "unknown_system"})
    g.add_edge("unknown_system", END)
    g.add_edge("gate", "research")
    g.add_edge("research", "replan_check")
    g.add_conditional_edges("replan_check", _after_replan_check,
                            {"analysis": "analysis", "research": "research"})
    g.add_edge("analysis", "write")
    g.add_edge("write", "validate")
    g.add_conditional_edges("validate", _after_validate,
                            {"finalize": "finalize", "write": "write"})
    g.add_edge("finalize", END)
    return g.compile()
