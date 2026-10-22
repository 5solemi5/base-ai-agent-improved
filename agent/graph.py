"""v2 Supervisor — 실제 langgraph.StateGraph (Wave 4·5·6·7·8·10b·15 통합).

route → (chat 직행 | incident 파이프라인)
incident: plan(MCP Prompt 절차 반영) → gate(권위 게이트)
          → research_supervisor ⇄ {metrics_agent, log_agent, runbook_agent}  ([개선 10] 서브에이전트)
          → analysis(needs_analysis=True일 때만 실제 실행) → write → validate(반려 시 재작성)
          → approval([개선 11] 위험 조치면 사람 승인) → finalize

모든 노드는 SupervisorDeps를 functools.partial로 주입받는 순수 함수다(Wave 13).
각 노드에는 @traceable이 붙어 있어, LANGCHAIN_TRACING_V2=true와 유효한 LangSmith 키가 있을 때 계층 트리로
기록된다(개선 1~14 동안은 추적이 꺼져 있었고, 개선 15에서 처음 켰다).

[개선 요약 — Mini-Project 과제 1번]
  1 성능   : route/plan/analyze의 동기 LLM 호출을 asyncio.to_thread로 — 이벤트 루프 블로킹 해소
  2 정확도 : 조사 대상은 agent/systems.py가 쿼리 기준으로 결정, 모니터링 대상이 아닌 시스템만
             물으면 unknown_system 노드에서 조사 없이 정직하게 안내(다른 시스템으로 바꿔치기 금지)
  3 RAG    : incident_type 선택을(현재는 runbook_agent 안) 키워드 규칙 → runbook 벡터 검색(agent/runbook_retriever.py)
  4 검증   : validate가 형식 + 근거(agent/grounding.py)를 함께 검사, 최대 시도 후 통과 시 경고 문구 첨부
  8 모듈화 : 노드 구현은 agent/nodes/ 패키지로 옮기고, 이 파일에는 '노드 연결(엣지)'만 남겼다
  9 되묻기 : 시스템을 특정할 수 없으면 기본 시스템(auth-service)으로 찍지 않고 clarify 노드에서 되묻는다
  10 멀티에이전트 구조 : research + replan_check를 LLM 없이 도구를 조회하는 메트릭·로그·런북 서브그래프
             (compiled subgraph) 3개와, 부를 순서를 코드가 정하는 research_supervisor 노드로 교체 (agent/research/)
  11 HITL  : 검수를 통과한 답변에 재시작·롤백 같은 위험 조치가 있으면 approval 노드에서 interrupt()로
             멈추고 사람의 승인/거절을 받은 뒤 finalize (agent/nodes/approval.py)
"""
from functools import partial

from langgraph.graph import END, StateGraph

from .deps import SupervisorDeps
from .nodes.analysis import analysis_node
from .nodes.approval import approval_node
from .nodes.planning import clarify_node, gate_node, plan_node, unknown_system_node
from .nodes.routing import chat_answer_node, route_gate, route_node
from .nodes.state import AgentState
from .nodes.writing import (MAX_VALIDATION_ATTEMPTS, finalize_node, validate_node,
                            write_node)
from .research.log_agent import build_log_agent
from .research.metrics_agent import build_metrics_agent
from .research.runbook_agent import RETRIEVER, build_runbook_agent
from .research.supervisor import (AGENTS, log_agent_node, metrics_agent_node,
                                  research_supervisor_node, runbook_agent_node)

# 예전 import 경로(from agent.graph import route_gate 등)를 쓰는 코드가 그대로 동작하게 다시 내보낸다
__all__ = ["build_graph", "AgentState", "route_gate", "RETRIEVER", "MAX_VALIDATION_ATTEMPTS"]


def _after_route(state: AgentState) -> str:
    return "chat_answer" if state["route"] == "chat" else "plan"


def _after_plan(state: AgentState) -> str:
    """[개선 2] 조사할 시스템이 하나도 없으면(모니터링 대상이 아닌 시스템만 물음) 안내 노드로."""
    if state.get("needs_clarification"):   # [개선 9] 대상 불명확 → 되묻기
        return "clarify"
    has_research = any(t["agent"] == "research" for t in state.get("tasks", []))
    return "gate" if has_research else "unknown_system"


def _after_supervisor(state: AgentState) -> str:
    """[개선 10] research_supervisor가 고른 서브에이전트로, 다 끝났으면 analysis로."""
    return "analysis" if state["next_agent"] == "done" else state["next_agent"]


def _after_validate(state: AgentState) -> str:
    # [개선 11] 합격한 답변은 finalize 전에 approval(위험 조치 사람 승인)을 거친다
    return "approval" if state.get("validation_feedback") is None else "write"


def build_graph(deps: SupervisorDeps, checkpointer=None):
    """checkpointer: [개선 11] interrupt()로 멈춘 상태를 저장할 곳. HITL을 켰는데 안 주면 메모리 저장소를 쓴다."""
    if deps.hitl and checkpointer is None:
        from langgraph.checkpoint.memory import InMemorySaver
        checkpointer = InMemorySaver()
    g = StateGraph(AgentState)
    node = lambda fn: partial(fn, deps=deps)   # Wave 13 — 명시적 의존성 주입
    g.add_node("route", node(route_node))
    g.add_node("chat_answer", node(chat_answer_node))
    g.add_node("plan", node(plan_node))
    g.add_node("unknown_system", node(unknown_system_node))
    g.add_node("clarify", node(clarify_node))
    g.add_node("gate", node(gate_node))
    # [개선 10] 서브에이전트는 각자 compile된 StateGraph — 어댑터 노드가 메인 상태와 주고받는다
    g.add_node("research_supervisor", node(research_supervisor_node))
    g.add_node("metrics_agent", partial(metrics_agent_node, deps=deps, agent=build_metrics_agent(deps.mcp)))
    g.add_node("log_agent", partial(log_agent_node, deps=deps, agent=build_log_agent(deps.mcp)))
    g.add_node("runbook_agent", partial(runbook_agent_node, deps=deps, agent=build_runbook_agent(deps.mcp)))
    g.add_node("analysis", node(analysis_node))
    g.add_node("write", node(write_node))
    g.add_node("validate", node(validate_node))
    g.add_node("approval", node(approval_node))
    g.add_node("finalize", node(finalize_node))

    g.set_entry_point("route")
    g.add_conditional_edges("route", _after_route,
                            {"chat_answer": "chat_answer", "plan": "plan"})
    g.add_edge("chat_answer", END)
    g.add_conditional_edges("plan", _after_plan,
                            {"gate": "gate", "unknown_system": "unknown_system",
                             "clarify": "clarify"})
    g.add_edge("unknown_system", END)
    g.add_edge("clarify", END)
    g.add_edge("gate", "research_supervisor")
    g.add_conditional_edges("research_supervisor", _after_supervisor,
                            {**{a: a for a in AGENTS}, "analysis": "analysis"})
    for a in AGENTS:   # 서브에이전트가 끝나면 항상 supervisor로 돌아가 다음을 고른다
        g.add_edge(a, "research_supervisor")
    g.add_edge("analysis", "write")
    g.add_edge("write", "validate")
    g.add_conditional_edges("validate", _after_validate,
                            {"approval": "approval", "write": "write"})
    g.add_edge("approval", "finalize")
    g.add_edge("finalize", END)
    return g.compile(checkpointer=checkpointer)
