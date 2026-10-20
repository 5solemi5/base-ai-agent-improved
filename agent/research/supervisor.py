"""research_supervisor — 메인 그래프에서 서브에이전트 3개를 지휘하는 노드와, 서브에이전트를
메인 그래프 노드로 감싸는 어댑터.

흐름(매 단계 supervisor로 돌아와 다음을 고른다):
    research_supervisor → metrics_agent → research_supervisor
                        → log_agent(저하 시스템이 있을 때만) → research_supervisor
                        → runbook_agent → research_supervisor → analysis

'다음에 누구를 부를지'는 LLM이 아니라 코드가 정한다. 이유: 순서가 정해진 3단계라 LLM에게 물으면
지연·비용만 늘고, 0.6b가 같은 에이전트를 반복 호출하는 무한 루프 위험이 생긴다. 대신 각 에이전트는
한 번씩만 부르고(MAX_CALLS), 결과에 따라 log_agent를 건너뛰는 '조건부' 판단만 한다.
"""
from langsmith import traceable

from ..deps import SupervisorDeps
from ..nodes.state import AgentState

AGENTS = ("metrics_agent", "log_agent", "runbook_agent")


def next_agent(state: AgentState) -> str:
    """지금까지 수집된 것을 보고 다음 서브에이전트(또는 'done')를 고른다. 순수 함수 — 단위 테스트 대상."""
    called = state.get("agents_called", [])
    if len(called) >= len(AGENTS):   # [리뷰 반영] 하드 상한 — 어떤 경우에도 서브에이전트 호출은 3번을 넘지 않는다
        return "done"
    if "metrics_agent" not in called:
        return "metrics_agent"
    degraded = state.get("degraded", [])
    if degraded and "log_agent" not in called:
        return "log_agent"
    if "runbook_agent" not in called:
        return "runbook_agent"
    return "done"


@traceable(name="research_supervisor")
async def research_supervisor_node(state: AgentState, deps: SupervisorDeps) -> dict:
    nxt = next_agent(state)
    event = {"type": "SupervisorDecision", "next": nxt}
    if nxt == "runbook_agent" and "log_agent" not in state.get("agents_called", []):
        event["skipped"] = "log_agent (저하된 시스템 없음)"
    return {"next_agent": nxt, "events": state["events"] + [event]}


async def metrics_agent_node(state: AgentState, deps: SupervisorDeps, agent) -> dict:
    """메인 상태 → 서브에이전트 입력(targets), 서브에이전트 출력 → 메인 상태(collected, events)."""
    tasks = [dict(t) for t in state["tasks"]]
    pending = [t for t in tasks if t["agent"] == "research" and t["status"] == "pending"]
    out = await agent.ainvoke({"targets": [t["target"] for t in pending]})
    collected = list(state.get("collected", []))
    events = list(state["events"])
    for task, data, retries in zip(pending, out["results"], out["retries"]):
        task["status"] = "done"
        collected.append({"source": "monitoring", "data": data})
        events.append({"type": "TaskCompleted", "id": task["id"], "source": "monitoring",
                       "target": task["target"], "retries": retries, "agent": "metrics_agent"})
    events.append({"type": "AgentFinished", "agent": "metrics_agent", "degraded": out["degraded"]})
    return {"tasks": tasks, "collected": collected, "degraded": out["degraded"],
            "retry_count": state.get("retry_count", 0) + sum(out["retries"]),
            "parallel_elapsed": out["elapsed"],
            "agents_called": state.get("agents_called", []) + ["metrics_agent"],
            "events": events}


async def log_agent_node(state: AgentState, deps: SupervisorDeps, agent) -> dict:
    out = await agent.ainvoke({"systems": state.get("degraded", [])})
    collected = list(state.get("collected", []))
    events = list(state["events"])
    for f in out["findings"]:
        collected.append({"source": "log", "data": f})
        events.append({"type": "TaskCompleted", "source": "log", "target": f["system_name"],
                       "errors": f["error_count"], "warns": f["warn_count"], "agent": "log_agent"})
    events.append({"type": "AgentFinished", "agent": "log_agent"})
    return {"collected": collected,
            "agents_called": state.get("agents_called", []) + ["log_agent"], "events": events}


async def runbook_agent_node(state: AgentState, deps: SupervisorDeps, agent) -> dict:
    out = await agent.ainvoke({"query": state["query"], "collected": state.get("collected", [])})
    picked = out["picked"]
    itype = picked["incident_type"]
    events = state["events"] + [
        {"type": "RunbookSelected", "target": itype, "method": picked["method"],
         "scores": picked["scores"], "agent": "runbook_agent"},
        {"type": "TaskCompleted", "id": "task-runbook", "source": "runbook", "target": itype,
         "retries": 0, "agent": "runbook_agent"},
        {"type": "AgentFinished", "agent": "runbook_agent"}]
    tasks = state["tasks"] + [{"id": "task-runbook", "agent": "research", "kind": "runbook",
                               "target": itype, "status": "done"}]
    return {"tasks": tasks,
            "collected": state.get("collected", []) + [{"source": "runbook", "data": out["runbook"]}],
            "agents_called": state.get("agents_called", []) + ["runbook_agent"], "events": events}
