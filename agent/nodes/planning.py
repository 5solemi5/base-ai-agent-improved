"""plan(조사 계획) · unknown_system(미등록 시스템 안내) · gate(권위 게이트) 노드."""
import asyncio

from langsmith import traceable

from ..deps import SupervisorDeps
from ..systems import KNOWN_SYSTEMS, resolve_query_systems
from .state import AgentState, add_cost


@traceable(name="plan_node")
async def plan_node(state: AgentState, deps: SupervisorDeps) -> dict:
    """Wave 4 — Plan/Task 분해. Wave 2의 MCP Prompt가 표준 절차를 깔아준다.

    resource와 마찬가지로 이름을 어느 서버가 들고 있는지 미리 가정하지 않고,
    find_prompt()로 연결된 서버들을 먼저 조회해 찾은 뒤에만 get_prompt를 보낸다."""
    resolved = resolve_query_systems(state["query"])
    if resolved["source"] == "default":
        # [개선 9] 시스템 이름도 별칭도 없다 — 원본처럼 auth-service로 '찍지' 않고 사용자에게 되묻는다
        event = {"type": "PlanCreated", "task_count": 0, "targets": [], "needs_clarification": True}
        return {"tasks": [], "needs_analysis": False, "needs_clarification": True,
                "events": state["events"] + [event]}
    if not resolved["systems"]:
        # [개선 2] 모니터링 대상이 아닌 시스템만 물었다 — LLM을 부르지 않고 바로 안내 경로로.
        # 원본은 여기서 auth-service로 바꿔치기해 엉뚱한 진단서를 냈다.
        event = {"type": "PlanCreated", "task_count": 0, "targets": [],
                 "unknown_systems": resolved["unknown"]}
        return {"tasks": [], "needs_analysis": False, "unknown_systems": resolved["unknown"],
                "events": state["events"] + [event]}

    prompt_server = await deps.mcp.find_prompt("incident_analysis_prompt")
    procedure = await deps.mcp.get_prompt(
        "incident_analysis_prompt",
        # MCP Prompt의 "시스템:" 줄에는 쿼리에서 찾은 시스템 전부를 넘긴다(원본은 첫 번째 하나만)
        {"system_name": ", ".join(resolved["systems"]), "symptom": state["query"]},
        server=prompt_server,
    )
    result, usage = await asyncio.to_thread(deps.fast_llm.plan, state["query"], procedure)
    targets = [t["target"] for t in result["tasks"]
               if t["agent"] == "research" and t.get("target")]
    unknown = result.get("unknown_systems", resolved["unknown"])
    event = {"type": "PlanCreated", "task_count": len(result["tasks"]),
             "targets": targets, "target_source": result.get("target_source", resolved["source"])}
    if unknown:
        event["unknown_systems"] = unknown
    return {"tasks": result["tasks"], "needs_analysis": result["needs_analysis"],
            "unknown_systems": unknown,
            "cost_usd": add_cost(state, usage), "events": state["events"] + [event]}


@traceable(name="unknown_system_node")
async def unknown_system_node(state: AgentState, deps: SupervisorDeps) -> dict:
    """[개선 2] 모니터링 대상이 아닌 시스템에 대한 정직한 안내. LLM 호출 없음(비용 0, 즉시 응답)."""
    names = ", ".join(state.get("unknown_systems", []))
    final = (f"{names}는(은) 현재 모니터링 대상이 아니라 상태를 조회할 수 없습니다. "
             f"조회 가능한 시스템: {', '.join(sorted(KNOWN_SYSTEMS))}. "
             f"위 시스템 중 관련된 것이 있으면 이름을 함께 알려 주세요.")
    return {"final": final,
            "events": state["events"] + [{"type": "Finalize", "pipeline": "unknown-system"}]}


CLARIFY_PREFIX = "조사할 시스템을 먼저 알려 주세요."   # 다음 턴에서 '되물은 직후인가'를 알아보는 표식


@traceable(name="clarify_node")
async def clarify_node(state: AgentState, deps: SupervisorDeps) -> dict:
    """[개선 9] 조사 대상이 불명확할 때 되묻기. LLM 호출 없음.

    사용자가 다음 메시지에서 시스템을 말하면 route_node가 이 질문 직전의 증상과 합쳐
    원래 장애 질문으로 이어서 진단한다(routing.resume_after_clarify)."""
    final = (f"{CLARIFY_PREFIX} 증상만으로는 어느 시스템인지 알 수 없어서, 임의의 시스템을 조사하지 않았습니다. "
             f"조회 가능한 시스템: {', '.join(sorted(KNOWN_SYSTEMS))}. "
             f"예: \"payments-service요\"")
    return {"final": final,
            "events": state["events"] + [{"type": "ClarifyAsked"},
                                         {"type": "Finalize", "pipeline": "clarify"}]}


@traceable(name="gate_node")
async def gate_node(state: AgentState, deps: SupervisorDeps) -> dict:
    """Wave 6 — 권위 게이트. needs_analysis=False인데 섞여 들어온 analysis task를
    프롬프트가 아니라 코드로 강제 제거한다."""
    tasks = state["tasks"]
    if not state.get("needs_analysis", False):
        kept = [t for t in tasks if t["agent"] != "analysis"]
        removed = len(tasks) - len(kept)
    else:
        kept, removed = tasks, 0
    event = {"type": "GateApplied", "removed_analysis_tasks": removed}
    return {"tasks": kept, "gate_removed": removed,
            "events": state["events"] + [event]}
