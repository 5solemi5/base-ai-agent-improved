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
"""
import asyncio
import time
from functools import partial
from typing import TypedDict

from langgraph.config import get_stream_writer
from langgraph.graph import END, StateGraph
from langsmith import traceable

from .deps import SupervisorDeps
from .grounding import check_grounding
from .retry import with_retry
from .runbook_retriever import RunbookRetriever
from .systems import (INCIDENT_PATTERN, KNOWN_SYSTEMS, alias_system, resolve_query_systems,
                      systems_in_text, unknown_systems_in_text)


RETRIEVER = RunbookRetriever()   # [개선 3] runbook 인덱스는 프로세스당 1번만 만든다(W1 싱글턴과 같은 이유)
MAX_VALIDATION_ATTEMPTS = 3


class AgentState(TypedDict, total=False):
    query: str
    thread_id: str
    history: list[dict]
    route: str
    needs_analysis: bool
    tasks: list[dict]
    gate_removed: int
    collected: list[dict]
    replan_rounds: int
    retry_count: int
    parallel_elapsed: float
    draft: str
    validation_attempts: int
    validation_feedback: str
    final: str
    last_prompt: str
    cost_usd: float
    events: list[dict]
    unknown_systems: list[str]   # [개선 2] 쿼리에 나왔지만 모니터링 대상이 아닌 시스템
    grounding_issues: list[str]  # [개선 4] 마지막 검증에서 남은 근거 문제(최대 시도 후 통과 시 안내)


def _cost(state: AgentState, usage: dict) -> float:
    return round(state.get("cost_usd", 0.0) + usage["usd"], 6)


def route_gate(query: str, llm_route: str) -> str:
    """[개선 1-보완] 라우팅 권위 게이트 — thinking을 끈 0.6b가 "제 이름은 민수입니다"를 incident로
    보낸 회귀(실측)를 코드로 막는다. 장애 신호(증상 단어 또는 시스템 이름)가 '하나도 없으면' chat,
    증상 단어와 시스템 이름이 '둘 다 있으면' incident로 확정하고, 그 사이 애매한 경우만 모델을 따른다."""
    has_symptom = bool(INCIDENT_PATTERN.search(query))
    # 시스템 이름 그대로 또는 별칭("로그인"→auth-service)이 있으면 '대상이 있다'로 본다
    has_system = bool(systems_in_text(query) or unknown_systems_in_text(query) or alias_system(query))
    if not has_symptom and not has_system:
        return "chat"
    if has_symptom and has_system:
        return "incident"
    return llm_route


@traceable(name="route_node")
async def route_node(state: AgentState, deps: SupervisorDeps) -> dict:
    """Wave 5 — 모든 쿼리가 무거운 파이프라인을 탈 필요는 없다."""
    # [개선 1] fast_llm.route는 동기 HTTP 호출 — 스레드로 넘겨 이벤트 루프(다른 요청·SSE)를 막지 않는다
    result, usage = await asyncio.to_thread(deps.fast_llm.route, state["query"])
    route = route_gate(state["query"], result["route"])
    event = {"type": "RouteDecided", "route": route}
    if route != result["route"]:   # 게이트가 모델 판단을 뒤집은 경우 기록(측정·디버깅용)
        event["llm_route"] = result["route"]
    return {"route": route, "needs_analysis": result["needs_analysis"],
            "cost_usd": _cost(state, usage),
            "events": state.get("events", []) + [event]}


@traceable(name="chat_answer_node")
async def chat_answer_node(state: AgentState, deps: SupervisorDeps) -> dict:
    """Wave 5 direct 경로 — 히스토리 기반 응답 (Wave 12 세션 증명 지점).

    토큰 단위 스트리밍 — get_stream_writer()는 /chat/stream처럼 custom 모드로 소비될
    때만 실제로 전송되고, /chat(ainvoke)처럼 스트리밍하지 않는 호출에서는 자동으로
    no-op이 된다."""
    writer = get_stream_writer()
    reply, usage = await deps.fast_llm.chat(
        state["query"], state.get("history", []),
        on_token=lambda piece: writer({"type": "AnswerToken", "delta": piece}))
    return {"final": reply, "last_prompt": usage.get("prompt"),
            "cost_usd": _cost(state, usage),
            "events": state["events"] + [{"type": "Finalize", "pipeline": "chat-direct"}]}


@traceable(name="plan_node")
async def plan_node(state: AgentState, deps: SupervisorDeps) -> dict:
    """Wave 4 — Plan/Task 분해. Wave 2의 MCP Prompt가 표준 절차를 깔아준다.

    resource와 마찬가지로 이름을 어느 서버가 들고 있는지 미리 가정하지 않고,
    find_prompt()로 연결된 서버들을 먼저 조회해 찾은 뒤에만 get_prompt를 보낸다."""
    resolved = resolve_query_systems(state["query"])
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
            "cost_usd": _cost(state, usage), "events": state["events"] + [event]}


@traceable(name="unknown_system_node")
async def unknown_system_node(state: AgentState, deps: SupervisorDeps) -> dict:
    """[개선 2] 모니터링 대상이 아닌 시스템에 대한 정직한 안내. LLM 호출 없음(비용 0, 즉시 응답)."""
    names = ", ".join(state.get("unknown_systems", []))
    final = (f"{names}는(은) 현재 모니터링 대상이 아니라 상태를 조회할 수 없습니다. "
             f"조회 가능한 시스템: {', '.join(sorted(KNOWN_SYSTEMS))}. "
             f"위 시스템 중 관련된 것이 있으면 이름을 함께 알려 주세요.")
    return {"final": final,
            "events": state["events"] + [{"type": "Finalize", "pipeline": "unknown-system"}]}


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


@traceable(name="research_node")
async def research_node(state: AgentState, deps: SupervisorDeps) -> dict:
    """Wave 15 가드 + Wave 10b 병렬 + Wave 9 재시도 + Wave 2 Tool/Resource 실호출."""
    tasks = state["tasks"]
    research_indices = [i for i, t in enumerate(tasks)
                        if t["agent"] == "research" and t["status"] == "pending"]
    # Wave 15 — 객체가 아니라 '실행할 목록'을 검사한다 (not research_agent 버그의 수정판)
    if not research_indices:
        return {"events": state["events"] +
                [{"type": "ResearchSkipped", "reason": "pending research task 0개"}]}

    retry_total = state.get("retry_count", 0)

    # URI를 코드에 하드코딩하지 않는다. 여러 MCP 서버가 연결돼 있고 그중 어느 서버가
    # "get_incident_runbook" resource를 제공하는지 모른다고 가정하고, 먼저 연결된
    # 서버들을 조회(list)해 유효한 서버를 찾은 뒤에만 그 서버로 요청한다.
    needs_runbook = any(tasks[i]["kind"] != "monitoring" for i in research_indices)
    runbook_server = runbook_uri_template = None
    if needs_runbook:
        runbook_server, runbook_uri_template = await deps.mcp.find_resource_template(
            "get_incident_runbook")

    async def run_one(idx: int) -> dict:
        task = tasks[idx]
        await asyncio.sleep(0.35)   # 태스크당 고정 지연 — 병렬 효과를 실측하기 위함
        if task["kind"] == "monitoring":
            data, retries = await with_retry(
                deps.mcp.call_tool, "get_monitoring_status",
                {"system_name": task["target"]})
            return {"idx": idx, "source": "monitoring", "data": data, "retries": retries}
        uri = runbook_uri_template.format(incident_type=task["target"])
        data = await deps.mcp.read_resource(uri, server=runbook_server)
        return {"idx": idx, "source": "runbook", "data": data, "retries": 0}

    started = time.perf_counter()
    results = await asyncio.gather(*(run_one(i) for i in research_indices))
    elapsed = round(time.perf_counter() - started, 2)

    new_tasks = [dict(t) for t in tasks]
    collected = list(state.get("collected", []))
    events = list(state["events"])
    for r in results:
        new_tasks[r["idx"]]["status"] = "done"
        collected.append({"source": r["source"], "data": r["data"]})
        retry_total += r["retries"]
        events.append({"type": "TaskCompleted", "id": new_tasks[r["idx"]]["id"],
                       "source": r["source"], "target": new_tasks[r["idx"]]["target"],
                       "retries": r["retries"]})
    return {"tasks": new_tasks, "collected": collected, "retry_count": retry_total,
            "parallel_elapsed": elapsed, "events": events}


@traceable(name="replan_check_node")
async def replan_check_node(state: AgentState, deps: SupervisorDeps) -> dict:
    """Wave 7 — 1차 수집 결과 평가. runbook이 없으면 gap task만 추가해 재조사."""
    has_runbook = any(c["source"] == "runbook" for c in state.get("collected", []))
    rounds = state.get("replan_rounds", 0)
    if has_runbook or rounds >= 1:
        verdict = {"type": "ReplanCheck", "round": rounds, "is_sufficient": True}
        return {"events": state["events"] + [verdict]}
    # [개선 3] 키워드 규칙 대신 runbook 벡터 검색 — 사용자 질문 + 저하된 시스템의 알림이 질의
    # [8단계 수정] embed 모드면 classify 안에서 동기 HTTP 호출 → 스레드로 넘겨 이벤트 루프를 막지 않는다
    picked = await asyncio.to_thread(RETRIEVER.classify, state["query"], state.get("collected", []))
    incident_type = picked["incident_type"]
    gap_task = {"id": "task-gap-runbook", "agent": "research", "kind": "runbook",
                "target": incident_type, "status": "pending"}
    verdict = {"type": "ReplanCheck", "round": rounds, "is_sufficient": False,
               "gap_task": gap_task["id"], "target": incident_type,
               "method": picked["method"], "scores": picked["scores"]}
    return {"tasks": state["tasks"] + [gap_task], "replan_rounds": rounds + 1,
            "events": state["events"] + [verdict]}


@traceable(name="analysis_node")
async def analysis_node(state: AgentState, deps: SupervisorDeps) -> dict:
    """Wave 6 보완 — needs_analysis=True가 정당한 경우(모델의 자기모순이 아니라
    실제로 패턴 비교가 필요하다고 판단한 경우)를 위한 실행 경로. gate가 살려둔
    agent=='analysis' task가 있을 때만 수집된 monitoring 결과들을 비교 분석한다."""
    tasks = state["tasks"]
    pending = [t for t in tasks if t["agent"] == "analysis" and t["status"] == "pending"]
    if not pending:
        return {"events": state["events"] +
                [{"type": "AnalysisSkipped", "reason": "analysis task 0개"}]}
    summary, usage = await asyncio.to_thread(deps.fast_llm.analyze, state.get("collected", []))
    new_tasks = [dict(t) for t in tasks]
    for t in new_tasks:
        if t["agent"] == "analysis" and t["status"] == "pending":
            t["status"] = "done"
    collected = state.get("collected", []) + [{"source": "analysis", "data": summary}]
    event = {"type": "AnalysisCompleted", "length": len(summary)}
    return {"tasks": new_tasks, "collected": collected,
            "cost_usd": _cost(state, usage), "events": state["events"] + [event]}


@traceable(name="write_node")
async def write_node(state: AgentState, deps: SupervisorDeps) -> dict:
    """Wave 8/10a — 고품질 모델로 초안 작성 (1차는 의도적 저품질).

    토큰마다 attempt 번호를 같이 실어 보낸다 — validate가 반려해 재작성(attempt 증가)
    될 때 프론트가 새 말풍선으로 끊어서 보여줄 수 있게 하기 위함(Wave 8 재작성 루프를
    그대로 노출)."""
    attempt = state.get("validation_attempts", 0) + 1
    writer = get_stream_writer()
    draft, usage = await deps.quality_llm.write(
        state["query"], state.get("collected", []),
        attempt, state.get("validation_feedback"),
        on_token=lambda piece: writer(
            {"type": "AnswerToken", "attempt": attempt, "delta": piece}))
    return {"draft": draft, "last_prompt": usage.get("prompt"),
            "cost_usd": _cost(state, usage),
            "events": state["events"] + [{"type": "DraftWritten", "attempt": attempt,
                                          "length": len(draft)}]}


@traceable(name="validate_node")
async def validate_node(state: AgentState, deps: SupervisorDeps) -> dict:
    """Wave 8 — 코드 기준 검증. [개선 4] 형식(30자 이상, 1) 2) 번호 절차) + 근거(grounding)."""
    draft = state["draft"]
    attempts = state.get("validation_attempts", 0) + 1
    problems = []
    if len(draft) < 30:
        problems.append(f"응답이 너무 짧습니다({len(draft)}자, 최소 30자).")
    if "1)" not in draft or "2)" not in draft:
        problems.append("절차를 1) 2) 3) 번호로 구분해 구체적으로 설명하세요.")
    issues = check_grounding(draft, state.get("collected", []), state.get("unknown_systems", []))
    problems += issues
    ok = not problems
    verdict = "합격" if ok else "불합격"
    if not ok and attempts >= MAX_VALIDATION_ATTEMPTS:   # 무한 재작성 방지 — 통과시키되 문제는 기록
        ok, verdict = True, "합격(최대 시도 도달 — 통과 처리)"
    feedback = None if ok else " ".join(problems)
    event = {"type": "Validation", "attempt": attempts, "verdict": verdict}
    if issues:
        event["grounding_issues"] = issues
    return {"validation_attempts": attempts, "validation_feedback": feedback,
            "grounding_issues": issues, "events": state["events"] + [event]}


@traceable(name="finalize_node")
async def finalize_node(state: AgentState, deps: SupervisorDeps) -> dict:
    final = state["draft"]
    notes = []
    if state.get("unknown_systems"):   # [개선 2] 일부만 모니터링 대상이 아닐 때도 숨기지 않는다
        notes.append(f"※ {', '.join(state['unknown_systems'])}는(은) 모니터링 대상이 아니라 조사하지 못했습니다.")
    if state.get("grounding_issues"):  # [개선 4] 3회 재작성 후에도 남은 근거 문제를 사용자에게 알린다
        notes.append("※ 자동 검증에서 근거를 확인하지 못한 내용이 있습니다: "
                     + " / ".join(state["grounding_issues"]))
    if notes:
        final = final.rstrip() + "\n\n" + "\n".join(notes)
    return {"final": final,
            "events": state["events"] + [{"type": "Finalize", "pipeline": "incident-v2"}]}


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
