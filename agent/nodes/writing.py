"""write(답변 작성) · validate(형식·근거 검사) · finalize(마무리) 노드."""
from langgraph.config import get_stream_writer
from langsmith import traceable

from ..deps import SupervisorDeps
from ..grounding import check_grounding
from .state import AgentState, add_cost

MAX_VALIDATION_ATTEMPTS = 3


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
            "cost_usd": add_cost(state, usage),
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
