"""route(잡담/장애 분류) · chat_answer(잡담 답변) 노드."""
import asyncio

from langgraph.config import get_stream_writer
from langsmith import traceable

from ..deps import SupervisorDeps
from ..systems import INCIDENT_PATTERN, alias_system, systems_in_text, unknown_systems_in_text
from .planning import CLARIFY_PREFIX
from .state import AgentState, add_cost


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


def resume_after_clarify(query: str, history: list[dict]) -> str | None:
    """[개선 9] 직전 답이 '되묻기'였고 이번 메시지가 시스템을 짚었다면, 되묻기 직전의 증상과 합친
    장애 질문을 돌려준다. 해당 없으면 None.
    예) "응답이 너무 느려요" → (되묻기) → "payments-service요"
        ⇒ "응답이 너무 느려요 (대상 시스템: payments-service요)" """
    if len(history) < 2 or history[-1].get("role") != "assistant":
        return None
    if not history[-1].get("content", "").startswith(CLARIFY_PREFIX):
        return None
    if not (systems_in_text(query) or unknown_systems_in_text(query) or alias_system(query)):
        return None   # "모르겠어요" 같은 답이면 평소처럼 라우팅
    symptom = history[-2].get("content", "")
    return f"{symptom} (대상 시스템: {query})"


@traceable(name="route_node")
async def route_node(state: AgentState, deps: SupervisorDeps) -> dict:
    """Wave 5 — 모든 쿼리가 무거운 파이프라인을 탈 필요는 없다."""
    resumed = resume_after_clarify(state["query"], state.get("history", []))
    if resumed:   # [개선 9] 되묻기에 대한 대답 — LLM 없이 원래 장애 질문으로 이어서 진단
        return {"query": resumed, "route": "incident", "needs_analysis": False,
                "events": state.get("events", []) + [
                    {"type": "RouteDecided", "route": "incident", "resumed_query": resumed}]}
    # [개선 1] fast_llm.route는 동기 HTTP 호출 — 스레드로 넘겨 이벤트 루프(다른 요청·SSE)를 막지 않는다
    result, usage = await asyncio.to_thread(deps.fast_llm.route, state["query"])
    route = route_gate(state["query"], result["route"])
    event = {"type": "RouteDecided", "route": route}
    if route != result["route"]:   # 게이트가 모델 판단을 뒤집은 경우 기록(측정·디버깅용)
        event["llm_route"] = result["route"]
    return {"route": route, "needs_analysis": result["needs_analysis"],
            "cost_usd": add_cost(state, usage),
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
            "cost_usd": add_cost(state, usage),
            "events": state["events"] + [{"type": "Finalize", "pipeline": "chat-direct"}]}
