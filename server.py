"""all_total 통합 실습 — FastAPI 백엔드 (Wave 1·3·12·14 축 + 전체 조립).

- Wave 1  : lifespan에서 Agent 번들(MCP 연결·LLM·그래프)을 프로세스당 정확히 1회 생성,
            app.state로 모든 요청이 공유 (creation_count/instance_id로 증명)
- Wave 2  : mcp_server.py를 subprocess(stdio)로 기동해 연결 유지
- Wave 3  : GET /chat/stream — 그래프 실행 과정을 SSE 이벤트로 스트리밍
- Wave 12 : 대화 이력을 SQLite(session.db)에 저장 — 재시작해도 유지
- Wave 14 : SUPERVISOR_VERSION(v1/v2) 환경변수로 legacy/신형 파이프라인 즉시 전환
- LangSmith: 요청당 chat_request 루트 run 아래로 노드·LLM·MCP 호출이 계층 기록됨

[개선 5] 포트를 BACKEND_PORT 환경변수로(기본 8100 — 다른 프로그램이 8100을 쓰면 바꿔서 기동),
/chat/stream 도중 예외가 나면 연결을 끊지 않고 Error 이벤트 + [DONE]을 보내며, 응답 metadata에
개선 지표(조사 대상 출처·미등록 시스템·runbook 선택 방식·근거 문제)를 함께 돌려준다.

[개선 12] 대화 이력은 agent/memory.load_history로 읽는다 — 오래된 턴은 요약본(system 메시지)으로 대체.

[개선 11] Human-in-the-loop: 답변에 위험 조치가 있으면 그래프가 approval 노드에서 멈춘다(interrupt).
  - /chat         → {"status": "pending_approval", "approval": {...}} 로 응답하고 결정을 기다린다
  - /chat/stream  → ApprovalRequired 이벤트를 보내고 스트림을 닫는다
  - POST /chat/approve, GET /chat/approve/stream 으로 승인/거절을 넘기면 멈춘 지점부터 이어서 실행한다
요청마다 체크포인트 키(run_key = thread_id:난수)를 새로 만든다 — 같은 키를 재사용하면 이전 요청의
상태(collected, draft …)가 다음 요청에 섞인다. 대화 이력은 지금처럼 session.db가 맡는다.
"""
import json
import os
import uuid
from contextlib import asynccontextmanager

from dotenv import load_dotenv

load_dotenv()   # LANGCHAIN_TRACING_V2 / API_KEY / PROJECT — import보다 먼저

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from langgraph.types import Command
from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree
from pydantic import BaseModel

from agent.deps import SupervisorDeps
from agent.fake_llm import FastLLM, QualityLLM
from agent.graph import build_graph
from agent.mcp_client import MCPClient
from agent.memory import load_history
from agent.ollama_llm import format_facts
from agent.session_store import SqliteSessionStore
from agent.supervisor_v1 import run_v1

CREATION_COUNT = 0   # Wave 1 — 프로세스 내 Agent 번들 생성 횟수 (항상 1이어야 정상)


class ChatRequest(BaseModel):   # Wave 1 — Pydantic 요청 검증
    message: str
    thread_id: str = "default"


class ApproveRequest(BaseModel):   # [개선 11]
    thread_id: str = "default"
    approved: bool
    comment: str = ""


def _build_llms():
    """LLM_BACKEND=fake(기본, 결정론) | ollama(로컬 실모델) | openai(실제 OpenAI API)."""
    backend = os.environ.get("LLM_BACKEND", "fake")
    if backend == "ollama":
        from agent.ollama_llm import OllamaFastLLM, OllamaQualityLLM
        fast = OllamaFastLLM(os.environ.get("OLLAMA_FAST_MODEL", "qwen3:0.6b"),
                             usd_per_1k_tokens=0.00015)
        quality = OllamaQualityLLM(os.environ.get("OLLAMA_QUALITY_MODEL", "qwen3:8b"),
                                   usd_per_1k_tokens=0.005)
        return backend, fast, quality
    if backend == "openai":
        from agent.openai_llm import OpenAIFastLLM, OpenAIQualityLLM
        fast = OpenAIFastLLM(os.environ.get("OPENAI_FAST_MODEL", "gpt-4o-mini"),
                             usd_per_1k_tokens=0.00015)
        quality = OpenAIQualityLLM(os.environ.get("OPENAI_QUALITY_MODEL", "gpt-4o"),
                                   usd_per_1k_tokens=0.005)
        return backend, fast, quality
    fast = FastLLM("fast-model-stub", usd_per_1k_tokens=0.00015)
    quality = QualityLLM("quality-model-stub", usd_per_1k_tokens=0.005)
    return backend, fast, quality


@asynccontextmanager
async def lifespan(app: FastAPI):
    global CREATION_COUNT
    CREATION_COUNT += 1
    mcp = MCPClient()
    await mcp.start()   # Wave 2 — MCP subprocess는 서버 수명 동안 딱 1개
    backend, fast_llm, quality_llm = _build_llms()
    app.state.llm_backend = backend
    deps = SupervisorDeps(
        fast_llm=fast_llm,
        quality_llm=quality_llm,
        mcp=mcp,
        session_store=SqliteSessionStore(),
        version=os.environ.get("SUPERVISOR_VERSION", "v2"),
        hitl=os.environ.get("HITL_APPROVAL", "1") == "1",   # [개선 11] 기본 켬, 자동 평가 땐 0
    )
    app.state.deps = deps
    app.state.graph = build_graph(deps) if deps.version == "v2" else None
    app.state.instance_id = id(deps)
    app.state.pending = {}   # [개선 11] thread_id → {"run_key", "message"} : 승인 대기 중인 요청
    yield
    await mcp.stop()
    deps.session_store.close()
    try:   # 종료 전 LangSmith 배치 전송 플러시 — kill 시 run이 pending으로 남는 것 방지
        from langchain_core.tracers.langchain import wait_for_all_tracers
        wait_for_all_tracers()
    except Exception:
        pass


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok", "instance_id": app.state.instance_id,
            "creation_count": CREATION_COUNT,
            "supervisor_version": app.state.deps.version,
            "llm_backend": app.state.llm_backend,
            "hitl": app.state.deps.hitl,
            "fast_llm": app.state.deps.fast_llm.name,
            "quality_llm": app.state.deps.quality_llm.name}


@app.get("/version")
async def version():
    return {"supervisor_version": app.state.deps.version}


def _metadata(result: dict, run_id: str | None) -> dict:
    return {
        "route": result.get("route"),
        "task_count": len(result.get("tasks", [])),
        "gate_removed": result.get("gate_removed", 0),
        "retry_count": result.get("retry_count", 0),
        "parallel_elapsed": result.get("parallel_elapsed"),
        "validation_attempts": result.get("validation_attempts", 0),
        "unknown_systems": result.get("unknown_systems", []),
        "grounding_issues": result.get("grounding_issues", []),
        "runbook_method": next((e.get("method") for e in result.get("events", [])
                                if e.get("type") == "RunbookSelected"), None),
        "agents_called": result.get("agents_called", []),   # [개선 10]
        "approval": result.get("approval"),                  # [개선 11]
        "memory": result.get("memory"),                      # [개선 12] 메시지 수·요약 범위
        "facts": format_facts(result.get("collected", [])),  # [개선 13] LLM-as-judge 채점 근거
        "cost_usd": result.get("cost_usd", 0.0),
        "llm_backend": app.state.llm_backend,
        "supervisor_version": app.state.deps.version,
        "instance_id": app.state.instance_id,
        "langsmith_run_id": run_id,
    }


def _run_config(run_key: str) -> dict:
    return {"configurable": {"thread_id": run_key}}


def _interrupt_payload(result: dict) -> dict | None:
    """ainvoke 결과에 __interrupt__가 있으면 = approval 노드에서 멈췄다."""
    intr = result.get("__interrupt__")
    return intr[0].value if intr else None


@traceable(name="chat_request")
async def _handle_chat(message: str, thread_id: str) -> tuple[dict, str | None]:
    deps = app.state.deps
    # Wave 12 + [개선 12] 오래된 턴은 요약본으로 — 요약이 필요하면 여기서 갱신된다
    history, memory = await load_history(deps.session_store, deps.fast_llm, thread_id)
    app.state.pending.pop(thread_id, None)   # 승인 대기 중에 새 질문이 오면 이전 대기는 버린다
    if deps.version == "v2":
        run_key = f"{thread_id}:{uuid.uuid4().hex[:8]}"
        result = await app.state.graph.ainvoke(
            {"query": message, "thread_id": thread_id, "history": history,
             "events": [], "cost_usd": 0.0}, config=_run_config(run_key))
        result = {**result, "memory": memory}
        if _interrupt_payload(result):   # [개선 11] 승인 대기 — 대화 이력은 결정이 난 뒤에 저장
            app.state.pending[thread_id] = {"run_key": run_key, "message": message}
            rt = get_current_run_tree()
            return result, (str(rt.id) if rt else None)
        _finish_run(thread_id, run_key)
    else:
        result = await run_v1(deps, message, thread_id, history)   # Wave 14
    deps.session_store.append(thread_id, "user", message)
    deps.session_store.append(thread_id, "assistant", result["final"])
    rt = get_current_run_tree()
    return result, (str(rt.id) if rt else None)


def _chat_response(result: dict, run_id: str | None) -> dict:
    payload = _interrupt_payload(result)
    if payload:
        return {"status": "pending_approval", "approval": payload,
                "reply": "⏸ 위험 조치가 포함된 답변입니다. 운영자 승인을 기다리는 중입니다.",
                "events": result.get("events", []), "metadata": _metadata(result, run_id)}
    return {"status": "done", "reply": result["final"], "events": result.get("events", []),
            "metadata": _metadata(result, run_id)}


@app.post("/chat")
async def chat(req: ChatRequest):
    result, run_id = await _handle_chat(req.message, req.thread_id)
    return _chat_response(result, run_id)


def _get_pending(thread_id: str) -> dict:
    """[리뷰 반영] 꺼내지(pop) 않고 보기만 한다 — 재개 중 예외가 나도 승인 대기가 사라지지 않아 다시 시도할 수 있다.
    재개가 끝까지 성공한 뒤에 _finish_run이 지운다."""
    pending = app.state.pending.get(thread_id)
    if pending is None:
        raise HTTPException(404, f"승인 대기 중인 요청이 없습니다: thread_id={thread_id}")
    return pending


def _finish_run(thread_id: str, run_key: str) -> None:
    """[리뷰 반영] 끝난 요청의 체크포인트를 지운다. 요청마다 새 run_key로 저장하므로, 지우지 않으면
    InMemorySaver가 요청 수만큼 계속 커진다. 승인 대기 중인 요청은 재개해야 하므로 지우지 않는다."""
    if app.state.pending.get(thread_id, {}).get("run_key") == run_key:
        app.state.pending.pop(thread_id, None)
    checkpointer = app.state.graph.checkpointer if app.state.graph is not None else None
    if checkpointer is not None:
        checkpointer.delete_thread(run_key)


@app.post("/chat/approve")
async def approve(req: ApproveRequest):
    """[개선 11] 멈춘 그래프를 사람의 결정으로 재개한다."""
    pending = _get_pending(req.thread_id)
    result = await app.state.graph.ainvoke(
        Command(resume={"approved": req.approved, "comment": req.comment}),
        config=_run_config(pending["run_key"]))
    _finish_run(req.thread_id, pending["run_key"])
    deps = app.state.deps
    deps.session_store.append(req.thread_id, "user", pending["message"])
    deps.session_store.append(req.thread_id, "assistant", result["final"])
    return _chat_response(result, None)


def _sse(obj) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


def _guarded(body):
    """[개선 5] 예외가 나도 연결을 그냥 끊지 않고 Error 이벤트 + [DONE]으로 끝낸다."""
    async def gen():
        try:
            async for chunk in body:
                yield chunk
        except Exception as e:   # 원본은 여기서 연결이 그냥 끊겨 화면이 영원히 ⏳ 상태였다
            yield _sse({"type": "Error", "message": f"{type(e).__name__}: {e}"})
        yield "data: [DONE]\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream")


async def _stream_graph(graph_input, run_key: str, thread_id: str, message: str, seen: int = 0):
    """그래프 실행을 SSE로 중계한다. 새 질문과 [개선 11] 승인 후 재개가 같은 코드를 쓴다.

    seen: 이미 화면에 보낸 이벤트 수 — 재개 시 앞부분 이벤트를 다시 보내지 않으려고 넘겨받는다."""
    deps = app.state.deps
    final = ""
    last_prompt = None   # finalize까지 살아남은 최종 write/chat 호출의 실제 프롬프트
    # stream_mode를 리스트로 주면 (mode, payload) 튜플이 나온다 — "updates"는
    # 노드가 끝날 때마다의 상태 델타, "custom"은 write_node/chat_answer_node가
    # get_stream_writer()로 실행 중에 즉시 흘려보내는 토큰(AnswerToken) 이벤트다.
    async for mode, payload in app.state.graph.astream(
            graph_input, config=_run_config(run_key), stream_mode=["updates", "custom"]):
        if mode == "custom":
            yield _sse(payload)
            continue
        if "__interrupt__" in payload:   # [개선 11] approval 노드에서 멈춤 → 결정을 기다린다
            app.state.pending[thread_id] = {"run_key": run_key, "message": message}
            yield _sse({**payload["__interrupt__"][0].value, "thread_id": thread_id})
            return   # FinalAnswer도, 대화 이력 저장도 하지 않는다 — 재개 스트림이 마무리한다
        for node_name, delta in payload.items():
            for ev in (delta.get("events") or [])[seen:]:
                yield _sse({**ev, "node": node_name})
            if delta.get("events"):
                seen = len(delta["events"])
            if delta.get("final"):
                final = delta["final"]
            if delta.get("last_prompt"):
                last_prompt = delta["last_prompt"]
    yield _sse({"type": "FinalAnswer", "reply": final, "prompt": last_prompt})
    _finish_run(thread_id, run_key)
    deps.session_store.append(thread_id, "user", message)
    deps.session_store.append(thread_id, "assistant", final)


@app.get("/chat/stream")
async def chat_stream(message: str, thread_id: str = "default"):
    """Wave 3 — 그래프의 노드별 진행 상황을 SSE로 스트리밍.

    이벤트마다 'data: {json}\\n\\n' 형태로 내려보내고 마지막에 data: [DONE] 센티널을
    보낸다 — 프론트엔드는 \\n\\n 경계로 이벤트를 분리해 파싱해야 한다(Wave 3b).
    """
    deps = app.state.deps

    async def body():
        history, memory = await load_history(deps.session_store, deps.fast_llm, thread_id)
        if memory["updated"]:   # [개선 12] 요약을 새로 만든 요청이면 화면에 알린다
            yield _sse({"type": "MemorySummarized", **memory})
        app.state.pending.pop(thread_id, None)
        if deps.version != "v2":
            result = await run_v1(deps, message, thread_id, history)
            for ev in result["events"]:
                yield _sse(ev)
            yield _sse({"type": "FinalAnswer", "reply": result["final"]})
            deps.session_store.append(thread_id, "user", message)   # 원본 v1 스트림은 저장을 빠뜨렸다
            deps.session_store.append(thread_id, "assistant", result["final"])
            return
        graph_input = {"query": message, "thread_id": thread_id, "history": history,
                       "events": [], "cost_usd": 0.0}
        async for chunk in _stream_graph(graph_input, f"{thread_id}:{uuid.uuid4().hex[:8]}",
                                         thread_id, message):
            yield chunk

    return _guarded(body())


@app.get("/chat/approve/stream")
async def approve_stream(thread_id: str, approved: bool, comment: str = ""):
    """[개선 11] 승인/거절을 넘기고, 멈췄던 지점(approval)부터 finalize까지를 SSE로 이어서 보낸다."""
    pending = _get_pending(thread_id)

    async def body():
        snapshot = await app.state.graph.aget_state(_run_config(pending["run_key"]))
        seen = len(snapshot.values.get("events", []))
        async for chunk in _stream_graph(Command(resume={"approved": approved, "comment": comment}),
                                         pending["run_key"], thread_id, pending["message"], seen):
            yield chunk

    return _guarded(body())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("BACKEND_PORT", "8100")),
                log_level="warning")
