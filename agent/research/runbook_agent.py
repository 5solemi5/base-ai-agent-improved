"""런북 분석 서브에이전트 — classify(RAG로 incident_type 선택) → fetch(runbook resource 조회)."""
import asyncio
from functools import partial
from typing import TypedDict

from langgraph.graph import END, StateGraph
from langsmith import traceable

from ..runbook_retriever import RunbookRetriever

RETRIEVER = RunbookRetriever()   # [개선 3] runbook 인덱스는 프로세스당 1번만 만든다(W1 싱글턴과 같은 이유)


class RunbookState(TypedDict, total=False):
    query: str                # 입력: 사용자 질문
    collected: list[dict]     # 입력: 지금까지 모은 사실(모니터링·로그) — 검색 질의에 쓴다
    picked: dict              # 출력: RETRIEVER.classify 결과 (incident_type, method, scores)
    runbook: str              # 출력: runbook 본문


def log_signals(collected: list[dict]) -> str:
    """[개선 10] 로그 에이전트가 찾은 ERROR/WARN 메시지도 RAG 질의에 넣는다(증상 근거가 늘어난다)."""
    return " ".join(t["message"] for c in collected if c.get("source") == "log"
                    for t in c["data"].get("top", []))


@traceable(name="runbook_agent.classify")
async def classify(state: RunbookState) -> dict:
    query = state["query"] + " " + log_signals(state.get("collected", []))
    # embed 모드면 classify 안에서 동기 HTTP 호출 → 스레드로 넘겨 이벤트 루프를 막지 않는다
    picked = await asyncio.to_thread(RETRIEVER.classify, query, state.get("collected", []))
    return {"picked": picked}


@traceable(name="runbook_agent.fetch")
async def fetch(state: RunbookState, mcp) -> dict:
    # URI를 코드에 하드코딩하지 않는다 — 연결된 서버들을 조회해 템플릿을 가진 서버를 찾는다
    server, template = await mcp.find_resource_template("get_incident_runbook")
    uri = template.format(incident_type=state["picked"]["incident_type"])
    return {"runbook": await mcp.read_resource(uri, server=server)}


def build_runbook_agent(mcp):
    g = StateGraph(RunbookState)
    g.add_node("classify", classify)
    g.add_node("fetch", partial(fetch, mcp=mcp))
    g.set_entry_point("classify")
    g.add_edge("classify", "fetch")
    g.add_edge("fetch", END)
    return g.compile()
