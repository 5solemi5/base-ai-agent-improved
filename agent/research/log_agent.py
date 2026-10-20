"""로그 분석 서브에이전트 — fetch(최근 WARN/ERROR 로그 조회) → summarize(오류 패턴 요약)."""
import asyncio
from collections import Counter
from functools import partial
from typing import TypedDict

from langgraph.graph import END, StateGraph
from langsmith import traceable

from ..retry import with_retry


class LogState(TypedDict, total=False):
    systems: list[str]        # 입력: 로그를 볼 시스템 (보통 저하된 시스템만)
    logs: list[dict]          # 출력: get_recent_logs 원본 결과
    findings: list[dict]      # 출력: 시스템별 요약 {system_name, error_count, warn_count, top}


@traceable(name="log_agent.fetch")
async def fetch(state: LogState, mcp) -> dict:
    async def one(system: str) -> dict:
        data, _ = await with_retry(mcp.call_tool, "get_recent_logs",
                                   {"system_name": system, "min_level": "WARN", "limit": 20})
        return data

    return {"logs": list(await asyncio.gather(*(one(s) for s in state["systems"])))}


def summarize(state: LogState) -> dict:
    """같은 메시지가 몇 번 반복됐는지 센다. 반복되는 ERROR가 원인 후보 1순위다.
    LLM을 쓰지 않는다 — 개수 세기는 코드가 정확하고 빠르다(0.6b에게 맡기면 틀리게 센다)."""
    findings = []
    for log in state["logs"]:
        counts = Counter((l["level"], l["message"]) for l in log["lines"])
        top = [{"level": lv, "message": msg, "count": n}
               for (lv, msg), n in sorted(counts.items(),
                                          key=lambda kv: (kv[0][0] != "ERROR", -kv[1]))[:3]]
        findings.append({"system_name": log["system_name"], "error_count": log["error_count"],
                         "warn_count": log["warn_count"], "top": top})
    return {"findings": findings}


def build_log_agent(mcp):
    g = StateGraph(LogState)
    g.add_node("fetch", partial(fetch, mcp=mcp))
    g.add_node("summarize", summarize)
    g.set_entry_point("fetch")
    g.add_edge("fetch", "summarize")
    g.add_edge("summarize", END)
    return g.compile()
