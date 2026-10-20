"""메트릭 분석 서브에이전트 — fetch(모니터링 조회) → assess(저하 시스템 판정)."""
import asyncio
import time
from functools import partial
from typing import TypedDict

from langgraph.graph import END, StateGraph
from langsmith import traceable

from ..retry import with_retry

TASK_DELAY_S = 0.35   # 태스크당 고정 지연 — 원본 research_node와 같게 둬서 병렬 효과를 실측할 수 있게


class MetricsState(TypedDict, total=False):
    targets: list[str]        # 입력: 조사할 시스템
    results: list[dict]       # 출력: 시스템별 모니터링 값
    retries: list[int]        # 출력: 시스템별 재시도 횟수 (targets와 같은 순서)
    elapsed: float            # 출력: 병렬 조회 소요(초)
    degraded: list[str]       # 출력: healthy가 아닌 시스템


@traceable(name="metrics_agent.fetch")
async def fetch(state: MetricsState, mcp) -> dict:
    async def one(system: str) -> tuple[dict, int]:
        await asyncio.sleep(TASK_DELAY_S)
        return await with_retry(mcp.call_tool, "get_monitoring_status", {"system_name": system})

    started = time.perf_counter()
    pairs = await asyncio.gather(*(one(s) for s in state["targets"]))
    return {"results": [data for data, _ in pairs],
            "retries": [r for _, r in pairs],
            "elapsed": round(time.perf_counter() - started, 2)}


def assess(state: MetricsState) -> dict:
    """판정은 코드로 — status가 healthy가 아니면 저하. (unknown은 데이터가 없는 것이라 제외)"""
    return {"degraded": [d["system_name"] for d in state["results"]
                         if d.get("status") not in ("healthy", "unknown")]}


def build_metrics_agent(mcp):
    g = StateGraph(MetricsState)
    g.add_node("fetch", partial(fetch, mcp=mcp))
    g.add_node("assess", assess)
    g.set_entry_point("fetch")
    g.add_edge("fetch", "assess")
    g.add_edge("assess", END)
    return g.compile()
