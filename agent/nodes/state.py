"""그래프 상태 정의 + 노드 공통 도우미."""
from typing import TypedDict


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


def add_cost(state: AgentState, usage: dict) -> float:
    """지금까지 쌓인 비용 + 이번 LLM 호출 비용 (원래 graph.py의 _cost)."""
    return round(state.get("cost_usd", 0.0) + usage["usd"], 6)
