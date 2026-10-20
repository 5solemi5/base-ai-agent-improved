"""[개선 10] Research 멀티 에이전트 — supervisor의 선택 규칙과 서브에이전트를 따로따로 검증한다."""
import asyncio

from agent.research.log_agent import build_log_agent, summarize
from agent.research.metrics_agent import build_metrics_agent
from agent.research.supervisor import next_agent


def test_supervisor_order_and_log_skip():
    assert next_agent({}) == "metrics_agent"
    assert next_agent({"agents_called": ["metrics_agent"], "degraded": ["auth-service"]}) == "log_agent"
    # 저하된 시스템이 없으면 로그 분석을 건너뛴다 (healthy 시스템 로그는 근거가 되지 않음)
    assert next_agent({"agents_called": ["metrics_agent"], "degraded": []}) == "runbook_agent"
    assert next_agent({"agents_called": ["metrics_agent", "runbook_agent"], "degraded": []}) == "done"
    # 이미 부른 에이전트는 다시 부르지 않는다 → 무한 루프 불가
    assert next_agent({"agents_called": ["metrics_agent", "log_agent", "runbook_agent"],
                       "degraded": ["x"]}) == "done"


def test_metrics_agent_subgraph_alone(stub_mcp):
    out = asyncio.run(build_metrics_agent(stub_mcp).ainvoke({"targets": ["web-01", "auth-service"]}))
    assert out["degraded"] == ["auth-service"] and out["retries"] == [0, 0]


def test_log_agent_ranks_repeated_errors_first(stub_mcp):
    out = asyncio.run(build_log_agent(stub_mcp).ainvoke({"systems": ["gateway-service"]}))
    top = out["findings"][0]["top"][0]
    assert top["level"] == "ERROR" and top["count"] == 2 and "upstream timeout" in top["message"]


def test_summarize_handles_empty_logs():
    out = summarize({"logs": [{"system_name": "web-01", "lines": [], "error_count": 0, "warn_count": 0}]})
    assert out["findings"][0]["top"] == []


def test_healthy_only_query_skips_log_agent(fake_graph):
    graph, mcp = fake_graph
    result = asyncio.run(graph.ainvoke({"query": "web-01 CPU 급증 확인해줘", "thread_id": "t",
                                        "history": [], "events": [], "cost_usd": 0.0}))
    assert "log_agent" not in result["agents_called"]
    assert not any(c[0] == "get_recent_logs" for c in mcp.calls)


def test_log_claims_allowed_once_logs_collected():
    # 로그를 실제로 수집했다면 "로그 확인 결과"를 써도 근거 위반이 아니다 (개선 4의 규칙과 연결)
    from agent.grounding import check_grounding
    collected = [{"source": "log", "data": {"system_name": "auth-service", "top": []}}]
    assert not any("로그" in i for i in check_grounding("로그 확인 결과 풀 고갈 반복", collected))


def test_supervisor_hard_cap():
    # [리뷰 반영] 이름이 겹치거나 상태가 이상해도 3번 부른 뒤에는 무조건 끝낸다
    assert next_agent({"agents_called": ["metrics_agent"] * 3, "degraded": ["x"]}) == "done"
