"""그래프 통합 테스트 — fake LLM + 가짜 MCP로 실제 LangGraph를 끝까지 돌린다(서버·Ollama 불필요)."""
import asyncio

from agent.graph import route_gate


def run(graph, query, history=None):
    return asyncio.run(graph.ainvoke({"query": query, "thread_id": "t", "history": history or [],
                                      "events": [], "cost_usd": 0.0}))


def probed(result):
    return [e["target"] for e in result["events"]
            if e["type"] == "TaskCompleted" and e["source"] == "monitoring"]


def test_unknown_system_skips_investigation(fake_graph):
    graph, mcp = fake_graph
    result = run(graph, "billing-service 응답이 느려요")
    assert probed(result) == []
    assert "billing-service" in result["final"] and "모니터링 대상이 아니" in result["final"]
    assert not any(c[0] == "call_tool" for c in mcp.calls)   # MCP 조회 자체를 하지 않음


def test_multi_system_query_investigates_all(fake_graph):
    graph, _ = fake_graph
    result = run(graph, "gateway-service에서 payments-service로 가는 요청이 타임아웃 나고 있어요")
    assert {"gateway-service", "payments-service"} <= set(probed(result))
    runbook = [e["target"] for e in result["events"]
               if e["type"] == "TaskCompleted" and e["source"] == "runbook"]
    assert runbook == ["network-timeout"]


def test_replan_event_records_retriever(fake_graph):
    graph, _ = fake_graph
    result = run(graph, "로그인 응답이 느려요, 원인 확인해주세요")
    replan = [e for e in result["events"] if e["type"] == "ReplanCheck" and not e["is_sufficient"]]
    assert replan and replan[0]["method"] == "tfidf" and replan[0]["target"] == "high-latency"


def test_validator_rejects_short_first_draft_then_passes(fake_graph):
    # fake quality 모델은 1차 초안을 일부러 짧게("재시작하면 됩니다.") 쓴다 → 반려 후 2차 합격
    graph, _ = fake_graph
    result = run(graph, "payments-service 결제 응답이 느려지고 CPU가 치솟았어요")
    verdicts = [e["verdict"] for e in result["events"] if e["type"] == "Validation"]
    assert verdicts == ["불합격", "합격"]
    assert result.get("grounding_issues") == []


def test_chat_route_remembers_name(fake_graph):
    graph, _ = fake_graph
    history = [{"role": "user", "content": "제 이름은 민수입니다"},
               {"role": "assistant", "content": "안녕하세요, 민수님!"}]
    result = run(graph, "제 이름이 뭐였죠?", history)
    assert result["route"] == "chat" and "민수" in result["final"]


def test_route_gate():
    assert route_gate("제 이름은 민수입니다", "incident") == "chat"          # think 끔 회귀 방지
    assert route_gate("billing-service 응답이 느려요", "chat") == "incident"
    assert route_gate("로그인 응답이 느려요", "chat") == "incident"          # 별칭도 대상으로 인정
    assert route_gate("web-01 상태 알려줘", "chat") == "chat"                # 애매하면 모델을 따름


def test_v1_rollback_investigates_incident(stub_mcp):
    # 원본 v1은 fake 응답 문구("도와드릴까요")에 의존해 실모델에서는 조사하지 않았다
    from agent.deps import SupervisorDeps
    from agent.fake_llm import FastLLM, QualityLLM
    from agent.supervisor_v1 import run_v1

    class PlainChat(FastLLM):   # 실모델처럼 "도와드릴까요"가 없는 답을 하는 가짜
        async def chat(self, query, history, on_token=None):
            return "확인해 보겠습니다.", self._usage(query, "")

    deps = SupervisorDeps(PlainChat("f", 0.0), QualityLLM("q", 0.0), stub_mcp, None, "v1")
    result = asyncio.run(run_v1(deps, "로그인 응답이 느려요", "t", []))
    assert any(e["type"] == "TaskCompleted" for e in result["events"])


def test_unclear_target_asks_back_then_resumes(fake_graph):
    # [개선 9] 시스템 단서가 없으면 auth-service로 찍지 않고 되묻는다 → 다음 턴에 시스템을 말하면 이어서 진단
    graph, mcp = fake_graph
    first = run(graph, "응답이 너무 느려요, 원인 확인해주세요")
    assert probed(first) == [] and first["final"].startswith("조사할 시스템을 먼저 알려 주세요.")
    assert not any(c[0] == "call_tool" for c in mcp.calls)
    history = [{"role": "user", "content": "응답이 너무 느려요, 원인 확인해주세요"},
               {"role": "assistant", "content": first["final"]}]
    second = run(graph, "payments-service요", history)
    assert second["route"] == "incident" and "payments-service" in probed(second)
    assert "응답이 너무 느려요" in second["query"]


def test_clarify_answer_without_system_routes_normally(fake_graph):
    graph, _ = fake_graph
    history = [{"role": "user", "content": "응답이 너무 느려요"},
               {"role": "assistant", "content": "조사할 시스템을 먼저 알려 주세요. …"}]
    result = run(graph, "잘 모르겠어요", history)
    assert result["route"] == "chat"
