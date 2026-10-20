"""[개선 11] Human-in-the-loop 승인 — 위험 조치 탐지 규칙과 interrupt → resume 흐름."""
import asyncio

import pytest
from langgraph.types import Command

from agent.approval import apply_decision, find_risky_actions


def test_find_risky_actions_from_runbook_sentences():
    text = ("1) CPU 91%입니다. 2) 해당 인스턴스를 오토스케일링 그룹에서 한 대 더 띄워 부하를 분산한다. "
            "3) 직전 배포를 롤백한다. 4) 프로세스를 즉시 재시작한다.")
    names = [a["action"] for a in find_risky_actions(text)]
    assert names == ["스케일 조정", "롤백", "재시작"]


def test_safe_answer_has_no_risky_actions():
    assert find_risky_actions("1) 모니터링 확인 2) 커넥션 풀 사용률 점검 3) 추가 확인 필요") == []


def test_reject_removes_risky_sentence():
    draft = "1) 지표 확인. 2) 직전 배포를 롤백한다. 3) 문서화."
    actions = find_risky_actions(draft)
    out = apply_decision(draft, actions, approved=False)
    assert "롤백한다" not in out and "[운영자 거절로 제외된 조치: 롤백]" in out and "⛔" in out


@pytest.fixture
def hitl_graph(stub_mcp):
    from agent.deps import SupervisorDeps
    from agent.fake_llm import FastLLM, QualityLLM
    from agent.graph import build_graph
    from tests.conftest import MemorySession

    deps = SupervisorDeps(FastLLM("f", 0.0), QualityLLM("q", 0.0), stub_mcp, MemorySession(), "v2",
                          hitl=True)
    return build_graph(deps)   # hitl=True면 InMemorySaver 체크포인터가 자동으로 붙는다


QUERY = "payments-service 결제 응답이 느려지고 CPU가 치솟았어요"   # cpu-spike runbook → 스케일 아웃 권고


def _start(graph, key):
    cfg = {"configurable": {"thread_id": key}}
    first = asyncio.run(graph.ainvoke({"query": QUERY, "thread_id": "t", "history": [],
                                       "events": [], "cost_usd": 0.0}, config=cfg))
    return cfg, first


@pytest.mark.parametrize("approved, mark", [(True, "✅ 운영자 승인 완료"), (False, "⛔")])
def test_interrupt_then_resume(hitl_graph, approved, mark):
    cfg, first = _start(hitl_graph, f"run-{approved}")
    assert "__interrupt__" in first and "final" not in first          # approval에서 멈췄다
    payload = first["__interrupt__"][0].value
    assert payload["type"] == "ApprovalRequired" and payload["actions"]
    done = asyncio.run(hitl_graph.ainvoke(Command(resume={"approved": approved}), config=cfg))
    assert mark in done["final"]
    assert [e for e in done["events"] if e["type"] == "ApprovalDecided"][0]["approved"] is approved


def test_hitl_off_passes_through(fake_graph):
    graph, _ = fake_graph   # conftest의 기본 deps는 hitl=False
    result = asyncio.run(graph.ainvoke({"query": QUERY, "thread_id": "t", "history": [],
                                        "events": [], "cost_usd": 0.0}))
    skipped = [e for e in result["events"] if e["type"] == "ApprovalSkipped"][0]
    assert skipped["risky"] > 0 and skipped["hitl"] is False and "final" in result


def test_sentence_split_ignores_decimal_point():
    # 화면 검증에서 발견: "CPU 36.8%"의 점에서 문장이 잘려 승인 카드에 "8% (알림 0건) 2) …"가 떴다
    draft = ("1) 모니터링: cache-service CPU 36.8% (알림 0건) "
             "2) OOM 직전(90% 이상)이면 해당 프로세스를 즉시 재시작해 서비스 중단을 막는다. 3) 문서화.")
    [a] = find_risky_actions(draft)
    assert a["sentence"] == "OOM 직전(90% 이상)이면 해당 프로세스를 즉시 재시작해 서비스 중단을 막는다"
    assert "36.8%" in apply_decision(draft, [a], approved=False)
