"""[개선 8] 모듈화 후 — 그래프 전체를 만들지 않고 노드 함수 하나만 직접 불러 테스트한다."""
import asyncio

from agent.nodes.planning import gate_node
from agent.nodes.writing import MAX_VALIDATION_ATTEMPTS, finalize_node, validate_node


def test_gate_node_removes_analysis_when_not_needed():
    state = {"needs_analysis": False, "events": [],
             "tasks": [{"agent": "research"}, {"agent": "analysis"}]}
    out = asyncio.run(gate_node(state, deps=None))   # deps를 안 쓰는 노드라 None으로 충분
    assert out["tasks"] == [{"agent": "research"}] and out["gate_removed"] == 1


def test_validate_node_passes_after_max_attempts_with_issue_recorded():
    state = {"draft": "짧음", "collected": [], "events": [],
             "validation_attempts": MAX_VALIDATION_ATTEMPTS - 1}
    out = asyncio.run(validate_node(state, deps=None))
    assert out["validation_feedback"] is None                  # 무한 재작성 방지 → 통과
    assert out["events"][-1]["verdict"].startswith("합격(최대")


def test_finalize_node_appends_notes():
    state = {"draft": "1) 확인 2) 조치", "events": [], "unknown_systems": ["billing-service"],
             "grounding_issues": ["수치 문제"]}
    out = asyncio.run(finalize_node(state, deps=None))
    assert "billing-service" in out["final"] and "수치 문제" in out["final"]
