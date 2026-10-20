"""[개선 2] 조사 대상 결정 규칙 테스트 — 실측에서 발견한 실패 사례를 그대로 테스트로 고정한다."""
import pytest

from agent.systems import (INCIDENT_PATTERN, KNOWN_SYSTEMS, as_bool, normalize_plan,
                           resolve_query_systems)


def test_known_systems_loaded_from_monitoring_json():
    # 원본은 경로가 틀어지면 조용히 2개짜리 목록으로 바뀌었다 → 6개 전부 읽혔는지 확인
    assert len(KNOWN_SYSTEMS) == 6
    assert {"auth-service", "billing-service"} & KNOWN_SYSTEMS == {"auth-service"}


@pytest.mark.parametrize("query, systems, source", [
    ("payments-service 결제 응답이 느려지고 CPU가 치솟았어요", ["payments-service"], "query"),
    ("gateway-service에서 payments-service로 가는 요청이 타임아웃", ["gateway-service", "payments-service"], "query"),
    ("web-01 CPU 급증, auth-service 커넥션 풀도 같이", ["web-01", "auth-service"], "query"),
    ("payments-service랑 cache-service 둘 다 저하", ["payments-service", "cache-service"], "query"),
    ("로그인 응답이 느려요", ["auth-service"], "alias"),
    ("billing-service 응답이 느려요", [], "unknown-only"),
    ("서버가 이상해요", ["auth-service"], "default"),
])
def test_resolve_query_systems(query, systems, source):
    r = resolve_query_systems(query)
    assert r["systems"] == systems
    assert r["source"] == source


def test_unknown_system_is_reported_not_replaced():
    r = resolve_query_systems("billing-service 응답이 느려요")
    assert r["unknown"] == ["billing-service"]   # 원본: auth-service로 바꿔치기


def test_sentence_target_does_not_drop_second_system():
    # 실측 원출력: target이 문장 → 원본은 web-01로 덮어쓰고 auth-service를 중복으로 버렸다
    raw = {"needs_analysis": False,
           "tasks": [{"agent": "research", "target": "check web-01 cpu"},
                     {"agent": "research", "target": "check auth-service connection pool status"}]}
    plan = normalize_plan(raw, "web-01 CPU 급증, auth-service 커넥션 풀도 같이 확인해줘")
    assert [t["target"] for t in plan["tasks"] if t["agent"] == "research"] == ["web-01", "auth-service"]


def test_model_extra_systems_are_dropped():
    # think 끔 실측: 쿼리에 없는 시스템까지 4~5개 계획 → 쿼리 범위로 제한
    raw = {"tasks": [{"agent": "research", "target": s} for s in sorted(KNOWN_SYSTEMS)]}
    plan = normalize_plan(raw, "로그인 응답이 느려요")
    assert [t["target"] for t in plan["tasks"]] == ["auth-service"]


def test_default_case_uses_only_first_model_system():
    # 단서가 없는 질문 — 모델이 시스템을 전부 나열해도 첫 번째 하나만 조사
    raw = {"tasks": [{"agent": "research", "target": f"check {s}"} for s in ["cache-service", "web-01"]]}
    plan = normalize_plan(raw, "서버가 이상해요")
    assert [t["target"] for t in plan["tasks"]] == ["cache-service"]


def test_analysis_only_when_two_or_more_systems():
    raw = {"needs_analysis": True, "tasks": [{"agent": "analysis"}]}
    single = normalize_plan(raw, "payments-service가 느려요")
    multi = normalize_plan(raw, "payments-service랑 cache-service 둘 다 저하")
    assert not any(t["agent"] == "analysis" for t in single["tasks"])
    assert any(t["agent"] == "analysis" for t in multi["tasks"]) and multi["needs_analysis"]


@pytest.mark.parametrize("value, expected", [("false", False), ("False", False), ("true", True),
                                             (True, True), (None, False), ("", False)])
def test_as_bool(value, expected):
    assert as_bool(value) is expected   # bool("false")는 True — 그 함정을 막는다


@pytest.mark.parametrize("query, incident", [
    ("payments-service 결제 응답이 느려지고 CPU가 치솟았어요", True),
    ("gateway-service에서 요청이 타임아웃 나고 있어요", True),
    ("payments-service랑 cache-service 둘 다 저하됐는데", True),
    ("로그인 응답이 느려요", True),
    ("안녕, 너는 누구야?", False),
    ("제 이름은 민수입니다", False),
    ("제 이름이 뭐였죠?", False),
])
def test_incident_pattern(query, incident):
    assert bool(INCIDENT_PATTERN.search(query)) is incident


def test_incident_pattern_covers_conjugations():
    # 7단계 실험: 패턴에서 "느려"를 빼도 기존 테스트가 모두 통과했다 → 빈틈을 메우는 문장
    assert INCIDENT_PATTERN.search("결제가 느려졌어")
