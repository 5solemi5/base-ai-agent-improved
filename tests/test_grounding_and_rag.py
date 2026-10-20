"""[개선 3·4] RAG 검색기와 근거 검사기 테스트."""
import pytest

from agent.grounding import check_grounding
from agent.runbook_retriever import RunbookRetriever

AUTH = {"source": "monitoring", "data": {
    "system_name": "auth-service", "status": "degraded", "cpu_percent": 80.9, "memory_percent": 62.5,
    "alerts": ["p95 latency 2.4s (threshold 1.0s)", "connection pool 92% used"]}}
WEB = {"source": "monitoring", "data": {
    "system_name": "web-01", "status": "healthy", "cpu_percent": 41.8, "memory_percent": 51.9, "alerts": []}}
RUNBOOK = {"source": "runbook", "data": "커넥션 풀 사용률이 90%를 넘으면 풀 크기를 늘린다. p95 임계치(1.0s)"}


# ---------- 근거 검사 (실측 Before 답변에서 나온 문장들) ----------

def test_grounded_answer_passes():
    draft = ("1) auth-service p95 2.4s, 커넥션 풀 92%로 임계치 초과 "
             "2) 90%를 넘었으므로 풀 크기 확장 3) 1.0s 이하로 내려오는지 확인")
    assert check_grounding(draft, [AUTH, RUNBOOK]) == []


def test_log_claim_is_rejected():
    draft = "1) CPU 80.9% 2) 최근 로그에서 반복 호출 패턴 없음 3) 확장"
    issues = check_grounding(draft, [AUTH])
    assert any("로그" in i for i in issues)


def test_login_word_is_not_a_log_claim():
    # "로그인 … 결과"는 로그 분석 단정이 아니다 (실제 오탐 수정)
    assert check_grounding("'로그인 응답이 느려요' 진단 결과: 1) auth-service CPU 80.9% 2) 확장", [AUTH]) == []


def test_unprobed_system_is_rejected():
    draft = "1) web-01 CPU 41.8% 2) auth-service 커넥션 풀 확인 필요"
    issues = check_grounding(draft, [WEB])
    assert any("auth-service" in i for i in issues)


def test_invented_number_is_rejected():
    draft = "1) auth-service CPU 80.9% 2) cache-service CPU 53.5% 3) 재시작"
    issues = check_grounding(draft, [AUTH])
    assert any("53.5%" in i for i in issues)
    assert any("cache-service" in i for i in issues)


def test_unknown_system_may_be_mentioned_when_allowed():
    draft = "1) billing-service는 모니터링 대상이 아닙니다 2) auth-service CPU 80.9%"
    assert check_grounding(draft, [AUTH], allowed_systems=["billing-service"]) == []


# ---------- RAG 검색기 ----------

@pytest.fixture(scope="module")
def retriever():
    return RunbookRetriever(method="tfidf")


@pytest.mark.parametrize("query, collected, expected", [
    ("gateway-service에서 payments-service로 가는 요청이 타임아웃",
     [{"source": "monitoring", "data": {"status": "degraded",
       "alerts": ["upstream timeout to payments-service (504)", "network packet loss 3%"]}},
      {"source": "monitoring", "data": {"status": "degraded",
       "alerts": ["CPU 91% (threshold 80%)", "payments-service-3 instance responding slow"]}}],
     "network-timeout"),   # 원본 키워드 규칙은 cpu-spike
    ("web-01 CPU가 갑자기 올라갔어요", [WEB], "cpu-spike"),          # 원본: db-down 폴백
    ("web-01 디스크가 꽉 찼어요", [WEB], "disk-full"),               # 원본: db-down 폴백
    ("로그인 응답이 느려요", [AUTH], "high-latency"),
])
def test_retriever_picks_expected_runbook(retriever, query, collected, expected):
    assert retriever.classify(query, collected)["incident_type"] == expected


def test_healthy_system_alerts_are_ignored(retriever):
    # healthy 시스템의 정보는 질의에 넣지 않는다 → 질의 = 사용자 질문 + 저하된 시스템 알림만
    picked = retriever.classify("web-01 CPU 급증", [WEB, AUTH])
    assert picked["query_text"].strip() == "web-01 CPU 급증 " + " ".join(AUTH["data"]["alerts"])


def test_embed_falls_back_to_tfidf_when_model_missing(monkeypatch):
    # 임베딩 서버/모델이 없으면 죽지 않고 tfidf로 전환되어야 한다
    monkeypatch.setenv("OLLAMA_EMBED_URL", "http://127.0.0.1:9/api/embed")
    r = RunbookRetriever(method="embed")
    assert r.method == "tfidf" and hasattr(r, "fallback_reason")


# ---------- 8단계: 코드 리뷰에서 찾은 버그의 회귀 테스트 ----------

def test_runbook_chunk_keeps_numbers_inside_text(retriever):
    # "4. 5분 내 p95가 …" → 앞 번호 "4."만 떼고 본문의 "5분"은 남아야 한다
    texts = [c["text"] for c in retriever.chunks if c["incident_type"] == "high-latency"]
    assert any(t.startswith("5분 내") for t in texts)
    assert not any(t.startswith("분 내") for t in texts)


def test_last_checked_time_is_not_a_fact():
    # 조회 시각 10:41의 "41"이 사실로 들어가 지어낸 "41%"가 통과하던 버그
    mon = {"source": "monitoring", "data": {**AUTH["data"], "last_checked": "2026-10-20T10:41:07.123456"}}
    issues = check_grounding("1) auth-service CPU 41% 2) 커넥션 풀 92%", [mon])
    assert any("41%" in i for i in issues)
