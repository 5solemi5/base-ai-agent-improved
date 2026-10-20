"""[개선 13] LLM-as-judge 보조 로직 — LLM 없이 확인할 수 있는 부분만 (라벨 세트·사실 구성·권고문 필터)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))
import judge  # noqa: E402

CASES = json.loads((Path(judge.HERE) / "judge_cases.json").read_text(encoding="utf-8"))["cases"]


def test_label_set_is_balanced_enough():
    assert len(CASES) == 12 and sum(c["faithful"] for c in CASES) == 4


def test_faithful_answers_pass_rule_check():
    # 라벨이 '충실'인 답은 규칙 검사도 통과해야 한다 — 라벨 세트 자체의 일관성 확인
    for c in CASES:
        if c["faithful"]:
            assert judge.rule(c["answer"], judge.build_collected(c))["pass"], c["id"]


def test_build_collected_matches_agent_shape():
    collected = judge.build_collected(CASES[0])
    assert [c["source"] for c in collected] == ["monitoring", "log", "runbook"]
    assert collected[1]["data"]["error_count"] == 2


def test_recommendation_filter():
    assert judge.RECOMMENDATION.search("runbook에 따라 헬스체크를 점검하세요.")
    assert judge.RECOMMENDATION.search("타임아웃 임계값 상향 및 재시도 정책 재검토 필요")
    assert judge.RECOMMENDATION.search("추가 확인 필요: p95 latency 값 및 리전별 트래픽 분포.")
    assert not judge.RECOMMENDATION.search("gateway-service의 CPU 과부하가 원인입니다.")


def test_system_note_lines_removed():
    text = "1) CPU 91%입니다.\n\n✅ 운영자 승인 완료 — 위험 조치: 재시작"
    assert judge.SYSTEM_NOTE.sub("", text).strip() == "1) CPU 91%입니다."


def test_recommendation_with_number_is_still_checked():
    # [리뷰 반영] 숫자가 든 권고문은 거르지 않는다(틀린 수치가 '권고' 모양으로 숨는 것 방지)
    class StubJudge(judge.Judge):
        def __init__(self):
            pass

        def _ask(self, system, facts, answer, schema):
            return {"claims": [{"claim": "CPU 97%로 증설이 필요", "supported": False, "evidence": ""},
                               {"claim": "헬스체크 점검 필요", "supported": False, "evidence": ""}]}

    out = StubJudge().claims("CPU 91%", "1) CPU 97%로 증설이 필요 2) 헬스체크 점검 필요")
    assert [c["claim"] for c in out["detail"]] == ["CPU 97%로 증설이 필요"]
