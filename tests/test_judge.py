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
    assert not judge.RECOMMENDATION.search("gateway-service의 CPU 과부하가 원인입니다.")
