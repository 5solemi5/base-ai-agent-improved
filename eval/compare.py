"""두 측정 결과(eval/results/<label>.json)를 Markdown 표로 비교한다.

사용법: python eval/compare.py before after
"""
import json
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"

LABELS = [
    ("route_accuracy", "라우팅 정확도"),
    ("incident_total_s", "장애 쿼리 총 소요(초)"),
    ("all_total_s", "전체 9건 총 소요(초)"),
    ("missing_systems", "조사 누락 시스템 수"),
    ("extra_systems", "불필요하게 조사한 시스템 수"),
    ("runbook_accuracy", "runbook 정확도"),
    ("unknown_handled", "미등록 시스템 정직하게 안내"),
    ("flagged_answers", "근거 없는 내용이 섞인 답변 수"),
    ("name_memory", "이름 기억"),
    ("cost_usd", "가상 비용 합계(USD)"),
    ("retry_total", "재시도 합계"),
    ("validation_attempts_total", "검증 시도 합계"),
]


def main():
    a, b = (json.loads((RESULTS / f"{name}.json").read_text()) for name in sys.argv[1:3])
    print(f"| 지표 | {a['label']} | {b['label']} |")
    print("|---|---|---|")
    for key, name in LABELS:
        print(f"| {name} | {a['summary'][key]} | {b['summary'][key]} |")
    print()
    print(f"| # | 쿼리 | {a['label']} 조사·runbook·시간 | {b['label']} 조사·runbook·시간 |")
    print("|---|---|---|---|")
    for ra, rb in zip(a["rows"], b["rows"]):
        cell = lambda r: (f"{', '.join(r['probed']) or '-'} · {', '.join(r['runbooks']) or '-'} · "
                          f"{r['elapsed_s']}s" + (" ⚠️" if r.get("flags") else ""))
        print(f"| {ra['id']} | {ra['query'][:28]} | {cell(ra)} | {cell(rb)} |")


if __name__ == "__main__":
    main()
