"""[개선 15] 답변 형식 점검 — OpenAI로 다듬기 전후를 같은 잣대로 센다.
사용: python3 eval/format_check.py after_openai after_openai_polished
- 장애 답변(조사한 시스템이 있는 것)만 본다.
- 덧붙인 문단: 1)~3) 항목, '추가 확인 필요' 줄, 승인 표시 줄이 아닌 문단
- 실행 약속: 에이전트는 조치를 실행하지 않는데 '~하겠습니다/진행합니다'라고 씀
- 빈 마무리: '추가 확인 필요'로 끝나고 무엇을 확인할지 적지 않음
"""
import json
import re
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
ALLOWED = re.compile(r"^\s*(\d\)|추가 확인 필요|✅|⛔|⚠️)")


def check(label: str) -> dict:
    rows = json.loads((RESULTS / f"{label}.json").read_text())["rows"]
    inc = [r for r in rows if r["probed"]]
    extra = markdown = vague = promise = 0
    lengths = []
    for r in inc:
        text = r["reply"]
        paras = [p for p in re.split(r"\n\s*\n|\n(?=\d\))", text) if p.strip()]
        extra += any(not ALLOWED.match(p) for p in paras)
        markdown += bool(re.search(r"\*\*|^#", text, re.M))
        vague += bool(re.search(r"추가 확인 필요\s*\.?\s*$", text.split("✅")[0].split("⛔")[0].strip()))
        body = text.split("✅")[0].split("⛔")[0]
        promise += bool(re.search(r"하겠습니다|진행합니다|분산합니다|적용합니다|재시작합니다", body))
        lengths.append(len(text.split("✅")[0].split("⛔")[0]))
    return {"label": label, "n": len(inc), "덧붙인 문단이 있는 답": extra, "마크다운": markdown,
            "'추가 확인 필요'만 쓰고 무엇인지 없음": vague,
            "조치를 직접 하겠다고 씀": promise,
            "평균 길이(자)": round(sum(lengths) / len(lengths))}


if __name__ == "__main__":
    out = [check(l) for l in sys.argv[1:]]
    keys = [k for k in out[0] if k != "label"]
    print("| 지표 | " + " | ".join(o["label"] for o in out) + " |")
    print("|---|" + "---|" * len(out))
    for k in keys:
        print(f"| {k} | " + " | ".join(str(o[k]) for o in out) + " |")
