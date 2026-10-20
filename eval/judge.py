"""[개선 13] LLM-as-judge — "답변이 수집된 사실(수치·상태)을 정확히 인용했는가"를 채점하고,
채점기 자체가 믿을 만한지 정답 라벨 세트(eval/judge_cases.json)로 먼저 검증한다.

비교하는 채점기 4가지:
  rule    : agent/grounding.py (개선 4의 규칙 검사 — 시스템 이름·단위 있는 수치·로그 단정)
  score   : LLM에게 1~5점을 매기게 함 (흔한 방식, 4점 이상이면 통과)
  claims  : LLM에게 답변의 사실 주장을 하나씩 뽑아 '수집된 사실로 확인되는가'를 판정하게 함
            → 하나라도 확인 안 되면 불통과 (점수 대신 근거를 요구하는 방식)
  rule+claims : 둘 중 하나라도 문제를 찾으면 불통과

사용법:
    python eval/judge.py                               # 라벨 세트로 채점기 검증 (qwen3:8b)
    python eval/judge.py --results after_ollama_v2     # bench 결과의 실제 답변을 claims 채점
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from agent.grounding import check_grounding   # noqa: E402
from agent.ollama_llm import OllamaLLM, format_facts   # noqa: E402
from agent.research.log_agent import summarize as summarize_logs   # noqa: E402

SCORE_SCHEMA = {"type": "object", "properties": {"score": {"type": "integer"}, "reason": {"type": "string"}},
                "required": ["score", "reason"]}
CLAIMS_SCHEMA = {"type": "object", "properties": {"claims": {"type": "array", "items": {
    "type": "object", "properties": {"claim": {"type": "string"}, "supported": {"type": "boolean"},
                                     "evidence": {"type": "string"}},
    "required": ["claim", "supported", "evidence"]}}}, "required": ["claims"]}

# 권고문("…점검하세요")은 사실 주장이 아니다 — 프롬프트로 빼라고 해도 8b가 뽑아 와서(실측: 멀쩡한 답 2개를
# 오판) 코드로 한 번 더 거른다. 문장 끝이 명령·권고형이면 주장에서 제외.
RECOMMENDATION = re.compile(r"^추가 확인 필요|(?:하세요|하십시오|하라|바랍니다|권장합니다|권고합니다|필요합니다|필요)[.!\s]*$")
# 시스템이 덧붙인 안내 줄(✅ 승인 기록, ⛔ 거절 기록, ※ 검증 경고)은 모델이 쓴 주장이 아니다
SYSTEM_NOTE = re.compile(r"^\s*[✅⛔※].*$", re.M)
NUMBER = re.compile(r"\d+(?:\.\d+)?")
SCORE_SYSTEM = ("당신은 운영 보고서 채점자다. '수집된 사실'과 '답변'을 비교해 답변이 사실에 충실한지 "
                "1~5점으로 채점하라(5=모든 내용이 사실과 일치, 1=사실과 다르거나 지어낸 내용이 많음). "
                "JSON {score, reason}으로 답하라.")
CLAIMS_SYSTEM = ("당신은 사실 확인 담당자다. '답변'에서 사실 주장(수치, 개수, 시스템 상태, 원인 단정)을 "
                 "하나씩 뽑아라. 각 주장마다 '수집된 사실'에 그 내용이 실제로 있는지 확인해 supported를 정하라. "
                 "수치·개수는 정확히 같아야 하고, 사실에 없는 원인 단정이나 사실과 반대되는 상태는 false다. "
                 "'확인하세요' 같은 권고·'추가 확인 필요'는 주장이 아니므로 뽑지 마라. "
                 "evidence에는 근거가 된 사실 문장을 그대로 적어라(없으면 빈 문자열).")


def build_collected(case: dict) -> list[dict]:
    """라벨 세트의 사실을 data/ 파일에서 만든다 — 에이전트가 수집하는 것과 같은 구조, jitter 없는 기준값."""
    mon = json.loads((ROOT / "data" / "monitoring.json").read_text(encoding="utf-8"))
    logs = json.loads((ROOT / "data" / "logs.json").read_text(encoding="utf-8"))
    collected = [{"source": "monitoring", "data": {
        "system_name": s, "status": mon[s]["status"], "cpu_percent": mon[s]["cpu_base"],
        "memory_percent": mon[s]["memory_base"], "alerts": mon[s]["alerts"]}} for s in case["systems"]]
    raw = [{"system_name": s, "lines": [l for l in logs[s] if l["level"] != "INFO"]} for s in case["logs"]]
    for r in raw:
        r["error_count"] = sum(l["level"] == "ERROR" for l in r["lines"])
        r["warn_count"] = sum(l["level"] == "WARN" for l in r["lines"])
    collected += [{"source": "log", "data": f} for f in summarize_logs({"logs": raw})["findings"]]
    collected.append({"source": "runbook",
                      "data": (ROOT / "data" / "runbooks" / f"{case['runbook']}.md").read_text(encoding="utf-8")})
    return collected


class Judge:
    def __init__(self, model: str):
        self.llm = OllamaLLM(model, usd_per_1k_tokens=0.005)

    def _ask(self, system: str, facts: str, answer: str, schema: dict) -> dict:
        user = f"## 수집된 사실\n{facts}\n\n## 답변\n{answer}"
        content, _ = self.llm._chat(system, user, schema=schema)
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {}

    def score(self, facts: str, answer: str) -> dict:
        out = self._ask(SCORE_SYSTEM, facts, answer, SCORE_SCHEMA)
        return {"pass": int(out.get("score", 0)) >= 4, "detail": out}

    def claims(self, facts: str, answer: str) -> dict:
        answer = SYSTEM_NOTE.sub("", answer).strip()
        out = self._ask(CLAIMS_SYSTEM, facts, answer, CLAIMS_SCHEMA)
        answer_numbers = set(NUMBER.findall(answer))
        bad, misquoted = [], []
        for c in out.get("claims", []):
            text = c.get("claim", "").strip()
            if c.get("supported", False) or RECOMMENDATION.search(text):
                continue
            # 채점기가 답변을 잘못 옮겨 적은 경우(실측: 답변의 95%를 93%로 인용) — 답변에 없는 숫자가
            # 들어간 주장은 '답변의 문제'가 아니라 '채점기의 착오'로 따로 센다
            if set(NUMBER.findall(text)) - answer_numbers:
                misquoted.append(c)
            else:
                bad.append(c)
        return {"pass": not bad, "detail": bad, "misquoted": misquoted}


def rule(answer: str, collected: list[dict]) -> dict:
    issues = check_grounding(answer, collected)
    return {"pass": not issues, "detail": issues}


def validate_judges(judge: Judge) -> dict:
    cases = json.loads((HERE / "judge_cases.json").read_text(encoding="utf-8"))["cases"]
    names = ["rule", "score", "claims", "rule+claims"]
    rows, timing = [], {"score": 0.0, "claims": 0.0}
    for case in cases:
        collected = build_collected(case)
        facts = format_facts(collected)
        r = rule(case["answer"], collected)
        t0 = time.perf_counter(); s = judge.score(facts, case["answer"]); timing["score"] += time.perf_counter() - t0
        t0 = time.perf_counter(); c = judge.claims(facts, case["answer"]); timing["claims"] += time.perf_counter() - t0
        verdicts = {"rule": r["pass"], "score": s["pass"], "claims": c["pass"],
                    "rule+claims": r["pass"] and c["pass"]}
        rows.append({"id": case["id"], "faithful": case["faithful"], "note": case.get("note", ""),
                     "verdicts": verdicts, "rule": r["detail"], "score": s["detail"], "claims": c["detail"]})
        marks = " ".join(f"{n}={'O' if verdicts[n] == case['faithful'] else 'X'}" for n in names)
        print(f"#{case['id']:>2} 정답={'충실' if case['faithful'] else '문제'}  {marks}", flush=True)

    bad = [r for r in rows if not r["faithful"]]
    good = [r for r in rows if r["faithful"]]
    summary = {}
    for n in names:
        summary[n] = {
            "accuracy": f"{sum(r['verdicts'][n] == r['faithful'] for r in rows)}/{len(rows)}",
            "caught_bad": f"{sum(not r['verdicts'][n] for r in bad)}/{len(bad)}",      # 문제 답을 잡아냄
            "false_alarm": f"{sum(not r['verdicts'][n] for r in good)}/{len(good)}",   # 멀쩡한 답을 반려
        }
    summary["avg_seconds"] = {k: round(v / len(rows), 1) for k, v in timing.items()}
    return {"summary": summary, "rows": rows}


def judge_results(judge: Judge, label: str) -> dict:
    data = json.loads((HERE / "results" / f"{label}.json").read_text(encoding="utf-8"))
    rows = []
    for r in data["rows"]:
        facts = (r.get("metadata") or {}).get("facts")
        if not facts or "missing" not in r:   # 장애 답변이고 서버가 사실을 돌려준 경우만
            continue
        c = judge.claims(facts, r["reply"])
        rows.append({"id": r["id"], "pass": c["pass"], "unsupported": c["detail"],
                     "judge_misquoted": c["misquoted"]})
        print(f"#{r['id']} {'통과' if c['pass'] else '근거 없음 ' + str(len(c['detail'])) + '건'}", flush=True)
    return {"label": label, "judged": len(rows), "passed": sum(r["pass"] for r in rows),
            "judge_misquotes": sum(len(r["judge_misquoted"]) for r in rows), "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3:8b")
    ap.add_argument("--results", help="bench 결과 label — 주면 그 답변들을 claims 방식으로 채점")
    args = ap.parse_args()
    judge = Judge(args.model)
    if args.results:
        out = judge_results(judge, args.results)
        path = HERE / "results" / f"judge_{args.results}.json"
    else:
        out = validate_judges(judge)
        print("\n| 채점기 | 정확도 | 문제 답 잡아냄 | 멀쩡한 답 오판 |")
        print("|---|---|---|---|")
        for n in ["rule", "score", "claims", "rule+claims"]:
            s = out["summary"][n]
            print(f"| {n} | {s['accuracy']} | {s['caught_bad']} | {s['false_alarm']} |")
        print("평균 채점 시간(초):", out["summary"]["avg_seconds"])
        path = HERE / "results" / f"judge_validation_{args.model.replace(':', '_')}.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print("saved:", path)


if __name__ == "__main__":
    main()
