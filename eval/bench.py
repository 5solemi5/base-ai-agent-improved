"""Before/After 측정 도구 — 같은 쿼리 세트를 /chat에 보내고 지표를 계산해 JSON으로 저장한다.

사용법:
    python eval/bench.py --url http://127.0.0.1:8101 --label before
    python eval/bench.py --url http://127.0.0.1:8101 --label after

지표(쿼리별 + 합계):
- elapsed_s       : 요청 1건의 왕복 시간(초)
- route_ok        : 기대한 경로(chat/incident)로 갔는가
- missing/extra   : 기대 시스템 중 조사 안 한 것 / 기대하지 않았는데 조사한 것
- runbook_ok      : 붙은 runbook이 기대 목록 안에 있는가 (runbook이 없어야 하는 쿼리는 안 붙었는가)
- unknown_ok      : 모니터링 대상이 아닌 시스템을 다른 시스템으로 바꿔치기하지 않았는가
- flags           : 답변에 조사하지 않은 시스템/수집하지 않은 로그 분석 결과가 등장했는가
- cost_usd, retry_count, validation_attempts : 서버가 metadata로 돌려준 값
"""
import argparse
import json
import re
import time
import uuid
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
SYSTEM_RE = re.compile(r"(?<![a-z0-9-])([a-z][a-z0-9]*-(?:service|\d+))(?![a-z0-9])")
LOG_CLAIM_RE = re.compile(r"로그(?!인)[^.\n]{0,25}(?:결과|없음|없었|확인됨|확인되었|발견되지|나타나지)")


def investigated(events: list[dict]) -> list[str]:
    return [e["target"] for e in events
            if e.get("type") == "TaskCompleted" and e.get("source") == "monitoring"]


def runbooks(events: list[dict]) -> list[str]:
    return [e["target"] for e in events
            if e.get("type") == "TaskCompleted" and e.get("source") == "runbook"]


def flags(reply: str, probed: list[str]) -> list[str]:
    out = [f"조사 안 한 시스템 언급: {s}" for s in dict.fromkeys(SYSTEM_RE.findall(reply))
           if s not in probed]
    if LOG_CLAIM_RE.search(reply):
        out.append("수집하지 않은 로그 분석 결과 언급")
    return out


def score(case: dict, res: dict) -> dict:
    events = res.get("events", [])
    meta = res.get("metadata", {})
    reply = res.get("reply", "")
    probed = investigated(events)
    rbs = runbooks(events)
    row = {"id": case["id"], "query": case["query"], "elapsed_s": res["elapsed_s"],
           "route": meta.get("route"), "route_ok": meta.get("route") == case["route"],
           "cost_usd": meta.get("cost_usd", 0.0), "retry_count": meta.get("retry_count", 0),
           "validation_attempts": meta.get("validation_attempts", 0),
           "probed": probed, "runbooks": rbs, "reply": reply,
           "events": events, "metadata": meta}   # 원자료도 저장 → --rescore로 재채점 가능
    if case["route"] == "incident":
        exp = set(case["systems"])
        row["missing"] = sorted(exp - set(probed))
        row["extra"] = sorted(set(probed) - exp)
        row["runbook_ok"] = (set(rbs) <= set(case["runbooks"])) and (bool(rbs) == bool(case["runbooks"]))
        if case.get("unknown"):
            # 미등록 시스템 질문: 아무것도 조사하지 않고, 그 이름을 답변에서 짚어 줘야 정답
            # (안내문에 '조회 가능한 시스템 목록'이 나오므로 flags 검사는 하지 않는다)
            row["unknown_ok"] = not probed and all(u in reply for u in case["unknown"])
            row["flags"] = []
        else:
            row["flags"] = flags(reply, probed)
    if case.get("expect_in_reply"):
        row["reply_ok"] = case["expect_in_reply"] in reply
    return row


def _events_from_row(r: dict) -> list[dict]:
    """원자료(events)를 저장하지 않던 초기 버전 결과 파일을 재채점할 때 쓰는 복원."""
    return ([{"type": "TaskCompleted", "source": "monitoring", "target": t} for t in r["probed"]]
            + [{"type": "TaskCompleted", "source": "runbook", "target": t} for t in r["runbooks"]])


def _meta_from_row(r: dict) -> dict:
    return {k: r.get(k) for k in ("route", "cost_usd", "retry_count", "validation_attempts")}


def summarize(rows: list[dict]) -> dict:
    inc = [r for r in rows if "missing" in r]
    return {
        "queries": len(rows),
        "route_accuracy": f"{sum(r['route_ok'] for r in rows)}/{len(rows)}",
        "incident_total_s": round(sum(r["elapsed_s"] for r in inc), 1),
        "all_total_s": round(sum(r["elapsed_s"] for r in rows), 1),
        "missing_systems": sum(len(r["missing"]) for r in inc),
        "extra_systems": sum(len(r["extra"]) for r in inc),
        "runbook_accuracy": f"{sum(r['runbook_ok'] for r in inc)}/{len(inc)}",
        "unknown_handled": f"{sum(r.get('unknown_ok', False) for r in rows)}/{sum('unknown_ok' in r for r in rows)}",
        "flagged_answers": sum(bool(r["flags"]) for r in inc),
        "name_memory": all(r.get("reply_ok", True) for r in rows),
        "cost_usd": round(sum(r["cost_usd"] for r in rows), 6),
        "retry_total": sum(r["retry_count"] for r in rows),
        "validation_attempts_total": sum(r["validation_attempts"] for r in rows),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8100")
    ap.add_argument("--label", required=True)
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--rescore", action="store_true",
                    help="서버에 다시 묻지 않고 저장된 결과를 현재 채점 규칙으로 다시 계산")
    args = ap.parse_args()

    cases = json.loads((HERE / "queries.json").read_text())
    path = HERE / "results" / f"{args.label}.json"
    if args.rescore:   # 채점 규칙을 고친 뒤 Before/After를 같은 기준으로 맞출 때
        old = json.loads(path.read_text())
        rows = [score(c, {"reply": r["reply"], "elapsed_s": r["elapsed_s"],
                          "events": r.get("events") or _events_from_row(r),
                          "metadata": r.get("metadata") or _meta_from_row(r)})
                for c, r in zip(cases, old["rows"])]
        old.update(summary=summarize(rows), rows=rows)
        path.write_text(json.dumps(old, ensure_ascii=False, indent=1))
        print(json.dumps(old["summary"], ensure_ascii=False, indent=1))
        return

    run_id = uuid.uuid4().hex[:6]   # 실행마다 새 thread_id → 이전 측정의 대화 이력이 섞이지 않게
    rows = []
    for case in cases:
        thread = f"bench-{run_id}-{case.get('thread', case['id'])}"
        t0 = time.perf_counter()
        try:
            res = httpx.post(f"{args.url}/chat", json={"message": case["query"], "thread_id": thread},
                             timeout=args.timeout).json()
        except Exception as e:   # 실패도 결과로 남긴다
            res = {"reply": f"[ERROR] {e!r}", "events": [], "metadata": {}}
        res["elapsed_s"] = round(time.perf_counter() - t0, 1)
        row = score(case, res)
        rows.append(row)
        print(f"[{row['elapsed_s']:>6}s] #{case['id']} route={row['route']} probed={row['probed']} "
              f"runbooks={row['runbooks']} flags={row.get('flags', [])}", flush=True)

    out = {"label": args.label, "url": args.url, "summary": summarize(rows), "rows": rows}
    (HERE / "results").mkdir(exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps(out["summary"], ensure_ascii=False, indent=1))
    print("saved:", path)


if __name__ == "__main__":
    main()
