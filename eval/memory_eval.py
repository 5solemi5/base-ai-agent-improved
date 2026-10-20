"""[개선 12] 대화 요약 메모리 Before/After — 긴 대화 끝에 처음 말한 사실을 기억하는가, 프롬프트는 얼마나 큰가.

같은 대화(이름·소속을 밝힌 뒤 장애 문의 12턴)를 세 가지 방식으로 모델에 넘기고, 마지막 질문
2개("제 이름이 뭐였죠?", "제가 어느 팀이라고 했죠?")에 대한 답과 프롬프트 토큰 수를 잰다.
  window : 최근 10개 메시지만 (개선 12 이전 동작 = Before)
  full   : 전체 이력을 전부 (창 제한 없음 — 기억은 하지만 프롬프트가 계속 커진다)
  summary: 요약 메모리 (After)

사용법:
    python eval/memory_eval.py                 # fake (동작 확인용 — fake chat은 창 크기와 무관하게 이력 전체를 본다)
    python eval/memory_eval.py --backend ollama  # qwen3:0.6b 실측 (ollama serve 필요)
"""
import argparse
import asyncio
import json
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import agent.memory as memory   # noqa: E402
from agent.memory import load_history   # noqa: E402
from agent.session_store import SqliteSessionStore   # noqa: E402

OPENING = [("user", "제 이름은 민수입니다. 결제팀에서 일해요."),
           ("assistant", "안녕하세요, 민수님! 결제팀이시군요. 무엇을 도와드릴까요?")]
FILLER = [
    ("payments-service CPU가 높아요", "payments-service CPU 91%로 저하 상태입니다. cpu-spike 절차를 확인하세요."),
    ("web-01은 괜찮나요?", "web-01은 healthy입니다."),
    ("gateway 타임아웃도 있나요?", "gateway-service에서 payments-service로 504가 반복됩니다."),
    ("캐시는요?", "cache-service 메모리가 89%로 계속 오르고 있습니다."),
    ("로그 서비스 디스크는?", "log-service 디스크가 95%입니다."),
    ("auth 커넥션 풀은?", "auth-service 커넥션 풀 92% 사용 중입니다."),
    ("롤백하면 될까요?", "배포 직후 시작됐다면 롤백을 검토하되 승인이 필요합니다."),
    ("스케일 아웃은요?", "CPU 90% 이상 지속 시 한 대 증설을 권고합니다."),
    ("알림 임계값은 몇이에요?", "CPU 임계값은 80%입니다."),
    ("p95 지연 기준은?", "p95 임계치는 1.0s입니다."),
    ("오늘 배포 있었나요?", "payments-service v2.14.0이 09:58에 배포됐습니다."),
    ("고마워요", "천만에요."),
]
QUESTIONS = [("제 이름이 뭐였죠?", "민수"), ("제가 어느 팀이라고 했죠?", "결제")]


def build_llm(backend: str):
    if backend == "ollama":
        from agent.ollama_llm import OllamaFastLLM
        return OllamaFastLLM("qwen3:0.6b", usd_per_1k_tokens=0.00015)
    from agent.fake_llm import FastLLM
    return FastLLM("fast-stub", 0.00015)


async def run_condition(llm, mode: str, repeat: int = 1) -> dict:
    rows = []
    for question, expect in QUESTIONS:
        convo = OPENING + [m for pair in FILLER * repeat for m in (("user", pair[0]), ("assistant", pair[1]))]
        history = [{"role": r, "content": c} for r, c in convo]
        summarize_s = 0.0
        if mode == "summary":
            with tempfile.TemporaryDirectory() as d:
                store = SqliteSessionStore(Path(d) / "m.db")
                for m in history:
                    store.append("t", m["role"], m["content"])
                t0 = time.perf_counter()
                history, info = await load_history(store, llm, "t")
                summarize_s = round(time.perf_counter() - t0, 1)
                store.close()
        # full 조건만 창 제한을 사실상 없앤다(MEMORY_CHAT_WINDOW 환경변수와 같은 설정값)
        memory.CHAT_WINDOW = 10_000 if mode == "full" else 10
        t0 = time.perf_counter()
        reply, usage = await llm.chat(question, history)
        rows.append({"question": question, "reply": reply.strip()[:120], "ok": expect in reply,
                     # ollama는 실제 프롬프트 토큰(prompt_eval_count), fake는 문자 수 기반 추정치
                     "prompt_tokens": usage.get("prompt_tokens") or usage.get("tokens", 0),
                     "chat_s": round(time.perf_counter() - t0, 1), "summarize_s": summarize_s,
                     # 실제로 모델에 들어간 이력 메시지 수(system 지시문·이번 질문 제외)
                     "messages_sent": len(memory.chat_messages("", history, question)) - 2})
    return {"mode": mode, "turns": 1 + len(FILLER) * repeat, "recall": f"{sum(r['ok'] for r in rows)}/{len(rows)}",
            "avg_prompt_tokens": round(sum(r["prompt_tokens"] for r in rows) / len(rows)),
            "rows": rows}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="fake", choices=["fake", "ollama"])
    ap.add_argument("--repeat", type=int, default=1, help="장애 문의 12턴을 몇 번 반복할지 (대화 길이)")
    args = ap.parse_args()
    llm = build_llm(args.backend)
    results = [await run_condition(llm, m, args.repeat) for m in ("window", "full", "summary")]
    print(f"대화 길이: {results[0]['turns']}턴")
    print("| 방식 | 기억(이름·팀) | 평균 프롬프트 토큰 | 모델에 보낸 메시지 수 | 요약 생성 시간 |")
    print("|---|---|---|---|---|")
    for r in results:
        print(f"| {r['mode']} | {r['recall']} | {r['avg_prompt_tokens']} | "
              f"{r['rows'][0]['messages_sent']} | {r['rows'][0]['summarize_s']}s |")
    for r in results:
        for row in r["rows"]:
            print(f"  [{r['mode']}] {row['question']} → {row['reply']!r}")
    suffix = f"_x{args.repeat}" if args.repeat > 1 else ""
    out = HERE / "results" / f"memory_{args.backend}{suffix}.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=1))
    print("saved:", out)


if __name__ == "__main__":
    asyncio.run(main())
