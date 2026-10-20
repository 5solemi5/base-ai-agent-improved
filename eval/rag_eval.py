"""[개선 3 측정] incident_type 선택 정확도: 원본 키워드 규칙 vs RAG 검색기.

LLM·서버 없이 바로 돌아간다(결정론). 사용법:
    python eval/rag_eval.py                 # tfidf
    RUNBOOK_RETRIEVER=embed python eval/rag_eval.py   # nomic-embed-text를 받아 둔 경우

정답 라벨은 사람이 붙였다(이 파일을 쓴 사람 = 메타데이터를 쓴 사람이라 낙관적일 수 있음 →
발표 때 '한계'로 밝힐 것). 각 케이스는 (사용자 질문, 조사된 시스템, 허용 정답 집합)이다.
조사된 시스템의 알림은 data/monitoring.json에서 그대로 가져온다.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.runbook_retriever import RunbookRetriever   # noqa: E402

MON = json.loads((ROOT / "data" / "monitoring.json").read_text(encoding="utf-8"))

CASES = [
    # (질문, 조사된 시스템, 정답으로 인정하는 incident_type들)
    ("payments-service 결제 응답이 느려지고 CPU가 치솟았어요", ["payments-service"], {"cpu-spike"}),
    ("gateway-service에서 payments-service로 가는 요청이 타임아웃 나고 있어요", ["gateway-service", "payments-service"], {"network-timeout"}),
    ("web-01 CPU 급증, auth-service 커넥션 풀도 같이 확인해줘", ["web-01", "auth-service"], {"cpu-spike", "high-latency"}),
    ("payments-service랑 cache-service 둘 다 저하됐는데 공통 원인이 있는지", ["payments-service", "cache-service"], {"cpu-spike", "memory-leak"}),
    ("로그인 응답이 느려요, 원인 확인해주세요", ["auth-service"], {"high-latency"}),
    ("web-01 서버가 느려요", ["web-01"], {"high-latency", "cpu-spike"}),
    ("web-01 CPU가 갑자기 올라갔어요", ["web-01"], {"cpu-spike"}),
    ("cache-service 메모리 사용률이 계속 올라가고 있어요", ["cache-service"], {"memory-leak"}),
    ("cache-service가 자꾸 죽었다 살아나요", ["cache-service"], {"memory-leak"}),
    ("log-service 디스크 공간이 부족하다는 알림이 왔어요", ["log-service"], {"disk-full"}),
    ("log-service 상태 좀 봐주세요", ["log-service"], {"disk-full"}),
    ("gateway-service 상태 확인해줘", ["gateway-service"], {"network-timeout"}),
    ("gateway-service에서 504가 계속 떠요", ["gateway-service"], {"network-timeout"}),
    ("auth-service 상태 확인해줘", ["auth-service"], {"high-latency"}),
    ("auth-service 토큰 발급이 오래 걸려요", ["auth-service"], {"high-latency"}),
    ("payments-service 인스턴스 하나가 응답을 늦게 해요", ["payments-service"], {"cpu-spike", "high-latency"}),
    ("cache-service 힙이 거의 다 찼대요", ["cache-service"], {"memory-leak"}),
    ("web-01에서 다른 서비스 호출할 때 연결 시간 초과가 나요", ["web-01"], {"network-timeout"}),
    ("web-01 디스크가 꽉 찼어요", ["web-01"], {"disk-full"}),
    ("web-01에서 DB 연결이 안 돼요", ["web-01"], {"db-down"}),
    ("log-service 로그가 너무 많이 쌓여요", ["log-service"], {"disk-full"}),
    ("payments-service 결제 API가 타임아웃 나요", ["payments-service"], {"network-timeout", "cpu-spike"}),
]

# 원본 graph.py의 규칙을 그대로 옮긴 것 (Before)
_RULES = [("latency", "high-latency"), ("pool", "high-latency"), ("cpu", "cpu-spike"),
          ("memory", "memory-leak"), ("heap", "memory-leak"), ("timeout", "network-timeout"),
          ("network", "network-timeout"), ("disk", "disk-full"), ("down", "db-down")]


def keyword_rule(collected: list[dict]) -> str:
    text = " ".join(f"{c['data'].get('status', '')} " + " ".join(c["data"].get("alerts", []))
                    for c in collected).lower()
    for kw, itype in _RULES:
        if kw in text:
            return itype
    return "db-down"


def main():
    retriever = RunbookRetriever()
    print(f"retriever method = {retriever.method}"
          + (f" (fallback: {retriever.fallback_reason})" if hasattr(retriever, "fallback_reason") else ""))
    hits = {"keyword": 0, "rag": 0}
    print("| # | 질문 | 정답 | 키워드 규칙 | RAG |")
    print("|---|---|---|---|---|")
    for i, (q, systems, gold) in enumerate(CASES, 1):
        collected = [{"source": "monitoring", "data": {"system_name": s, **MON[s]}} for s in systems]
        kw = keyword_rule(collected)
        rag = retriever.classify(q, collected)["incident_type"]
        hits["keyword"] += kw in gold
        hits["rag"] += rag in gold
        mark = lambda x: f"{x} {'✅' if x in gold else '❌'}"
        print(f"| {i} | {q[:30]} | {'/'.join(sorted(gold))} | {mark(kw)} | {mark(rag)} |")
    n = len(CASES)
    print(f"\n키워드 규칙 정확도: {hits['keyword']}/{n} = {hits['keyword'] / n:.0%}")
    print(f"RAG({retriever.method}) 정확도: {hits['rag']}/{n} = {hits['rag'] / n:.0%}")


if __name__ == "__main__":
    main()
