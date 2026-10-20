"""[개선 1 측정] 이벤트 루프 블로킹 — 장애 질문 1건을 처리하는 동안 /health가 얼마나 늦게 답하나.

FastAPI는 이벤트 루프 하나로 모든 요청을 처리한다. 노드 안에서 동기 HTTP(httpx.post)를 부르면
그동안 루프가 멈춰서 다른 사용자의 요청·SSE 전송도 같이 멈춘다. /health는 원래 1ms 안에 끝나는
요청이라, 그 응답 시간이 곧 '루프가 막힌 시간'이다.

사용법: python eval/block_probe.py --url http://127.0.0.1:8101
"""
import argparse
import statistics
import threading
import time

import httpx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8100")
    ap.add_argument("--query", default="로그인 응답이 느려요, 원인 확인해주세요")
    args = ap.parse_args()

    done = threading.Event()

    def heavy():
        httpx.post(f"{args.url}/chat", json={"message": args.query, "thread_id": f"probe-{time.time()}"},
                   timeout=600)
        done.set()

    threading.Thread(target=heavy, daemon=True).start()
    samples = []
    while not done.is_set():
        t0 = time.perf_counter()
        httpx.get(f"{args.url}/health", timeout=600)
        samples.append((time.perf_counter() - t0) * 1000)
        time.sleep(0.2)
    print(f"/health 측정 {len(samples)}회 — 중앙값 {statistics.median(samples):.1f}ms, "
          f"최대 {max(samples):.0f}ms, 100ms 넘은 횟수 {sum(s > 100 for s in samples)}")


if __name__ == "__main__":
    main()
