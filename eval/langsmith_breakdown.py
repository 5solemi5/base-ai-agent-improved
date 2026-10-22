"""[개선 15] LangSmith에 기록된 실행에서 노드별 소요 시간을 모은다.
사용: (LANGCHAIN_TRACING_V2=true로 서버를 띄워 eval/bench.py를 한 번 돌린 뒤)
    set -a; source .env; set +a
    python3 eval/langsmith_breakdown.py capstone-all-total-ollama capstone-all-total-openai
결과: eval/results/langsmith_breakdown.json + 표 출력. API 키는 .env에서만 읽고 저장하지 않는다.
"""
import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

from langsmith import Client

warnings.filterwarnings("ignore", category=DeprecationWarning)
OUT = Path(__file__).resolve().parent / "results" / "langsmith_breakdown.json"


def breakdown(client: Client, project: str) -> dict:
    """메인 LangGraph 실행 바로 아래(=그래프 노드)의 시간을 노드 이름별로 합친다."""
    runs = list(client.list_runs(project_name=project))
    by_id = {r.id: r for r in runs}
    # 메인 그래프만 — 서브에이전트(compiled subgraph)도 이름이 LangGraph라서, 부모가 chat_request인 것만 고른다.
    # 서브그래프 안쪽 단계(fetch 등)까지 세면 바깥 노드(metrics_agent) 시간과 겹쳐 두 번 세어진다.
    graph_ids = {r.id for r in runs if r.name == "LangGraph"
                 and (r.parent_run_id is None or by_id.get(r.parent_run_id) is not None
                      and by_id[r.parent_run_id].name == "chat_request")}
    total = defaultdict(float)
    count = defaultdict(int)
    for r in runs:
        if r.parent_run_id in graph_ids and r.end_time:
            total[r.name] += (r.end_time - r.start_time).total_seconds()
            count[r.name] += 1
    whole = sum(total.values())
    nodes = sorted(total, key=lambda n: -total[n])
    return {"project": project, "runs_recorded": len(runs), "graph_runs": len(graph_ids),
            "node_total_s": round(whole, 2),
            "nodes": [{"node": n, "seconds": round(total[n], 2), "share": round(total[n] / whole, 3),
                       "calls": count[n]} for n in nodes]}


def main():
    client = Client()
    out = [breakdown(client, p) for p in sys.argv[1:]]
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    for b in out:
        print(f"== {b['project']} (기록된 실행 {b['runs_recorded']}개, 그래프 실행 {b['graph_runs']}회, 노드 합계 {b['node_total_s']}초)")
        for n in b["nodes"][:6]:
            print(f"  {n['node']:20s} {n['seconds']:6.2f}초 {n['share'] * 100:5.1f}%  호출 {n['calls']}회")
    print("saved:", OUT)


if __name__ == "__main__":
    main()
