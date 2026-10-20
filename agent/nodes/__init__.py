"""[개선 8] 그래프 노드 모듈 패키지 — 원래 agent/graph.py 한 파일(374줄)에 있던 노드들을 역할별로 나눴다.

    state.py    : 그래프 상태(AgentState)와 노드들이 같이 쓰는 작은 도우미
    routing.py  : route(잡담/장애 분류) · chat_answer(잡담 답변)
    planning.py : plan(조사 계획) · unknown_system(미등록 시스템 안내) · gate(권위 게이트)
    research.py : research(MCP 조회) · replan_check(runbook RAG 선택)
    analysis.py : analysis(여러 시스템 비교)
    writing.py  : write(답변 작성) · validate(형식·근거 검사) · finalize(마무리)

agent/graph.py에는 '노드를 어떤 순서로 잇는가(엣지)'만 남는다. 노드 하나를 고칠 때 다른 노드
코드를 스크롤할 필요가 없고, 노드별로 따로 import해 단위 테스트할 수 있다.
"""
