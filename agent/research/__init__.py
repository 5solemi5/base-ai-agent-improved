"""[개선 10] Research 멀티 에이전트 '구조' — 단일 research_node를 서브그래프(compiled subgraph) 3개로 나눴다.
서브그래프 안에는 LLM 호출이 없다(도구 조회 + 상태 판정·개수 세기·TF-IDF 검색). 부를 순서도 코드가 정한다.

    metrics_agent.py : 메트릭 분석 — 모니터링 지표 조회(병렬·재시도) → 저하 시스템 판정
    log_agent.py     : 로그 분석   — 저하된 시스템의 WARN/ERROR 로그 조회 → 오류 패턴 요약
    runbook_agent.py : 런북 분석   — 질문 + 지표 알림 + 로그 오류로 RAG 검색 → runbook 조회
    supervisor.py    : 메인 그래프의 research_supervisor 노드 — 지금까지 모은 결과를 보고
                       다음에 부를 서브에이전트를 '코드로' 고른다 (모두 끝나면 analysis로)

각 서브에이전트는 자기 상태(TypedDict)와 노드를 가진 독립된 StateGraph를 compile()한 것이다.
메인 그래프는 서브에이전트 내부를 모르고, 입력(대상 시스템·질문)을 넘기고 결과(collected·events)만 받는다.
"""
