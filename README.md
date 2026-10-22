# Base AI Agent 개선 — 운영 인시던트 대응 에이전트

AI 서비스 Mini-Project **과제 1번**. 제공된 Base AI Agent(`all_total_vue`)를 실행·측정해 찾은 문제를 고치고 기능을 더했다.
같은 질문 9개로 개선 전/후를 실측했다.

- **결과 보고서 (문제 → 개선 → Before/After → 한계):** [`doc/IMPROVEMENTS.md`](doc/IMPROVEMENTS.md)
- **발표 자료 (18쪽):** [`doc/presentation.md`](doc/presentation.md)(웹에서 바로 보기) · [`doc/presentation.pdf`](doc/presentation.pdf)(PDF) — 1\~10쪽 기능·결과, 11쪽 그래프 구성, 12\~14쪽 한계·느낀점, 15\~18쪽 부록(코드)
- **원래 사용 설명서:** [`doc/README.md`](doc/README.md)
- **로컬로 만들고 OpenAI로 다듬기 + LangSmith 추적:** 보고서 [2-1절](doc/IMPROVEMENTS.md#2-1-로컬로-만들고-openai로-다듬기-개선-15--langsmith-추적) — 같은 9개 질문에서 정확도는 그대로, 장애 질문 소요 50.0초 → 26.1초

## 한눈에 보기

| 지표 | Before | After |
|---|---|---|
| 장애 질문 6개 총 소요 | 269.4초 | 51.4초 |
| 조사 누락 / 불필요한 조사 | 3 / 1 | 0 / 0 |
| runbook 정확도 | 4/6 | 6/6 |
| 질문에 나온 시스템을 조사하지 않고 언급한 답변 (조사 누락 3건과 같은 사례) | 3 | 0 |
| RAG incident_type 정확도 (22문항) | 77% | 95% |
| 37턴 합성 대화 후 이름·소속 기억 / 프롬프트 토큰 | 0/2 (창 10개 조건 재현) | 2/2, 328토큰 (정규식으로 고정한 항목) |
| pytest | 0 | 80건 (테스트 함수 59개, 단위·통합) |

- 각 조건 1회 측정. After 열의 소요·정확도는 `eval/results/after_ollama_v2.json`(개선 15의 프롬프트 변경 전)이다. 최종 코드로 다시 잰 `after_ollama_polished.json`도 정확도 지표는 같고 장애 질문 총 소요는 50.0초다.
- '조사하지 않고 언급한 답변'은 `eval/compare.py`의 '조사 안 한 시스템·로그를 언급한 답변 수' 지표다(예전 이름표는 '근거 없는 내용이 섞인 답변 수'였는데 실제로 세는 것보다 넓게 읽혀 바꿨다). Before의 3건은 모두 사용자가 말한 시스템을 조사하지 않고 언급한 경우로, 조사 누락 3건과 같은 사례다(지어낸 내용이 아님).
- 메모리 행: Before는 원본 코드가 아니라 창 10개 조건을 개선 코드로 재현한 것이다. 이름·소속은 정규식으로 고정한 항목이라 2/2는 정규식 효과이고, LLM 요약 품질은 측정하지 않았다.

## 구조

```
route ─┬─ chat_answer                         (잡담·대화 기억: 요약 메모리)
       └─ plan ─┬─ clarify                    (대상 불명확 → 되묻기)
                ├─ unknown_system             (모니터링 대상 아님 → 정직한 안내)
                └─ gate → research_supervisor ⇄ metrics_agent / log_agent / runbook_agent
                          → analysis → write ⇄ validate(근거 검사) → approval(위험 조치 사람 승인) → finalize
```

| 개선 | 파일 |
|---|---|
| 1 thinking 제어·비동기 호출 / 2 조사 대상 결정 / 3 runbook RAG / 4 근거 검사 | `agent/ollama_llm.py`, `agent/systems.py`, `agent/runbook_retriever.py`, `agent/grounding.py` |
| 5 실행 스크립트·서버·화면 안정화 / 6 테스트 / 7 리뷰 버그 수정 | `sh/`, `server.py`, `web_chat.html`, `tests/` |
| 8 노드 모듈화 / 9 되묻기 | `agent/nodes/` |
| 10 Research를 LLM 없이 도구를 조회하는 서브그래프 3개 + 순서를 코드가 정하는 감독 노드로 분리(멀티 에이전트 '구조') + 로그 MCP Tool | `agent/research/`, `mcp_server.py` |
| 11 Human-in-the-loop 승인 | `agent/approval.py`, `agent/nodes/approval.py` |
| 12 대화 요약 메모리 | `agent/memory.py` |
| 13 LLM-as-judge 평가 | `eval/judge.py` |
| 리뷰 반영: 체크포인트 정리, 승인 대기 유지, 소속 오탐, 채점 필터, 감독자 상한 | `server.py`, `agent/memory.py`, `eval/judge.py`, `agent/research/supervisor.py` |

## 실행

```bash
bash sh/setup_venv.sh                         # 최초 1회
bash sh/setup_ollama.sh                       # qwen3:0.6b / qwen3:8b
BACKEND_PORT=8100 bash sh/run_agent_ollma.sh  # 백엔드 (다른 프로그램이 8100을 쓰면 포트 변경)
bash sh/run_frontend.sh                       # http://localhost:8600 채팅 화면
```

- **위험 조치 승인(HITL):** 기본으로 켜져 있다. 답변에 재시작·롤백·스케일 조정 등이 있으면 화면에 승인/거절 카드가 뜬다.
  - API로 승인: `POST /chat/approve {"thread_id": "...", "approved": true}`
  - 끄기: `HITL_APPROVAL=0`

## 테스트와 측정

```bash
.venv/bin/python3 -m pip install -r requirements-dev.txt
.venv/bin/python3 -m pytest -q                                          # LLM·MCP 없이 실행
.venv/bin/python3 eval/bench.py --url http://127.0.0.1:8100 --label after   # 같은 9개 질문 측정
.venv/bin/python3 eval/compare.py before_ollama after
.venv/bin/python3 eval/memory_eval.py --backend ollama                 # 요약 메모리
.venv/bin/python3 eval/judge.py                                        # 채점기 검증
```
