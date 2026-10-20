# Mini-Project 과제 1번: Base AI Agent 개선 결과

원본 `all_total_vue_student`(운영 인시던트 대응 에이전트)를 실행·측정해 찾은 문제를 고치고,
같은 입력 9개로 개선 전/후를 실측했다. 측정 환경: macOS(Apple Silicon, RAM 16GB), Ollama 0.33,
fast=`qwen3:0.6b`, quality=`qwen3:8b`, 각 조건 1회 측정(LLM이라 실행마다 수치는 조금씩 달라짐).

## 1. 문제 → 개선

| # | 영역 | 문제 (원본 실측) | 개선 | 파일 |
|---|---|---|---|---|
| 1 | 성능 | qwen3 thinking이 기본으로 켜져 장애 6건 269~356초. 동기 LLM 호출이 이벤트 루프를 막아 /health가 최대 1.6초 정지 | 요청마다 `think` 명시(기본 끔, `OLLAMA_THINK=1`로 켬), 동기 호출을 `asyncio.to_thread`로 | `agent/ollama_llm.py`, `agent/graph.py` |
| 1+ | 라우팅 | thinking을 끄면 0.6b가 "제 이름은 민수입니다"를 장애로 분류 | 라우팅 권위 게이트: 장애 신호가 하나도 없으면 chat, 증상+대상이 있으면 incident | `agent/graph.py` `route_gate` |
| 2 | 정확도 | 시스템 2개를 말해도 1개만 조사(누락 3건). billing-service를 auth-service로 바꿔치기 | 조사 대상은 쿼리가 결정(`agent/systems.py`), 모니터링 대상이 아니면 조사 없이 정직하게 안내 | `agent/systems.py`, `graph.py` `unknown_system` |
| 3 | RAG | incident_type 키워드 규칙: healthy 시스템이면 db-down 폴백, 질문 무시 | runbook 조각 + incident_type 메타데이터 벡터 인덱스(TF-IDF, 임베딩 선택) | `agent/runbook_retriever.py`, `data/runbook_meta.json` |
| 4 | 검증 | validator가 "30자 + 1)"만 확인 → 지어낸 로그 분석, 조사 안 한 시스템 단정이 통과 | 근거 검사(조사 안 한 시스템·없는 수치·로그 단정) 후 재작성 요청, 남으면 경고 첨부 | `agent/grounding.py`, `graph.py` `validate_node` |
| 5 | 운영 | 실행 스크립트가 8100 포트 주인을 무조건 `kill -9`(Docker 백엔드), 기동마다 session.db 삭제, 포트 하드코딩, 화면 검수 표시 버그, 새로고침 시 대화 잊음 | 이 폴더의 서버일 때만 종료, 세션 보존, `BACKEND_PORT`/`FRONT_PORT`, 화면 버그 수정 | `sh/*.sh`, `server.py`, `web_demo_server.py`, `web_chat.html` |
| 6 | 테스트 | 테스트 없음 | pytest 45개(LLM·MCP 없이 실행) | `tests/` |

## 2. Before / After (같은 9개 질문, `eval/bench.py`)

| 지표 | Before (원본) | After (개선) |
|---|---|---|
| 라우팅 정확도 | 9/9 | 9/9 |
| 장애 질문 6개 총 소요 | 269.4초 | 51.7초 (약 5.2배 ↓) |
| 조사 누락 시스템 | 3 | 0 |
| 불필요하게 조사한 시스템 | 1 | 0 |
| runbook 정확도 | 4/6 | 6/6 |
| 미등록 시스템 정직하게 안내 | 0/1 | 1/1 |
| 근거 없는 내용이 섞인 답변 | 3 | 0 |
| 가상 비용 합계(USD) | 0.035891 | 0.016716 (53% ↓) |

| 추가 측정 | Before | After |
|---|---|---|
| RAG: incident_type 정확도(22문항, `eval/rag_eval.py`) | 키워드 규칙 77% | TF-IDF 검색 95% |
| 이벤트 루프 블로킹: 장애 질문 처리 중 /health 최대 지연(`eval/block_probe.py`) | 1657ms (thinking만 꺼도 569ms) | 27ms |
| fake 백엔드 라우팅 정확도 | 4/9 | 9/9 |

## 3. 한계와 트레이드오프

- **RAG 정확도는 낙관적일 수 있다.** 평가 문항과 메타데이터를 같은 사람이 썼다. 틀린 1건("web-01 서버가 느려요" → disk-full)처럼 증상이 모호하면 여전히 틀린다. 임베딩(`nomic-embed-text`)을 쓰면 의미 유사도는 좋아지지만 모델 설치·호출 지연·인덱스 재생성 비용이 생긴다.
- **근거 검사는 규칙 기반이다.** 숫자·시스템 이름·로그 단정만 본다. "공통 원인은 메모리"처럼 추론이 틀린 문장은 잡지 못한다. 반대로 정상 문장을 오탐하면 재작성 시간이 늘어난다(실제로 "로그인 … 결과" 오탐을 발견해 고쳤다). 최대 3회 후에는 통과시키고 경고를 붙인다.
- **조사 대상을 쿼리가 결정하므로**, 사용자가 말하지 않은 연관 시스템(예: 원인이 gateway인데 payments만 말함)은 먼저 찾아보지 않는다. 대신 정확하고 예측 가능하다.
- **thinking을 끄면** 0.6b의 분류·계획 품질이 흔들린다 → 라우팅 게이트와 코드 정규화로 보완했지만, 대화 답변 품질("민수님의 답변입니다.")은 여전히 어색하다. fast 모델을 `qwen3:1.7b`로 올리는 것이 다음 개선 후보.
- 측정은 조건별 1회다. LLM 출력과 모니터링 jitter 때문에 수치는 실행마다 달라진다.

## 4. 재현 방법

```bash
bash sh/setup_venv.sh && .venv/bin/python3 -m pip install -r requirements-dev.txt
.venv/bin/python3 -m pytest -q                       # 45 passed
.venv/bin/python3 eval/rag_eval.py                    # RAG 정확도
BACKEND_PORT=8101 bash sh/run_agent_ollma.sh          # 다른 터미널
.venv/bin/python3 eval/bench.py --url http://127.0.0.1:8101 --label after
.venv/bin/python3 eval/compare.py before_ollama after
```
