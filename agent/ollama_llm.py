"""LLM_BACKEND=ollama — 로컬 Ollama 실모델 백엔드 (fake_llm과 동일 인터페이스).

Wave 10a 구도를 실물로: fast(기본 qwen3:0.6b)는 분류·계획·단순 대화, quality(기본
qwen3:8b)는 최종 답변 작성 전담. Ollama의 구조화 출력(format=<JSON Schema>)과
temperature=0으로 변동을 줄이지만 실모델이므로 결정론은 아니다 — 그래서 gate(W6)·
replan(W7)·validator(W8)가 연출이 아니라 실제 방어선으로 동작한다.

비용(cost_usd)은 '가상 단가 × 실측 토큰 수(prompt_eval_count+eval_count)'다 — 로컬
추론이라 실제 과금은 0이지만, W10a의 모델 분리 효과를 같은 잣대로 비교하기 위함이다.

[개선 1] qwen3 계열은 Ollama에서 기본으로 thinking(생각 과정 생성)이 켜진다. "안녕"
한 마디에도 thinking 447자를 만들었고, 장애 쿼리 6건이 356초 걸렸다. 요청마다 "think"
값을 명시해(기본 false, OLLAMA_THINK=1이면 true) 끄고 켤 수 있게 했다. 타임아웃도
환경변수(OLLAMA_TIMEOUT, 기본 120초)로 뺐다.
[개선 2] 시스템 목록·plan 정규화는 agent/systems.py 한 곳에서 가져온다.
[개선 6] 대화 이력을 "user: …" 텍스트 한 덩어리가 아니라 messages 배열의 role로 넘긴다
(0.6b가 "제 이름이 뭐였죠?"에 "저의 이름은 민수입니다"라고 답하던 역할 혼동 완화).

사전 조건: `ollama serve` 실행 중 + `ollama pull qwen3:0.6b qwen3:8b`.
"""
import json
import os

import httpx
from langsmith import traceable

from .memory import SUMMARIZE_SYSTEM, chat_messages, summarize_prompt
from .systems import INCIDENT_PATTERN, KNOWN_SYSTEMS, as_bool, normalize_plan

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434/api/chat")
OLLAMA_THINK = os.environ.get("OLLAMA_THINK", "0") == "1"
OLLAMA_TIMEOUT = float(os.environ.get("OLLAMA_TIMEOUT", "120"))


ROUTE_SCHEMA = {
    "type": "object",
    "properties": {
        "route": {"type": "string", "enum": ["chat", "incident"]},
        "needs_analysis": {"type": "boolean"},
    },
    "required": ["route", "needs_analysis"],
}

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "needs_analysis": {"type": "boolean"},
        "tasks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "agent": {"type": "string",
                              "enum": ["research", "analysis"]},
                    "target": {"type": "string"},
                },
                "required": ["agent"],
            },
        },
    },
    "required": ["needs_analysis", "tasks"],
}


class OllamaLLM:
    def __init__(self, model: str, usd_per_1k_tokens: float, think: bool = OLLAMA_THINK):
        self.name = f"ollama:{model}"
        self.model = model
        self.usd_per_1k = usd_per_1k_tokens
        self.think = think

    def _payload(self, messages: list[dict], stream: bool) -> dict:
        return {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "think": self.think,   # [개선 1] 명시하지 않으면 qwen3는 thinking을 켠다
            "options": {"temperature": 0.0},
        }

    def _usage(self, data: dict) -> dict:
        tokens = (data.get("prompt_eval_count") or 0) + (data.get("eval_count") or 0)
        return {"model": self.name, "tokens": tokens,
                "prompt_tokens": data.get("prompt_eval_count") or 0,   # [개선 12] 메모리 측정용
                "usd": round(tokens / 1000 * self.usd_per_1k, 6)}

    def _chat(self, system: str, user: str, schema: dict | None = None) -> tuple[str, dict]:
        """동기 호출. graph.py가 asyncio.to_thread()로 감싸 부르므로 이벤트 루프를 막지 않는다."""
        payload = self._payload([{"role": "system", "content": system},
                                 {"role": "user", "content": user}], stream=False)
        if schema is not None:
            payload["format"] = schema
        resp = httpx.post(OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        return data["message"]["content"], self._usage(data)

    async def _chat_stream(self, messages: list[dict], on_token=None) -> tuple[str, dict]:
        """사용자 대면 텍스트(최종 답변류) 전용 — Ollama NDJSON 스트림을 토큰 단위로
        받아 on_token(조각)을 즉시 호출해 SSE로 실시간 중계할 수 있게 한다. 분류/계획
        등 JSON 스키마 출력(route/plan/analyze)은 부분 출력이 의미가 없어 그대로 _chat을
        쓴다."""
        chunks: list[str] = []
        usage = {"model": self.name, "tokens": 0, "usd": 0.0}
        async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
            async with client.stream("POST", OLLAMA_URL,
                                     json=self._payload(messages, stream=True)) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    data = json.loads(line)
                    piece = data.get("message", {}).get("content", "")
                    if piece:
                        chunks.append(piece)
                        if on_token is not None:
                            on_token(piece)
                    if data.get("done"):
                        usage = self._usage(data)
        usage["prompt"] = "\n\n".join(f"[{m['role'].upper()}]\n{m['content']}" for m in messages)
        return "".join(chunks), usage


class OllamaFastLLM(OllamaLLM):
    """분류·계획·단순 대화 전담 (저사양 소형 모델)."""

    @traceable(run_type="llm", name="ollama_fast.route")
    def route(self, query: str) -> tuple[dict, dict]:
        system = ("당신은 ICT 운영 지원 라우터다. 사용자 요청을 분류하라. "
                  "장애·성능 저하·지연·진단·모니터링·티켓 등 운영 이슈면 route=incident, "
                  "인사·자기소개·잡담·일반 질문이면 route=chat. "
                  "needs_analysis는 패턴 비교/통계 분석이 꼭 필요할 때만 true.")
        content, usage = self._chat(system, query, schema=ROUTE_SCHEMA)
        try:
            raw = json.loads(content)
            route = raw.get("route") if raw.get("route") in ("chat", "incident") else None
            if route is None:   # 스키마 밖의 값("INCIDENT" 등)도 chat으로 떨어뜨리지 않고 규칙으로 판정
                route = "incident" if INCIDENT_PATTERN.search(query) else "chat"
            result = {"route": route, "needs_analysis": as_bool(raw.get("needs_analysis", False))}
        except (json.JSONDecodeError, AttributeError):
            # 소형 모델의 형식 붕괴 대비 — 코드 레벨 fallback (W6 정신)
            result = {"route": "incident" if INCIDENT_PATTERN.search(query) else "chat",
                      "needs_analysis": False}
        return result, usage

    @traceable(run_type="llm", name="ollama_fast.plan")
    def plan(self, query: str, procedure_prompt: str) -> tuple[dict, dict]:
        system = ("당신은 ICT 운영 Supervisor의 planning 노드다. 표준 진단 절차에 따라 "
                  "실행 계획(tasks)을 JSON으로 세워라. "
                  "사용자 요청에 등장한 시스템마다 agent=research task를 하나씩 세우고, "
                  "target에는 시스템 이름만 정확히 적어라 "
                  f"(알려진 시스템: {', '.join(sorted(KNOWN_SYSTEMS))}). "
                  "여러 시스템을 비교해야 할 때만 agent=analysis task를 추가하라.")
        user = f"## 표준 절차\n{procedure_prompt}\n\n## 사용자 요청\n{query}"
        content, usage = self._chat(system, user, schema=PLAN_SCHEMA)
        try:
            raw = json.loads(content)
        except json.JSONDecodeError:
            raw = {}
        # [개선 2] 대상 시스템은 쿼리가 결정하고, 모델 출력은 그 범위 안에서만 받아들인다
        return normalize_plan(raw, query), usage

    @traceable(run_type="llm", name="ollama_fast.analyze")
    def analyze(self, collected: list[dict]) -> tuple[str, dict]:
        system = ("당신은 ICT 운영 패턴 분석가다. 여러 시스템의 모니터링 결과를 비교해 "
                  "공통 원인이 의심되는 패턴이 있는지 한두 문장으로 짧게 답하라. "
                  "패턴이 없으면 '특이 패턴 없음'이라고만 답하라.")
        mon = [c["data"] for c in collected if c["source"] == "monitoring"]
        facts = "\n".join(
            f"- {d['system_name']}: 상태={d['status']}, CPU {d['cpu_percent']}%, "
            f"메모리 {d['memory_percent']}%, 알림 {d['alerts']}" for d in mon) or "(모니터링 데이터 없음)"
        return self._chat(system, facts)

    @traceable(run_type="llm", name="ollama_fast.chat")
    async def chat(self, query: str, history: list[dict], on_token=None) -> tuple[str, dict]:
        system = ("당신은 운영 인시던트 대응 에이전트의 간결한 한국어 비서다. 한두 문장으로 답하라. "
                  "대화 이력에서 사용자가 밝힌 이름이 있으면 정확히 기억해 답하라. "
                  "사용자의 이름과 당신 자신을 혼동하지 마라. "
                  "'[사용자 정보]'에 적힌 이름·소속은 사용자가 직접 밝힌 확정 사실이다.")
        # [개선 6] 이력을 role이 구분된 messages로 — 최근 10개만(컨텍스트 크기 제한)
        # [개선 12] 대화 요약(system 역할)이 있으면 함께 넣는다 — 10개 창 밖의 사실도 기억
        return await self._chat_stream(chat_messages(system, history, query), on_token=on_token)

    @traceable(run_type="llm", name="ollama_fast.summarize")
    def summarize(self, prev_summary: str, messages: list[dict]) -> tuple[str, dict]:
        """[개선 12] 대화 요약 메모리 — 동기 호출(agent/memory.py가 to_thread로 부른다)."""
        content, usage = self._chat(SUMMARIZE_SYSTEM, summarize_prompt(prev_summary, messages))
        return content.strip(), usage


class OllamaQualityLLM(OllamaLLM):
    """최종 답변 작성 전담 (상대적으로 큰 모델)."""

    @traceable(run_type="llm", name="ollama_quality.write")
    async def write(self, query: str, collected: list[dict], attempt: int,
                    feedback: str | None, on_token=None) -> tuple[str, dict]:
        system = ("당신은 ICT 운영 보고 작성자다. 수집된 사실만 사용해 한국어로 "
                  "간결한 대응 보고를 작성하라. 반드시 '1) ... 2) ... 3) ...' 형식의 "
                  "번호 절차로 답하고, 수치는 수집된 값을 그대로 인용하라. "
                  "수집된 사실에 없는 시스템·로그·수치는 언급하지 말고, 확인이 필요하면 "
                  "'추가 확인 필요'라고만 써라.")
        return await self._chat_stream(
            [{"role": "system", "content": system},
             {"role": "user", "content": build_write_prompt(query, collected, feedback)}],
            on_token=on_token)


def build_write_prompt(query: str, collected: list[dict], feedback: str | None) -> str:
    """write 프롬프트의 '사실' 부분 — openai_llm.py도 같이 쓴다(원본은 두 파일에 복사돼 있었다)."""
    facts = []
    for c in collected:
        if c["source"] == "monitoring":
            d = c["data"]
            facts.append(f"- 모니터링 {d['system_name']}: 상태={d['status']}, "
                         f"CPU {d['cpu_percent']}%, 메모리 {d['memory_percent']}%, "
                         f"알림 {len(d['alerts'])}건 {d['alerts']}")
        elif c["source"] == "log":   # [개선 10] 로그 분석 서브에이전트의 요약
            d = c["data"]
            top = "; ".join(f"{t['level']} {t['message']} ×{t['count']}" for t in d["top"])
            facts.append(f"- 로그 {d['system_name']}: ERROR {d['error_count']}건, "
                         f"WARN {d['warn_count']}건 — {top or '특이 로그 없음'}")
        elif c["source"] == "runbook":
            facts.append(f"- runbook 발췌:\n{c['data'].strip()}")
        elif c["source"] == "analysis":
            facts.append(f"- 패턴 분석: {c['data'].strip()}")
    user = f"## 사용자 요청\n{query}\n\n## 수집된 사실\n" + "\n".join(facts)
    if feedback:
        user += f"\n\n## 이전 초안에 대한 검증 반려 사유 (반드시 반영)\n{feedback}"
    return user
