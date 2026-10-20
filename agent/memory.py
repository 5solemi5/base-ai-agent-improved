"""[개선 12] 대화 요약 메모리 — 긴 대화의 오래된 턴을 LLM으로 요약해 압축 저장한다.

원본(그리고 개선 6까지)은 대화 이력 중 '최근 10개 메시지'만 모델에 넘겼다. 그래서
  - 6턴(12개 메시지) 전에 말한 이름은 잊는다 ("제 이름이 뭐였죠?" → 모름)
  - 창 크기를 늘리면 기억은 하지만 매 요청 프롬프트가 대화 길이만큼 계속 커진다(토큰 비용·지연)

방식: 최근 KEEP_RECENT개 메시지는 원문 그대로 두고, 그보다 오래된 메시지가 SUMMARIZE_EVERY개
쌓일 때마다 '이전 요약 + 새로 밀려난 메시지'를 fast 모델로 다시 요약해 session.db에 저장한다.
모델에 넘기는 이력 = [system: 이전 대화 요약] + 요약되지 않은 최근 메시지들.

요약은 SUMMARIZE_EVERY개마다 한 번만 만든다(매 요청 요약하면 요청마다 LLM 호출이 하나 늘어난다).

요약본 = [사용자 정보](코드가 정규식으로 뽑아 고정) + [대화 요약](LLM).
처음엔 전부 LLM에 맡겼는데, qwen3:0.6b가 "민수님, 결제팀입니다. CPU 91%로…"처럼 이름·소속을 다른
내용과 한 문장에 뭉개서, 요약을 받은 모델이 "어느 팀이라고 했죠?"에 답하지 못했다(실측 1/2).
이름·소속처럼 '틀리면 안 되는 사실'은 코드가 뽑아 별도 줄로 고정한다(W6 정신 — 결정은 코드가).
"""
import asyncio
import os
import re

KEEP_RECENT = int(os.environ.get("MEMORY_KEEP_RECENT", "6"))          # 원문 유지할 최근 메시지 수
SUMMARIZE_EVERY = int(os.environ.get("MEMORY_SUMMARIZE_EVERY", "8"))  # 이만큼 밀려나면 요약 갱신
CHAT_WINDOW = int(os.environ.get("MEMORY_CHAT_WINDOW", "10"))       # 요약과 별개로 넘기는 최근 메시지 상한
SUMMARY_PREFIX = "이전 대화 요약:"
PROFILE_HEADER, NOTES_HEADER = "[사용자 정보]", "[대화 요약]"

# 사용자가 자기에 대해 말한 사실 — (항목 이름, 정규식). 첫 번째 괄호가 값이다.
PROFILE_PATTERNS = [
    ("이름", re.compile(r"제 이름은 ([가-힣A-Za-z]+?)(?:입니다|이에요|예요|이라고|라고)")),
    # [리뷰 반영] 자기 소속을 말하는 표현만 — "인프라팀에서 연락 왔어요"처럼 남의 팀 언급은 제외
    ("소속", re.compile(r"(?:저는 |제가 |저 )?([가-힣A-Za-z]+팀)\s?(?:에서 일|에서 근무|에 있어|에 있습|소속|이에요|입니다)")),
]


def extract_profile(messages: list[dict], profile: dict | None = None) -> dict:
    """user 메시지에서 이름·소속을 뽑는다. 나중에 다시 말하면 새 값으로 덮어쓴다."""
    profile = dict(profile or {})
    for m in messages:
        if m.get("role") != "user":
            continue
        for key, pattern in PROFILE_PATTERNS:
            if (found := pattern.search(m["content"])):
                profile[key] = found.group(1)
    return profile


MAX_NOTE_LINES = 5


def compose_summary(profile: dict, notes: str) -> str:
    """요약 줄 수 상한은 코드가 지킨다 — 0.6b는 '5줄 이내' 지시를 무시하고 9줄을 냈다(실측)."""
    lines = [PROFILE_HEADER] + [f"- {k}: {v}" for k, v in profile.items()] if profile else []
    note_lines = [l.strip() for l in notes.splitlines() if l.strip()][:MAX_NOTE_LINES]
    return "\n".join(lines + [NOTES_HEADER] + note_lines).strip()


def parse_summary(summary: str) -> tuple[dict, str]:
    """저장된 요약본 → (profile, notes). 헤더가 없는 옛 형식이면 전부 notes로 본다."""
    if NOTES_HEADER not in summary:
        return {}, summary
    head, notes = summary.split(NOTES_HEADER, 1)
    profile = dict(re.findall(r"^- (\S+): (.+)$", head, re.M))
    return profile, notes.strip()


async def load_history(store, llm, thread_id: str) -> tuple[list[dict], dict]:
    """모델에 넘길 이력과 메모리 상태(info)를 돌려준다. 필요하면 요약을 갱신·저장한다."""
    messages = store.load(thread_id)
    summary, covered = store.load_summary(thread_id)   # covered: 요약에 이미 반영된 메시지 수
    info = {"messages": len(messages), "summarized": covered, "updated": False}
    pending_old = messages[covered:max(covered, len(messages) - KEEP_RECENT)]
    if len(pending_old) >= SUMMARIZE_EVERY:
        # fast_llm.summarize는 동기 HTTP 호출 — 이벤트 루프를 막지 않게 스레드로 (개선 1과 같은 이유)
        profile, notes = parse_summary(summary)
        notes, _usage = await asyncio.to_thread(llm.summarize, notes, pending_old)
        summary = compose_summary(extract_profile(pending_old, profile), notes)
        covered += len(pending_old)
        store.save_summary(thread_id, summary, covered)
        info.update(summarized=covered, updated=True)
    history = messages[covered:]
    if summary:
        history = [{"role": "system", "content": f"{SUMMARY_PREFIX}\n{summary}"}] + history
    info["sent"] = len(history)
    return history, info


def chat_messages(system: str, history: list[dict], query: str, window: int | None = None) -> list[dict]:
    """ollama·openai chat이 같이 쓰는 messages 조립. 요약(system 역할)은 항상 넣고,
    user/assistant 메시지는 최근 window개만 넣는다(요약이 없을 때의 안전장치)."""
    window = window or CHAT_WINDOW
    summaries = [m for m in history if m.get("role") == "system"]
    turns = [m for m in history if m.get("role") in ("user", "assistant")][-window:]
    return ([{"role": "system", "content": system}]
            + [{"role": m["role"], "content": m["content"]} for m in summaries + turns]
            + [{"role": "user", "content": query}])


SUMMARIZE_SYSTEM = ("당신은 대화 기록 요약기다. '이전 요약'과 '새 대화'를 합쳐 한국어 요약을 다시 써라. "
                    "보존: 문의했던 시스템과 증상, 수치, 결정·승인된 조치, 아직 해결되지 않은 요청. "
                    "인사말·잡담은 버려라. '- '로 시작하는 5줄 이내 목록으로만 답하라. "
                    "(사용자 이름·소속은 따로 저장하므로 쓰지 않아도 된다)")


def summarize_prompt(prev_summary: str, messages: list[dict]) -> str:
    convo = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
    return f"## 이전 요약\n{prev_summary or '(없음)'}\n\n## 새 대화\n{convo}"
