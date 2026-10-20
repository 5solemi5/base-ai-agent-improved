"""[개선 11] 위험 조치 탐지 — Human-in-the-loop 승인 노드가 '사람에게 물어볼지'를 정하는 규칙.

runbook에는 재시작·롤백·스케일 아웃·failover·트래픽 제외·파일 삭제 같은 '되돌리기 어렵거나
서비스에 영향을 주는' 조치가 들어 있고, 답변 작성 모델은 이것을 그대로 권고문에 옮긴다.
운영자가 확인하지 않은 채 이런 문장이 최종 답변으로 나가지 않도록, 답변에서 해당 문장을 찾는다.

판정은 정규식(코드)이다. LLM에게 "위험한가?"를 물으면 같은 문장도 실행마다 답이 달라지고,
놓치면(false negative) 승인 없이 나가기 때문이다. 대신 오탐(위험하지 않은데 묻는 것)은 감수한다.
"""
import re

RISKY_ACTIONS: list[tuple[str, re.Pattern]] = [
    ("재시작", re.compile(r"재시작|재기동|restart|reboot", re.I)),
    ("롤백", re.compile(r"롤백|rollback", re.I)),
    ("스케일 조정", re.compile(r"오토스케일|스케일 ?(?:아웃|업)|scale ?out|증설|한 대 더", re.I)),
    ("failover 전환", re.compile(r"failover|페일오버|장애 조치 전환", re.I)),
    ("트래픽 제외·우회", re.compile(r"로드밸런서에서[^.\n]{0,10}제외|우회하는 라우팅|트래픽을 차단", re.I)),
    ("데이터 삭제", re.compile(r"삭제|지운다|지워|정리해 여유", re.I)),
]

# 문장 경계: 마침표·물음표·느낌표(단 "36.8%"처럼 숫자 사이의 점은 제외), 줄바꿈, " 2) " 같은 번호 앞.
# 처음엔 [^.!?\n]+ 로 잘랐는데, 화면 검증에서 "CPU 36.8%"의 점에서 잘려 엉뚱한 조각이 승인 카드에 떴다.
_BOUNDARY = re.compile(r"(?<!\d)[.!?](?!\d)|\n|\s(?=\d+\)\s)")
_STEP_NO = re.compile(r"^\d+\)\s*")


def split_sentences(text: str) -> list[str]:
    """원문에 그대로 있는 조각만 돌려준다(앞의 번호 "2) "는 뗀다) — 거절 시 replace로 지울 수 있어야 하므로."""
    return [p for p in (_STEP_NO.sub("", piece.strip()) for piece in _BOUNDARY.split(text)) if p]


def find_risky_actions(text: str) -> list[dict]:
    """[{action, sentence}] — 같은 문장에 여러 조치가 있으면 조치마다 한 번씩, 같은 조치는 첫 문장만."""
    found, seen = [], set()
    for sentence in split_sentences(text):
        for name, pattern in RISKY_ACTIONS:
            if name not in seen and pattern.search(sentence):
                seen.add(name)
                found.append({"action": name, "sentence": sentence})
    return found


def apply_decision(draft: str, actions: list[dict], approved: bool, comment: str = "") -> str:
    """승인: 답변은 그대로 두고 승인 기록을 붙인다.
    거절: 위험 조치가 든 문장을 '[운영자 거절로 제외된 조치: …]'로 바꿔 실행 권고가 나가지 않게 한다."""
    names = ", ".join(a["action"] for a in actions)
    note = f" (메모: {comment})" if comment else ""
    if approved:
        return draft.rstrip() + f"\n\n✅ 운영자 승인 완료 — 위험 조치: {names}{note}"
    out = draft
    for a in actions:
        out = out.replace(a["sentence"], f"[운영자 거절로 제외된 조치: {a['action']}]")
    return out.rstrip() + f"\n\n⛔ 운영자가 위험 조치({names})를 승인하지 않아 답변에서 제외했습니다{note}"
