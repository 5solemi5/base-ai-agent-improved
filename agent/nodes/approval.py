"""[개선 11] Human-in-the-loop 승인 노드 — 위험 조치가 든 답변은 사람 승인 후에만 내보낸다.

위치: validate(합격) → approval → finalize.
PDF 예시는 'write → validate 사이'였지만, 그 자리에 두면 validate가 반려해 재작성할 때마다
사람에게 다시 묻게 된다. 검수를 통과한 '최종 후보'에 대해 한 번만 묻도록 validate 뒤에 뒀다.

LangGraph interrupt()는 그래프 실행을 이 지점에서 멈추고(체크포인트에 상태 저장) 호출자에게
payload를 돌려준다. 서버가 사람의 결정을 Command(resume={...})로 넘기면 같은 노드가 처음부터
다시 실행되고, 이번에는 interrupt()가 그 결정을 반환값으로 돌려준다.
"""
from langgraph.types import interrupt
from langsmith import traceable

from ..approval import apply_decision, find_risky_actions
from ..deps import SupervisorDeps
from .state import AgentState


@traceable(name="approval_node")
async def approval_node(state: AgentState, deps: SupervisorDeps) -> dict:
    actions = find_risky_actions(state["draft"])
    if not actions or not deps.hitl:
        # 위험 조치가 없거나 HITL을 끈 경우(HITL_APPROVAL=0, 자동 평가 등) — 그대로 통과
        return {"events": state["events"] + [{"type": "ApprovalSkipped",
                                              "risky": len(actions), "hitl": deps.hitl}]}
    # ⚠️ interrupt() 앞의 코드는 재개(resume) 때 한 번 더 실행된다 — 부작용(외부 호출·저장)을 두지 않는다
    decision = interrupt({"type": "ApprovalRequired", "actions": actions, "draft": state["draft"]})
    approved = bool(decision.get("approved"))
    draft = apply_decision(state["draft"], actions, approved, decision.get("comment", ""))
    event = {"type": "ApprovalDecided", "approved": approved,
             "actions": [a["action"] for a in actions]}
    return {"draft": draft, "approval": {"approved": approved, "actions": actions},
            "events": state["events"] + [event]}
