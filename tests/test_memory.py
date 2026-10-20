"""[개선 12] 대화 요약 메모리 — 오래된 턴을 요약해도 이름 같은 핵심 사실은 남고, 모델에 보내는 양은 줄어든다."""
import asyncio

from agent.fake_llm import FastLLM
from agent.memory import (KEEP_RECENT, SUMMARIZE_EVERY, chat_messages, compose_summary,
                          extract_profile, load_history, parse_summary)
from agent.session_store import SqliteSessionStore


def _long_conversation(store, thread, turns=12):
    store.append(thread, "user", "제 이름은 민수입니다")
    store.append(thread, "assistant", "안녕하세요, 민수님!")
    for i in range(turns):
        store.append(thread, "user", f"web-01 상태 질문 {i}")
        store.append(thread, "assistant", f"web-01은 healthy입니다 ({i})")


def test_summary_created_and_persisted(tmp_path):
    store = SqliteSessionStore(tmp_path / "s.db")
    _long_conversation(store, "t")
    history, info = asyncio.run(load_history(store, FastLLM("f", 0.0), "t"))
    assert info["updated"] and info["messages"] == 26
    assert history[0]["role"] == "system" and "- 이름: 민수" in history[0]["content"]
    assert len(history) == 1 + (26 - info["summarized"])     # 요약 1개 + 요약 안 된 최근 메시지
    assert len(history) < 26
    # 서버를 다시 띄워도(새 연결) 요약은 session.db에 남아 있다
    assert SqliteSessionStore(tmp_path / "s.db").load_summary("t")[1] == info["summarized"]


def test_no_summary_for_short_conversation(tmp_path):
    store = SqliteSessionStore(tmp_path / "s.db")
    for i in range(KEEP_RECENT + SUMMARIZE_EVERY - 1):   # 아직 요약 기준에 못 미침
        store.append("t", "user", f"m{i}")
    history, info = asyncio.run(load_history(store, FastLLM("f", 0.0), "t"))
    assert not info["updated"] and all(m["role"] == "user" for m in history)


def test_chat_messages_keeps_summary_outside_window():
    history = [{"role": "system", "content": "이전 대화 요약:\n- 사용자 이름: 민수"}] + [
        {"role": "user", "content": f"q{i}"} for i in range(30)]
    msgs = chat_messages("SYS", history, "제 이름이 뭐였죠?", window=10)   # 요약 1 + 최근 10
    assert msgs[1]["content"].startswith("이전 대화 요약") and len(msgs) == 1 + 1 + 10 + 1


def test_name_recalled_after_long_chat_via_graph(tmp_path, fake_graph):
    # 원래(최근 10개 창)라면 13턴 전의 이름은 창 밖이라 잊는다 → 요약 덕분에 기억
    graph, _ = fake_graph
    store = SqliteSessionStore(tmp_path / "s.db")
    _long_conversation(store, "t")
    history, _ = asyncio.run(load_history(store, FastLLM("f", 0.0), "t"))
    assert not any("민수" in m["content"] for m in history[-10:] if m["role"] != "system")
    result = asyncio.run(graph.ainvoke({"query": "제 이름이 뭐였죠?", "thread_id": "t",
                                        "history": history, "events": [], "cost_usd": 0.0}))
    assert "민수" in result["final"]


def test_profile_pinned_by_code_and_round_trips():
    msgs = [{"role": "user", "content": "제 이름은 민수입니다. 결제팀에서 일해요."},
            {"role": "assistant", "content": "제 이름은 봇입니다"},          # assistant 말은 무시
            {"role": "user", "content": "아 참, 지금은 인프라팀 소속이에요"}]   # 나중 값으로 덮어씀
    profile = extract_profile(msgs)
    assert profile == {"이름": "민수", "소속": "인프라팀"}
    text = compose_summary(profile, "- 문의: payments CPU")
    assert parse_summary(text) == (profile, "- 문의: payments CPU")


def test_other_team_mention_is_not_my_team():
    # [리뷰 반영] 남의 팀 언급이 내 소속을 덮어쓰면 안 된다
    msgs = [{"role": "user", "content": "결제팀에서 일해요"},
            {"role": "user", "content": "인프라팀에서 연락 왔어요"}]
    assert extract_profile(msgs) == {"소속": "결제팀"}
