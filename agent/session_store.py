"""Wave 12 — 프로세스 밖(SQLite 파일)에 두는 세션 저장소.

서버가 재시작돼도(새 프로세스, 새 인스턴스) 같은 session.db 파일을 다시 열므로
대화 이력이 유지된다. in-memory dict였다면 재시작 즉시 소실됐을 상태다.

[개선 12] summaries 테이블 — 대화 요약 메모리(agent/memory.py)가 쓰는 요약본과 '몇 번째
메시지까지 요약했는지(covered)'를 thread별로 저장한다.
"""
import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "session.db"


class SqliteSessionStore:
    def __init__(self, db_path: Path = DB_PATH):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS messages ("
            " thread_id TEXT, role TEXT, content TEXT, created TEXT)"
        )
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS summaries ("
            " thread_id TEXT PRIMARY KEY, summary TEXT, covered INTEGER, updated TEXT)"
        )
        self.conn.commit()

    def load(self, thread_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT role, content FROM messages WHERE thread_id=? ORDER BY rowid",
            (thread_id,),
        ).fetchall()
        return [{"role": r, "content": c} for r, c in rows]

    def append(self, thread_id: str, role: str, content: str) -> None:
        self.conn.execute(
            "INSERT INTO messages VALUES (?,?,?,?)",
            (thread_id, role, content, datetime.now().isoformat()),
        )
        self.conn.commit()

    def load_summary(self, thread_id: str) -> tuple[str, int]:
        row = self.conn.execute(
            "SELECT summary, covered FROM summaries WHERE thread_id=?", (thread_id,)).fetchone()
        return (row[0], row[1]) if row else ("", 0)

    def save_summary(self, thread_id: str, summary: str, covered: int) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO summaries VALUES (?,?,?,?)",
            (thread_id, summary, covered, datetime.now().isoformat()),
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()
