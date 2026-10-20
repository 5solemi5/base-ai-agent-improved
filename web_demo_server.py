"""web_demo_server.py — all_total_vue 에이전트와 브라우저 채팅창으로 대화해보는 데모.

web_chat.html이 바닐라 JS 대신 Vue 3(CDN 글로벌 빌드, 별도 빌드 단계 없음)로 되어
있다는 점만 all_total과 다르다. server.py(8100, 기존 통합 테스트 서버)는 전혀
건드리지 않는다. 이 스크립트는
- web_chat.html 한 장을 같은 출처(origin)로 띄우고
- /chat/stream 요청을 그대로 8100으로 중계(프록시)해서

브라우저의 CORS 제약 없이 실제 에이전트와 대화할 수 있게 해주는 아주 얇은 레이어다.
server.py는 평소처럼(fake/ollama/openai 아무 백엔드로나) 8100에 미리 띄워 놓아야 한다.

실행:
    python3 web_demo_server.py        # http://localhost:8600 에서 채팅 화면 오픈
"""
import os
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

# [개선 5] 백엔드 주소·프론트 포트를 환경변수로 (원본은 8100/8600 하드코딩)
BACKEND = os.environ.get("BACKEND_URL", f"http://127.0.0.1:{os.environ.get('BACKEND_PORT', '8100')}")
FRONT_PORT = int(os.environ.get("FRONT_PORT", "8600"))
HERE = Path(__file__).parent

app = FastAPI()


@app.get("/")
async def index():
    # 수정한 HTML이 브라우저 캐시 때문에 반영 안 되던 문제 방지
    return FileResponse(HERE / "web_chat.html", headers={"Cache-Control": "no-store"})


@app.get("/health")
async def proxy_health():
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(f"{BACKEND}/health")
    return JSONResponse(r.json(), status_code=r.status_code)


@app.post("/chat")
async def proxy_chat(req: Request):
    body = await req.body()
    async with httpx.AsyncClient(timeout=600.0) as client:   # 실모델 장애 진단은 100초 가까이 걸릴 수 있다
        r = await client.post(f"{BACKEND}/chat", content=body,
                              headers={"content-type": "application/json"})
    return JSONResponse(r.json(), status_code=r.status_code)


@app.get("/chat/stream")
async def proxy_stream(message: str, thread_id: str = "web-demo"):
    async def relay():
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "GET", f"{BACKEND}/chat/stream",
                params={"message": message, "thread_id": thread_id},
            ) as upstream:
                async for chunk in upstream.aiter_bytes():
                    yield chunk

    return StreamingResponse(relay(), media_type="text/event-stream")


@app.post("/chat/approve")
async def proxy_approve(req: Request):   # [개선 11] 승인/거절 (JSON)
    body = await req.body()
    async with httpx.AsyncClient(timeout=600.0) as client:
        r = await client.post(f"{BACKEND}/chat/approve", content=body,
                              headers={"content-type": "application/json"})
    return JSONResponse(r.json(), status_code=r.status_code)


@app.get("/chat/approve/stream")
async def proxy_approve_stream(thread_id: str, approved: bool, comment: str = ""):   # [개선 11]
    async def relay():
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "GET", f"{BACKEND}/chat/approve/stream",
                params={"thread_id": thread_id, "approved": str(approved).lower(), "comment": comment},
            ) as upstream:
                async for chunk in upstream.aiter_bytes():
                    yield chunk

    return StreamingResponse(relay(), media_type="text/event-stream")


if __name__ == "__main__":
    print(f"브라우저에서 http://localhost:{FRONT_PORT} 을 열어보세요 (백엔드: {BACKEND})")
    # 0.0.0.0이면 같은 네트워크의 다른 기기에서도 접속된다 — 데모 PC 밖으로 열 이유가 없어 127.0.0.1로
    uvicorn.run(app, host="127.0.0.1", port=FRONT_PORT)
