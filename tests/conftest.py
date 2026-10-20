"""pytest 공통 설정 — 프로젝트 루트를 import 경로에 넣고, 테스트용 가짜 MCP를 제공한다.

LangSmith 추적은 테스트에서 끈다(키가 없어도 경고 없이 돌게).
"""
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ.setdefault("LANGSMITH_TRACING", "false")

MONITORING = json.loads((ROOT / "data" / "monitoring.json").read_text(encoding="utf-8"))


class StubMCP:
    """mcp_client.MCPClient와 같은 메서드를 가진 가짜. subprocess 없이 data/ 파일을 그대로 읽는다.

    jitter(무작위 흔들림)가 없어서 테스트 결과가 항상 같다."""

    def __init__(self):
        self.calls = []

    async def find_prompt(self, name):
        return "ops"

    async def get_prompt(self, name, arguments, server="ops"):
        self.calls.append(("get_prompt", arguments))
        return f"시스템: {arguments['system_name']}\n증상: {arguments['symptom']}"

    async def call_tool(self, name, arguments, server="ops"):
        self.calls.append(("call_tool", arguments["system_name"]))
        base = MONITORING.get(arguments["system_name"])
        if base is None:
            return {"system_name": arguments["system_name"], "status": "unknown",
                    "cpu_percent": 0.0, "memory_percent": 0.0, "alerts": []}
        return {"system_name": arguments["system_name"], "status": base["status"],
                "cpu_percent": base["cpu_base"], "memory_percent": base["memory_base"],
                "alerts": base["alerts"]}

    async def find_resource_template(self, name):
        return "ops", "runbook://incident-response/{incident_type}"

    async def read_resource(self, uri, server="ops"):
        self.calls.append(("read_resource", uri))
        itype = uri.rsplit("/", 1)[-1]
        return (ROOT / "data" / "runbooks" / f"{itype}.md").read_text(encoding="utf-8")


class MemorySession:
    def __init__(self):
        self.rows = {}

    def load(self, thread_id):
        return list(self.rows.get(thread_id, []))

    def append(self, thread_id, role, content):
        self.rows.setdefault(thread_id, []).append({"role": role, "content": content})


@pytest.fixture
def stub_mcp():
    return StubMCP()


@pytest.fixture
def fake_graph(stub_mcp):
    """fake LLM + 가짜 MCP로 만든 실제 LangGraph. 반환값: (graph, mcp)"""
    from agent.deps import SupervisorDeps
    from agent.fake_llm import FastLLM, QualityLLM
    from agent.graph import build_graph

    deps = SupervisorDeps(fast_llm=FastLLM("fast-stub", 0.00015),
                          quality_llm=QualityLLM("quality-stub", 0.005),
                          mcp=stub_mcp, session_store=MemorySession(), version="v2")
    return build_graph(deps), stub_mcp
