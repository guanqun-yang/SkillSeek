import asyncio
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest

from conftest import FIXTURES, REPO_ROOT

_TMP = Path(tempfile.mkdtemp(prefix="skillseek-test-"))
os.environ.update({
    "SKILLS_DIR": str(FIXTURES / "pool"),
    "MCP_TEXT_VARIANT": "toolrex",
    "MCP_TOOLREX_PATH": str(FIXTURES / "toolrex.json"),
    "MCP_LOG_PATH": str(_TMP / "calls.jsonl"),
    "MCP_EMBED_CACHE": str(_TMP / "cache.npy"),
})
from mcp_server import skills_mcp  # noqa: E402


def log_rows():
    return [json.loads(line) for line in Path(os.environ["MCP_LOG_PATH"]).read_text().splitlines()]


def test_load_skills():
    skills = {s["name"]: s for s in skills_mcp.store.skills}
    assert sorted(skills) == ["lowercase-skill", "pdf-excel-diff", "untitled-notes", "video-silence-remover", "weather-lookup"]
    assert skills["weather-lookup"]["description"].startswith("Look up the current weather")
    assert skills["untitled-notes"]["description"] == ""
    assert skills["pdf-excel-diff"]["body"].startswith("# PDF vs Excel diff")


def test_build_text_toolrex():
    by_name = {s["name"]: s for s in skills_mcp.store.skills}
    text = skills_mcp.build_text(by_name["pdf-excel-diff"])
    assert text.startswith("pdf-excel-diff: Compare a PDF document")
    assert "Tags: pdf, compare, excel, xlsx" in text
    assert "When to use: When checking a PDF report" in text
    assert skills_mcp.build_text(by_name["weather-lookup"]) == f"weather-lookup: {by_name['weather-lookup']['description']}"


def test_bm25_lookup():
    hits = skills_mcp.store.lookup("compare pdf with excel spreadsheet", 3, "bm25")
    assert hits[0]["name"] == "pdf-excel-diff"
    assert len(hits) == 3


def test_rerank_lookup_and_cache():
    hits = skills_mcp.store.lookup("strip the quiet parts out of my screen recording", 3, "rerank")
    assert hits[0]["name"] == "video-silence-remover"
    assert [h["score"] for h in hits] == sorted((h["score"] for h in hits), reverse=True)
    cache = np.load(os.environ["MCP_EMBED_CACHE"])
    assert cache.shape == (len(skills_mcp.store.skills), 768)
    fresh = skills_mcp.SkillStore(FIXTURES / "pool")
    fresh._ensure_embeddings()
    assert np.array_equal(fresh._embeddings, cache)


def test_unknown_method():
    with pytest.raises(ValueError, match="unknown method"):
        skills_mcp.store.lookup("anything", 3, "voyage")


def test_tools_and_log():
    listed = skills_mcp.skill_list()
    assert "- pdf-excel-diff: Compare a PDF document" in listed
    looked = skills_mcp.skill_lookup("total spending per category from a csv", k=2, method="bm25")
    assert looked.splitlines()[0].startswith("- lowercase-skill (score=")
    assert skills_mcp.skill_load("lowercase-skill").startswith("# CSV spending report")
    assert skills_mcp.skill_load("no-such-skill") == "skill not found: no-such-skill"
    rows = log_rows()
    tools = [r["tool"] for r in rows]
    assert {"skill_list", "skill_lookup", "skill_load"} <= set(tools)
    lookup = [r for r in rows if r["tool"] == "skill_lookup"][-1]
    assert lookup["method"] == "bm25" and lookup["method_override_applied"] is False
    assert lookup["returned"][0]["name"] == "lowercase-skill"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_http_headers(tmp_path):
    import httpx
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    port = _free_port()
    log_path = tmp_path / "http.jsonl"
    env = {**os.environ, "MCP_TRANSPORT": "http", "MCP_HOST": "127.0.0.1", "MCP_PORT": str(port),
           "MCP_LOG_PATH": str(log_path)}
    server = subprocess.Popen([sys.executable, "-m", "mcp_server.skills_mcp"], cwd=REPO_ROOT, env=env,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    async def call():
        headers = {"X-Skill-Method": "bm25", "X-Trial-ID": "trial-42"}
        async with httpx.AsyncClient(headers=headers, timeout=30) as client:
            async with streamable_http_client(f"http://127.0.0.1:{port}/mcp", http_client=client) as (r, w, _):
                async with ClientSession(r, w) as s:
                    await s.initialize()
                    tools = {t.name for t in (await s.list_tools()).tools}
                    res = await s.call_tool("skill_lookup", {"query": "compare pdf and excel", "k": 2, "method": "embed"})
                    return tools, res.content[0].text

    try:
        for _ in range(120):
            try:
                tools, text = asyncio.run(call())
                break
            except Exception:
                time.sleep(0.5)
        else:
            pytest.fail("MCP server did not come up")
    finally:
        server.terminate()
        server.wait(timeout=30)
    assert tools == {"skill_list", "skill_lookup", "skill_load"}
    assert text.startswith("- pdf-excel-diff (score=")
    row = json.loads(log_path.read_text().splitlines()[-1])
    assert row["method"] == "bm25"
    assert row["method_override_applied"] is True
    assert row["trial_id"] == "trial-42"


def test_stdio_script(tmp_path):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(
        command=sys.executable,
        args=[str(REPO_ROOT / "mcp_server/skills_mcp.py")],
        env={**os.environ, "MCP_LOG_PATH": str(tmp_path / "stdio.jsonl")},
        cwd=str(tmp_path),
    )

    async def call():
        async with stdio_client(params) as (r, w):
            async with ClientSession(r, w) as s:
                await s.initialize()
                listed = await s.call_tool("skill_list", {})
                loaded = await s.call_tool("skill_load", {"name": "pdf-excel-diff"})
                return listed.content[0].text, loaded.content[0].text

    listed, loaded = asyncio.run(call())
    assert listed.count("\n- ") == 4
    assert loaded.startswith("# PDF vs Excel diff")
