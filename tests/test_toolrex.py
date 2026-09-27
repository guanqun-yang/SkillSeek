import argparse
import asyncio
import json

import pytest
from pydantic import ValidationError

import build_toolrex_profiles as tr
from conftest import FIXTURES, REPO_ROOT, requires_env
from llm import LLMConfig


def test_profile_schema():
    ok = tr.ToolRexProfile.model_validate({"function_description": "Diffs PDFs.", "tags": ["pdf", "diff", "compare", "report"]})
    assert ok.model_dump(exclude_none=True) == {"function_description": "Diffs PDFs.", "tags": ["pdf", "diff", "compare", "report"]}
    with pytest.raises(ValidationError):
        tr.ToolRexProfile.model_validate({"function_description": "Diffs PDFs.", "tags": ["pdf", "diff", "compare"]})
    with pytest.raises(ValidationError):
        tr.ToolRexProfile.model_validate({"tags": ["pdf", "diff", "compare", "report"]})


def test_load_pool():
    assert tr.strip_frontmatter("---\nname: x\n---\n# Body") == "\n# Body"
    assert tr.strip_frontmatter("# No frontmatter") == "# No frontmatter"
    names = [n for n, _ in tr.load_pool(FIXTURES / "pool")]
    assert names == ["pdf-excel-diff", "untitled-notes", "video-silence-remover", "weather-lookup"]


@requires_env("OPENROUTER_API_KEY")
def test_live_build(tmp_path):
    out = tmp_path / "profiles.json"
    out.write_text(json.dumps({"untitled-notes": {"function_description": "cached", "tags": ["a", "b", "c", "d"]}}))
    args = argparse.Namespace(pool=str(FIXTURES / "pool"), out=str(out), concurrency=4, save_every=2)
    config = LLMConfig(model="openrouter/openai/gpt-oss-120b", max_context_length=131072, max_output_tokens=1500,
                       reasoning=True, reasoning_strength="medium", temperature=0.0, top_p=1.0, top_k=1)
    before = set((REPO_ROOT / "logs").glob("*-toolrex")) if (REPO_ROOT / "logs").exists() else set()
    asyncio.run(tr.run(args, config))

    profiles = json.loads(out.read_text())
    assert profiles["untitled-notes"]["function_description"] == "cached"
    for name in ("pdf-excel-diff", "video-silence-remover", "weather-lookup"):
        tr.ToolRexProfile.model_validate(profiles[name])
    assert profiles["pdf-excel-diff"]["tags"][0] == "pdf"

    log_dir = (set((REPO_ROOT / "logs").glob("*-toolrex")) - before).pop()
    rows = [json.loads(line) for line in (log_dir / "calls.jsonl").read_text().splitlines()]
    done = [r for r in rows if "profile" in r]
    assert {r["skill"] for r in done} == {"pdf-excel-diff", "video-silence-remover", "weather-lookup"}
    for r in done:
        assert r["prompt"].startswith("You are extracting structured metadata")
        assert r["reasoning"] and r["reasoning"] not in r["output"]
        assert tr.parse_structured(r["output"], tr.ToolRexProfile).model_dump(exclude_none=True) == r["profile"]
    assert json.loads((log_dir / "config.json").read_text())["temperature"] == 0.0
