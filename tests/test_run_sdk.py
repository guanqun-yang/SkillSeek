import pytest

pytest.importorskip("openhands.sdk")
import run_sdk  # noqa: E402


@pytest.mark.parametrize("stdout,reward", [
    ("Writing default score: 0.75", 0.75),
    ("reward: 1.0", 1.0),
    ("FINAL SCORE: 3/4 = 0.75", 0.75),
    ("===== 3 passed, 1 failed in 2.1s =====", 0.75),
    ("tests/test_outputs.py::test_a PASSED [ 50%]\ntests/test_outputs.py::test_b FAILED [100%]", 0.5),
    ("tests/test_outputs.py ..F.  [100%]", 0.75),
    ("All tests passed", 1.0),
    ("nothing useful", None),
])
def test_parse_reward(stdout, reward):
    assert run_sdk.parse_reward(stdout) == reward


def test_find_skills_nudges():
    system, user = run_sdk.find_skills_nudges("http://172.17.0.1:8743", "192")
    assert "192-skill index" in system and "http://172.17.0.1:8743/hybrid" in system
    assert "http://172.17.0.1:8743/hybrid?q=" in user and "http://172.17.0.1:8743/detail/" in user
    assert "8742" not in system + user and "34000" not in system + user


def test_mcp_nudges():
    assert "top_k" not in "".join(run_sdk.mcp_nudges(None))
    system, user = run_sdk.mcp_nudges(3)
    assert "`skill_lookup` with top_k=3" in system and "`skill_lookup` with top_k=3" in user


def test_cli_defaults():
    args = run_sdk.parse_args(["some/task"])
    assert args.model == "openrouter/qwen/qwen3.5-397b-a17b"
    assert (args.temperature, args.top_p, args.top_k) == (1.0, 1.0, 0)
    assert args.reasoning_effort == "high"
    assert args.max_output_tokens == 32768
    assert args.max_iterations == 30 and args.timeout == 600


def test_resolve_api_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert run_sdk.resolve_api_key("openrouter/qwen/qwen3.5-397b-a17b", None) == "or-key"
    assert run_sdk.resolve_api_key("openrouter/qwen/qwen3.5-397b-a17b", "explicit") == "explicit"
    assert run_sdk.resolve_api_key("hosted_vllm/openai/gpt-oss-120b", None) is None
    with pytest.raises(SystemExit, match="ANTHROPIC_API_KEY"):
        run_sdk.resolve_api_key("anthropic/claude-haiku-4-5", None)
