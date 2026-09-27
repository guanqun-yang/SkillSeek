"""Run one SkillsBench task with the OpenHands SDK agent in a Docker workspace."""

import argparse
import base64
import io
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import time
from pathlib import Path

os.environ.setdefault("OPENHANDS_SUPPRESS_BANNER", "1")

from openhands.sdk import LLM, Agent, AgentContext, Conversation, Event
from openhands.sdk.context.skills import load_skills_from_dir
from openhands.sdk.tool import Tool
from openhands.tools.file_editor import FileEditorTool
from openhands.tools.terminal import TerminalTool
from openhands.workspace import DockerDevWorkspace

REPO_ROOT = Path(__file__).resolve().parents[1]
TOKEN_KEYS = ("prompt_tokens", "completion_tokens", "reasoning_tokens", "cache_read_tokens", "cache_write_tokens")
# The agent loop runs in the container, so the key travels with the LLM config.
PROVIDER_KEY_ENV = {
    "openrouter": "OPENROUTER_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "moonshot": "MOONSHOT_API_KEY",
    "zai": "ZAI_API_KEY",
    "minimax": "MINIMAX_API_KEY",
}


def resolve_api_key(model: str, api_key: str | None) -> str | None:
    if api_key:
        return api_key
    env = PROVIDER_KEY_ENV.get(model.split("/", 1)[0])
    if env and not os.environ.get(env):
        sys.exit(f"--model {model} needs ${env} (or pass --api-key)")
    return os.environ.get(env) if env else None


def build_task_image(task_dir: Path) -> str:
    raw_tag = f"skillsbench/{task_dir.name}:raw"
    open_tag = f"skillsbench/{task_dir.name}:open"
    env_dir = task_dir / "environment"
    if not (env_dir / "Dockerfile").exists():
        sys.exit(f"no Dockerfile in {env_dir}")
    print(f"[driver] building raw task image {raw_tag}")
    subprocess.run(["docker", "build", "-t", raw_tag, str(env_dir)], check=True)
    print(f"[driver] building open overlay {open_tag} (chmod /root for non-root agent-server)")
    overlay = (
        f"FROM {raw_tag}\n"
        "RUN chmod -R o+rwX /root && (chmod o+rx /root /home 2>/dev/null || true)\n"
    )
    subprocess.run(["docker", "build", "-t", open_tag, "-"], input=overlay.encode(), check=True)
    return open_tag


def parse_reward(stdout: str) -> float | None:
    m = re.search(r"Writing default score:\s*([0-9.]+)", stdout)
    if m:
        return float(m.group(1))
    m = re.search(r"reward:\s*([0-9.]+)", stdout)
    if m:
        return float(m.group(1))
    m = re.search(r"(?:Final score|FINAL SCORE):\s*(?:\d+/\d+\s*=\s*)?([0-9.]+)", stdout)
    if m:
        return float(m.group(1))
    passed = sum(int(x) for x in re.findall(r"(\d+)\s+passed", stdout))
    failed = sum(int(x) for x in re.findall(r"(\d+)\s+failed", stdout))
    failed += sum(int(x) for x in re.findall(r"(\d+)\s+error", stdout))
    if passed + failed == 0:
        verbose = re.findall(r"::\S+\s+(PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)\s+\[", stdout)
        passed = sum(1 for s in verbose if s in ("PASSED", "XPASS"))
        failed = sum(1 for s in verbose if s in ("FAILED", "ERROR"))
    if passed + failed == 0:
        for line in stdout.splitlines():
            mm = re.match(r".+\.py\s+([.FsExX]+)\s+\[\s*\d+%\]\s*$", line.strip())
            if mm:
                passed += mm.group(1).count(".")
                failed += sum(mm.group(1).count(c) for c in "FE")
    if passed + failed > 0:
        return round(passed / (passed + failed), 3)
    if "All tests passed" in stdout:
        return 1.0
    if "Tests failed" in stdout:
        return 0.0
    return None


def mcp_nudges(topk: int | None) -> tuple[str, str]:
    topk_phrase = f" with top_k={topk}" if topk else ""
    system = (
        "IMPORTANT: A skill retrieval tool named `skill_lookup` is available. "
        "Before exploring the workspace or running commands, you MUST call "
        f"`skill_lookup`{topk_phrase} with a short query summarizing the task to discover "
        "relevant domain skills. Then call `skill_load` on the most relevant "
        "result to read its full content."
    )
    user = (
        "\n\n[REQUIRED FIRST STEP] Your very first tool call must be "
        f"`skill_lookup`{topk_phrase} with a 5-15 word query that summarizes this task. "
        "Then call `skill_load` on the most relevant result. Only after at "
        "least one `skill_load` call may you start working on the task."
    )
    return system, user


def find_skills_nudges(url: str, pool_size: str) -> tuple[str, str]:
    system = (
        "IMPORTANT: A finding-skills skill is loaded into your context. "
        "Before exploring the workspace or running commands, you MUST use it "
        f"to discover relevant domain skills from a {pool_size}-skill index. "
        "Specifically: use `curl -s` to query the hybrid search endpoint "
        f"({url}/hybrid?q=...) with a short summary of the task, "
        "then fetch full SKILL.md content for the most promising 1-3 results "
        "via the /detail/SKILL_ID endpoint."
    )
    user = (
        "\n\n[REQUIRED FIRST STEPS]\n"
        f"1. Use `curl -s '{url}/hybrid?q=<5-15 word task summary>&top_k=10'` "
        "to retrieve 10 candidate skills.\n"
        "2. Pick the 1-3 most relevant skill_ids and fetch full content via "
        f"`curl -s '{url}/detail/<skill_id>'`.\n"
        "3. Read the `skill_md` field from each detail response carefully.\n"
        "4. Only after that may you start working on the task, applying the "
        "knowledge from the loaded skills."
    )
    return system, user


def extract_refined_skills(workspace, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    r = workspace.execute_command(
        "cd /root && if [ -d refined_skills ]; then tar -czf - refined_skills | base64 -w0; else echo NOREFINED; fi",
        timeout=120,
    )
    payload = (r.stdout or "").strip()
    if payload == "NOREFINED":
        print("[driver] no refined_skills dir found in container")
        return
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(payload)), mode="r:gz") as tar:
        tar.extractall(out)
    inner = out / "refined_skills"
    for child in inner.iterdir():
        target = out / child.name
        if target.exists():
            shutil.rmtree(target)
        child.rename(target)
    inner.rmdir()
    print(f"[driver] extracted {sum(1 for d in out.iterdir() if d.is_dir())} refined skills to {out}")


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("task_path", type=Path)
    p.add_argument("--model", default="openrouter/qwen/qwen3.5-397b-a17b", help="LiteLLM model string")
    p.add_argument("--base-url", default=None, help="Endpoint for self-hosted models (vLLM)")
    p.add_argument("--api-key", default=None, help="Defaults to the provider's key in the environment")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=0, help="0 disables top-k truncation")
    p.add_argument("--reasoning-effort", choices=["none", "low", "medium", "high"], default="high")
    p.add_argument("--max-input-tokens", type=int, default=262144)
    p.add_argument("--max-output-tokens", type=int, default=32768)
    p.add_argument("--max-iterations", type=int, default=30)
    p.add_argument("--timeout", type=int, default=600, help="Wall-clock cap (s) for the conversation")
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / f"logs/{time.strftime('%Y%m%d-%H%M%S')}-run")
    p.add_argument("--trial-id", default=None)
    p.add_argument("--skills-dir", default=None, help="Load these skills into the agent context")
    p.add_argument("--mcp-url", default=None, help="SkillSeek MCP endpoint reachable from the container")
    p.add_argument("--mcp-method", default=None, help="Server-side skill_lookup method (X-Skill-Method header)")
    p.add_argument("--nudge-mcp", action="store_true", help="Tell the agent to call skill_lookup first")
    p.add_argument("--nudge-mcp-topk", type=int, default=None)
    p.add_argument("--nudge-find-skills", action="store_true", help="Liu et al. baseline: tell the agent to use finding-skills first")
    p.add_argument("--find-skills-url", default="http://172.17.0.1:8742")
    p.add_argument("--find-skills-pool-size", default="34000")
    p.add_argument("--instruction-override", default=None, help="Replaces the instruction; {ORIGINAL_TASK} is substituted")
    p.add_argument("--refinement-mode", action="store_true", help="Skip the verifier and extract /root/refined_skills/")
    p.add_argument("--refined-skills-out", type=Path, default=None)
    args = p.parse_args(argv)
    if args.refinement_mode and not args.refined_skills_out:
        sys.exit("--refinement-mode requires --refined-skills-out")
    return args


def main():
    args = parse_args()
    api_key = resolve_api_key(args.model, args.api_key)

    def _on_timeout(signum, frame):
        raise TimeoutError(f"trial exceeded --timeout={args.timeout}s")
    signal.signal(signal.SIGALRM, _on_timeout)
    signal.alarm(args.timeout)

    task = args.task_path.resolve()
    if not (task / "instruction.md").exists():
        sys.exit(f"no instruction.md in {task}")
    instruction = (task / "instruction.md").read_text()
    if args.instruction_override:
        instruction = args.instruction_override.replace("{ORIGINAL_TASK}", instruction)

    trial_id = args.trial_id or f"{task.name}-{int(time.time())}"
    out_dir = args.out_dir / trial_id
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[driver] trial_id={trial_id} out_dir={out_dir}")

    task_image = build_task_image(task)

    llm_config = {
        "model": args.model,
        "base_url": args.base_url,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "reasoning_effort": args.reasoning_effort,
        "max_input_tokens": args.max_input_tokens,
        "max_output_tokens": args.max_output_tokens,
    }
    print(f"[driver] llm={json.dumps(llm_config)}")
    llm = LLM(usage_id="agent", api_key=api_key, **llm_config)

    skills = []
    if args.skills_dir:
        _, _, agent_skills = load_skills_from_dir(Path(args.skills_dir).resolve())
        skills = list(agent_skills.values())
        print(f"[driver] loaded {len(skills)} skills from {args.skills_dir}")
    system_nudge = user_nudge = ""
    if args.nudge_mcp and args.mcp_url:
        system_nudge, user_nudge = mcp_nudges(args.nudge_mcp_topk)
    elif args.nudge_find_skills:
        system_nudge, user_nudge = find_skills_nudges(args.find_skills_url, args.find_skills_pool_size)

    agent_kwargs = {"llm": llm, "tools": [Tool(name=TerminalTool.name), Tool(name=FileEditorTool.name)]}
    if skills or system_nudge:
        agent_kwargs["agent_context"] = AgentContext(
            skills=skills,
            load_public_skills=False,
            system_message_suffix=system_nudge,
            user_message_suffix=user_nudge,
        )
    if args.mcp_url:
        headers = {"X-Trial-ID": trial_id}
        if args.mcp_method:
            headers["X-Skill-Method"] = args.mcp_method
        agent_kwargs["mcp_config"] = {"mcpServers": {"skills": {"url": args.mcp_url, "headers": headers}}}
        print(f"[driver] mcp_url={args.mcp_url} method={args.mcp_method or 'default'}")
    agent = Agent(**agent_kwargs)

    events_path = out_dir / "events.jsonl"
    events_log = events_path.open("a")

    def event_cb(event: Event) -> None:
        try:
            data = event.model_dump(mode="json")
        except Exception:
            data = str(event)
        events_log.write(json.dumps({"ts": time.time(), "event_type": type(event).__name__, "data": data}, default=str) + "\n")
        events_log.flush()

    started = time.time()
    error = None
    reward = None
    try:
        with DockerDevWorkspace(
            base_image=task_image,
            target="source",
            platform="linux/amd64",
            working_dir="/root",
            volumes=[f"{task}/tests:/tests:ro"],
        ) as workspace:
            conversation = Conversation(
                agent=agent, workspace=workspace, callbacks=[event_cb],
                max_iteration_per_run=args.max_iterations,
            )
            conversation.send_message(instruction)
            try:
                conversation.run()
                print("[driver] conversation done")
            except Exception as conv_e:
                error = f"{type(conv_e).__name__}: {conv_e}"
                print(f"[driver] conversation failed: {error}")
            if args.refinement_mode:
                extract_refined_skills(workspace, args.refined_skills_out)
            else:
                print("[driver] running verifier")
                r = workspace.execute_command('export PATH="$HOME/.local/bin:$PATH" && bash /tests/test.sh', timeout=1800)
                test_stdout = r.stdout or ""
                (out_dir / "test_stdout.log").write_text(test_stdout)
                reward = parse_reward(test_stdout)
    except Exception as e:
        if error is None:
            error = f"{type(e).__name__}: {e}"
        print(f"[driver] error: {error}")
    finally:
        signal.alarm(0)
        events_log.close()

    usage = getattr(llm.metrics, "accumulated_token_usage", None)
    tokens = {k: int(getattr(usage, k, 0) or 0) for k in TOKEN_KEYS} if usage is not None else {}
    if not any(tokens.values()) and events_path.exists():
        last = None
        for line in events_path.open():
            m = re.search(r'"accumulated_token_usage"\s*:\s*(\{[^}]*\})', line)
            if m:
                try:
                    last = json.loads(m.group(1))
                except json.JSONDecodeError:
                    pass
        if last:
            tokens = {k: int(last.get(k, 0) or 0) for k in TOKEN_KEYS}
    result = {
        "trial_id": trial_id,
        "task_name": task.name,
        "model": args.model,
        "llm_config": llm_config,
        "skills_dir": args.skills_dir,
        "mcp_url": args.mcp_url,
        "mcp_method": args.mcp_method,
        "reward": reward,
        "error": error,
        "duration_sec": time.time() - started,
        "cost": float(getattr(llm.metrics, "accumulated_cost", 0) or 0),
        "tokens": tokens,
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    print(f"[driver] RESULT: {json.dumps(result)}")


if __name__ == "__main__":
    main()
