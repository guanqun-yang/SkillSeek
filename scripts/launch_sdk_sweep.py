"""Run a sweep of scripts/run_sdk.py over trials, conditions, and tasks."""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DRIVER = REPO_ROOT / "scripts/run_sdk.py"
SDK_DIR = Path(os.environ.get("OPENHANDS_SDK_DIR", REPO_ROOT / "third_party/software-agent-sdk"))
TASKS_DIR = Path(os.environ.get("SKILLSBENCH_TASKS", REPO_ROOT / "third_party/skillsbench/tasks"))
MCP_METHODS = ("bm25", "rerank", "rerank_qwen0_6b", "rerank_qwen4b", "rerank_qwen8b")
CONDITIONS = ("none", *MCP_METHODS, "liu_hybrid", "liu_refine", "liu_refined")
LIU = {
    "192": {"url": "http://172.17.0.1:8743", "size": "192", "skills": REPO_ROOT / "liu_baseline/skills/192"},
    "34k": {"url": "http://172.17.0.1:8742", "size": "34000", "skills": REPO_ROOT / "liu_baseline/skills/34k"},
}

LIU_REFINEMENT_PROMPT = """You are a skill refinement agent. Your goal is to discover useful skills, briefly attempt the target task with them, and then write refined task-specific skills that will help a future agent solve the same task.

## Target task (for reflection only — DO NOT actually solve it now)

{ORIGINAL_TASK}

## Your workflow

### Step 1 — discover skills via search
Use `curl -s '<LIU_URL>/hybrid?q=<5-15 word query>&top_k=10'` to retrieve 10 candidate skills, then `curl -s '<LIU_URL>/detail/<skill_id>'` to fetch full `skill_md` content for the 3 most relevant ones.

### Step 2 — briefly attempt the task using these skills
Read the retrieved skills, try a small portion of the task to understand which skill content is useful, which is misleading, and what is missing. DO NOT spend more than 10 commands exploring.

### Step 3 — write refined skills
Based on what you learned, write 1-3 refined SKILL.md files to `/root/refined_skills/<skill-name>/SKILL.md`. Each refined skill MUST start with YAML frontmatter:

```
---
name: skill-name
description: One-line summary of what this skill does (max 200 chars)
---

# Skill Name

(body content tailored to this task)
```

Each refined skill:
- Keeps the parts that actually worked when you tried them
- Fixes or removes parts that were wrong or misleading
- Adds task-specific tips you discovered
- Focuses ONLY on this target task — drop irrelevant sections

Write your final refined skills using the file editor to `/root/refined_skills/<name>/SKILL.md`. Use `mkdir -p /root/refined_skills/<name>` first. Do NOT attempt the actual task solution — your only job is to produce refined skills.
"""


def condition_args(cond: str, task: str, args) -> list[str]:
    liu = LIU[args.pool]
    find_skills = ["--skills-dir", str(liu["skills"]), "--nudge-find-skills",
                   "--find-skills-url", liu["url"], "--find-skills-pool-size", liu["size"]]
    refined = args.liu_refined_dir / task
    if cond == "none":
        return []
    if cond in MCP_METHODS:
        return ["--mcp-url", args.mcp_url, "--nudge-mcp", "--mcp-method", cond]
    if cond == "liu_hybrid":
        return find_skills
    if cond == "liu_refine":
        prompt = LIU_REFINEMENT_PROMPT.replace("<LIU_URL>", liu["url"])
        return find_skills + ["--instruction-override", prompt, "--refinement-mode", "--refined-skills-out", str(refined)]
    if cond == "liu_refined":
        if refined.is_dir() and any(refined.iterdir()):
            return ["--skills-dir", str(refined)]
        return find_skills
    raise ValueError(cond)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--conditions", required=True, help=f"Comma-separated subset of {', '.join(CONDITIONS)}")
    p.add_argument("--pool", choices=sorted(LIU), required=True, help="Pool served to the retrieval conditions")
    p.add_argument("--tasks", default=None, help="Comma-separated task names (default: all SkillsBench tasks)")
    p.add_argument("--trials", type=int, default=1)
    p.add_argument("--trial-start", type=int, default=1)
    p.add_argument("--parallelism", type=int, default=3, help="Concurrent trials; keep under the provider rate limit")
    p.add_argument("--tag", default="sdk", help="Log directory suffix and trial-id prefix")
    p.add_argument("--mcp-url", default="http://172.17.0.1:8002/mcp")
    p.add_argument("--liu-refined-dir", type=Path, default=None,
                   help="Where liu_refine writes and liu_refined reads refined skills (default: logs/liu_refined_skills/<pool>-<tag>)")
    p.add_argument("--driver-python", default=str(SDK_DIR / ".venv/bin/python"))
    p.add_argument("--log-dir", type=Path, default=None)
    args, driver_args = p.parse_known_args()

    conds = [c.strip() for c in args.conditions.split(",") if c.strip()]
    unknown = set(conds) - set(CONDITIONS)
    if unknown:
        sys.exit(f"unknown conditions: {sorted(unknown)}")
    tasks = args.tasks.split(",") if args.tasks else sorted(
        d.name for d in TASKS_DIR.iterdir() if (d / "instruction.md").exists() and (d / "tests/test.sh").exists())
    args.liu_refined_dir = args.liu_refined_dir or REPO_ROOT / f"logs/liu_refined_skills/{args.pool}-{args.tag}"
    log_dir = args.log_dir or REPO_ROOT / f"logs/{time.strftime('%Y%m%d-%H%M%S')}-sweep-{args.tag}"
    log_dir.mkdir(parents=True, exist_ok=True)

    plan = []
    for trial in range(args.trial_start, args.trial_start + args.trials):
        for cond in conds:
            for task in tasks:
                trial_id = f"{args.tag}-{cond}-{task}-t{trial}"
                cmd = [args.driver_python, str(DRIVER), str(TASKS_DIR / task), "--trial-id", trial_id,
                       "--out-dir", str(log_dir), *driver_args, *condition_args(cond, task, args)]
                plan.append((trial_id, cmd))
    print(f"[sweep] {len(plan)} trials -> {log_dir}")

    running: list[tuple[str, subprocess.Popen]] = []
    done = 0
    started = time.time()

    def reap(block: bool) -> None:
        nonlocal running, done
        still = []
        for tid, proc in running:
            if block:
                proc.wait()
            if proc.poll() is None:
                still.append((tid, proc))
            else:
                done += 1
                print(f"[{time.strftime('%H:%M:%S')}] DONE {tid} rc={proc.returncode} ({done}/{len(plan)})")
        running = still

    for trial_id, cmd in plan:
        while len(running) >= args.parallelism:
            time.sleep(2)
            reap(block=False)
        with (log_dir / f"{trial_id}.log").open("w") as f:
            running.append((trial_id, subprocess.Popen(cmd, cwd=SDK_DIR, stdout=f, stderr=subprocess.STDOUT)))
        print(f"[{time.strftime('%H:%M:%S')}] LAUNCH {trial_id}")
    reap(block=True)
    print(f"[sweep] all done in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
