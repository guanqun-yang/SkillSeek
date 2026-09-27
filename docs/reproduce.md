# Reproducing the Experiments

In each trial, an OpenHands agent attempts one SkillsBench task inside its own Docker workspace, and one retrieval condition decides which skills the agent can read. Rerunning the experiments takes the five steps below. They need Docker, [uv](https://docs.astral.sh/uv/), an OpenRouter key, and a GPU for the Liu et al. baseline on the 192-skill pool.

## Fetch the Benchmarks, Pools, and Agent SDK

`scripts/setup.sh` clones [SkillsBench](https://github.com/benchflow-ai/skillsbench), the gold data of [Liu et al. (2026)](https://github.com/UCSB-NLP-Chang/Skill-Usage), and the [OpenHands SDK](https://github.com/OpenHands/software-agent-sdk) at pinned commits. It installs the SDK into its own environment from `scripts/openhands-sdk-uv.lock`, downloads the 34K pool from [Hugging Face](https://huggingface.co/datasets/Shiyu-Lab/Skill-Usage) (about 3 gigabytes, skipped when `SKIP_34K=1`), and builds the 192-skill pool:

```bash
bash scripts/setup.sh
```

## Start the Servers for One Pool

A sweep evaluates one pool at a time. Start the SkillSeek server for that pool:

```bash
# 192-skill SkillsBench pool
MCP_TRANSPORT=http MCP_PORT=8002 MCP_EMBED_CACHE=cache/192.npy python mcp_server/skills_mcp.py

# 34K pool
SKILLS_DIR=third_party/skill_usage_34k/skills MCP_TOOLREX_PATH=data/toolrex_profiles_34k.json \
  MCP_TRANSPORT=http MCP_PORT=8002 MCP_EMBED_CACHE=cache/34k.npy python mcp_server/skills_mcp.py
```

The Liu et al. conditions also need that pool's search server, on port 8742 for the 34K pool or 8743 for the 192-skill pool. [liu_baseline/README.md](../liu_baseline/README.md) shows how to start it.

## Run the Agent Sweep

`scripts/launch_sdk_sweep.py` runs each combination of trial, condition, and task, and it queues trial 1 of each condition and task before any trial 2. The conditions are `none`, the server methods (`bm25`, `rerank`, `rerank_qwen0_6b`, `rerank_qwen4b`, `rerank_qwen8b`), and the Liu et al. conditions `liu_hybrid`, `liu_refine`, and `liu_refined`. Because `liu_refined` loads the skills that `liu_refine` writes, run it afterwards with the same `--pool` and `--tag`:

```bash
export OPENROUTER_API_KEY=...
python scripts/launch_sdk_sweep.py --pool 192 --tag qwen192 \
  --conditions none,bm25,rerank,rerank_qwen0_6b,liu_hybrid --parallelism 8
python scripts/launch_sdk_sweep.py --pool 192 --tag qwen192 --conditions liu_refine --parallelism 8
python scripts/launch_sdk_sweep.py --pool 192 --tag qwen192 --conditions liu_refined --parallelism 8
python scripts/aggregate_results.py logs/*-sweep-qwen192
```

The launcher passes any argument it does not recognize to the agent driver, `scripts/run_sdk.py`. The driver's default backbone is Qwen3.5 on OpenRouter (`--model openrouter/qwen/qwen3.5-397b-a17b`). The second backbone in our experiments was `openrouter/minimax/minimax-m2.7`. It sends temperature 1.0, top-p 1.0, top-k 0 (disabled), and reasoning effort `high`, which match the OpenRouter defaults of our runs, and it caps each trial at 30 agent turns and 600 seconds.

Each trial writes `events.jsonl` (each agent turn, tool call, and tool result), `result.json`, and the verifier output under `logs/<timestamp>-sweep-<tag>/`. The reward is the score the task's verifier reports, or the fraction of its tests that pass when it reports no score. A trial that errors or writes no reward counts as 0 in `aggregate_results.py`.

## Measure Retrieval Quality Against Gold Skills

`scripts/eval_retrieval.py` queries a running server and reports R@K, P@K, MAP, and nDCG against gold skills. On the 192-skill pool, the gold skills of a task are the ones SkillsBench ships with it, and the query is the task name. On the 34K pool, both come from Liu et al.'s data:

```bash
python scripts/eval_retrieval.py --pool skillsbench --methods bm25,embed,rerank   # server on the 192-skill pool
python scripts/eval_retrieval.py --pool liu34k --methods bm25,embed,rerank        # server on the 34K pool
```

## Run the Tests

The unit tests use a small fixture pool and the real BGE models. The live tests call each provider whose key is set, check that the answer validates against its Pydantic schema, and check that no reasoning text leaks into it. The driver tests run in the SDK environment:

```bash
pytest
third_party/software-agent-sdk/.venv/bin/python -m pytest tests/test_run_sdk.py
```

Set `VLLM_API_BASE`, and optionally `VLLM_MODEL`, to include a local vLLM server in the live tests.
