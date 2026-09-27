# Configuring SkillSeek

Environment variables configure the MCP server, and command-line flags configure the Tool-REX profile builder. The defaults reproduce the setup used in our experiments.

## The Server Reads Its Settings From Environment Variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `SKILLS_DIR` | `pools/skillsbench_192` | Folder of `<skill>/SKILL.md` directories to index |
| `MCP_TEXT_VARIANT` | `toolrex` | `toolrex` adds the profile to name and description, and `baseline` indexes name and description only |
| `MCP_TOOLREX_PATH` | `data/toolrex_profiles_192.json` | Profiles, keyed by skill name or folder name |
| `MCP_EMBED_CACHE` | unset | `.npy` file that stores the bi-encoder embeddings after the first lookup |
| `MCP_EMBED_MODEL` | `BAAI/bge-base-en-v1.5` | Bi-encoder |
| `MCP_RERANK_MODEL` | `BAAI/bge-reranker-v2-m3` | Cross-encoder used by `rerank` |
| `MCP_RERANK_TOPK` | `20` | Bi-encoder candidates passed to the reranker |
| `MCP_RERANK_DEVICE` | `cpu` | Reranker device (`cuda` suits the Qwen3 rerankers) |
| `MCP_TRANSPORT` | `stdio` | `stdio` or `http` |
| `MCP_HOST`, `MCP_PORT` | `0.0.0.0`, `8002` | HTTP address |
| `MCP_LOG_PATH` | `logs/<timestamp>-mcp/calls.jsonl` | One JSON line per tool call, with the query, method, and returned skills |

A skill without a profile is indexed by name and description alone. For a pool of tens of thousands of skills, set `MCP_EMBED_CACHE` so that later starts skip the encoding step.

## The Ranking Method Can Change per Call

`skill_lookup` takes a `method` argument, and over HTTP an `X-Skill-Method` header overrides it on each request. Experiments use the header to switch rankers while the agent's tool description stays the same. The methods are:

- `rerank`, the SkillSeek default: the bi-encoder's top candidates, reordered by bge-reranker-v2-m3.
- `rerank_qwen0_6b`, `rerank_qwen4b`, and `rerank_qwen8b`: the same candidates, reordered by Qwen3-Reranker-0.6B, 4B, or 8B.
- `embed`: bi-encoder similarity alone.
- `bm25`: BM25 over name and description, with no model.

An `X-Trial-ID` header, when present, is copied into each log line.

## The Profile Builder Accepts Any LiteLLM Model

`scripts/build_toolrex_profiles.py` sends the body of each `SKILL.md` to an LLM, validates the answer against a Pydantic schema (a function description and exactly four tags, plus optional when-to-use and limitations fields), and writes the profiles keyed by folder name. It keeps profiles already in `--out`, so an interrupted run resumes where it stopped.

`--model` takes a LiteLLM model string: `openrouter/...`, `anthropic/...`, `openai/...`, `gemini/...`, `moonshot/...` for Kimi, `zai/...`, `minimax/...`, or `hosted_vllm/...` together with `--api-base` for a local vLLM server. The flags `--temperature`, `--top-p`, `--top-k`, `--reasoning` or `--no-reasoning`, `--reasoning-strength`, `--max-context-length`, and `--max-output-tokens` are always sent explicitly. Models without a reasoning setting, such as MiniMax-M2, receive none, and the log records each request as sent. `--concurrency` sets the number of parallel requests, which should stay under the provider's rate limit.

The profiles in `data/` came from gpt-oss-120b at temperature 0, and the builder uses that model and temperature by default. Each run logs its configuration and each call, with prompt, answer, and reasoning trace in separate fields, under `logs/<timestamp>-toolrex/`.
