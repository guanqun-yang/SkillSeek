"""Generate Tool-REX v3 profiles for the skills in a pool."""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm import LLMConfig, agenerate, fit_to_context, parse_structured  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
MAX_BODY_CHARS = 8000

PROMPT = """You are extracting structured metadata for a SKILL.md document so it can be retrieved by a search system. Return a single JSON object with these keys, grounded ONLY in the SKILL.md content (no fabrication; omit any optional key whose answer is not in the source):

REQUIRED keys:
- "function_description": one sentence, max 20 words. State (a) what file type or domain object the skill operates on and (b) what specific operation it performs. Use literal nouns, not marketing language.
- "tags": list of exactly 4 short tags (1-3 words each), all lowercase, ordered by query relevance: tag 1 = the primary file/format type the skill operates on (e.g., "pdf", "xlsx", "video"); tag 2 = the primary user-facing operation (e.g., "fill form", "extract text", "merge documents"); tags 3-4 = secondary operations or domain terms a user might query with. Do NOT include library names (no "pymupdf", "ffmpeg", "pandas"). Do NOT paraphrase the function_description.

OPTIONAL keys (include ONLY if the SKILL.md explicitly addresses them):
- "when_to_use": one sentence, max 20 words, the trigger scenario for invoking this skill, using user-facing terms.
- "limitations": one sentence, what the skill cannot do or assumes about its input.

Output a JSON object only, parseable by json.loads(). No prose preface, no <think>, no code fence, no trailing commentary.

SKILL.md content:
{body}

JSON:"""

RETRY_SUFFIX = "\n\nIMPORTANT: output ONLY a JSON object with the required keys, no prose."


class ToolRexProfile(BaseModel):
    function_description: str
    tags: list[str] = Field(min_length=4, max_length=4)
    when_to_use: str | None = None
    limitations: str | None = None


def strip_frontmatter(text: str) -> str:
    if not text.startswith("---"):
        return text
    parts = text.split("---", 2)
    return parts[2] if len(parts) >= 3 else text


def load_pool(pool: Path) -> list[tuple[str, str]]:
    skills = []
    for d in sorted(pool.iterdir()):
        md = d / "SKILL.md"
        if d.is_dir() and md.exists():
            skills.append((d.name, strip_frontmatter(md.read_text(errors="replace"))))
    return skills


async def profile_one(config: LLMConfig, name: str, body: str, log) -> dict:
    body = fit_to_context(config, PROMPT + RETRY_SUFFIX, body[:MAX_BODY_CHARS])
    prompt = PROMPT.format(body=body)
    for attempt in range(2):
        record = {"ts": time.time(), "skill": name, "attempt": attempt, "prompt": prompt}
        try:
            result = await agenerate(config, prompt)
            record.update(result.model_dump())
            profile = parse_structured(result.output, ToolRexProfile)
            record["profile"] = profile.model_dump(exclude_none=True)
            log(record)
            return record["profile"]
        except Exception as e:
            record["error"] = f"{type(e).__name__}: {e}"
        log(record)
        prompt = prompt + RETRY_SUFFIX
    return {}


async def run(args, config: LLMConfig) -> None:
    out_path = Path(args.out)
    profiles = json.loads(out_path.read_text()) if out_path.exists() else {}
    todo = [(n, b) for n, b in load_pool(Path(args.pool)) if n not in profiles]
    log_dir = REPO_ROOT / f"logs/{time.strftime('%Y%m%d-%H%M%S')}-toolrex"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "config.json").write_text(json.dumps({**vars(args), **config.model_dump()}, indent=2, default=str))
    log_file = (log_dir / "calls.jsonl").open("a")

    def log(record: dict) -> None:
        log_file.write(json.dumps(record, default=str) + "\n")
        log_file.flush()

    print(f"[toolrex] {len(profiles)} cached, {len(todo)} to generate; log={log_dir}")
    sem = asyncio.Semaphore(args.concurrency)

    async def bounded(name: str, body: str):
        async with sem:
            return name, await profile_one(config, name, body, log)

    failures = 0
    for i, fut in enumerate(asyncio.as_completed([bounded(n, b) for n, b in todo]), 1):
        name, profile = await fut
        profiles[name] = profile
        failures += not profile
        if i % args.save_every == 0 or i == len(todo):
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(profiles, indent=2))
            print(f"[toolrex] {i}/{len(todo)} done, {failures} failed")
    log_file.close()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pool", required=True, help="Directory of <skill>/SKILL.md folders")
    p.add_argument("--out", required=True, help="Output JSON, keyed by skill directory name")
    p.add_argument("--model", default="openrouter/openai/gpt-oss-120b", help="LiteLLM model string")
    p.add_argument("--api-base", default=None, help="Endpoint for self-hosted models, e.g. http://localhost:8000/v1")
    p.add_argument("--max-context-length", type=int, default=131072)
    p.add_argument("--max-output-tokens", type=int, default=1500)
    p.add_argument("--reasoning", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--reasoning-strength", choices=["minimal", "low", "medium", "high"], default="medium")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=1)
    p.add_argument("--concurrency", type=int, default=16, help="Parallel requests; keep under the provider rate limit")
    p.add_argument("--save-every", type=int, default=64)
    args = p.parse_args()
    config = LLMConfig(
        model=args.model,
        api_base=args.api_base,
        max_context_length=args.max_context_length,
        max_output_tokens=args.max_output_tokens,
        reasoning=args.reasoning,
        reasoning_strength=args.reasoning_strength,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )
    asyncio.run(run(args, config))


if __name__ == "__main__":
    main()
