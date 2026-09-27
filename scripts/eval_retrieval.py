"""Retrieval quality (R@K, P@K, MAP, nDCG@K) of a running SkillSeek server against gold skills."""

import argparse
import asyncio
import csv
import json
import math
import statistics
import time
from pathlib import Path

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_gold(pool: str, tasks_dir: Path, liu_data: Path) -> tuple[dict[str, set[str]], dict[str, str]]:
    if pool == "skillsbench":
        gold = {}
        for d in sorted(tasks_dir.iterdir()):
            sd = d / "environment/skills"
            names = {x.name for x in sd.iterdir() if x.is_dir()} if sd.is_dir() else set()
            if names:
                gold[d.name] = names
        return gold, {t: t.replace("-", " ") for t in gold}
    gold = {k: set(v) for k, v in json.loads((liu_data / "task_skill_mapping.json").read_text()).items()}
    raw = json.loads((liu_data / "task_queries.json").read_text())
    return gold, {k: " ".join(v) if isinstance(v, list) else str(v) for k, v in raw.items()}


async def lookup(mcp_url: str, query: str, k: int, method: str) -> list[str]:
    timeout = httpx.Timeout(30, read=300)
    async with httpx.AsyncClient(headers={"X-Skill-Method": method}, timeout=timeout) as client:
        async with streamable_http_client(mcp_url, http_client=client) as (r, w, _):
            async with ClientSession(r, w) as s:
                await s.initialize()
                res = await s.call_tool("skill_lookup", {"query": query, "k": k})
    return [line[2:].split(" (score=")[0] for line in res.content[0].text.splitlines() if line.startswith("- ")]


def precision_at_k(ranked, gold, k):
    return len(set(ranked[:k]) & gold) / k


def recall_at_k(ranked, gold, k):
    return len(set(ranked[:k]) & gold) / len(gold)


def average_precision(ranked, gold):
    hits, total = 0, 0.0
    for i, item in enumerate(ranked, 1):
        if item in gold:
            hits += 1
            total += hits / i
    return total / len(gold)


def ndcg_at_k(ranked, gold, k):
    dcg = sum(1.0 / math.log2(i + 1) for i, item in enumerate(ranked[:k], 1) if item in gold)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(gold), k) + 1))
    return dcg / ideal


def score(ranked_by_task: dict[str, list[str]], gold: dict[str, set[str]], ks: list[int]) -> dict:
    tasks = sorted(ranked_by_task)
    row = {}
    for k in ks:
        row[f"P@{k}"] = round(statistics.mean(precision_at_k(ranked_by_task[t], gold[t], k) for t in tasks), 4)
        row[f"R@{k}"] = round(statistics.mean(recall_at_k(ranked_by_task[t], gold[t], k) for t in tasks), 4)
    row["MAP"] = round(statistics.mean(average_precision(ranked_by_task[t], gold[t]) for t in tasks), 4)
    row[f"nDCG@{max(ks)}"] = round(statistics.mean(ndcg_at_k(ranked_by_task[t], gold[t], max(ks)) for t in tasks), 4)
    return row


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pool", choices=["skillsbench", "liu34k"], required=True)
    p.add_argument("--methods", default="bm25,embed,rerank")
    p.add_argument("--mcp-url", default="http://127.0.0.1:8002/mcp")
    p.add_argument("--k", default="5,10")
    p.add_argument("--tasks-dir", type=Path, default=REPO_ROOT / "third_party/skillsbench/tasks")
    p.add_argument("--liu-data", type=Path, default=REPO_ROOT / "third_party/Skill-Usage/data")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    gold, queries = load_gold(args.pool, args.tasks_dir, args.liu_data)
    ks = [int(x) for x in args.k.split(",")]
    out = args.out or REPO_ROOT / f"logs/{time.strftime('%Y%m%d-%H%M%S')}-eval-retrieval/{args.pool}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for method in [m.strip() for m in args.methods.split(",") if m.strip()]:
        t0 = time.time()
        ranked = {t: asyncio.run(lookup(args.mcp_url, queries[t], max(ks), method)) for t in sorted(gold) if t in queries}
        row = {"method": method, "pool": args.pool, "n_tasks": len(ranked), "wall_s": round(time.time() - t0, 1),
               **score(ranked, gold, ks)}
        rows.append(row)
        print(row, flush=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
