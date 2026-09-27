"""Mean reward per condition from sweep directories; missing or failed trials count as 0."""

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

TRIAL_ID = re.compile(r"^(?:.+?-)?(?P<cond>none|bm25|rerank(?:_qwen(?:0_6b|4b|8b))?|liu_hybrid|liu_refine|liu_refined)-(?P<task>.+)-t(?P<trial>\d+)$")


def collect(dirs: list[Path]) -> dict[tuple[str, int], dict[str, dict]]:
    out: dict[tuple[str, int], dict[str, dict]] = defaultdict(dict)
    for d in dirs:
        for rj in sorted(d.glob("*/result.json")):
            result = json.loads(rj.read_text())
            m = TRIAL_ID.match(result["trial_id"])
            if m:
                out[(m["cond"], int(m["trial"]))][m["task"]] = result
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sweep_dirs", nargs="+", type=Path)
    p.add_argument("--n-tasks", type=int, default=89)
    args = p.parse_args()

    grouped = collect(args.sweep_dirs)
    print(f"{'condition':18s} {'trial':>5s} {'n_result':>8s} {'n_error':>7s} {'mean_reward':>11s} {'cost_usd':>9s}")
    for (cond, trial), by_task in sorted(grouped.items()):
        rewards = [float(r["reward"] or 0.0) for r in by_task.values()]
        rewards += [0.0] * (args.n_tasks - len(rewards))
        n_error = sum(1 for r in by_task.values() if r.get("error"))
        cost = sum(r.get("cost") or 0.0 for r in by_task.values())
        print(f"{cond:18s} {trial:>5d} {len(by_task):>8d} {n_error:>7d} {statistics.mean(rewards):>11.3f} {cost:>9.2f}")


if __name__ == "__main__":
    main()
