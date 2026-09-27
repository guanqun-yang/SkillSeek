import json
from argparse import Namespace
from pathlib import Path

import pytest

import aggregate_results as agg
import eval_retrieval as ev
import launch_sdk_sweep as sweep


def test_metrics():
    ranked, gold = ["a", "x", "b", "y", "z"], {"a", "b", "c"}
    assert ev.precision_at_k(ranked, gold, 5) == pytest.approx(2 / 5)
    assert ev.recall_at_k(ranked, gold, 5) == pytest.approx(2 / 3)
    assert ev.recall_at_k(ranked, gold, 1) == pytest.approx(1 / 3)
    assert ev.average_precision(ranked, gold) == pytest.approx((1 / 1 + 2 / 3) / 3)
    ideal = 1 + 1 / 1.584962500721156 + 1 / 2
    assert ev.ndcg_at_k(ranked, gold, 5) == pytest.approx((1 + 1 / 2) / ideal)


def test_load_gold_skillsbench(tmp_path):
    for task, skills in {"pdf-diff": ["pdf", "xlsx"], "no-skills": [], "video-cut": ["ffmpeg"]}.items():
        (tmp_path / task / "environment/skills").mkdir(parents=True)
        for s in skills:
            (tmp_path / task / "environment/skills" / s).mkdir()
    gold, queries = ev.load_gold("skillsbench", tmp_path, tmp_path)
    assert gold == {"pdf-diff": {"pdf", "xlsx"}, "video-cut": {"ffmpeg"}}
    assert queries == {"pdf-diff": "pdf diff", "video-cut": "video cut"}
    row = ev.score({"pdf-diff": ["pdf", "zip"], "video-cut": ["gif", "ffmpeg"]}, gold, [1, 2])
    assert row["R@1"] == pytest.approx(0.25) and row["R@2"] == pytest.approx(0.75)


def _write_trial(sweep_dir: Path, trial_id: str, reward, error=None, cost=0.1):
    d = sweep_dir / trial_id
    d.mkdir(parents=True)
    (d / "result.json").write_text(json.dumps({"trial_id": trial_id, "reward": reward, "error": error, "cost": cost}))


def test_aggregate(tmp_path, capsys, monkeypatch):
    _write_trial(tmp_path, "qwen-192-rerank-task-a-t1", 1.0)
    _write_trial(tmp_path, "qwen-192-rerank-task-b-t1", None, error="TimeoutError: ...")
    _write_trial(tmp_path, "qwen-192-none-task-a-t1", 0.5)
    grouped = agg.collect([tmp_path])
    assert set(grouped) == {("rerank", 1), ("none", 1)}
    assert set(grouped[("rerank", 1)]) == {"task-a", "task-b"}
    monkeypatch.setattr("sys.argv", ["aggregate_results.py", str(tmp_path), "--n-tasks", "4"])
    agg.main()
    lines = {line.split()[0]: line.split() for line in capsys.readouterr().out.splitlines()[1:]}
    assert lines["rerank"][2:5] == ["2", "1", "0.250"]
    assert lines["none"][4] == "0.125"


def _sweep_args(tmp_path, pool):
    return Namespace(pool=pool, mcp_url="http://172.17.0.1:8002/mcp", liu_refined_dir=tmp_path / "refined")


@pytest.mark.parametrize("pool,port,size", [("192", "8743", "192"), ("34k", "8742", "34000")])
def test_liu_conditions(tmp_path, pool, port, size):
    args = _sweep_args(tmp_path, pool)
    for cond in ("liu_hybrid", "liu_refine", "liu_refined"):
        argv = sweep.condition_args(cond, "task-a", args)
        joined = " ".join(argv)
        assert f"--find-skills-url http://172.17.0.1:{port}" in joined
        assert f"--find-skills-pool-size {size}" in joined
        assert str(sweep.LIU[pool]["skills"]) in joined
        other = "8742" if port == "8743" else "8743"
        assert other not in joined
        skill_md = (sweep.LIU[pool]["skills"] / "finding-skills/SKILL.md").read_text()
        assert f"172.17.0.1:{port}" in skill_md and f"172.17.0.1:{other}" not in skill_md
    refine = sweep.condition_args("liu_refine", "task-a", args)
    prompt = refine[refine.index("--instruction-override") + 1]
    assert f"http://172.17.0.1:{port}/hybrid" in prompt and "<LIU_URL>" not in prompt
    assert refine[refine.index("--refined-skills-out") + 1] == str(tmp_path / "refined/task-a")


def test_liu_refined_fallback(tmp_path):
    args = _sweep_args(tmp_path, "192")
    assert "--nudge-find-skills" in sweep.condition_args("liu_refined", "task-a", args)
    (tmp_path / "refined/task-a/tip").mkdir(parents=True)
    assert sweep.condition_args("liu_refined", "task-a", args) == ["--skills-dir", str(tmp_path / "refined/task-a")]


def test_mcp_conditions(tmp_path):
    args = _sweep_args(tmp_path, "34k")
    assert sweep.condition_args("none", "t", args) == []
    for method in sweep.MCP_METHODS:
        assert sweep.condition_args(method, "t", args) == [
            "--mcp-url", "http://172.17.0.1:8002/mcp", "--nudge-mcp", "--mcp-method", method]
