"""Liu et al. search server over the 192-skill pool."""

import argparse
import json
import os
import pickle
import re
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI

REPO_ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR = Path(os.environ.get("LIU_192_INDEX_DIR", REPO_ROOT / "third_party/liu_192_index"))
SKILLS_DIR = Path(os.environ.get("SKILLS_DIR", REPO_ROOT / "pools/skillsbench_192"))

app = FastAPI(title="Liu-192 Search")

_state = {}


def _tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 1]


def _load() -> None:
    with (INDEX_DIR / "bm25.pkl").open("rb") as f:
        bm = pickle.load(f)
    _state["bm25"] = bm["bm25"]
    _state["skill_ids"] = bm["skill_ids"]
    _state["embeddings"] = np.load(INDEX_DIR / "embeddings.npy")
    meta = {}
    with (INDEX_DIR / "meta.jsonl").open() as f:
        for line in f:
            r = json.loads(line)
            meta[r["skill_id"]] = r
    _state["meta"] = meta
    from sentence_transformers import SentenceTransformer
    device = os.environ.get("LIU_192_DEVICE", "cuda")
    _state["encoder"] = SentenceTransformer("Qwen/Qwen3-Embedding-4B", device=device)


def _record(skill_id: str, score: float) -> dict:
    m = _state["meta"].get(skill_id, {})
    return {
        "skill_id": skill_id,
        "name": m.get("name", skill_id),
        "description": m.get("description", ""),
        "skill_md_snippet": m.get("snippet", ""),
        "score": float(score),
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "n_skills": len(_state["skill_ids"])}


@app.get("/keyword")
def keyword(q: str, top_k: int = 10) -> dict:
    scores = np.asarray(_state["bm25"].get_scores(_tokenize(q)))
    idx = np.argsort(scores)[::-1][:top_k]
    return {"results": [_record(_state["skill_ids"][int(i)], scores[int(i)]) for i in idx]}


@app.get("/semantic")
def semantic(q: str, top_k: int = 10) -> dict:
    qvec = _state["encoder"].encode([q], normalize_embeddings=True, show_progress_bar=False)[0]
    scores = _state["embeddings"] @ qvec
    idx = np.argsort(scores)[::-1][:top_k]
    return {"results": [_record(_state["skill_ids"][int(i)], scores[int(i)]) for i in idx]}


@app.get("/hybrid")
def hybrid(q: str, top_k: int = 10, keyword_weight: float = 0.5, semantic_weight: float = 0.5) -> dict:
    bm = np.asarray(_state["bm25"].get_scores(_tokenize(q)))
    qvec = _state["encoder"].encode([q], normalize_embeddings=True, show_progress_bar=False)[0]
    sem = _state["embeddings"] @ qvec
    n = len(_state["skill_ids"])
    rank_bm = {int(i): r for r, i in enumerate(np.argsort(bm)[::-1])}
    rank_sem = {int(i): r for r, i in enumerate(np.argsort(sem)[::-1])}
    k_rrf = 60
    rrf = np.zeros(n)
    for i in range(n):
        rrf[i] = keyword_weight / (k_rrf + rank_bm[i]) + semantic_weight / (k_rrf + rank_sem[i])
    idx = np.argsort(rrf)[::-1][:top_k]
    return {"results": [_record(_state["skill_ids"][int(i)], rrf[int(i)]) for i in idx]}


@app.get("/detail/{skill_id}")
def detail(skill_id: str) -> dict:
    md_path = SKILLS_DIR / skill_id / "SKILL.md"
    if not md_path.exists():
        md_path = SKILLS_DIR / skill_id / "skill.md"
    if not md_path.exists():
        return {"error": f"unknown skill_id: {skill_id}"}
    body = md_path.read_text(errors="replace")
    m = _state["meta"].get(skill_id, {})
    return {
        "skill_id": skill_id,
        "name": m.get("name", skill_id),
        "description": m.get("description", ""),
        "skill_md": body,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8743)
    args = p.parse_args()
    _load()
    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
