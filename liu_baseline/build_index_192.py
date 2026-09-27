"""Build the BM25 and Qwen3-Embedding-4B indexes of the Liu et al. search server over the 192-skill pool."""

import json
import os
import pickle
import re
from pathlib import Path

import numpy as np
import yaml
from rank_bm25 import BM25Okapi

REPO_ROOT = Path(__file__).resolve().parents[1]
SKILLS_DIR = Path(os.environ.get("SKILLS_DIR", REPO_ROOT / "pools/skillsbench_192"))
OUT_DIR = Path(os.environ.get("LIU_192_INDEX_DIR", REPO_ROOT / "third_party/liu_192_index"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
SNIPPET_CHARS = 600
BODY_CHARS_FOR_EMBED = 4000


def _split_frontmatter(text: str) -> tuple[str, str]:
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end > 0:
            return text[3:end], text[end + 4 :]
    return "", text


def _tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 1]


def load_skills() -> list[dict]:
    records = []
    for d in sorted(SKILLS_DIR.iterdir()):
        md = d / "SKILL.md"
        if not md.exists():
            md = d / "skill.md"
        if not md.exists():
            continue
        text = md.read_text(errors="replace")
        front, body = _split_frontmatter(text)
        try:
            meta = yaml.safe_load(front) or {}
        except yaml.YAMLError:
            meta = {}
        name = meta.get("name") or d.name
        desc = meta.get("description", "")
        records.append({
            "skill_id": d.name,
            "name": name,
            "description": desc,
            "body": body,
            "snippet": body[:SNIPPET_CHARS],
        })
    return records


def main() -> None:
    print(f"Loading skills from {SKILLS_DIR}")
    records = load_skills()
    print(f"  loaded {len(records)} skills")

    meta_path = OUT_DIR / "meta.jsonl"
    with meta_path.open("w") as f:
        for r in records:
            f.write(json.dumps({
                "skill_id": r["skill_id"],
                "name": r["name"],
                "description": r["description"],
                "snippet": r["snippet"],
                "body_chars": len(r["body"]),
            }) + "\n")
    print(f"  wrote {meta_path}")

    corpus_texts = [f"{r['name']} {r['description']} {r['body'][:BODY_CHARS_FOR_EMBED]}" for r in records]
    tokenized = [_tokenize(t) for t in corpus_texts]
    bm25 = BM25Okapi(tokenized)
    with (OUT_DIR / "bm25.pkl").open("wb") as f:
        pickle.dump({"bm25": bm25, "tokenized": tokenized, "skill_ids": [r["skill_id"] for r in records]}, f)
    print(f"  wrote bm25.pkl")

    from sentence_transformers import SentenceTransformer
    device = os.environ.get("LIU_192_DEVICE", "cuda")
    print(f"  loading Qwen/Qwen3-Embedding-4B on {device}")
    model = SentenceTransformer("Qwen/Qwen3-Embedding-4B", device=device)
    embs = model.encode(corpus_texts, normalize_embeddings=True, show_progress_bar=True, batch_size=8)
    embs = embs.astype(np.float32)
    np.save(OUT_DIR / "embeddings.npy", embs)
    print(f"  wrote embeddings.npy with shape {embs.shape}")
    print("done.")


if __name__ == "__main__":
    main()
