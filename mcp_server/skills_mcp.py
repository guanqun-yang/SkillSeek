"""SkillSeek MCP server: skill_lookup, skill_load, and skill_list over stdio or HTTP."""

import contextvars
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from mcp.server.fastmcp import FastMCP
from rank_bm25 import BM25Okapi


_TRIAL_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar("_TRIAL_ID", default=None)
_METHOD_OVERRIDE: contextvars.ContextVar[str | None] = contextvars.ContextVar("_METHOD_OVERRIDE", default=None)

REPO_ROOT = Path(__file__).resolve().parents[1]
SKILLS_DIR = Path(os.environ.get("SKILLS_DIR", REPO_ROOT / "pools/skillsbench_192"))
LOG_PATH = Path(os.environ.get(
    "MCP_LOG_PATH", REPO_ROOT / f"logs/{time.strftime('%Y%m%d-%H%M%S')}-mcp/calls.jsonl"))
EMBED_MODEL = os.environ.get("MCP_EMBED_MODEL", "BAAI/bge-base-en-v1.5")
EMBED_CACHE = os.environ.get("MCP_EMBED_CACHE")
RERANK_MODEL = os.environ.get("MCP_RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
RERANK_TOPK = int(os.environ.get("MCP_RERANK_TOPK", "20"))
RERANK_DEVICE = os.environ.get("MCP_RERANK_DEVICE", "cpu")
QWEN_RERANKER_MODELS = {
    "rerank_qwen0_6b": "Qwen/Qwen3-Reranker-0.6B",
    "rerank_qwen4b": "Qwen/Qwen3-Reranker-4B",
    "rerank_qwen8b": "Qwen/Qwen3-Reranker-8B",
}
METHODS = ("bm25", "embed", "rerank", *QWEN_RERANKER_MODELS)
TEXT_VARIANT = os.environ.get("MCP_TEXT_VARIANT", "toolrex")
TOOLREX_PATH = os.environ.get("MCP_TOOLREX_PATH", REPO_ROOT / "data/toolrex_profiles_192.json")
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

if TEXT_VARIANT not in ("baseline", "toolrex"):
    raise ValueError(f"MCP_TEXT_VARIANT must be 'baseline' or 'toolrex', got {TEXT_VARIANT!r}")
_TOOLREX: dict[str, dict] = {}
if TEXT_VARIANT == "toolrex":
    _TOOLREX = json.loads(Path(TOOLREX_PATH).read_text())


def build_text(s: dict[str, Any]) -> str:
    base = f"{s['name']}: {s['description']}"
    if TEXT_VARIANT != "toolrex":
        return base
    profile = _TOOLREX.get(s["name"]) or _TOOLREX.get(Path(s["path"]).name) or {}
    parts = []
    if profile.get("function_description"):
        parts.append(f"Function: {profile['function_description']}")
    if isinstance(profile.get("tags"), list):
        parts.append("Tags: " + ", ".join(str(t) for t in profile["tags"]))
    if profile.get("when_to_use"):
        parts.append(f"When to use: {profile['when_to_use']}")
    if profile.get("limitations"):
        parts.append(f"Limitations: {profile['limitations']}")
    return base + "\n\n" + "\n".join(parts) if parts else base


def _tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 1]


def _stringify(x: Any) -> str:
    if isinstance(x, list):
        return " ".join(str(i) for i in x)
    return str(x or "")


def _split_frontmatter(text: str) -> tuple[str, str]:
    if not text.startswith("---"):
        return "", text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return "", text
    return parts[1], parts[2]


class SkillStore:
    def __init__(self, root: Path) -> None:
        self.skills: list[dict[str, Any]] = []
        self._load(root)
        self._by_name = {}
        for s in self.skills:
            self._by_name.setdefault(s["name"], s)
        self._bm25 = BM25Okapi([_tokenize(f"{s['name']} {s['description']}") for s in self.skills])
        self._embedder = None
        self._embeddings = None
        self._rerankers: dict[str, Any] = {}

    def _load(self, root: Path) -> None:
        for d in sorted(root.iterdir()):
            if not d.is_dir():
                continue
            md = d / "SKILL.md"
            if not md.exists():
                md = d / "skill.md"
            if not md.exists():
                continue
            front, body = _split_frontmatter(md.read_text(errors="replace"))
            try:
                meta = yaml.safe_load(front) or {}
            except yaml.YAMLError:
                meta = {}
            if not isinstance(meta, dict):
                meta = {}
            self.skills.append({
                "name": _stringify(meta.get("name") or d.name).strip(),
                "description": _stringify(meta.get("description")).strip(),
                "body": body.strip(),
                "path": str(d),
            })

    def _ensure_embeddings(self) -> None:
        if self._embeddings is not None:
            return
        from sentence_transformers import SentenceTransformer
        self._embedder = SentenceTransformer(EMBED_MODEL)
        if EMBED_CACHE and Path(EMBED_CACHE).exists():
            cached = np.load(EMBED_CACHE)
            if cached.shape[0] == len(self.skills):
                self._embeddings = cached
                return
        texts = [build_text(s) for s in self.skills]
        self._embeddings = self._embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        if EMBED_CACHE:
            Path(EMBED_CACHE).parent.mkdir(parents=True, exist_ok=True)
            np.save(EMBED_CACHE, self._embeddings)

    def _ensure_reranker(self, method: str):
        if method not in self._rerankers:
            from sentence_transformers import CrossEncoder
            model = RERANK_MODEL if method == "rerank" else QWEN_RERANKER_MODELS[method]
            self._rerankers[method] = CrossEncoder(model, device=RERANK_DEVICE)
        return self._rerankers[method]

    def lookup(self, query: str, k: int, method: str) -> list[dict[str, Any]]:
        if method not in METHODS:
            raise ValueError(f"unknown method: {method!r} (expected one of {', '.join(METHODS)})")
        if method == "bm25":
            scores = np.asarray(self._bm25.get_scores(_tokenize(query)))
            idx = np.argsort(scores)[::-1][:k]
            return [self._hit(int(i), scores[int(i)]) for i in idx]
        self._ensure_embeddings()
        qvec = self._embedder.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
        bi_scores = self._embeddings @ qvec
        if method == "embed":
            idx = np.argsort(bi_scores)[::-1][:k]
            return [self._hit(int(i), bi_scores[int(i)]) for i in idx]
        cand_idx = np.argsort(bi_scores)[::-1][:RERANK_TOPK]
        pairs = [(query, build_text(self.skills[int(i)])) for i in cand_idx]
        ce = self._ensure_reranker(method)
        if method == "rerank":
            ce_scores = ce.predict(pairs, show_progress_bar=False)
        else:
            import torch
            ce_scores = ce.predict(pairs, activation_fn=torch.nn.Sigmoid(), show_progress_bar=False)
        order = np.argsort(ce_scores)[::-1][:k]
        return [self._hit(int(cand_idx[int(j)]), ce_scores[int(j)]) for j in order]

    def _hit(self, i: int, score: float) -> dict[str, Any]:
        s = self.skills[i]
        return {"name": s["name"], "description": s["description"], "score": float(score)}

    def get(self, name: str) -> dict[str, Any] | None:
        return self._by_name.get(name)


def _log(record: dict[str, Any]) -> None:
    record.setdefault("trial_id", _TRIAL_ID.get() or os.environ.get("MCP_TRIAL_ID"))
    with LOG_PATH.open("a") as f:
        f.write(json.dumps(record, default=str) + "\n")


store = SkillStore(SKILLS_DIR)
mcp = FastMCP(
    "skills",
    host=os.environ.get("MCP_HOST", "0.0.0.0"),
    port=int(os.environ.get("MCP_PORT", "8002")),
)


@mcp.tool()
def skill_list() -> str:
    """List every available skill as `name: description` lines."""
    t0 = time.time()
    out = "\n".join(f"- {s['name']}: {s['description']}" for s in store.skills)
    _log({"ts": time.time(), "tool": "skill_list", "n_returned": len(store.skills), "latency_ms": (time.time() - t0) * 1000})
    return out


@mcp.tool()
def skill_lookup(query: str, k: int = 3, method: str = "bm25") -> str:
    """Return the top-k skills (name + description + score) for `query`.

    method: "bm25" (default, no model) or "embed" (BGE-base-en-v1.5).
    """
    t0 = time.time()
    override = _METHOD_OVERRIDE.get()
    effective = override or method
    hits = store.lookup(query, k, effective)
    _log({
        "ts": time.time(),
        "tool": "skill_lookup",
        "query": query,
        "k": k,
        "method": effective,
        "method_override_applied": override is not None and override != method,
        "returned": [{"name": h["name"], "score": h["score"]} for h in hits],
        "latency_ms": (time.time() - t0) * 1000,
    })
    return "\n".join(f"- {h['name']} (score={h['score']:.3f}): {h['description']}" for h in hits)


@mcp.tool()
def skill_load(name: str) -> str:
    """Return the full SKILL.md body for `name` (no truncation)."""
    t0 = time.time()
    s = store.get(name)
    body = s["body"] if s else f"skill not found: {name}"
    _log({"ts": time.time(), "tool": "skill_load", "name": name, "found": s is not None, "n_chars": len(body), "latency_ms": (time.time() - t0) * 1000})
    return body


if __name__ == "__main__":
    if os.environ.get("MCP_TRANSPORT", "stdio") == "http":
        import uvicorn
        from starlette.middleware.base import BaseHTTPMiddleware

        class TrialIdMiddleware(BaseHTTPMiddleware):
            async def dispatch(self, request, call_next):
                t1 = _TRIAL_ID.set(request.headers.get("x-trial-id") or request.query_params.get("trial"))
                t2 = _METHOD_OVERRIDE.set(request.headers.get("x-skill-method") or request.query_params.get("method"))
                try:
                    return await call_next(request)
                finally:
                    _METHOD_OVERRIDE.reset(t2)
                    _TRIAL_ID.reset(t1)

        app = mcp.streamable_http_app()
        app.add_middleware(TrialIdMiddleware)
        uvicorn.run(app, host=os.environ.get("MCP_HOST", "0.0.0.0"),
                    port=int(os.environ.get("MCP_PORT", "8002")), log_level="info")
    else:
        mcp.run()
