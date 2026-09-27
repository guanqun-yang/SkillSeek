import json
import os
import sqlite3
import threading

import numpy as np
import requests


DEFAULT_EMBEDDING_MODEL = "qwen/qwen3-embedding-4b"

QUERY_INSTRUCTION = "Find skills matching this query: "

OPENROUTER_URL = "https://openrouter.ai/api/v1/embeddings"


class SemanticSearch:
    def __init__(self, embeddings_path: str, skill_ids_path: str, db_path: str,
                 model_name: str = DEFAULT_EMBEDDING_MODEL,
                 content_embeddings_path: str | None = None,
                 content_weight: float = 0.05):
        self.embeddings = np.load(embeddings_path)
        with open(skill_ids_path) as f:
            self.skill_ids = json.load(f)
        self.db_path = db_path
        self._model_name = model_name
        self._local = threading.local()
        self.content_weight = content_weight

        if content_embeddings_path and os.path.exists(content_embeddings_path):
            self.content_embeddings = np.load(content_embeddings_path)
        else:
            self.content_embeddings = None

        self._api_key = os.environ.get("OPENROUTER_API_KEY")
        if not self._api_key:
            raise RuntimeError("OPENROUTER_API_KEY env var required for query encoding")

    def _get_db(self):
        if not hasattr(self._local, 'db'):
            self._local.db = sqlite3.connect(self.db_path)
            self._local.db.row_factory = sqlite3.Row
        return self._local.db

    def _encode_query(self, text: str) -> np.ndarray:
        resp = requests.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json={"model": self._model_name, "input": [text]},
            timeout=30,
        )
        resp.raise_for_status()
        vec = np.array(resp.json()["data"][0]["embedding"], dtype=np.float32)
        vec /= np.linalg.norm(vec) + 1e-12
        return vec

    def search(self, query: str, top_k: int = 10,
               content_weight: float | None = None) -> list[dict]:
        q_emb = self._encode_query(QUERY_INSTRUCTION + query)

        summary_scores = self.embeddings @ q_emb
        if self.content_embeddings is not None:
            content_scores = self.content_embeddings @ q_emb
            w = content_weight if content_weight is not None else self.content_weight
            scores = (1 - w) * summary_scores + w * content_scores
        else:
            scores = summary_scores
        top_indices = np.argpartition(scores, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        db = self._get_db()
        results = []
        for idx in top_indices:
            row = db.execute(
                """SELECT name, description, skill_md_snippet,
                          skill_id, github_stars
                   FROM skills WHERE rowid = ?""",
                (int(idx),),
            ).fetchone()
            if row:
                r = dict(row)
                r["score"] = float(scores[idx])
                results.append(r)
        return results

    def close(self):
        if hasattr(self._local, 'db'):
            self._local.db.close()
            del self._local.db
