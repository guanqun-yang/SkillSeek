# Liu et al. (2026) Baseline

This directory reimplements the LLM-mediated skill retrieval of Liu et al., *How Well Do Agentic Skills Work in the Wild: Benchmarking LLM Skill Usage in Realistic Settings* ([arXiv:2604.04323](https://arxiv.org/abs/2604.04323)), as used for the `liu_hybrid` and `liu_refined` conditions in the SkillSeek experiments.

| File | Role |
| --- | --- |
| `http_server.py`, `bm25_search.py`, `semantic_search.py` | Search server over the 34K pool (BM25 plus Qwen3-Embedding-4B, fused by reciprocal rank), reading Liu et al.'s prebuilt index. Serves `/keyword`, `/semantic`, `/hybrid`, `/detail/<id>`. |
| `build_index_192.py`, `serve_192.py` | The same server design rebuilt over the 192-skill SkillsBench pool. |
| `skills/{34k,192}/finding-skills/SKILL.md` | The `finding-skills` skill that tells the agent how to query the server for each pool. |

The refinement prompt for `liu_refine` lives in `scripts/launch_sdk_sweep.py`.

## Running

```bash
# 34K pool on port 8742 (query encoding goes through OpenRouter's Qwen3-Embedding-4B)
OPENROUTER_API_KEY=... python liu_baseline/http_server.py --port 8742

# 192 pool on port 8743 (index build and query encoding run Qwen3-Embedding-4B locally)
LIU_192_DEVICE=cuda:0 python liu_baseline/build_index_192.py
LIU_192_DEVICE=cuda:0 python liu_baseline/serve_192.py --port 8743
```

Both ports are fixed in the matching `finding-skills/SKILL.md`, which reaches the host through the Docker bridge address `172.17.0.1`.

## Attribution

`http_server.py`, `bm25_search.py`, `semantic_search.py`, and both `finding-skills/SKILL.md` files are adapted from [UCSB-NLP-Chang/Skill-Usage](https://github.com/UCSB-NLP-Chang/Skill-Usage) at commit `03446d1`, released under the MIT License. Our changes: query embeddings come from OpenRouter instead of a local model, index and skill paths are configurable, and the skill text points at the Docker host address and states the pool size.

```
MIT License

Copyright (c) 2026 the Skill-Usage authors (Liu et al.)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
