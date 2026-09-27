---
name: finding-skills
description: Discovers relevant agent skills from a local index of 192 skills for a given task. Breaks complex tasks into sub-tasks and finds 10 skills across keyword, semantic, and hybrid search. Use when starting a new task, looking for specialized capabilities, or wanting to find best practices for a domain.
---

# Finding Skills

Searches a local index of 192 agent skills to find the most relevant ones for a given task. Skills are pre-downloaded — no installation needed.

## When to Use

- Starting a new task that may benefit from specialized skills
- Looking for best practices, patterns, or workflows for a specific domain
- Wanting to find tools or templates for a task (testing, deployment, design, etc.)

## Search API

Three search endpoints are available. Use `curl -s` to query them. The server runs on the Docker host gateway at `172.17.0.1:8743`.

### Keyword search

Best for exact term matching when you know specific skill names or technologies.

```bash
curl -s "http://172.17.0.1:8743/keyword?q=QUERY&top_k=10"
```

Supports FTS5 syntax:
- Prefix: `react*`
- Phrase: `"code review"`
- Boolean: `react OR vue`

### Semantic search

Best for conceptual queries where you describe what you need in natural language.

```bash
curl -s "http://172.17.0.1:8743/semantic?q=QUERY&top_k=10"
```

Example: `q=help me build and deploy containerized applications`

### Hybrid search

Combines keyword and semantic search with reciprocal rank fusion. Best general-purpose option.

```bash
curl -s "http://172.17.0.1:8743/hybrid?q=QUERY&top_k=10&keyword_weight=0.5&semantic_weight=0.5"
```

### Response format

All search endpoints return a JSON array. Each result contains:

- `name`: skill name (may be duplicated across authors)
- `description`: what the skill does
- `skill_md_snippet`: first 100 words of the skill's documentation
- `skill_id`: unique identifier (author--name format), use with the detail endpoint
- `github_stars`: popularity of the source repository
- `score`: relevance score (higher = better for semantic / hybrid; more negative = better for keyword BM25)

### Skill detail

Fetches full SKILL.md content for a skill. Pass the `skill_id` from search results.

```bash
curl -s "http://172.17.0.1:8743/detail/SKILL_ID"
```

The response includes a `skill_md` field with the full skill documentation.

## Workflow

### Step 1: Analyze the task

Break the task into concrete sub-tasks.

### Step 2: Search for each sub-task

For each sub-task, run search queries to find skills. Hybrid search is the best default.

```bash
curl -s "http://172.17.0.1:8743/hybrid?q=convert+PDF+to+Excel&top_k=10"
```

Refine queries if results are too broad or miss the mark.

### Step 3: Review and select

Select the most relevant skills. Prioritize:
- High relevance to the target task
- Higher `github_stars` when multiple skills cover the same topic
- Informative `skill_md_snippet` content

### Step 4: Fetch full content and apply

For the most promising skills, fetch full content via `/detail/SKILL_ID` and read the SKILL.md instructions. Then apply those instructions to solve the task.
