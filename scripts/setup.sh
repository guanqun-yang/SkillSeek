#!/bin/bash
# Fetch the benchmarks, skill pools, and pinned OpenHands SDK. SKIP_34K=1 skips the 34K pool (about 3 GB).
set -e
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO"
mkdir -p third_party pools

clone() {
  if [ -d "$1/.git" ]; then echo "[skip] $1"; return; fi
  git clone -q "$2" "$1"
  git -C "$1" checkout -q "$3"
  echo "[clone] $1 @ ${3:0:7}"
}

clone third_party/skillsbench        https://github.com/benchflow-ai/skillsbench.git      a8e53eab028d88669cd8e3722ec1200d181e35a9
clone third_party/Skill-Usage        https://github.com/UCSB-NLP-Chang/Skill-Usage.git    03446d16f7b659ccc93ac5bd512f62e9b7fabb45
clone third_party/software-agent-sdk https://github.com/OpenHands/software-agent-sdk.git  3e5a6e644854c2e485446ad380c8fff3ccf42035
cp scripts/openhands-sdk-uv.lock third_party/software-agent-sdk/uv.lock
(cd third_party/software-agent-sdk && uv sync -q --frozen)

if [ -z "$SKIP_34K" ] && [ ! -d third_party/skill_usage_34k/skills ]; then
  mkdir -p third_party/skill_usage_34k
  cd third_party/skill_usage_34k
  hf download Shiyu-Lab/Skill-Usage skills-34k/skills.zip search_index/search_index.zip \
    --repo-type dataset --local-dir .
  unzip -q skills-34k/skills.zip -d skills
  unzip -q search_index/search_index.zip -d search_index
  cd "$REPO"
fi

python scripts/build_pool_192.py --tasks third_party/skillsbench/tasks --out pools/skillsbench_192
