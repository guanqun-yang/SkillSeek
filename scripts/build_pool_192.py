"""Build the 192-skill SkillsBench pool; a skill shared by several tasks keeps the first task's copy."""

import argparse
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", type=Path, default=REPO_ROOT / "third_party/skillsbench/tasks")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "pools/skillsbench_192")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    seen = set()
    for task in sorted(args.tasks.iterdir()):
        skills_dir = task / "environment/skills"
        if not skills_dir.is_dir():
            continue
        for skill in sorted(skills_dir.iterdir()):
            if skill.name in seen or not (skill / "SKILL.md").exists():
                continue
            seen.add(skill.name)
            shutil.copytree(skill, args.out / skill.name, dirs_exist_ok=True)
    print(f"wrote {len(seen)} skills to {args.out}")


if __name__ == "__main__":
    main()
