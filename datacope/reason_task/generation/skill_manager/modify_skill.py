"""
Parallel skill modification: refine existing skills based on iteration comparison results.
"""

import argparse
import json
import threading
from pathlib import Path
from pydantic import BaseModel

import anyio
import anyio.to_thread
from .expert_model import SkillGeneratorAgent, skill_generator_modify_system_prompt
import os

os.environ["CLAUDE_CODE_FILE_READ_MAX_OUTPUT_TOKENS"] = "40000"

# ── Constants ─────────────────────────────────────────────────────────────────

SKILL_GENERATE_PROMPT = '''\
# Objective:
Please modify a Skill for questions of the {category} category in the {task} \
dataset to help models effectively solve this type of problem.

# Role
You are a Skill Refinement Agent. You must analyze trajectories, identify which behaviors help or hurt performance, and convert the findings into an improved Skill.

# Inputs
- Dataset Directory: `@{data_dir}`
- Trajectory Directory: `@{traj_dir}`
- Target Task: `{task}`
- Target Category: `{category}`
- Output Skill Directory: `{skill_dir}`

# Terminology for Trajectory Analysis:
The trajectories are grouped by changes in Answer.
1. For different tasks, trajectories with different answers are collected in the folders named Answer1, Answer2, Answer 3 and so on. Each answer folder contains one trajectory which is the most representative trajectory among all trajectories with the same answer.
2. Each task has summary.json file which contains the self-consistency of each answer. You should analyze the self-consistency of different answers and compare the trajectories with different answers to find out what leads to the change of answer.
3. We provide trajectory information before and after the iteration, including their answer clustering and self-consistency information. Changes in answer clustering and self-consistency may occur either because correct information has been found or because a wrong pattern has been fitted. You need to carefully analyze and discover reusable successful experiences or patterns.
4. The trajectories currently provided are those that have undergone significant changes after adding the current Skill. They may be moving in a positive direction or a negative direction. You need to carefully diagnose the current Skill and the trajectories to ensure that the information in the Skill is consistent with the data file and can provide reasonable solutions.

# Important:
1. Before refining the Skill, you should actively explore the relevant dataset files available under the dataset directory and referenced by the exploration trajectories.
2. Answers and Self-Consistency do not guarantee correctness! You should analyze every trajectory, corresponding self-consistency and related files to find out the effective solution strategies for the {category} category.
3. Ensure that the final Skill is consistent with the conditions in the problem and the information in the data file, rather than originating from conjecture or assumption.
4. The provided SKILL is not completely correct. If there is any conflict with the current data or trajectories, please carefully analyze and make corrections.

# Note:
1. Self-Consistency does not necessarily correlate with correctness, but it can provide insights into the reliability of the trajectories.
2. You should analyze every trajectory and the corresponding self-consistency to find out the effective solution strategies for the {category} category. Your goal is to maximize the performance of the refined Skill and make the self-consistency as high as possible.
3. Summarize the effective solution strategies, and modify the existing Skill guide based on your analysis. Save the modified Skill into the {skill_dir} directory.

# Output:
Modify the Skill in {skill_dir} directory. \
Don\'t save it in ".claude/skills" or any other directory.
'''

# ── Helpers ───────────────────────────────────────────────────────────────────

def _log(msg: str, lock: threading.Lock) -> None:
    with lock:
        print(msg, flush=True)


# ── Per-category async pipeline ───────────────────────────────────────────────

async def process_category(
    category: str,
    skill_generator_agent: SkillGeneratorAgent,
    task_name: str,
    traj_base_dir: str,
    skill_output_dir: str,
    data_dir: str,
    print_lock: threading.Lock,
) -> tuple[str, str]:
    skill_dir = os.path.join(skill_output_dir, category)
    Path(skill_dir).mkdir(parents=True, exist_ok=True)

    traj_dir = f"{traj_base_dir}/{category}"

    prompt = SKILL_GENERATE_PROMPT.format(
        category=category,
        task=task_name,
        traj_dir=traj_dir,
        skill_dir=skill_dir,
        data_dir=data_dir,
    )

    _log(f"\n[{category}] Starting skill refinement ...", print_lock)
    result = await skill_generator_agent.run_async(prompt)
    _log(f"\n[{category}] Skill refinement complete.", print_lock)

    return category, result


# ── Main ──────────────────────────────────────────────────────────────────────

async def main_async(
    task_name: str = "dabstep",
    traj_base_dir: str = "",
    skill_output_dir: str = "",
    data_dir: str = "",
    target_categories: list[str] | None = None,
    max_concurrent: int = 3,
) -> None:
    if not traj_base_dir:
        traj_base_dir = str(Path(__file__).parent / "workspace" / "dabstep")
    if not skill_output_dir:
        skill_output_dir = str(Path(__file__).parent / "current_skills")
    if not data_dir:
        data_dir = str(Path(__file__).parent / "workspace" / "data")

    if target_categories is None:
        target_categories = sorted(
            p.name for p in Path(traj_base_dir).iterdir() if p.is_dir()
        )
    print(f"\nCategories to process ({len(target_categories)}): {target_categories}")

    class ClaudeCodeResponse(BaseModel):
        generated_skill: str
        reasoning: str

    skill_generator_agent = SkillGeneratorAgent(
        cwd=str(Path(__file__).parent),
        allowed_tools=[
            "Read", "Write", "Bash", "Glob", "Grep", "Edit",
            "TodoWrite", "BashOutput", "Skill",
        ],
        permission_mode="bypassPermissions",
        output_format={
            "type": "json_schema",
            "schema": ClaudeCodeResponse.model_json_schema(),
        },
        system_prompt=skill_generator_modify_system_prompt,
    )

    print_lock = threading.Lock()
    all_results: dict[str, str] = {}
    semaphore = anyio.Semaphore(max_concurrent)

    async def run_one(category: str) -> None:
        async with semaphore:
            cat, result = await process_category(
                category=category,
                skill_generator_agent=skill_generator_agent,
                task_name=task_name,
                traj_base_dir=traj_base_dir,
                skill_output_dir=skill_output_dir,
                data_dir=data_dir,
                print_lock=print_lock,
            )
            all_results[cat] = result

    async with anyio.create_task_group() as tg:
        for category in target_categories:
            tg.start_soon(run_one, category)

    print(f"\n{'='*60}")
    print(f"All {len(all_results)} categories processed.")
    for cat, res in sorted(all_results.items()):
        preview = (res or "")[:120].replace("\n", " ")
        print(f"  [{cat}] {preview}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Modify/refine existing skills")
    parser.add_argument("--task-name", default="dabstep", help="Task name")
    parser.add_argument("--traj-base-dir", default="", help="Trajectory base directory")
    parser.add_argument("--skill-output-dir", default="", help="Skill output directory")
    parser.add_argument("--data-dir", default="", help="Dataset directory")
    parser.add_argument("--categories", nargs="*", default=None, help="Target categories")
    parser.add_argument("--max-concurrent", type=int, default=3, help="Max concurrent tasks")
    args = parser.parse_args()

    anyio.run(
        main_async,
        args.task_name,
        args.traj_base_dir,
        args.skill_output_dir,
        args.data_dir,
        args.categories,
        args.max_concurrent,
    )


if __name__ == "__main__":
    main()
