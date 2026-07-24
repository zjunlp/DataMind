#!/usr/bin/env python3
import os
import sys
import warnings
import argparse
from pathlib import Path
from prompt import SYSTEM_PROMPT
# Suppress common warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.cuda")
warnings.filterwarnings("ignore", category=UserWarning, module="multiprocessing.resource_tracker")
os.environ.setdefault("PYTHONWARNINGS", "ignore::FutureWarning,ignore::UserWarning")
import json
# Set multiprocessing start method to avoid resource tracking issues
import multiprocessing
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set

# Add DSGym to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dsgym.datasets import DatasetRegistry, load_tasks_from_dataset, create_custom_task
from dsgym.agents import ReActDSAgent, DSPredictReActAgent, MultiTurnReActDSAgent
from dsgym.eval import Evaluator
from dsgym.eval.utils import EvaluationConfig
import shutil
from datetime import datetime
from prompt import JUDGE_PROMPT
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
# Dataset configuration - now using ReActDSAgent as default for all datasets
DATASET_CONFIG = {
    "longds": {
        "agent_type": "multiturn",
        "extra_params": [],
        "result_processor": "longds"
    },
}


def llm_judge_evaluate(task_trajectories, api_key=None, base_url=None, judge_model="deepseek-v4-pro", max_workers=15):
    """Evaluate each task's solution against ground_truth using LLM as judge."""
    
    api_key = api_key or os.environ.get("JUDGE_API_KEY")
    base_url = base_url or os.environ.get("JUDGE_BASE_URL")
    if not api_key:
        raise ValueError("Judge API key is required. Pass --judge-api-key or set JUDGE_API_KEY.")
    if not base_url:
        raise ValueError("Judge base URL is required. Pass --judge-base-url or set JUDGE_BASE_URL.")
    
    client = OpenAI(
        base_url= base_url,
        api_key= api_key
    )


    def judge_one(task):
        if not task['ground_truth'] or not task['solution']:
            return {'score': 0.0, 'reasoning': '', 'error_detail': 'Empty solution or ground truth', 'judge_response': '', 'error': None}
        gt_str = json.dumps(task['ground_truth'], ensure_ascii=False) if isinstance(task['ground_truth'], dict) else str(task['ground_truth'])
        prompt = JUDGE_PROMPT.format(
            question=task['question'],
            ground_truth=gt_str,
            solution=task['solution'],
        )
        try:
            try_no = 0
            while try_no < 3:
                resp = client.chat.completions.create(
                    model=judge_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                )
                text = resp.choices[0].message.content
                
                score_m = re.search(r'<score>\s*(\d)\s*</score>', text)
                reasoning_m = re.search(r'<reasoning>(.*?)</reasoning>', text, re.DOTALL)
                error_m = re.search(r'<error>(.*?)</error>', text, re.DOTALL)
                score = int(score_m.group(1)) if score_m else None
                if score is None:
                    try_no += 1
                    print(f"⚠️  Judge response parsing failed, retrying... (attempt {try_no}/3)")
                    continue
                reasoning = reasoning_m.group(1).strip() if reasoning_m else ''
                error_detail = error_m.group(1).strip() if error_m else ''
                return {'score': score, 'reasoning': reasoning, 'error_detail': error_detail, 'judge_response': text, 'error': None}
            return {
                'score': None,
                'reasoning': '',
                'error_detail': 'Judge response parsing failed after 3 attempts',
                'judge_response': text,
                'error': 'Judge response parsing failed after 3 attempts',
            }
        except Exception as e:
            return {'score': None, 'judge_response': None, 'error': str(e)}

    results = [None] * len(task_trajectories)
    summary = {'correct': [], 'incorrect': []}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(judge_one, t): i for i, t in enumerate(task_trajectories)}
        for future in as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()
            t = task_trajectories[idx]
            s = results[idx]['score']
            if s == 1:
                summary['correct'].append(t['turn_id'])
            elif s == 0:
                summary['incorrect'].append(t['turn_id'])
            print(f"  Turn {t['turn_id']}: score={s}")
    summary['correct'].sort()
    summary['incorrect'].sort()
    # Attach judge results back to trajectories
    for t, r in zip(task_trajectories, results):
        t['judge'] = r

    scores = [r['score'] for r in results if r['score'] is not None]
    avg = sum(scores) / len(scores) if scores else 0.0
    print(f"📊 LLM Judge: avg_score={avg:.3f} ({len(scores)}/{len(results)} judged)")
    summary['avg_score'] = avg
    task_trajectories.append({'summary': summary})
    return task_trajectories


def create_parser():
    """Create the argument parser with all possible parameters."""
    parser = argparse.ArgumentParser(description="Evaluate agent on DSGym datasets")
    
    # Required arguments
    parser.add_argument("--dataset", type=str, required=True,
                       choices=list(DATASET_CONFIG.keys()),
                       help="Dataset to evaluate on")
    parser.add_argument("--model", type=str, required=True, 
                       help="Model name (e.g., 'gpt-4', 'together_ai/Qwen/Qwen3-235B-A22B-Instruct-2507-tput')")

    # Common arguments
    parser.add_argument("--dataset-path", type=str,
                       default=str(Path(__file__).resolve().parents[3] / "dataset" / "task" / "longds"),
                       help="Path to the LongDS task directory containing task_list.json")
    parser.add_argument("--backend", type=str, default="litellm", 
                       choices=["litellm", "vllm", "sglang"],
                       help="Backend to use for model inference")
    parser.add_argument("--task-limit", "--limit", dest="task_limit", type=int, default=None,
                       help="Maximum number of LongDS task directories to evaluate")
    parser.add_argument("--start-index", type=int, default=0,
                       help="Index of first LongDS task directory to evaluate (0-based)")
    parser.add_argument("--turn-limit", type=int, default=None,
                        help="Maximum number of LongDS turns to load from each task.json")
    parser.add_argument("--output-dir", type=str, default="./results",
                       help="Output directory for results")
    parser.add_argument("--manager-url", type=str, default="http://localhost:5000",
                       help="Code sandbox manager URL")
    parser.add_argument("--temperature", type=float, default=0.0,
                       help="Sampling temperature")
    parser.add_argument("--max-steps", type=int, default=40,
                       help="Maximum agent steps per turn")
    parser.add_argument("--api-key", type=str, default=None,
                       help="API key (uses environment variable if not provided)")
    parser.add_argument("--judge-model", type=str, default="deepseek-v4-pro",
                       help="Model name used by the LLM-as-judge evaluator")
    parser.add_argument("--judge-max-workers", type=int, default=15,
                       help="Maximum number of parallel judge requests per task. Default: 15")
    parser.add_argument("--run-parallel", type=int, default=1, metavar="N",
                       help="Number of LongDS tasks to run concurrently. Default: 1")
    parser.add_argument("--max-model-len", type=int, default=32768,
                       help="Maximum model sequence length for vLLM backend (default: 32768)")
    
    # Dataset-specific arguments
    parser.add_argument("--split", type=str, default="test",
                       choices=["train", "validation", "test"],
                       help="Dataset split to use (for discoverybench)")
    parser.add_argument("--dataset-type", type=str, default="original",
                       choices=["original", "synthetic"],
                       help="Type of dataset to use (for qrdata)")
    parser.add_argument("--synthetic-path", type=str, default=None,
                       help="Path to synthetic dataset (for qrdata)")

    parser.add_argument("--reset_env_times", type=int, default=0,
                       help="Number of turns after which to reset the environment")

    return parser


def create_agent(args, agent_type):
    """Create agent instance based on configuration."""
    # Agent configuration
    agent_config = {
        "manager_url": args.manager_url,
        "max_steps": args.max_steps,
        "temperature": args.temperature,
        "output_dir": args.output_dir,
    }
    
    # Add vLLM-specific parameters
    if args.backend in ["vllm", "sglang"]:
        agent_config["max_model_len"] = args.max_model_len
    
    if args.backend == "litellm" and args.api_key:
        agent_config["api_key"] = args.api_key
    
    # Choose agent type
    if agent_type == "multiturn":
        agent_class = MultiTurnReActDSAgent
    elif agent_type == "react":
        agent_class = ReActDSAgent
    elif agent_type == "dspredict":
        agent_class = DSPredictReActAgent
        agent_config["submission_dir"] = "./submissions"
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    return agent_class(
        backend=args.backend,
        model=args.model,
        **agent_config
    )



def load_dataset(args):
    """Load dataset with appropriate configuration."""
    Base_Path = args.dataset_path
    task_root = Path(Base_Path)
    task_list_path = task_root / "task_list.json"

    with open(task_list_path, "r", encoding="utf-8") as f:
        task_list = json.load(f)

    if args.start_index < 0:
        raise ValueError("--start-index must be non-negative")
    if args.task_limit is not None and args.task_limit < 0:
        raise ValueError("--task-limit must be non-negative")
    if args.turn_limit is not None and args.turn_limit < 0:
        raise ValueError("--turn-limit must be non-negative")

    task_list = task_list[args.start_index:]
    if args.task_limit is not None:
        task_list = task_list[:args.task_limit]

    for task_info in task_list:
        system_prompt = SYSTEM_PROMPT.format(PATH=f"/data/longds/{task_info['task_domain']}/{task_info['dataset_name']}/{task_info['task_id']}/data")
        task_path = task_root / task_info['task_domain'] / task_info['dataset_name'] / task_info['task_id'] / "task.json"
        task_info['task_path'] = str(task_path)
        with open(task_path, 'r', encoding='utf-8') as f:
            tasks_json = json.load(f)
        if args.turn_limit is not None:
            tasks_json = tasks_json[:args.turn_limit]
        for item in tasks_json:
            item['turn_id'] = item.pop('turn_id')
            if item['turn_id'] == 1:
                item['prompt'] = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"{item['context']}\nQuestion: {item['question']}"}
                ]
            else:
                item['prompt'] = [
                    {"role": "user", "content": f"{item['context']}\nQuestion: {item['question']}"}
                ]
        task_info['tasks'] = tasks_json
        
    return "Success", task_list


def run_one_task(args, agent_type, task_info):
    """Run and judge one LongDS task in an isolated agent session."""
    turns = task_info["tasks"]
    domain = task_info["task_domain"]
    dataset_name = task_info["dataset_name"]
    task_id = task_info["task_id"]
    task_key = f"{domain}/{dataset_name}/{task_id}"

    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    model_name = args.model.replace("/", "_")
    output_dir = (
        f"{args.output_dir}/{args.dataset}/{domain}/{dataset_name}/{task_id}/"
        f"{model_name}_{timestamp}"
    )

    print(f"📝 Starting task {task_key} with {len(turns)} turns...")
    print(f"🤖 Initializing agent for {task_key}...")
    agent = create_agent(args, agent_type)

    bak_path = f"{output_dir}/bak"
    os.makedirs(bak_path, exist_ok=True)
    result = agent.solve_task(
        turns,
        bak_path=bak_path,
        reset_env_times=args.reset_env_times,
    )

    traj_path = f"{output_dir}/traj.json"
    with open(traj_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    conversation = result.get("conversation", [])
    solutions = result.get("solutions", [])
    task_trajectories = []
    msg_idx = 0
    for solution in solutions:
        turn_msgs = []
        assistant_count = 0
        while msg_idx < len(conversation) and assistant_count < solution["steps"]:
            message = conversation[msg_idx]
            turn_msgs.append(message)
            if message["role"] == "assistant":
                assistant_count += 1
            msg_idx += 1
        task_trajectories.append({
            "turn_id": solution["turn_id"],
            "question": f"{solution['context']}\nQuestion: {solution['question']}",
            "ground_truth": solution["ground_truth"],
            "solution": solution["solution"],
            "success": solution["success"],
            "steps": solution["steps"],
            "trajectory": turn_msgs,
        })

    all_result_path = f"{output_dir}/results.json"
    with open(all_result_path, "w") as f:
        json.dump(task_trajectories, f, indent=2, ensure_ascii=False)
    print(f"📁 Results for {task_key} saved to: {all_result_path}")

    code_lines = []
    for trajectory in task_trajectories:
        code_lines.append(f"############## turn {trajectory['turn_id']}")
        for message in trajectory["trajectory"]:
            if message.get("role") == "assistant":
                for match in re.finditer(
                    r"<python>(.*?)</python>",
                    message.get("content", ""),
                    re.DOTALL,
                ):
                    code_lines.append(match.group(1).strip())
                    code_lines.append("")
    code_path = f"{output_dir}/code.py"
    with open(code_path, "w") as f:
        f.write("\n".join(code_lines))
    print(f"📁 Code for {task_key} saved to: {code_path}")

    print(f"⚖️ Running LLM judge for {task_key}...")
    evaluated_trajectories = llm_judge_evaluate(
        task_trajectories,
        judge_model=args.judge_model,
        max_workers=args.judge_max_workers,
    )
    all_eval_path = f"{output_dir}/results_eval.json"
    with open(all_eval_path, "w") as f:
        json.dump(evaluated_trajectories, f, indent=2, ensure_ascii=False)
    print(f"✅ Completed task {task_key}")
    print(f"📁 Judged results saved to: {all_eval_path}")

    return {
        "task": task_key,
        "output_dir": output_dir,
        "avg_score": evaluated_trajectories[-1]["summary"]["avg_score"],
    }


def main():
    parser = create_parser()
    args = parser.parse_args()

    dataset_config = DATASET_CONFIG[args.dataset]

    if args.run_parallel < 1:
        parser.error("--run-parallel must be at least 1")
    if args.judge_max_workers < 1:
        parser.error("--judge-max-workers must be at least 1")

    print(f"🚀 Starting {args.dataset.upper()} Evaluation with DSGym")
    print(f"Model: {args.model}")
    print(f"Backend: {args.backend}")
    print(f"Task limit: {args.task_limit if args.task_limit is not None else 'all'}")
    print(f"Turn limit: {args.turn_limit}")
    print(f"Start index: {args.start_index}")
    print(f"Parallel task workers: {args.run_parallel}")
    print(f"Judge workers per task: {args.judge_max_workers}")
    print("-" * 50)

    # Load dataset
    print(f"📊 Loading {args.dataset} dataset...")
    try:
        dataset, all_tasks = load_dataset(args)
        print(f"✅ Loaded {len(all_tasks)} tasks from {args.dataset}")
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return 1

    if args.reset_env_times > 0:
        args.output_dir += f"_reset_{args.reset_env_times}"

    if not all_tasks:
        print("No tasks selected.")
        return 0

    worker_count = min(args.run_parallel, len(all_tasks))
    failures = []
    completed = 0

    if worker_count == 1:
        for task_info in all_tasks:
            task_key = (
                f"{task_info['task_domain']}/"
                f"{task_info['dataset_name']}/"
                f"{task_info['task_id']}"
            )
            try:
                run_one_task(args, dataset_config["agent_type"], task_info)
                completed += 1
            except Exception as exc:
                failures.append((task_key, str(exc)))
                print(f"❌ Task {task_key} failed: {exc}")
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            futures = {
                pool.submit(
                    run_one_task,
                    args,
                    dataset_config["agent_type"],
                    task_info,
                ): (
                    f"{task_info['task_domain']}/"
                    f"{task_info['dataset_name']}/"
                    f"{task_info['task_id']}"
                )
                for task_info in all_tasks
            }
            for future in as_completed(futures):
                task_key = futures[future]
                try:
                    future.result()
                    completed += 1
                    print(
                        f"📊 Progress: {completed}/{len(all_tasks)} tasks completed "
                        f"({task_key})"
                    )
                except Exception as exc:
                    failures.append((task_key, str(exc)))
                    print(f"❌ Task {task_key} failed: {exc}")

    print("-" * 50)
    print(f"Completed tasks: {completed}/{len(all_tasks)}")
    if failures:
        print(f"Failed tasks: {len(failures)}")
        for task_key, error in failures:
            print(f"  - {task_key}: {error}")
        return 1

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
