"""
Unified evaluation runner for DABStep and ScienceAgentBench.
"""

import argparse
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

from tqdm import tqdm

from llm import LLM
from prompt import JUDGE_SYS_PROMPT, JUDGE_USR_PROMPT


def format_reasoning_messages(question, reasoning):
    if reasoning is None or reasoning == {}:
        return None
    pattern = r'(<Execute>.*?</Execute>)'
    parts = re.split(pattern, reasoning, flags=re.DOTALL)
    parts = [p.strip() for p in parts if p.strip()]
    msgs = [{"role": "user", "content": question}]
    for i, part in enumerate(parts):
        role = "assistant" if i % 2 == 0 else "user"
        msgs.append({"role": role, "content": part})
    return msgs


def get_judge_prompt(messages, current_step_index, final_step_index, previous_predictions=None):
    question = messages[0]["content"]
    file_list = question.split("Files:\n")[1].strip()

    para_list = []
    for index, msg in enumerate(messages[:current_step_index * 2 + 1]):
        if msg["role"] == "assistant" and index != len(messages) - 1:
            para = msg["content"].strip() + "\n" + messages[index + 1]["content"].strip()
        elif msg["role"] == "assistant" and index == len(messages) - 1:
            para = msg["content"].strip()
        elif msg["role"] == "user":
            continue
        else:
            raise Exception(f"Unknown role: {msg['role']}")

        para = para.replace("<extra_0>", "").strip()
        para_list.append(para)

    previous_para_list = para_list[:-1]
    current_paragraph = para_list[-1]

    assert len(previous_para_list) == len(previous_predictions), (
        f"Paragraphs and previous predictions length mismatch: "
        f"{len(previous_para_list)} vs {len(previous_predictions)}"
    )

    paragraph_list_str = ""
    for idx, (para, pred) in enumerate(zip(previous_para_list, previous_predictions)):
        paragraph_list_str += (
            f"[Paragraph_{idx + 1}]\n{para}\n[/Paragraph_{idx + 1}]\n"
            f"[Verification_{idx + 1}]\n{pred}\n[/Verification_{idx + 1}]\n\n"
        )

    current_paragraph_str = (
        f"[Paragraph_{current_step_index}]\n{current_paragraph}\n[/Paragraph_{current_step_index}]\n"
    )
    if current_step_index == final_step_index:
        current_paragraph_str += "Note: This is the final step.\n"

    return JUDGE_USR_PROMPT.format(
        problem=question,
        file_list=file_list,
        paragraph_list=paragraph_list_str,
        current_paragraph=current_paragraph_str,
    )


def _parse_prediction(reasoning):
    if "<score>" in reasoning and "</score>" in reasoning:
        raw_score = reasoning.split("<score>")[-1].split("</score>")[0].strip()
        summary = "No summary found."
        if "<summary>" in reasoning and "</summary>" in reasoning:
            summary = reasoning.split("<summary>")[-1].split("</summary>")[0].strip()
        try:
            score = float(raw_score)
            if score not in (0.0, 0.5, 1.0):
                score = -1.0
        except Exception as e:
            score = -1.0
            summary = "No score found."
            print(f"Score parse error: {e}")
    else:
        score = -1.0
        summary = "No score found."
    return score, summary


def _parallel_sample(
    prompt, workspace_dir, sys_prompt_tools,
    tool_functions, temperature, top_p, top_k,
    context_codes, step_index,
    base_url, api_key, model_name,
):
    analyzer = LLM(
        base_url=base_url or os.getenv("OPENAI_BASE_URL"),
        api_key=api_key or os.getenv("OPENAI_API_KEY"),
        model_name=model_name or os.getenv("MODEL_NAME", "dataprm"),
        max_rounds=10,
    )
    try:
        step_start = time.time()
        prompt_msg = [
            {"role": "system", "content": sys_prompt_tools},
            {"role": "user", "content": prompt},
        ]
        response = analyzer.generate(
            prompt=prompt_msg,
            workspace=workspace_dir,
            temperature=temperature,
            execute_context_code=True,
            context_codes=context_codes,
            top_p=top_p,
            top_k=top_k,
            use_tool=True,
            tool_functions=tool_functions,
        )
        elapsed = time.time() - step_start
        reasoning = response["reasoning"]
        messages = response["messages"]
        token_usage = response.get(
            "token_usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        )
        score, summary = _parse_prediction(reasoning)
        return {
            "step_index": step_index,
            "prediction": f"Score: {score}\nSummary: {summary}",
            "prediction_score": score,
            "prediction_summary": summary,
            "messages": messages,
            "reasoning": reasoning,
            "token_usage": token_usage,
            "elapsed_seconds": elapsed,
        }
    except Exception as e:
        print(f"Error in _parallel_sample (step {step_index}): {e}")
        return {
            "step_index": step_index,
            "prediction": f"Score: -1.0\nSummary: [Error]: {e}",
            "prediction_score": -1.0,
            "prediction_summary": f"[Error]: {e}",
            "messages": "",
            "reasoning": "",
            "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "elapsed_seconds": 0.0,
        }


def _process_sample_path(
    item, index_upper, workspace_path, data_file_dir, sample_index,
    sys_prompt_tools, tool_functions, temperature, top_p, top_k,
    final_step_index, link_files_fn: Callable,
    base_url, api_key, model_name,
):
    task_id = item["task_id"]
    reasoning_trace = item["reasoning_trace"]
    if not reasoning_trace:
        return {"results": None}

    question = item["prompt"][-1]["content"]
    messages = format_reasoning_messages(question, reasoning_trace)
    sample_results = []
    previous_predictions = []

    for step_num in range(len(messages) // 2):
        step_index = step_num + 1
        context_codes = messages[: step_index * 2]
        sample_workspace_dir = workspace_path / str(index_upper) / f"{task_id}_step_{step_index}"
        sample_workspace_dir.mkdir(parents=True, exist_ok=True)

        link_files_fn(item, data_file_dir, sample_workspace_dir)

        prompt = get_judge_prompt(messages, step_index, final_step_index, previous_predictions)
        result = _parallel_sample(
            prompt, sample_workspace_dir, sys_prompt_tools, tool_functions,
            temperature, top_p, top_k, context_codes, step_index,
            base_url, api_key, model_name,
        )
        sample_results.append(result)
        previous_predictions.append(result["prediction"])

    return {"results": sample_results}


def _process_item(
    item, workspace_path, data_file_dir, index_upper, sample_num,
    sys_prompt_tools, tool_functions, temperature, top_p, top_k,
    final_step_index, link_files_fn: Callable,
    base_url, api_key, model_name,
):
    task_id = item["task_id"]
    sample_args = [
        (item, index_upper, workspace_path, data_file_dir, i,
         sys_prompt_tools, tool_functions, temperature, top_p, top_k,
         final_step_index, link_files_fn, base_url, api_key, model_name)
        for i in range(sample_num)
    ]
    sample_paths_data = []
    with ThreadPoolExecutor(max_workers=sample_num) as executor:
        futures = {executor.submit(_process_sample_path, *args): args for args in sample_args}
        for future in as_completed(futures):
            try:
                sample_paths_data.append(future.result())
            except Exception as e:
                print(f"Error processing sample path: {e}")

    all_samples_data = []
    for sp in sample_paths_data:
        if sp["results"] is not None:
            all_samples_data.extend(sp["results"])

    return {"task_id": task_id, "answer": item["agent_answer"], "samples": all_samples_data}


def _run_eval(args, sys_prompt_tools, tool_functions, final_step_index, link_files_fn):
    data_list = [
        json.load(open(args.input_files.format(i), "r"))
        for i in range(args.number, args.number + args.skip)
    ]

    global_lock = threading.Lock()
    global_stats = {
        "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
        "eval_count": 0, "total_step_seconds": 0.0,
    }
    dataset_start = time.time()

    for idx, data in enumerate(data_list):
        file_index = idx + args.number
        output_file = os.path.join(args.output_dir, f"result_{file_index}.jsonl")
        os.makedirs(args.output_dir, exist_ok=True)
        data_file_dir = Path(args.data_file_dir)
        workspace_path = Path(args.workspace_root) / f"sample_{file_index}"
        workspace_path.mkdir(parents=True, exist_ok=True)

        processed_ids = _load_processed_ids(output_file)
        data = [item for item in data if str(item["task_id"]) not in processed_ids]
        print(f"[{file_index}] Remaining items: {len(data)}")

        processed_count = 0
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            future_to_item = {
                executor.submit(
                    _process_item, item, workspace_path, data_file_dir,
                    file_index, args.sample_num,
                    sys_prompt_tools, tool_functions,
                    args.temperature, args.top_p, args.top_k,
                    final_step_index, link_files_fn,
                    args.base_url, args.api_key, args.model_name,
                ): item
                for item in data
            }
            with open(output_file, "a", encoding="utf-8") as f_out:
                for future in tqdm(as_completed(future_to_item), total=len(data), desc=f"File {file_index}"):
                    item = future_to_item[future]
                    try:
                        result = future.result()
                        with global_lock:
                            for step_result in result.get("samples", []):
                                tu = step_result.get("token_usage", {})
                                global_stats["prompt_tokens"] += tu.get("prompt_tokens", 0)
                                global_stats["completion_tokens"] += tu.get("completion_tokens", 0)
                                global_stats["total_tokens"] += tu.get("total_tokens", 0)
                                global_stats["eval_count"] += 1
                                global_stats["total_step_seconds"] += step_result.get("elapsed_seconds", 0.0)
                        f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                        f_out.flush()
                        processed_count += 1
                    except Exception as e:
                        import traceback
                        print(f"Error processing item {item.get('task_id', 'unknown')}: {e}")
                        traceback.print_exc()

        print(f"[{file_index}] Done. New: {processed_count}, Total: {processed_count + len(processed_ids)}")

    _print_stats(global_stats, time.time() - dataset_start)


def _link_files_dabstep(item, data_file_dir: Path, workspace_dir: Path):
    """Link every file in data_file_dir into the step workspace."""
    if data_file_dir.exists():
        for file in data_file_dir.iterdir():
            link_path = workspace_dir / file.name
            if not link_path.exists():
                link_path.symlink_to(file.resolve())
    else:
        print(f"Data file directory does not exist: {data_file_dir}")


def _link_files_sab(item, data_file_dir: Path, workspace_dir: Path):
    """Link only the files listed in item['file_list'] into the step workspace."""
    for file_path in item.get("file_list", []):
        file = data_file_dir / file_path
        link_path = workspace_dir / file.name
        if not link_path.exists():
            os.symlink(file.resolve(), link_path)


def run_dabstep(args):
    from tools.tool_prompt import QUERY_DOCUMENT_PROMPT, QUERY_DOCUMENT_FILE_PATH

    with open(QUERY_DOCUMENT_FILE_PATH, "r", encoding="utf-8") as f:
        tool_functions = f.read()
    sys_prompt_tools = JUDGE_SYS_PROMPT.format(tools=QUERY_DOCUMENT_PROMPT.strip())
    final_step_index = args.final_step_index or 10

    _run_eval(args, sys_prompt_tools, tool_functions, final_step_index, _link_files_dabstep)


def run_scienceagentbench(args):
    from tools.tool_prompt import QUERY_IMAGE_PROMPT, QUERY_IMAGE_FILE_PATH

    with open(QUERY_IMAGE_FILE_PATH, "r", encoding="utf-8") as f:
        tool_functions = f.read()
    sys_prompt_tools = JUDGE_SYS_PROMPT.format(tools=QUERY_IMAGE_PROMPT.strip())
    final_step_index = args.final_step_index or 15

    _run_eval(args, sys_prompt_tools, tool_functions, final_step_index, _link_files_sab)


def _load_processed_ids(output_file):
    processed_ids = set()
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    processed_ids.add(str(item["task_id"]))
                except json.JSONDecodeError:
                    continue
        print(f"Resuming from {len(processed_ids)} processed items.")
    else:
        print("Starting fresh processing.")
    return processed_ids


def _print_stats(stats, dataset_elapsed):
    avg_step = stats["total_step_seconds"] / max(stats["eval_count"], 1)
    avg_tokens = stats["total_tokens"] / max(stats["eval_count"], 1)
    print("\n========== Token & Timing Statistics ==========")
    print(f"  Total eval steps       : {stats['eval_count']}")
    print(f"  Prompt tokens          : {stats['prompt_tokens']:,}")
    print(f"  Completion tokens      : {stats['completion_tokens']:,}")
    print(f"  Total tokens           : {stats['total_tokens']:,}")
    print(f"  Avg tokens / step      : {avg_tokens:.1f}")
    print(f"  Avg latency / step     : {avg_step:.2f}s")
    print(f"  Total dataset latency  : {dataset_elapsed:.2f}s  ({dataset_elapsed / 60:.1f} min)")
    print("================================================")

def build_parser():
    parser = argparse.ArgumentParser(
        description="Unified evaluation runner for DABStep and ScienceAgentBench",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="task", required=True)

    def add_common_args(p):
        p.add_argument(
            "--input_files", required=True,
            help="Path template for input JSON files. Use {} as the index placeholder, "
                 "e.g. '/data/results_{}.json'",
        )
        p.add_argument("--number", type=int, default=0, help="Starting file index")
        p.add_argument("--skip", type=int, default=1, help="Number of files to process")
        p.add_argument("--output_dir", required=True, help="Directory to write output JSONL files")
        p.add_argument("--data_file_dir", required=True, help="Path to benchmark dataset files")
        p.add_argument("--workspace_root", required=True, help="Root directory for per-run workspaces")
        p.add_argument("--sample_num", type=int, default=1, help="Number of samples per item")
        p.add_argument("--max_workers", type=int, default=8, help="Thread pool size")
        p.add_argument("--temperature", type=float, default=0.7)
        p.add_argument("--top_p", type=float, default=0.9)
        p.add_argument("--top_k", type=int, default=20)
        p.add_argument(
            "--final_step_index", type=int, default=15,
            help="Override the final step index (default: 10 for dabstep, 15 for scienceagentbench)",
        )
        p.add_argument(
            "--base_url", default=None,
            help="LLM API base URL (overrides OPENAI_BASE_URL env var)",
        )
        p.add_argument(
            "--api_key", default=None,
            help="LLM API key (overrides OPENAI_API_KEY env var)",
        )
        p.add_argument(
            "--model_name", default=None,
            help="Model name (overrides MODEL_NAME env var, fallback: dataprm)",
        )
        

    add_common_args(subparsers.add_parser("dabstep", help="Run DABStep evaluation"))
    add_common_args(subparsers.add_parser("scienceagentbench", help="Run ScienceAgentBench evaluation"))

    return parser


def main():
    args = build_parser().parse_args()
    {"dabstep": run_dabstep, "scienceagentbench": run_scienceagentbench}[args.task](args)


if __name__ == "__main__":
    main()
