import re
import os
import sys
sys.path.append('..')
from typing import Dict, Tuple, Optional
from func_timeout import func_timeout, FunctionTimedOut

from .exec_eval import eval_exec_match
import signal
import pandas as pd
import math

def extract_solution(solution_str: str) -> Tuple[Optional[str], str]:
    """Extracts the final answer from the model's response string.
    
    Args:
        solution_str: Raw response string from the language model
        
    Returns:
        Tuple containing (extracted_answer, processed_string)
    """
    # Split response to isolate assistant output
    if "Assistant:" in solution_str:
        processed_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        processed_str = solution_str.split("<|im_start|>assistant", 1)[1]
    else:
        print("[Error] Failed to locate model response header")
        return None, solution_str

    # Extract final answer using XML-style tags
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(answer_pattern, processed_str, re.DOTALL))
    think_pattern = r'<think>(.*?)</think>'
    think_matches = list(re.finditer(think_pattern, processed_str, re.DOTALL))

    if not think_matches:
        print("[Error] No valid think tags found")
        final_think = None
    else:
        final_think = think_matches[-1].group(1).strip()
    
    if not matches:
        print("[Error] No valid answer tags found")
        return None, final_think, processed_str
        
    final_answer = matches[-1].group(1).strip()

    return final_answer, final_think, processed_str

def parse_sql_from_answer(answer_text: str) -> Optional[str]:
    """Parses SQL from the model's answer text.
    
    Args:
        answer_text: Text extracted from model's <answer> tags
        
    Returns:
        SQL string, or None if no SQL is found
    """
    sql_pattern = r'```sql(.*?)```'
    matches = list(re.finditer(sql_pattern, answer_text, re.DOTALL))
    
    if not matches:
        print("[Error] No valid SQL tags found")
        return None
    
    print(f"[Parsed SQL]: {matches[-1].group(1).strip()}")
    return matches[-1].group(1).strip()

def validate_response_structure(answer_str: str, processed_str: str) -> bool:
    """Performs comprehensive validation of response structure.
    
    Args:
        processed_str: Processed response string from the model
        
    Returns:
        Boolean indicating whether all formatting requirements are met
    """
    print("\n[Structure Validation]")
    validation_passed = True

    # Check required tags
    tags = {
        'think_start': ('<think>', 1),
        'think_end': ('</think>', 1),
        'answer_start': ('<answer>', 1),
        'answer_end': ('</answer>', 1)
    }

    positions = {}
    for tag_name, (tag_str, expected_count) in tags.items():
        count = processed_str.count(tag_str)
        positions[tag_name] = pos = processed_str.find(tag_str)
        
        print(f"  {tag_str}: count={count}, position={pos}")
        
        if count != expected_count:
            print(f"  [Error] {tag_str} appears {count} times (expected {expected_count})")
            validation_passed = False

    # Verify tag order
    if (positions['think_start'] > positions['think_end'] or
        positions['think_end'] > positions['answer_start'] or
        positions['answer_start'] > positions['answer_end']):
        print("  [Error] Incorrect tag order: Expected <think>...</think><answer>...</answer>")
        validation_passed = False
    else:
        print("Tag sequence validation passed")

    # Extract SQL from answer text
    if validation_passed:
        pred_sql = parse_sql_from_answer(answer_str)
        if not pred_sql:
            validation_passed = False
    else:
        pred_sql = None

    return pred_sql, validation_passed

def validate_format(text: str) -> tuple[bool, str]:
    # check if <think></think>, <answer></answer> is paired
    if text.count('<think>') != text.count('</think>'):
        return False, f"<think> </think> not paired, {text.count('<think>')} <think>, {text.count('</think>')} </think>\n"
    
    if text.count('<answer>') != 1 or text.count('</answer>') != 1:
        return False, "<answer> or </answer> not found\n"
    
    # check the order of code/interpreter tags

    code_pattern = re.compile(r'<code>(.*?)</code>', re.DOTALL)
    result_pattern = re.compile(r'<interpreter>(.*?)</interpreter>', re.DOTALL)

    code_matches = list(code_pattern.finditer(text))
    result_matches = list(result_pattern.finditer(text))

    if len(code_matches) != len(result_matches):
        return False, f"The number of <code> and <interpreter> blocks do not match. {len(code_matches)} code, {len(result_matches)} result\n"
    
    last_end = 0
    for code_match, result_match in zip(code_matches, result_matches):
        code_start, code_end = code_match.span()
        result_start, result_end = result_match.span()

        # Ensure order: <code>...</code><interpreter>...</interpreter>
        if not (code_start >= last_end and code_end <= result_start and result_end > result_start):
            return False, "code/interpreter blocks are in the wrong order or overlapping.\n"
        
        last_end = result_end  # Move pointer forward to prevent overlapping
    
    answer_start = text.find('<answer>')
    answer_end = text.find('</answer>')
    if answer_start > answer_end:
        return False, "<answer> must be before </answer>\n"
    
    return True, "format is correct\n"


def extract_answer_sql(text: str):
    text = text.strip()

    answer_pattern = r"<answer>(.*?)</answer>"
    answer_match = re.search(answer_pattern, text, re.DOTALL)
    if not answer_match:
        return ""
    
    answer_content = answer_match.group(1)

    return answer_content

def extract_solution(solution_str: str):
    if "<|im_start|>assistant\n" in solution_str:
        tag = "<|im_start|>assistant\n"
    else:
        tag = "Assistant:"
    tag_user = "<|im_start|>user\n"
    tag_end = "<|im_end|>"
    index = solution_str.find(tag)
    if index != -1:
        after_first = solution_str[index + len(tag):]
        cleaned = after_first.replace(tag, "")
        cleaned = cleaned.replace(tag_user, "")
        cleaned = cleaned.replace(tag_end, "")
        return cleaned.strip()
    else:
        return ""  # 如果没有找到 <result>，返回空字符串
    
def length_penalty(answer: str) -> float:
    MAX_LEN = 1025  # 自定义阈值，防止内存炸裂
    if len(answer) > MAX_LEN:
        answer = answer[:MAX_LEN]

    L = len(answer)
    if L <= 256:
        return 1.0
    elif L >= 1024:
        return 0.5
    else:
        return 1.0 - 0.5 * ((L - 256) / 768)

def compute_execution_score(response, success_socre=0, failure_score=-0.5):
    # 正则表达式匹配 <result></result> 标签中的内容
    result_pattern = r'<interpreter>(.*?)</interpreter>'
    
    # 提取所有<result></result>标签中的内容
    results = re.findall(result_pattern, response, re.DOTALL)

    if len(results) == 0:
        score = failure_score
        reason = "no execution result\n"
    else:
        result = results[-1]
        if 'The code run failed' in result:
            score = failure_score
            reason = f"The code run failed\n"
        elif 'The code run successfully' in result:
            score = success_socre
            reason = f"The code run successfully\n"
        else:
            score = failure_score
            reason = f"The code run failed..\n"

    return score, reason

def compute_format_score(response, format_score=0.):
    """
    Args:
        solution_str: the solution text
        tokenizer: tokenizer
        format_score: the score for the format
    """

    # check format
    valid_template, reason = validate_format(response)
    if not valid_template:
        return -1.0, f'bad format: {reason}'
    else:
        return format_score, f"The correct format\n"

def compare_multi_pandas_table(pred, multi_gold, multi_condition_cols=[], multi_ignore_order=False):
    if multi_condition_cols == [] or multi_condition_cols == [[]] or multi_condition_cols == [None] or multi_condition_cols == None:
        multi_condition_cols = [[] for _ in range(len(multi_gold))]
    elif len(multi_gold) > 1 and not all(isinstance(sublist, list) for sublist in multi_condition_cols):
        multi_condition_cols = [multi_condition_cols for _ in range(len(multi_gold))]
    multi_ignore_order = [multi_ignore_order for _ in range(len(multi_gold))]

    for i, gold in enumerate(multi_gold):
        if compare_pandas_table(pred, gold, multi_condition_cols[i], multi_ignore_order[i]):
            return 1
    return 0

def compare_pandas_table(pred, gold, condition_cols=[], ignore_order=False):
    """_summary_

    Args:
        pred (Dataframe): _description_
        gold (Dataframe): _description_
        condition_cols (list, optional): _description_. Defaults to [].
        ignore_order (bool, optional): _description_. Defaults to False.

    """
    # print('condition_cols', condition_cols)
    
    tolerance = 1e-2

    def vectors_match(v1, v2, tol=tolerance, ignore_order_=False):
        if ignore_order_:
            v1, v2 = (sorted(v1, key=lambda x: (x is None, str(x), isinstance(x, (int, float)))),
                    sorted(v2, key=lambda x: (x is None, str(x), isinstance(x, (int, float)))))
        if len(v1) != len(v2):
            return False
        for a, b in zip(v1, v2):
            if pd.isna(a) and pd.isna(b):
                continue
            elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
                if not math.isclose(float(a), float(b), abs_tol=tol):
                    return False
            elif a != b:
                return False
        return True
    
    if condition_cols != []:
        gold_cols = gold.iloc[:, condition_cols]
    else:
        gold_cols = gold
    pred_cols = pred

    t_gold_list = gold_cols.transpose().values.tolist()
    t_pred_list = pred_cols.transpose().values.tolist()
    score = 1
    for _, gold in enumerate(t_gold_list):
        if not any(vectors_match(gold, pred, ignore_order_=ignore_order) for pred in t_pred_list):
            score = 0
        else:
            for j, pred in enumerate(t_pred_list):
                if vectors_match(gold, pred, ignore_order_=ignore_order):
                    break

    return score

def get_csv_name_from_answer(content: str):
    if "<answer>" in content and "</answer>" in content:
        answer = content.split("<answer>")[1].split("</answer>")[0].strip()
        # extract csv name
        match = re.search(r"['\"]([^'\"]*\.csv)['\"]", answer)
        if match:
            csv_file = match.group(1)
        else:
            csv_file = None  # default name if not found
        return csv_file
    else:
        match = re.search(r"['\"]([^'\"]*\.csv)['\"]", content)
        if match:
            csv_file = match.group(1)
        else:
            csv_file = None  # default name if not found
        return csv_file

def compute_answer_score(answer, task_id, pred_csv_result_dir_parent, gold_csv_results_dir,incorrect_score=-1.0, correct_score=1.0):
    if answer.strip() == "":
        return incorrect_score, 0.0, f'no answer extracted\n'

    try:
        print(f"[DEBUG] pred_csv_result_dir_parent: {pred_csv_result_dir_parent}, gold_csv_results_dir: {gold_csv_results_dir}")
        csv_name = get_csv_name_from_answer(answer)
        if csv_name is None:
            return incorrect_score, 0.0, f'no csv file name found in answer\n'
        pred_csv_path = os.path.join(pred_csv_result_dir_parent, task_id, csv_name)
        gold_csv_path = os.path.join(gold_csv_results_dir, f"{task_id}.csv")

        score = compare_pandas_table(
            pd.read_csv(pred_csv_path),
            pd.read_csv(gold_csv_path),
            ignore_order=True
        )

        if score == 1:
            penalty = length_penalty(answer)
            final_score = correct_score * penalty
            return final_score, 1.0, f'Match\n'
        else:
            return incorrect_score, 0.0, f'Mismatch\n'
    except Exception as e:
        print(f"[ERROR] Error in reward function: {e}")
        score = 0
        return incorrect_score, 0.0, f'Error in reward function: {e}\n'

def compute_tool_score(response, success_score=1, failure_score=-1):
    """
    Args:
        response: the response text
        success_score: the score for the success
        failure_score: the score for the failure
    """
    
    matches = re.findall(r"<code>.*?</code>", response, re.DOTALL)
    
    return 0.1 * len(matches)

def compute_score(solution_str: str, 
                 ground_truth: Dict[str, str],
                 question: str,
                 pred_csv_result_dir_parent: Optional[str] = None,
                 gold_csv_results_dir: Optional[str] = None):
    """Computes comprehensive score for model response.
    
    Args:
        solution_str: Raw model response string
        ground_truth: Dictionary containing ground truth data
        question: question
    """
    # Parse ground truth data
    task_id = ground_truth.get('task_id', '').replace('\n', '').strip()

    response = extract_solution(solution_str)

    to_remove = f"The SQL query you provided didn't return any output. Please double-check your query logic, table names, and filters, and rewrite the code"

    cleaned_response = response.replace(to_remove, "")

    answer = extract_answer_sql(cleaned_response)

    answer_score, acc, answer_reason = compute_answer_score(answer, task_id, pred_csv_result_dir_parent, gold_csv_results_dir, -1.0, 1.0)
    execution_score, execution_reason = compute_execution_score(cleaned_response, 1.0, -1.0)
    format_score, format_reason = compute_format_score(cleaned_response, 1.0)
    tool_score = compute_tool_score(cleaned_response, 1, -1)

    reason = '[format reason]' + format_reason + '[execution reason]' + execution_reason + '[answer reason]' + answer_reason 
    return format_score, execution_score, answer_score, acc, tool_score, reason