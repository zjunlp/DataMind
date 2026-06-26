# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import time
import openai
import numpy as np

def get_judge_prompt(question, pred_answer, gt_answer):

    JUDGE_PROMPT = (
        f"Evaluate the correctness (0 for incorrect, 1 for correct) of the predicted answer to the question: \n\n"
        f"Question: {question}\n\n"
        f"Predicted answer: {pred_answer}\n\n"
        f"Ground truth answer: {gt_answer}\n\n"
        f"Rules for judgment:\n"
        f"1. For numerical questions, any result within 3% of the ground truth answer is considered correct. Please compare abs(Predicted answer)/abs(True answer) with 3% to make your decision.\n"
        f"2. For multiple choice questions, exact match is required\n"
        f"3. The answer should be clear and complete\n"
        f"4. Calculation process alone is not considered correct\n\n"
        f"Wrap your reasoning inside <thought></thought> and warp accuracy score inside <score></score> tags."
        f"Keep your reasoning concise, no more than 3-5 clear and informative sentences. Avoid repetition or unnecessary elaboration. Only output the reasoning and score using the required tags."
        f"Follow the output format as shown in the example below:"
        f"Example response:"
        f"<thought>The predicted answer is 115624, which exactly matches the ground truth. The relative error is 0, well within the 3% threshold. The answer is clear, correct, and directly responds to the question.</thought><score>1</score>"
    )

    JUDGE_SYSTEM_PROMPT = (
        f"You are a fair and professional evaluator. Your task is to assess how closely an AI assistant's answer matches the provided ground truth for a given question. You are to provide a numerical score for how well the response answers the question based on the ground truth answer."
        f"You evaluation should focus on the assistant's answer to the question. Begin your evaluation by comparing the assistant's answer with the ground_truth answer. Identify and correct any mistakes. Be as objective as possible."
    )

    return JUDGE_PROMPT, JUDGE_SYSTEM_PROMPT

def validate_format(text: str) -> tuple[bool, str]:
    # check if <think></think>, <answer></answer> is paired
    if text.count('<think>') != text.count('</think>'):
        return False, "<think> </think> not paired\n"

    if text.count('<code>') != text.count('</code>'):
        return False, "<code> </code> not paired\n"
        
    if text.count('<code>') == 0 or text.count('</code>') == 0:
        return False, "<code> or </code> not found\n"    
    
    if text.count('<answer>') != 1 or text.count('</answer>') != 1:
        return False, "<answer> or </answer> not found\n"
        
    code_pattern = re.compile(r'<code>(.*?)</code>', re.DOTALL)
    result_pattern = re.compile(r'<interpreter>(.*?)</interpreter>', re.DOTALL)

    code_matches = list(code_pattern.finditer(text))
    result_matches = list(result_pattern.finditer(text))

    if len(code_matches) != len(result_matches):
        return False, "The number of <code> and <interpreter> blocks do not match.\n"
    
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

def extract_answer(text: str):
    text = text.strip()

    pattern = r"<answer>(.*?)</answer>"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return ""
    
    return match.group(1)

def extract_solution(solution_str: str):
    if "<|im_start|>assistant\n" in solution_str:
        tag = "<|im_start|>assistant\n"
        tag_user = "<|im_start|>user\n"
        tag_end = "<|im_end|>"
    elif "<|start|>assistant" in solution_str:
        tag = "<|start|>assistant"
        tag_user = "<|start|>user"
        tag_end = "<|end|>"
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
        return ""

def judge_with_retry(answer, gt_answer, question, model_name='gpt-4o-mini', max_retries=10):
    last_error_reason = "[Unknown error]"
    for attempt in range(max_retries):
        score, reason = pre_judge(answer, gt_answer, question, model_name)
        if not reason.startswith("[GPT JUDGE ERROR]"):
            return score, reason
        else:
            last_error_reason = reason 
            time.sleep(1)  
    print(f"Max retries reached. Returning default score.")
    return 0, last_error_reason

def pre_judge(answer, gt_answer, question, model_name='gpt-4o-mini'):
    import os
    prompt, system_prompt = get_judge_prompt(question, answer, gt_answer)

    client = openai.OpenAI()

    try:
        response = client.chat.completions.create(
            model='gpt-4o-mini',
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0,
            max_tokens=1024,
            top_p=1,
        )
        response_str = response.choices[0].message.content

        score_match = re.search(r"<score>(.*?)</score>", response_str)
        if score_match:
            score_str = score_match.group(1).strip()
        else:
            score_str = "[Missing score]"

        if score_str == "[Missing score]":
            reason = "[MISSING SCORE TEMPLATE] GPT judge failed to return a score\n" + response_str
        else:
            reason_match = re.search(r"<thought>(.*?)</thought>", response_str, re.DOTALL)
            reason = reason_match.group(1).strip() if reason_match else "[Missing thought]"
        score = score_str.strip()
        if score == "":
            score = 0
        elif score == "[Missing score]":
            score = 0
        else:
            score = score.replace(" ", "")
        try:
            score = float(score)
        except ValueError:
            score = float(score.splitlines()[0])
        except Exception as e:
            score = 0
    except openai.RateLimitError as e:
        score = 0
        reason = "[GPT JUDGE ERROR] RateLimitError\n"
    except openai.APIError as e:
        score = 0
        reason = "[GPT JUDGE ERROR] APIError\n"
    except Exception as e:
        score = 0
        reason = f"[GPT JUDGE ERROR] {str(e)}"
    return score, reason

def length_penalty(answer: str) -> float:
    MAX_LEN = 1025 
    if len(answer) > MAX_LEN:
        answer = answer[:MAX_LEN]

    L = len(answer)
    if L <= 256:
        return 1.0
    elif L >= 1024:
        return 0.5
    else:
        return 1.0 - 0.5 * ((L - 256) / 768)

def compute_execution_socre(response, success_socre=0, failure_score=-0.5):
    result_pattern = r'<interpreter>(.*?)</interpreter>'
    
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

def compute_answer_score(answer, ground_truth, question, incorrect_score=0, correct_score=1.):

    if answer == "":
        return incorrect_score, 0.0, f'no answer extracted\n'
    
    if len(answer) > 10240:
        return incorrect_score, 0.0, f'answer too long ({len(answer)} chars), skipped\n'
    
    answer_score, reason = judge_with_retry(answer, ground_truth, question, max_retries=3)

    if answer_score > 0 and answer_score <= 1:
        penalty = length_penalty(answer)
        final_score = answer_score * penalty
        return final_score, 1.0, reason
    else:
        return incorrect_score, 0.0, reason

def compute_tool_score(response, success_score=1, failure_score=-1):
    """
    Args:
        response: the response text
        success_score: the score for the success
        failure_score: the score for the failure
    """
    
    matches = re.findall(r"<code>.*?</code>", response, re.DOTALL)
    
    return 0.1 * len(matches)
    
def compute_score(solution_str, ground_truth, question, pred_csv_result_dir_parent=None, gold_csv_results_dir=None):
    response = extract_solution(solution_str)

    to_remove = f'Your previous action is invalid. \
If you want to execute the code for the execution result, you should put the code between <code>\n```python and ```\n</code>. \
If you want to give the final answer, you should put the answer between <answer> and </answer>. Please try again.'

    cleaned_response = response.replace(to_remove, "")

    answer = extract_answer(cleaned_response)

    gt = ground_truth.get('ground_truth', '')

    answer_score, acc, answer_reason = compute_answer_score(answer, gt, question, -1.0, 1.0)
    execution_score, execution_reason = compute_execution_socre(cleaned_response, 1.0, -1.0)
    format_score, format_reason = compute_format_score(cleaned_response, 1.0)
    tool_score = compute_tool_score(cleaned_response, 1, -1)

    reason = '[format reason]' + format_reason + '[execution reason]' + execution_reason + '[answer reason]' + answer_reason 
    return format_score, execution_score, answer_score, acc, tool_score, reason