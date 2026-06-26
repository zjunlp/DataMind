import torch
from verl.utils.reward_score import model_judge
from verl.utils.reward_score import sql

def get_my_score_fn(source):
    if source == 'darl/sql' or source == 'BIRD_DEV' or "sql" in source or "bird" in source.lower():
        return sql.compute_score
    else:
        return model_judge.compute_score

def process_single_item(sequences_str, ground_truth, extra_info, data_source, valid_response_length, pred_csv_result_dir_parent, gold_csv_results_dir):
    question = extra_info['question']
    compute_score_fn = get_my_score_fn(data_source)

    template_score, execution_score, answer_score, acc, tool_score, reason = compute_score_fn(
        solution_str=sequences_str,
        ground_truth=ground_truth,
        question=question,
        pred_csv_result_dir_parent=pred_csv_result_dir_parent,
        gold_csv_results_dir=gold_csv_results_dir,
    )

    if answer_score > 0.0:
        score = answer_score
    elif answer_score == -1.0 and template_score == 1.0:
        score = 0.0
    elif answer_score == -1.0 and template_score == -1.0:
        score = -0.1
    else:
        score = -0.1

    print('-' * 20)
    print(f"sequences_str: \n{sequences_str}")
    print(f"ground_truth: \n{ground_truth}")
    print(f"template_score: \n{template_score}")
    print(f"answer_score: \n{answer_score}")
    print(f"accuracy: \n{acc}")
    print(f"tool_score: \n{tool_score}")
    print(f"score: \n{score}")
    print(f"reason: \n{reason}")
    print('-' * 20)

    reward_dict = {
        "total_score": score,
        "valid_response_length":valid_response_length, 
        "answer_score": answer_score,
        "template_score": template_score,
        "accuracy": acc,
        "tool_score": tool_score,
    }

    return reward_dict