import os
import re
import json
import time
import copy
import pandas as pd
from colorama import init, Fore  
from chat import Chat_With_LLM_Messages

import sys

sys.path.append(".")



def check_correct_flag(sample_id, question_index, base_dirs):
    """
    Check whether there are files with the correct flag in the specified directory

    Args:
        sample_id: Sample id
        question_index: Question index
        base_dirs: A list of base directory paths

    Returns:
        bool: Return True if the file is found and correct_flag is True; otherwise, return False
    """
    pattern = f"question_{question_index}_final_answer"

    for base_dir in base_dirs:
        try:
            sample_dir = os.path.join(base_dir, sample_id)
            
            # Check if the directory exists
            if not os.path.exists(sample_dir):
                print(f"Directory not found: {sample_dir}")
                continue

            # Search for matching json files
            matching_files = [f for f in os.listdir(sample_dir) 
                            if f.endswith('.json') and pattern in f]

            # If matching files are found
            if matching_files:
                file_path = os.path.join(sample_dir, matching_files[0])
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        if data.get("correct_flag", False):
                            return True
                except json.JSONDecodeError:
                    print(f"Error reading JSON file: {file_path}")
                except Exception as e:
                    print(f"Error processing file {file_path}: {str(e)}")

        except Exception as e:
            print(f"Error processing directory {base_dir}: {str(e)}")

    return False

def extract_code(response_content):    
    # Check the independent ```python block
    if '```python' in response_content:
        code_parts = response_content.split('```python', 1)
        if len(code_parts) > 1:
            code_block = code_parts[1].split('```', 1)
            code = code_block[0] if len(code_block) > 0 else code_parts[1]
            return code.strip() 
    return None

def check_answers_equiv(question, pred_answer, gt_answer, check_model = "gpt-4o-mini",dataset_name = "QRData", api_key = None):
    """
    Comprehensively check whether the answers are correct

    Args:
        question: Question
        pred_answer: Predicted answer
        gt_answer: Ground truth answer
        check_model: The model used for checking
        dataset_name: Dataset name
    """
    if dataset_name == "QRData":
        prompt = (
            f"Evaluate the correctness (0 for incorrect, 1 for correct) of the predicted answer to the question: \n\n"
            f"Question: {question}\n\n"
            f"Predicted answer: {pred_answer}\n\n"
            f"Ground truth answer: {gt_answer}\n\n"
            f"Rules for judgment:\n"
            f"1. For numerical questions, any result within 3% of the ground truth answer is considered correct. Please compare abs(Predicted answer)/abs(True answer) with 3% to make your decision.\n"
            f"2. For multiple choice questions, exact match is required\n"
            f"3. The answer should be clear and complete\n"
            f"4. Calculation process alone is not considered correct\n\n"
            f"Please reply in this format: \n"
            f"Thoughts:\n\n"
            f"## The accuracy score is:"
        )
    elif dataset_name == "DiscoveryBench":
        prompt = (
            f"Please judge whether the generated answer is right or wrong.\n\n"
            f"Question: {question}\n"
            f"True answer: {gt_answer}\n"
            f"Predicted answer: {pred_answer}\n\n"
            f"Rules for judgment:\n"
            
            f"1. For numerical questions, any result within 1% of the ground truth answer is considered correct. Please compare abs(Predicted answer)/abs(True answer) with 1% to make your decision.\n"
            f"2. The answer should be clear and complete\n"
            f"3. Calculation process alone is not considered correct\n\n"
            
            f"Please reply in this format: \n"
            f"Thoughts:\n\n"
            f"## The final answer is: <Output only True or False>"
        )

    max_retries = 5
    retries = 0
    while retries < max_retries:
        try:
            response = Chat_With_LLM_Messages(model_name=check_model, temperature=0.0, max_tokens=256, messages = [{"role": "user", "content": f"{prompt}"}], port = None, api_key = api_key)

            if dataset_name == "QRData":
                if not response or not response.strip():
                    print(f"Empty response received (attempt {retries+1}), retrying...")
                    retries += 1
                    time.sleep(3)  
                    continue
                
                if "## The accuracy score is:" not in response:
                    print(f"Invalid response format (attempt {retries+1}), retrying...")
                    retries += 1
                    time.sleep(3)
                    continue
                
                return "1" in response.split("## The accuracy score is:")[-1].lower(), response

            elif dataset_name == "DiscoveryBench":
                return "true" in response.split("## The final answer is:")[-1].lower(),response

        except Exception as e:
            print(f"Error in GPT checking (attempt {retries + 1}): {e}")
            retries += 1
            time.sleep(1)

    # If all attempts fail, return False
    return False, response



def print_colored(role, content, find_error=False):
    if role == 'user':
        if find_error:
            print(Fore.LIGHTRED_EX + "***Find the error step***: " + content)
        else:
            print(Fore.YELLOW + content)  
    elif role == 'assistant':
        print(Fore.CYAN + content)  


def save_final_messages(messages, sample_dir, question, answer, question_index, correct_flag, resp):
    final_file_path = os.path.join(sample_dir, f"question_{question_index}_final_answer.json")
    
    print(f"Saving final messages to {final_file_path}")
    os.makedirs(os.path.dirname(final_file_path), exist_ok=True)  # Make sure the directory exists
    final_data = {
        "question": question,
        "messages": messages,
        "answer": answer,
        "correct_flag": correct_flag,  # Add correctness flag
        "reason": resp
    }
    with open(final_file_path, "w") as f:
        json.dump(final_data, f, ensure_ascii=False, indent=4)
        f.write('\n')

def load_samples(data_root, dataset_name):
    if dataset_name == "QRData":
        data_path = os.path.join(data_root, "QRData/benchmark", "QRData.json")
        with open(data_path, "r") as f:
            samples = json.load(f)
        return samples, data_path

    elif dataset_name == "DiscoveryBench":
        id_to_metadata = {}
        for subject_name in os.listdir(data_root + '/DiscoveryBench'):
            file_dir = os.path.join(data_root, 'DiscoveryBench', subject_name)
            if file_dir.endswith('.csv'): continue
            for file_name in os.listdir(file_dir):
                file_path = os.path.join(file_dir, file_name)
                if not file_path.endswith('.json'): continue
                with open(file_path, 'r') as f:
                    metadata = json.load(f)
                file_path_split = file_path.split('/')
                # subject
                subject = subject_name
                # metadata id
                metadata_id = file_path_split[-1].split('_')[1].split('.')[0]
                queries = metadata['queries']
                if len(queries) != 1: print('Check what is in queries.')
                for query in queries[0]:
                    metadata['question'] = query['question']
                    metadata['qid'] = query['qid']
                    metadata['question_type'] = query['question_type']
                    id_to_metadata[subject + '_' + metadata_id + '_' + str(query['qid'])] = copy.deepcopy(metadata)

        DiscoveryBench = pd.read_csv(data_root + '/DiscoveryBench/answer_key_real.csv')
        data_samples = []
        for sample in DiscoveryBench.iterrows():
            sample = sample[1]
            sample_id = sample['dataset'] + '_' + str(sample['metadataid']) + '_' + str(sample['query_id'])
            metadata = id_to_metadata[sample_id]
            metadata['sample_id'] = sample_id

            for dataset in metadata['datasets']:
                if len(dataset['columns']) > 1: print('Check what is in the dataset columns')
            metadata['file_paths'] = [data_root + 'DiscoveryBench/' + sample['dataset'] + '/' + dataset['name'] for
                          dataset in metadata['datasets']]
            metadata['subject'] = sample['dataset']
            metadata['answer'] = sample['gold_hypo']
            metadata['description'] = [dataset['description'] for dataset in metadata['datasets']]
            metadata['column_metadata'] = [{'columns': dataset['columns']['raw']} for dataset in metadata['datasets']]
            data_samples.append(metadata)
        return data_samples, data_root


def extract_first_thought(response_content):
    """
    Extract the response content:
    1. If ## Thought exists, extract the content from the first ## Thought (inclusive) to the next delimiter
       (Delimiters include: ## Thought, ## Final Answer, ## Observation)
    2. If only ## Final Answer exists, return the content with ## Final Answer
    3. If neither exists, return None

    Args:
        response_content (str): Original response content

    Return:
        str: The extracted content
    """
    # First, try matching ## Thought to the next delimiter
    pattern = r'(## Thought:.*?)(?=## Thought:|## Final Answer:|## Observation:|$)'
    thought_match = re.search(pattern, response_content, re.DOTALL)
    
    if thought_match:
        return thought_match.group(1).strip()
    
    # If there is no ## Thought, try matching ## Final Answer
    final_answer_pattern = r'(## Final Answer:.*?)$'
    final_answer_match = re.search(final_answer_pattern, response_content, re.DOTALL)
    
    if final_answer_match:
        return final_answer_match.group(1).strip()
    
    # If there is neither ## Thought nor ## Final Answer, return None
    return None
