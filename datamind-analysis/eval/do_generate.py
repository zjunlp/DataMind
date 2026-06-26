import os
import time
import random
from pandas.core import base
from tqdm import tqdm
from colorama import init, Fore
import logging
import yaml

from prompt import TEMPLATE_MAP
from chat import Chat_With_LLM_Messages
from data_process import *
from python_executor import CodeRunner

# Initialize colorama
init(autoreset=True)

def run_analysis(
    model_name,
    check_model,
    dataset_name,
    output_path,
    max_steps=25,
    api_port=8000,
    temperature = 0.7,
    top_p = 0.95,
    test_range = None,
    add_random = False
):
    """
    Solve each problem in the dataset
    """

    config = "config.yaml"
    with open(config, 'r', encoding='utf-8') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    data_root = configs['data_root']
    api_key = configs['api_key']

    # Load the dataset
    samples, data_path = load_samples(data_root, dataset_name)

    if not samples or not data_path:
        logging.error("Failed to load samples")
        return

    cur_dir = os.getcwd()

    # Create output_path if it does not exist
    output_path = os.path.join("results", output_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    begin_idx = None
    end_idx = None

    # Test range
    if test_range is not None:
        begin_idx, end_idx = test_range
        if begin_idx is not None and end_idx is not None:
            samples = samples[begin_idx:end_idx]
        elif begin_idx is not None:
            samples = samples[begin_idx:]
        elif end_idx is not None:
            samples = samples[:end_idx]

    # Configure logging
    log_file = os.path.join(
        output_path, 
        f"{begin_idx if begin_idx else 'start'}_{end_idx if end_idx else 'end'}_execution.log"
    )
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    # Initialize the necessary components
    code_runner = CodeRunner()

    if dataset_name == "QRData":
        table_folder = os.path.join(data_root, "QRData/data")
    elif dataset_name == "DiscoveryBench":
        table_folder = os.path.join(data_root, "DiscoveryBench/tables")

    question_index = 0
    if begin_idx:
        question_index = begin_idx

    # Iterate through each problem
    for sample in tqdm(samples):

        # Clear state before processing new sample
        code_runner.clear_state()
        code_history = []


        if dataset_name == "QRData":
            # Get data from the sample
            introduction = sample.get("data_description", "")
            excel_name = sample.get("data_files", "")

            file_path = ""
            for file in excel_name:
                file_path += file + "\n"
            
            # Add random files
            if add_random:
                all_tables = [f for f in os.listdir(table_folder)]
                while 1:
                    random_filepath = random.choice(all_tables)
                    if random_filepath not in excel_name:
                        break
                file_path += random_filepath

            excel_content = "The excel file path is:\" \n" + file_path +"\""

            question = sample.get("question", "")

            meta_data = sample.get("meta_data", {})   
            gt_answer = sample.get("answer", "")
            
        elif dataset_name == "DiscoveryBench":
            # Get data from the sample
            sample_id = sample['sample_id']
            subject = sample['subject']
            question = sample['question']
            gt_answer = sample['answer']
            question_type = sample['question_type']
            domain_knowledge = sample.get('domain_knowledge', None)
            workflow_tags = sample['workflow_tags']
            descriptions = sample['description']
            column_metadata = sample['column_metadata']
            file_paths = sample['file_paths']
            meta_data = None

            new_file_paths = []

            instruction = ''
            instruction += 'Below are the descriptions of the datasets and dataset columns.\n'

            for file_path, description, _column_metadata in zip(file_paths, descriptions, column_metadata):
                file_path = file_path.split('/')[-1]
                new_file_paths.append(file_path)

                instruction += 'Dataset ' + file_path + ': ' + description + '\n'
                instruction += 'Descriptions for the columns:\n'
                column_description = ''
                for column in _column_metadata['columns']:
                    column_description += 'Column name \'' + column['name'] + '\': ' + column['description'] + '\n'
                instruction += column_description
                instruction += '\n'

            if domain_knowledge:
                instruction += 'Domain Knowledge: ' + domain_knowledge

            # Add random files
            if add_random:
                all_tables = [f for f in os.listdir(table_folder)]
                while 1:
                    random_filepath = random.choice(all_tables)
                    if random_filepath not in new_file_paths:
                        break

                new_file_paths.append(random_filepath)


        # Check whether the sample has been processed
        if any(f"_{question_index}_final_answer" in f for f in os.listdir(output_path) if f.endswith('.json')):
            logging.info(f"Skip the processing results of the existing sample {question_index}.")
            question_index += 1
            continue

         # Prompts
        system_prompt_template = TEMPLATE_MAP["system_prompt_template"]

        if dataset_name == "QRData":
            input_prompt_template = TEMPLATE_MAP["input_prompt_template_QR"]
            input_prompt = input_prompt_template.format(
                introduction=introduction,
                excel_content=excel_content,
                question=question
            )
        elif dataset_name == "DiscoveryBench":
            input_prompt_template = TEMPLATE_MAP["input_prompt_template_DB"]
            input_prompt = input_prompt_template.format(
                descriptions=instruction,
                file_paths=new_file_paths,
                question=question
            )

        if 'DataMind' not in model_name:
            logging.info(f"system: {system_prompt_template}")
        
        logging.info(f"User Input: {input_prompt}")
        
         # Initialize the messages and the correct_flag
        if 'DataMind' not in model_name:
            messages = [
                {"role": "system", "content": system_prompt_template},
                {"role": "user", "content": input_prompt}
            ]
        else:
            messages = [
                {"role": "user", "content": input_prompt}
            ]

        correct_flag = False
        resp = ""
        current_state = None
        # Process the steps for the sample
        for step_count in range(max_steps):
            try:
                # response = chatter.chat(messages)
                response = Chat_With_LLM_Messages(messages=messages, model_name=model_name, temperature=temperature,max_tokens = 1024, top_p = top_p, port=api_port, api_key=api_key)
                step = extract_first_thought(response)
                
                if step is None:
                    logging.warning("Failed to get valid step, retrying...")
                    break
                    
                messages.append({"role": "assistant", "content": step})
                logging.info(f"Assistant Response: {step}")

                if "## Final Answer".lower() in step.lower():
                    if_correct, resp = check_answers_equiv(question, step, gt_answer, check_model, dataset_name, api_key)
                    correct_flag = if_correct
                    break
                    
                code_snippet = extract_code(step)
                if code_snippet:
                    current_state = code_runner.get_state()
                    out, err, has_error = code_runner.run_code(code_snippet,base_path=table_folder)
                    if has_error:
                        # Try to restore the state when an error occurs
                        if not code_runner.set_state(current_state, code_history):
                            logging.warning("Failed to restore state, clearing everything")
                            code_runner.clear_state()
                            code_history = []
                            break
                        obs_str = f"Error Code: {err}"
                    else:
                        # It is only added to the history record when the code execution is successful
                        code_history.append({"code": code_snippet})
                        obs_str = out if out else "[Executed Successfully with No Output]"
                    
                    messages.append({
                        "role": "user",
                        "content": f"## Observation: {obs_str}"
                    })
                    logging.info(f"## Observation: {obs_str}")

                else:
                    messages.append({
                        "role": "user",
                        "content": "## Observation: OK."
                    })
                    logging.info(f"## Observation: OK.")

                    
            except Exception as e:
                logging.error(f"Error in step {step_count}: {str(e)}")
                # Try to restore the state
                if current_state is not None:
                    if not code_runner.set_state(current_state, code_history):
                        logging.error("Failed to restore state after exception")
                        code_runner.clear_state()
                        code_history = []
                time.sleep(1)

        
        # Save the processing result of the current sample
        os.chdir(cur_dir)
        # resp
        save_final_messages(messages, output_path, question, gt_answer, question_index, correct_flag, resp)
        question_index += 1


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process samples for data analysis.")
    parser.add_argument('--model_name', type=str, default= "Qwen2.5-7B-Instruct", help='Model name to use')
    parser.add_argument('--check_model', type=str, default= "gpt-4o-mini", help='Check model to use')
    parser.add_argument('--output', type=str, default= "results", help='Output directory path')
    parser.add_argument('--dataset_name', type=str, default="QRData", help='Dataset name to use')
    parser.add_argument('--max_round', type=int, default=25, help='Maximum number of steps')
    parser.add_argument('--api_port', type=int, default=8000, help='API port number')
    parser.add_argument('--bidx', type=int, default=100, help='Begin index (inclusive)')
    parser.add_argument('--eidx', type=int, default=None, help='End index (exclusive)')
    parser.add_argument('--temperature', type = float, default=0.7, help='Temperature for sampling')
    parser.add_argument('--top_p', type = float, default=0.95, help='Top p for sampling')
    parser.add_argument('--add_random', type = bool, default=False, help='Whether to add random files')

    args = parser.parse_args()

    run_analysis(
        model_name=args.model_name,
        check_model=args.check_model,
        output_path=args.output,
        dataset_name=args.dataset_name,
        max_steps=args.max_round,
        api_port=args.api_port,
        temperature = args.temperature,
        top_p = args.top_p,
        test_range=(args.bidx, args.eidx),
        add_random = args.add_random
    )
