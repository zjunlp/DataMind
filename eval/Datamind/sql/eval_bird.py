import codecs
from dataclasses import dataclass
from datetime import datetime
import json
import math
from pathlib import Path
import re
import shutil
import sys
import time
import traceback
from typing import Any, Dict
from anthropic import Anthropic
from openai import OpenAI
from pprint import pprint
import logging
import os
import pandas as pd
from tqdm import tqdm
import tiktoken
from interpreter import SQLInterpreter
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import sqlite3
import argparse

class TqdmLogHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

console_handler = TqdmLogHandler()
console_formatter = logging.Formatter(
    fmt="%(asctime)s %(name)s:%(levelname)s:%(message)s",
    datefmt="%m-%Y %H:%M:%S"
)
console_handler.setFormatter(console_formatter)


os.makedirs('./logs', exist_ok=True)  
log_filename = os.path.join("logs", f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
file_handler = logging.FileHandler(log_filename, encoding='utf-8')

file_formatter = logging.Formatter(
    fmt="%(asctime)s [%(levelname)s] %(name)s:%(filename)s:%(lineno)d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
file_handler.setFormatter(file_formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)
logger.propagate = False

class OpenaiGenerator():
    def __init__(self, model_name, max_response_length):
        self.client = OpenAI(
            max_retries=10,
            timeout=300.0
        )
        self.model_name = model_name
        self.max_response_length = max_response_length

    def respond(self, messages, temperature, top_p):
        response = self.openai_chat(
            self.client,
            self.model_name,
            messages,
            temperature,
            top_p,
            self.max_response_length
        )
        if response is not None:
            result = response.choices[0].message.content
        else:
            result = ""
        return result

    def openai_chat(self, client, model_name, msg, temperature, top_p, max_response_length):
        if "gpt-5" in model_name:
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=msg,
                    temperature=temperature,
                )
            except Exception as e:
                logger.error(f"Error in OpenAI chat: {e}")
                response = None
        elif model_name.lower().startswith("o3") or model_name.lower().startswith('o4'):
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=msg,
                    temperature=temperature,
                )
            except Exception as e:
                logger.error(f"Error in OpenAI chat: {e}")
                response = None
        elif model_name.startswith("gpt"):
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=msg,
                    temperature=temperature,
                    max_tokens=max_response_length,
                    top_p=top_p,
                    frequency_penalty=0,
                    presence_penalty=0,
                    stop=["<interpreter>"]
                )
            except Exception as e:
                logger.error(f"Error in OpenAI chat: {e}")
                response = None
        else:
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=msg,
                    temperature=temperature,
                    max_tokens=max_response_length,
                    top_p=top_p,
                    stop=["<interpreter>"]
                )
            except Exception as e:
                logger.error(f"Error in OpenAI chat: {e}")
                response = None
        return response

class OssGenerator():
    def __init__(self, model_name, max_response_length):
        self.client = OpenAI(
            max_retries=10,
            timeout=120.0
        )
        self.model_name = model_name
        self.max_response_length = max_response_length

    def respond(self, messages, temperature, top_p):
        response = self.openai_chat(
            self.client,
            self.model_name,
            messages,
            temperature,
            top_p,
            self.max_response_length
        )
        result_text = response.choices[0].message.content
        if "assistantfinal" in result_text:
            result = result_text.split("assistantfinal")[1].strip()
        else:
            result = result_text
        return result

    def openai_chat(self, client, model_name, msg, temperature, top_p, max_response_length):

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=msg,
                temperature=temperature,
                max_tokens=max_response_length,
                top_p=top_p,
                stop=["<interpreter>"]
            )
        except Exception as e:
            logger.error(f"Error in OpenAI chat: {e}")
            response = None

        return response
    

class ClaudeGenerator():

    def __init__(self, model_name, max_response_length):
        self.client = Anthropic()
        self.model_name = model_name
        self.max_response_length = max_response_length

    def respond(self, user_input, temperature, top_p):
        response = self.claude_chat(
            self.client,
            self.model_name,
            user_input,
            temperature,
            top_p,
            self.max_response_length
        )

        if response is not None:
            result = response
        else:
            result = ""

        return result

    def claude_chat(self, client, engine, msg, temperature, top_p, max_response_length):
        try:
            response = client.messages.create(
                max_tokens=max_response_length,
                messages=msg,
                model=engine,
                temperature=temperature,
                top_p=top_p
            )
        except Exception as e:
            response = None

        return response.content[0].text


@dataclass
class LLMConfig:
    max_turns: int = 2,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_response_length: int = 4096,
    max_obs_length: int = 1024,
    working_dir: str = './workspace',
    working_temp_dir: str = './workspace/tmp',
    working_file_name: str = 'runfile.py',
    pred_csv_results_dir: str = './data'
    csv_folder: str = './data/files',
    max_prompt_length: int = 2048,
    format_tb_ipython: bool = False,
    skip_lib: bool = True,
    trace_back_len: int = 1,
    db_schema_data_path: str = './data/db_schema',

class LLM:
    def __init__(
            self,
            config: LLMConfig,
            model_name
    ):
        self.model_name = model_name
        self.config = config
        self.engine = None
        self.max_response_length = config.max_response_length
        self.interpreter = SQLInterpreter(config=config)
        if model_name.startswith("claude"):
            self.engine = ClaudeGenerator(model_name, self.max_response_length)
        else:
            self.engine = OpenaiGenerator(model_name, self.max_response_length)

    def respond(self, msg, temperature, top_p):
        response = self.engine.respond(msg, temperature, top_p)

        return response

    def prepare_workspace(self):
        working_path = Path(self.config.working_dir).resolve()

        if not os.path.exists(working_path):
            os.makedirs(working_path, exist_ok=True)

        origin_workdir = os.getcwd()

        os.chdir(str(working_path))

        # copy the csv from local to workspace
        src_folder = self.config.csv_folder
        dest_folder = working_path / 'data/files'
        self.copy_files(src_folder, dest_folder)

        os.chdir(str(origin_workdir))

    def count_tokens(self, messages, model="gpt-4"):
        encoding = tiktoken.encoding_for_model(model)
        if model.startswith("gpt-3.5"):
            tokens_per_message = 4
        elif model.startswith("gpt-4"):
            tokens_per_message = 3
        else:
            raise NotImplementedError("Token count rules not implemented for this model.")

        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for message in messages:
                num_tokens += len(encoding.encode(message['content']))
            num_tokens += tokens_per_message
        return num_tokens

    def val_prompt_length(self, eval_data):
        input_tokens = self.count_tokens(eval_data['prompt'])
        if input_tokens > self.config.max_prompt_length:
            logger.warning(
                f"[warning] Your prompt is over length, prompt len {input_tokens}, config.max_prompt_length {self.config.max_prompt_length}")

    def _postprocess_responses(self, gen_output):
        raw_responses = gen_output

        responses_str = raw_responses.split('</code>')[0] + '</code>' if '</code>' in raw_responses else raw_responses.split('</answer>')[0] + '</answer>' if '</answer>' in raw_responses else raw_responses
        return raw_responses, responses_str

    def parse_from_text(self, text: str):
        matches = re.findall(r'''execute_sql\s*\(\s*sql\s*=\s*(["']{1,3})(.*?)\1\s*,\s*output_path\s*=\s*(["']{1,3})(.*?)\3\s*\)''', text, re.VERBOSE | re.DOTALL)
        if matches:
            _, sql, _, output_path = (item.strip() for item in matches[-1])
            if sql.startswith('"') and sql.endswith('"'):
                sql = sql[1:-1]  # Remove surrounding quotes if present
            if output_path.startswith('"') and output_path.endswith('"'):
                output_path = output_path[1:-1]  # Remove surrounding quotes if present

            unescaped_sql = codecs.decode(sql, 'unicode_escape')
            return unescaped_sql, output_path
        else:
            print(f"[SQLInterpreter] No valid SQL command found in text: {text}")
            return None, None
        
    def _execute_sql(self, task_id, code: str, db_id: str = None) -> str:
        working_path = Path(self.config.working_dir).resolve()
        working_tmp_path = Path(self.config.working_temp_dir).resolve()

        # if working_dir doesn't exist, make it
        if not os.path.exists(working_path):
            os.makedirs(working_path, exist_ok=True)

        # if working_dir doesn't exist, make it
        if not os.path.exists(working_tmp_path):
            os.makedirs(working_tmp_path, exist_ok=True)

        origin_workdir = os.getcwd()

        os.chdir(str(working_path))

        if "get_db_info()" in code:
            result, report = self.interpreter.run_code(task_id, None, db_id, None, view_db=True)
        else:
            sql, output_path = self.parse_from_text(code)
            if sql is None:
                return "No valid SQL command found. You should use execute_sql(sql=\"a valid SQL query here\", output_path=\"result.csv\") function to execute your SQL command."

            result, report = self.interpreter.run_code(task_id, sql, db_id, output_path, view_db= False)

        result_str = str(result).strip()
        report_str = str(report).strip()
        if report_str == "Done":
            if result_str != '[]':
                final_result = f"The code run successfully:\n{result_str}"
            else:
                final_result = "The SQL query you provided didn't return any output. Please double-check your query logic, table names, and filters, and rewrite the code"
        else:
            final_result = f"The code run failed:\n{result_str}"

        os.chdir(str(origin_workdir))

        return final_result
    
    def execute_code(self, task_id, prediction: str, db_id: str):
        """
        Execute predictions across multiple environments.
        NOTE: the function is the actual `step` function in the environment
        NOTE penalty_for_invalid is not included in observation shown to the LLM

        Args:
            predictions: List of action responses
            pad_token: Token to use for padding

        Returns:
            List of observation strings
        """
        cur_action, content = self.postprocess_prediction(prediction)

        if cur_action == "code":
            generation_code = content
            if "```" in generation_code:
                pattern = r"```(?:sql\s*)?\n([\s\S]+?)```"
                matches = re.findall(pattern, generation_code)
                if matches:
                    generation_code = matches[0].strip()
            exec_result = self._execute_sql(task_id, generation_code, db_id)

        if cur_action == "answer":
            next_ob = ""
            done = 1
        elif cur_action == "code":
            next_ob = f"<interpreter>\n{exec_result.strip()}\n</interpreter>"
            done = 0
        else:
            next_ob = f"Your previous action is invalid. \
If you want to execute the code for the execution result, you should put the code between <code> and </code>. \
If you want to give the final answer, you should put the answer between <answer> and </answer>. Please try again."
            done = 0

        return next_ob, done

    def postprocess_prediction(self, prediction: Any):
        """
        Process (text-based) predictions from llm into actions and validity flags.

        Args:
            predictions: List of raw predictions

        Returns:
            Tuple of (actions list, validity flags list)
        """

        if isinstance(prediction, str):  # for llm output
            pattern = r'<(code|answer)>(.*?)</\1>'
            match = re.search(pattern, prediction, re.DOTALL)
            if match:
                content = match.group(2).strip()  # Return only the content inside the tags
                action = match.group(1)
            else:
                content = ''
                action = None
        else:
            raise ValueError(f"Invalid prediction type: {type(prediction)}")

        return action, content

    def extract_files_from_code(self, code):
        pattern = r'["\']([^"\']*\.(?:txt|csv|json|yaml|pickle|h5|xml|dat))["\']'
        return re.findall(pattern, code)

    def _execute(self, code: str = None) -> str:
        """
        Batchified execute for codes.
        Args:
            codes: codes to be executed
        Returns:
            execution results which is concatenated into a string
        """

        working_path = Path(self.config.working_dir).resolve()
        working_tmp_path = Path(self.config.working_temp_dir).resolve()

        # if working_dir doesn't exist, make it
        if not os.path.exists(working_path):
            os.makedirs(working_path, exist_ok=True)

        # if working_dir doesn't exist, make it
        if not os.path.exists(working_tmp_path):
            os.makedirs(working_tmp_path, exist_ok=True)

        origin_workdir = os.getcwd()

        os.chdir(str(working_path))

        # logger.info(f"code: {code}")

        result, report = self.interpreter.apply((0, code))

        if isinstance(result, str):
            if result.strip() != "" and report.strip() == "Done":
                exec_result = f"The code run successfully:\n{result}"
            elif report.strip() != "Done" and result.strip() == "":
                exec_result = f"The code run failed:\n{report}"
            elif report.strip() != "" and result.strip() != "":
                exec_result = f"The code run failed:\n{report}\n\nBut we capture part of your code output:\n{result}"
            else:
                exec_result = "We couldn't capture the output from your code. Please rewrite your last step code and modify it to explicitly use print() statements to display the values of any variables you want to inspect. Make sure to return the complete and corrected version of the code, and ensure that it can run successfully."
        else:
            exec_result = "We couldn't capture the output from your code. Please rewrite your last step code and modify it to explicitly use print() statements to display the values of any variables you want to inspect. Make sure to return the complete and corrected version of the code, and ensure that it can run successfully."

        os.chdir(str(origin_workdir))

        return exec_result

    def copy_files(self, src_folder, dest_folder):
        if not os.path.exists(dest_folder):
            os.makedirs(dest_folder)

        assert os.path.exists(src_folder), f"The src_folder {os.path.abspath(src_folder)} not exist!"

        for filename in os.listdir(src_folder):
            src_file = os.path.join(src_folder, filename)
            dest_file = os.path.join(dest_folder, filename)

            if not os.path.isfile(src_file):
                continue
            
            if os.path.exists(dest_file):
                continue
            
            shutil.copy(src_file, dest_file)

    def exception_summary(self, e, working_dir, exec_file_name, format_tb_ipython):
        """Generates a string that summarizes an exception and its stack trace (either in standard python repl or in IPython format)."""
        if format_tb_ipython:
            import IPython.core.ultratb

            # tb_offset = 1 to skip parts of the stack trace in weflow code
            tb = IPython.core.ultratb.VerboseTB(tb_offset=1, color_scheme="NoColor")
            tb_str = str(tb.text(*sys.exc_info()))
        else:
            tb_lines = traceback.format_exception(e)
            # skip parts of stack trace in weflow code
            tb_str = "".join(
                [l for l in tb_lines if "importlib" not in l]
            )

        # replace whole path to file with just filename (to remove agent workspace dir)
        tb_str = tb_str.replace(str(working_dir + '/' + exec_file_name), exec_file_name)

        exc_info = {}
        if hasattr(e, "args"):
            exc_info["args"] = [str(i) for i in e.args]
        for att in ["name", "msg", "obj"]:
            if hasattr(e, att):
                exc_info[att] = str(getattr(e, att))

        tb = traceback.extract_tb(e.__traceback__)
        exc_stack = [(t.filename, t.lineno, t.name, t.line) for t in tb]

        return tb_str, e.__class__.__name__, exc_info, exc_stack

    def extract_error_details(self, error_string):
        pattern = re.compile(r'File "(?P<file>.*?)", line (?P<line>\d+), in (?P<module>.*?)\n\s*(?P<code>.*?)\n')

        matches = pattern.finditer(error_string)

        error_details = []
        for match in matches:
            details = match.groupdict()
            details['line'] = int(details['line'])
            error_details.append(details)

        return error_details

    def exception_summary_str(self, tb_str, working_dir, exec_file_name):
        """Generates a string that summarizes an exception and its stack trace"""
        tb_str = tb_str.replace(str(working_dir + '/' + exec_file_name), exec_file_name)

        error_pattern = r'(\w+Error): (.+?)(?=\n\s+\w|$)'

        match_class = re.search(error_pattern + r'(?!.*' + error_pattern + ')', tb_str, re.DOTALL)

        if not match_class:
            exc_info = ''
        else:
            error_class = match_class.group(1)  # ZeroDivisionError
            error_message = match_class.group(2)  # division by zero

            exc_info = f"{error_class}: {error_message}"

        exc_stack = []
        error_details = self.extract_error_details(tb_str)

        for err in error_details:
            if self.config.skip_lib:
                if 'site-packages' in err['file']:
                    continue
                else:
                    exc_stack.append((err['file'], err['line'], err['module'], err['code']))
            else:
                exc_stack.append((err['file'], err['line'], err['module'], err['code']))

        err_msg = "Traceback (most recent call last):\n"

        if len(exc_stack) > 0:
            result_len = self.config.trace_back_len * -1
            concat_result = "\n".join([f"File {filename}, Line: {lineno}, in {module}\n{code}"
                                       for filename, lineno, module, code in exc_stack[result_len:]])

            err_msg = err_msg + concat_result + "\n"
        else:
            err_msg += "No traceback information available.\n"

        if exc_info:
            err_msg += exc_info

        return err_msg

    def _process_next_ob(self, next_ob: str):
        """Process next observations from environment."""
        if next_ob.startswith("<interpreter>\nThe code run successfully:\nThe output content is CREATE TABLE"):
            if len(next_ob) > 10 * self.config.max_obs_length:
                logger.warning(
                f"[WARNING] CREATE TABLE OBSERVATION TOO LONG, CONSIDER CHANGING YOUR CONFIG, {len(next_ob)} & {10 * self.config.max_obs_length}")
                next_ob = next_ob[: 10 * self.config.max_obs_length - 14] + "</interpreter>"
        else:
            if len(next_ob) > self.config.max_obs_length:
                logger.warning(
                    f"[WARNING] OBSERVATION TOO LONG, CONSIDER CHANGING YOUR CONFIG, {len(next_ob)} & {self.config.max_obs_length}")
                next_ob = next_ob[: self.config.max_obs_length - 14] + "</interpreter>"

        return next_ob

    def _update_rolling_state(self, prompt, cur_response, next_ob) -> Dict:
        """Update rolling state with new responses and observations."""
        # Concatenate and handle padding
        # breakpoint()

        assistant_prompt = {"role": "assistant", "content": cur_response}
        observation_prompt = {"role": "user", "content": next_ob}

        prompt.append(assistant_prompt)
        prompt.append(observation_prompt)

        return prompt

    def run_llm_loop(self, task_id, eval_data, db_id):
        active_mark = 1
        final_response = ""
        inputs = list(eval_data)

        # If db_id is provided, use SQLInterpreter
        # logger.info("[INFO] Using SQLInterpreter for database queries.")
        self.interpreter = SQLInterpreter(config=self.config)

        self.prepare_workspace()
        self.prepare_workspace()

        # Main generation loop
        for step in range(self.config.max_turns):
            if not active_mark:
                break

            # generate response
            logger.debug(f"[DEBUG] gen start")
            gen_output = self.respond(inputs, self.config.temperature, self.config.top_p)

            logger.debug(f"[DEBUG] gen end")

            # decode the answer and extract <code> <answer> or do no action if they don't appear in the response
            # generate new responses_ids by decode responses with skip-special-token and then encode it
            raw_response, response_str = self._postprocess_responses(gen_output)
        
            final_response += response_str

            logger.debug(f"[DEBUG] execute code start")
            # Execute in environment and process observations
            next_ob, done = self.execute_code(task_id, response_str, db_id)
            logger.debug(f"[DEBUG] execute code end")

            if done:
                active_mark = 0

            # tokenize obs and padding it
            next_ob = self._process_next_ob(next_ob)

            final_response += next_ob

            # Update states
            inputs = self._update_rolling_state(
                inputs,
                response_str,
                next_ob
            )

        # final LLM rollout
        if active_mark:
            logger.debug(f"[DEBUG] final LLM rollout")
            # generate response
            logger.debug(f"[DEBUG] gen start")
            gen_output = self.respond(inputs, self.config.temperature, self.config.top_p)
            logger.debug(f"[DEBUG] gen end")

            # decode the answer and extract <code> <answer> or do no action if they don't appear in the response
            # generate new responses_ids by decode responses with skip-special-token and then encode it
            raw_response, response_str = self._postprocess_responses(gen_output)

            final_response += response_str

            logger.debug(f"[DEBUG] execute code start")
            # Execute in environment and process observations
            next_ob, done = self.execute_code(task_id, response_str, db_id)
            logger.debug(f"[DEBUG] execute code end")

            # tokenize obs and padding it
            next_ob = self._process_next_ob(next_ob)

            final_response += next_ob

            # Update states
            inputs = self._update_rolling_state(
                inputs,
                response_str,
                next_ob
            )

        return final_response, inputs

def extract_answer(text: str):
    text = text.strip()

    pattern = r"<answer>(.*?)</answer>"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return ""

    return match.group(1)

def read_parquet(file_path):
    return pd.read_parquet(file_path).to_dict(orient='records')


def load_jsonl_to_dict(jsonl_file):
    data_dict = {}
    with open(jsonl_file, 'r') as file:
        for line in file:
            item = json.loads(line.strip())
            instance_id = item['instance_id']
            data_dict[instance_id] = item
    return data_dict

def reward_function(trajectory: list, prediction: str, id: str, gold_csv_results_dir: str, pred_csv_results_dir: str, test_bench: str) -> float:
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

    def get_csv_name_from_answer(id: str, content: str) -> str:
        csv_pattern = r"(?:['\"“]?)([^'\"\s]*\.csv)(?:['\"”]?)"
        if "<answer>" in content and "</answer>" in content:
            answer = content.split("<answer>")[1].split("</answer>")[0].strip()
            # extract csv name
            match = re.search(csv_pattern, answer)
            if match:
                csv_file = match.group(1)
                return csv_file
            else:
                logger.error(f"Extract {id} CSV file name error. CSV file name not found in the answer content: {content}")
                return "result.csv"
        else:
            match = re.search(csv_pattern, content)
            if match:
                csv_file = match.group(1)
                return csv_file
            else:
                logger.error(f"Extract {id} CSV file name error. CSV file name not found in the content: {content}")
                return "result.csv"
    
    def check():
        traj = trajectory
        last_conv = traj[-1] if traj else None
        if last_conv["role"] == "assistant":
            if "<answer>" not in last_conv["content"] or "</answer>" not in last_conv["content"]:
                logger.info(f"[INFO] last_conv error in trajectory: {last_conv['content']}")
                return False
        else:
            logger.info(f"[INFO] last_conv role error in trajectory: {last_conv['role']}")
            return False 

        for index, conv in enumerate(traj):
            if conv["role"] == "assistant":
                content = conv["content"]
                if "<think>" in content and "</think>" in content:
                    content_0 = content.split("<think>")[0].strip()
                    content_1 = content.split("</think>")[1].strip()
                    content = content_0 + content_1

                if index != len(traj)-1:
                    if "<code>" in content and "</code>" in content:
                        content_0 = content.split("<code>")[0].strip()
                        content_1 = content.split("</code>")[1].strip()
                        content = content_0 + content_1
                else:
                    if "<answer>" in content and "</answer>" in content:
                        content_0 = content.split("<answer>")[0].strip()
                        content_1 = content.split("</answer>")[1].strip()
                        content = content_0 + content_1
                
                if content.strip() != "":
                    logger.info(f"[INFO] Template error in trajectory: {content.strip()}")
                    return False
        return True
    
    template_reward = 1.0 if check() else 0.0
    answer_reward = 0.0
    if prediction.strip() == "":
        answer_reward = 0.0
        return template_reward * 0.1 + answer_reward * 0.9, template_reward, answer_reward

    try:
        pred_csv_results_dir = pred_csv_results_dir
        gold_csv_results_dir = gold_csv_results_dir
        csv_name = get_csv_name_from_answer(id, prediction)
        pred_csv_path = os.path.join(pred_csv_results_dir, id, csv_name)
        if os.path.exists(pred_csv_path) == False:
            answer_reward = 0.0
            logger.error(f"[ERROR] Predicted CSV file not found at path: {pred_csv_path}")
            return template_reward * 0.1 + answer_reward * 0.9, template_reward, answer_reward

        gold_csv_path = os.path.join(gold_csv_results_dir, f"{id}.csv")

        score = compare_pandas_table(
            pd.read_csv(pred_csv_path),
            pd.read_csv(gold_csv_path),
            ignore_order=True
        )
    except Exception as e:
        logger.error(f"[ERROR] Error in reward function: {e}")
        score = 0

    if score == 1:
        answer_reward = 1.0
    else:
        answer_reward = 0.0

    return template_reward * 0.1 + answer_reward * 0.9, template_reward, answer_reward


def get_final_answer(config, model, question, file_dir, tmp_dir, output_file, file_lock, result_list):
    llmconfig = LLMConfig(
        max_turns=config['max_turns'],
        temperature=config['temperature'],
        top_p=config['top_p'],
        max_response_length=config['max_response_length'],
        max_obs_length=config['max_obs_length'],
        working_dir=config['working_dir'],
        working_temp_dir=config['working_temp_dir'],
        working_file_name=config['working_file_name'],
        pred_csv_results_dir=config['pred_csv_results_dir'],
        csv_folder=config['csv_folder'],
        max_prompt_length=config['max_prompt_length'],
        format_tb_ipython=config['format_tb_ipython'],
        skip_lib=config['skip_lib'],
        trace_back_len=config['trace_back_len'],
        db_schema_data_path=config['db_schema_data_path'],
    )
    llm_model = LLM(llmconfig, model)
    
    output, prompt = llm_model.run_llm_loop(question["extra_info"]["index"], question["prompt"], question["extra_info"]["data_name"])
    if prompt[-1]["role"] != "assistant":
        prompt = prompt[:-1]  # remove the last prompt if it is not assistant role
    # answer = extract_answer(output)
    answer = extract_answer(prompt[-1]["content"])

    db_id = question["extra_info"]["data_name"] if question["extra_info"]["data_name"].endswith('.sqlite') else question["extra_info"]["data_name"] + '.sqlite'
    reward = reward_function(
        trajectory=prompt,
        prediction=answer,
        id=question["extra_info"]["index"],
        gold_csv_results_dir = config['gold_csv_results_dir'],
        pred_csv_results_dir = config['pred_csv_results_dir'],
        test_bench=config['test_bench']
    )
    
    response = {
        "id": question["data_source"] + "-" + question["extra_info"]["data_name"] + "-" + str(question["extra_info"]["index"]),
        "question": question["prompt"][-1]["content"],
        "answer": answer,
        "ground_truth": question["reward_model"]["ground_truth"],
        "total_score": reward[0],
        "template_score": reward[1],
        "answer_score": reward[2],
        "traj": prompt
    }
    
    with file_lock:
        result_list.append(response)
        
        if len(result_list) > 0:
            total_score = sum(item['total_score'] for item in result_list if 'total_score' in item)
            avg_score = total_score / len(result_list)
            logger.info(f"[INFO] Average score: {avg_score:.4f}")

            template_score = sum(item['template_score'] for item in result_list if 'template_score' in item)
            avg_template_score = template_score / len(result_list)
            logger.info(f"[INFO] Average template score: {avg_template_score:.4f}")

            answer_score = sum(item['answer_score'] for item in result_list if 'answer_score' in item)
            avg_answer_score = answer_score / len(result_list)
            logger.info(f"[INFO] Average answer score: {avg_answer_score:.4f}")

            score = {
                "total_score": avg_score,
                "template_score": avg_template_score,
                "answer_score": avg_answer_score
            }
        else:
            score = {
                "total_score": 0.0,
                "template_score": 0.0,
                "answer_score": 0.0
            }
        
        result_dict = {
            "score": score,
            "result": result_list.copy()
        }

        with open(output_file, "w", encoding='utf-8') as f:
            json.dump(result_dict, f, indent=4, ensure_ascii=False)
    
    return response

def main(config, model, question_file, output_file, gen_num, file_dir, tmp_dir):
    question_list = question_file
    output_data = []

    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            output_data = json.load(f)

    output_dict = output_data["result"] if "result" in output_data else output_data
    logger.info(f"[INFO] Loaded {len(output_dict)} previous results from {output_file}")
    
    if "numeric" in output_dict and "multiple_choice" in output_dict:
        output_list = output_dict["numeric"] + output_dict["multiple_choice"]
    elif "numeric" in output_dict:
        output_list = output_dict["numeric"]
    elif "multiple_choice" in output_dict:
        output_list = output_dict["multiple_choice"]
    else:
        output_list = output_dict if isinstance(output_dict, list) else []
    result = output_list if isinstance(output_list, list) else []

    solved_question_ids = [item['id'] for item in output_list if 'id' in item]
    logger.info("[INFO] Total solved questions: {}".format(len(solved_question_ids)))

    # get to solve questions from question_list
    to_solve_question_list = [q for q in question_list if q["data_source"] + "-" + q["extra_info"]["data_name"] + "-" + str(q["extra_info"]["index"]) not in solved_question_ids]
    if len(to_solve_question_list) == 0:
        logger.info("[INFO] All questions have been solved.")
        return
    
    logger.info(f"[INFO] Total questions to solve: {len(to_solve_question_list)}")

    file_lock = threading.Lock()
    
    with ThreadPoolExecutor(max_workers=min(os.cpu_count(), gen_num)) as executor:
        futures = []
        for question in to_solve_question_list:
            futures.append(executor.submit(get_final_answer, config, model, question, file_dir, tmp_dir, output_file, file_lock, result))
        
        with tqdm(total=len(futures), desc="Processing questions") as pbar:
            for future in as_completed(futures):
                try:
                    response = future.result()
                    logger.info(f"[INFO] Completed question: {response['id']}")
                except Exception as e:
                    logger.error(f"[Error] {e}")
                    traceback.print_exc()
                pbar.update(1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen2.5-coder-7b")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--bs", type=int, default=5)
    parser.add_argument("--test_bench", type=str, default="bird", choices=["bird"])
    parser.add_argument("--test_file", type=str)
    parser.add_argument("--csv_or_db_folder", type=str)
    parser.add_argument("--gold_csv_results_dir", type=str)
    parser.add_argument("--db_schema_data_path", type=str)
    return parser.parse_args()

if __name__ == "__main__":
    current_absolute_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_absolute_path)

    args = parse_args()

    model = args.model
    temperature = args.temperature
    top_p = args.top_p
    bs = args.bs

    test_bench = args.test_bench
    test_file = args.test_file

    working_dir = f'{current_dir}/eval_result/{test_bench}/workspace'
    working_data_dir = f'{current_dir}/eval_result/{test_bench}/workspace/data'
    working_temp_dir = f'{current_dir}/eval_result/{test_bench}/workspace/tmp'
    
    for index in range(3):
        output_filename = f"sql_{model}_traj_t{temperature}_topp{top_p}_bs{bs}_{test_bench}_test_{index}"
        output_file = f'{current_dir}/eval_result/{test_bench}/{output_filename}.json'

        if test_bench == "bird":
            question_file = read_parquet(test_file)
            csv_folder = args.csv_or_db_folder
            pred_csv_results_dir = f"{working_data_dir}/{output_filename}"
            gold_csv_results_dir = args.gold_csv_results_dir
            db_schema_data_path = args.db_schema_data_path
        else:
            raise ValueError(f"Unsupported test bench: {test_bench}")
        
        gen_num = bs


        config = {
            "test_bench": test_bench,
            "max_turns" : 9,
            "temperature": temperature,
            "top_p": top_p,
            "max_response_length": 4096,
            "max_obs_length": 2048,
            "working_dir": working_dir,
            "working_temp_dir": working_temp_dir,
            "working_file_name": 'runfile.py',
            "csv_folder": csv_folder,
            "max_prompt_length": 2048,
            "format_tb_ipython": False,
            "skip_lib": True,
            "trace_back_len": 1,
            "gold_csv_results_dir": gold_csv_results_dir,
            "pred_csv_results_dir": pred_csv_results_dir, 
            "db_schema_data_path": db_schema_data_path
        }
        file_dir = 'data/files'
        tmp_dir = 'data/tmp'
        
        main(config=config,model=model, question_file=question_file, output_file=output_file, gen_num=gen_num, file_dir=file_dir, tmp_dir=tmp_dir)
