import os
import io
import subprocess
import re
import pickle
import traceback
from contextlib import redirect_stdout
import tempfile
import asyncio
import sqlite3
import threading
from typing import Tuple, Any, List, Set
from itertools import product
from collections import defaultdict
import random
import time
from itertools import chain
import pdb
from func_timeout import func_timeout, FunctionTimedOut
import shutil
import uuid
import gc
import sqlparse
import json
import pandas as pd

def run_task(code, timeout):
    running_code = "\n".join(code)

    def limit_memory():
        import resource
        mem = 1024 * 1024 * 1024  # 1GB
        resource.setrlimit(resource.RLIMIT_AS, (mem, mem))

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp_file:
        tmp_file.write(running_code)
        tmp_file_path = tmp_file.name

    try:
        result = subprocess.run(
            ["/root/anaconda3/bin/conda", "run", "-n", "darl-run", "python", tmp_file_path],
            capture_output=True,
            preexec_fn=limit_memory,
            timeout=timeout
        )
        stdout = result.stdout.decode("utf-8")
        stderr = result.stderr.decode("utf-8")
        
        filtered_stderr = "\n".join(
            line for line in stderr.splitlines()
            if not line.strip().startswith("ERROR conda.cli.main_run")
        )
        return stdout, filtered_stderr
    except subprocess.TimeoutExpired as e:
        return "", "TimeoutError: 'Timed Out'"
    except Exception as e:
        return "", e
    finally:
        os.remove(tmp_file_path)  # 清理临时文件

def check_module_in_env(env_name: str, module: str) -> bool:
    try:
        res = subprocess.run(
            ["/root/anaconda3/bin/conda", "run", "-n", "darl-run", "python", "-c", f"import {module}"],
            capture_output=True
        )
        return res.returncode == 0
    except Exception as e:
        print(f"[Interpreter error] Failed to check module '{module}' in env '{env_name}'")
        return False


class CSVInterpreter:
    def __init__(
        self,
        timeout_length: int = 5,
    ) -> None:
        self.timeout_length = timeout_length
        self.codes = []

    def extra_code(self, code):
        match = re.search(r'```python\s*(.*?)```', code, re.DOTALL)
        if match:
            extra_code = match.group(1).strip()
        else:
            extra_code = code

        return extra_code

    def process_generation_to_code(self, gen):
        code = self.extra_code(gen)
        g_split = code.split("\n")
        run_code = self.codes.copy()
        for c in g_split:
            run_code.append(c)

        return run_code

    @staticmethod
    def execute(
        code,
        timeout_length=10,
    ):
        try:
            result, report = run_task(code, timeout_length)
            result = str(result).strip()
            report = str(report).strip()
            if not report:
                report = "Done"
                pickle.dumps(result)  # serialization check
        except:
            result = ""
            report = traceback.format_exc().split("\n")[-2]

        if 'No module named' in report:
            match = re.search(r"No module named ['\"]([^'\"]+)['\"]", report)
            if match:
                missing_full_module = match.group(1)
                top_module = missing_full_module.split('.')[0]

                install_module_name = 'scikit-learn' if top_module == 'sklearn' else top_module

                if check_module_in_env("darl-run", top_module):
                    print(f"[Interpreter warning] Module '{top_module}' is installed in darl-run, but '{missing_full_module}' not found. Skip auto-install.")
                else:
                    pip_res = subprocess.run(
                        ["/root/anaconda3/bin/conda", "run", "-n", "darl-run", "pip", "install", install_module_name],
                        capture_output=True
                    )
                    if pip_res.returncode != 0:
                        print(f"[Interpreter pip error] an error occured when using pip to install missing package: {pip_res.stderr.decode('utf-8')}")
                    else:
                        print(f"[Interpreter pip] successfully pip install module: {install_module_name}, restart the code")
                        try:
                            result, report = run_task(code, timeout_length)
                            result = str(result).strip()
                            report = str(report).strip()
                            if not report:
                                report = "Done"
                                pickle.dumps(result)  # serialization check
                        except:
                            result = ""
                            report = traceback.format_exc().split("\n")[-2]
        return result, report

    def apply(self, code):
        return self.run_code([code])[0]

    @staticmethod
    def truncate(s, max_length=1024):
        half = max_length // 2
        if len(s) > max_length:
            s = s[:half] + "..." + s[-half:]
        return s

    def run_code(self, code):
        all_code_snippets = self.process_generation_to_code(code)

        try:
            result, report = self.execute(all_code_snippets, timeout_length=self.timeout_length)
        except TimeoutError as error:
            result = ""
            report = "Timeout Error"

        res, report = str(result).strip(), str(report).strip()
        res, report = self.truncate(res), self.truncate(report)
        # self.update_code(report, code)
        self.update_code(report, all_code_snippets)

        gc.collect()

        return res, report
    
    def update_code(self, report, code):
        if report == 'Done':
            self.codes=[]
            for c in code:
                if not (c.strip().startswith("print") or "print(" in c):
                    self.codes.append(c)
                else:
                    indent = re.match(r"^\s*", c).group(0)
                    self.codes.append(f"{indent}pass")

import multiprocessing
import time

def run_sql_worker(task_id, sqlite_path, query, output_path, return_dict):
    runner = SQLInterpreter() 
    output, status = runner.exec_sql(task_id, sqlite_path, query, output_path)
    return_dict['output'] = output
    return_dict['status'] = status

class SQLInterpreter:
    def __init__(
        self,
        timeout_length: int = 180,
        db_schema_data_path: str = 'the_path_to_your_db_schema_data.json'
    ) -> None:
        self.timeout_length = timeout_length
        self.db_schema_data_path = db_schema_data_path

    def replace_cur_year(self, query: str) -> str:
        return re.sub(
            "YEAR\s*\(\s*CURDATE\s*\(\s*\)\s*\)\s*", "2025", query, flags=re.IGNORECASE
        )
    
    def extra_sql_code(self, code):
        match = re.search(r'```sql\s*(.*?)```', code, re.DOTALL)
        if match:
            extra_code = match.group(1).strip()
        else:
            extra_code = code

        return extra_code

    def is_not_select_query(self, sql: str) -> bool:
        # keep the pragma stat
        parsed = sqlparse.parse(sql)
        if not parsed:
            return False # if fail, return False

        first_statement = parsed[0]

        return first_statement.get_type() == "CREATE" or first_statement.get_type() == "INSERT" or first_statement.get_type() == "UPDATE" or first_statement.get_type() == "DELETE"
    
    def exec_sql(self, task_id: str, sqlite_path: str, query: str, output_path: str) -> Tuple[str, Any]:

        result_dir = "data/results"
        if not os.path.exists(result_dir):
            os.makedirs(result_dir, exist_ok=True)
        os.makedirs("data/results/"+task_id, exist_ok=True)

        query = self.replace_cur_year(query)
        conn = sqlite3.connect(sqlite_path)

        try:
            # Execute the SQL command and fetch the results
            if self.is_not_select_query(query):
                return "Error: Only SELECT statements are allowed.", "Error: Only SELECT statements are allowed."
            try:
                df = pd.read_sql_query(query, conn)
            except pd.io.sql.DatabaseError as db_err:
                return f"Error: {str(db_err)}", f"Error: {str(db_err)}"
            
            # Check if the output should be saved to a CSV file or printed directly
            if output_path.lower().endswith(".csv"):
                save_path = os.path.join(result_dir, task_id, output_path)
                df.to_csv(save_path, index=False)

                if "sql FROM sqlite_master WHERE type='table'" in query:
                    database_schemas = "\n".join(df["sql"])
                    return f"Output has been saved to: {output_path}. The output content is {database_schemas}", "Done"
                else:
                    return f"Output has been saved to: {output_path}. The output content is {df.to_string()}", "Done"
            else:
                return df.to_string(), "Done"
        except Exception as e:
            if "Cannot save file into a non-existent directory" in str(e):
                return f"Error: you should save the file directly in the current directory", f"Error: {e}"
            return f"Error: {e}", f"Error: {e}"
        finally:
            conn.close()

    def exec_on_db(self, task_id, sqlite_path, query, output_path, process_id=""):
        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        p = multiprocessing.Process(target=run_sql_worker, args=(task_id, sqlite_path, query, output_path, return_dict))

        p.start()
        p.join(self.timeout_length)

        if p.is_alive():
            p.terminate()
            p.join()
            return "Timeout", TimeoutError

        return return_dict.get("output", "Unknown Error"), return_dict.get("status", "Error")

    def postprocess(self, query: str) -> str:
        query = query.replace('> =', '>=').replace('< =', '<=').replace('! =', '!=')
        # Remove single-line comments
        query = re.sub(r"--.*", "", query)
        # Remove multi-line comments
        query = re.sub(r"/\*.*?\*/", "", query, flags=re.DOTALL)
        return query.strip()

    def view_db_schema(self, task_id: str):
        db_schema_data_path = self.db_schema_data_path
        db_schema_data = json.load(open(db_schema_data_path))

        pattern = re.compile(r"_Attempt\d+")
        match = pattern.search(task_id)
        if match:
            attempt_identifier = match.group(0)
            task_id = task_id.replace(attempt_identifier, "")

        for item in db_schema_data:
            if item["id"] == task_id:
                return "The output content is " + item["omni_db_ddl"], "Done"
        return "Error: Database schema not found.", "Error"

    def run_code(self, task_id: str, sql: str, db_id: str, output_path: str, view_db: bool = False) -> Tuple[str, str]:
        if view_db:
            return self.view_db_schema(task_id)
        
        if not sql:
            return "Invalid SQL code. Please ensure your code is correctly formatted.", "Invalid SQL code"
        if not output_path:
            return "Invalid output path. Please ensure your output path is correctly specified.", "Invalid output path"
        
        sql_code = self.extra_sql_code(sql)
        run_code = self.postprocess(sql_code)
        db_path = os.path.join('./data/files', db_id + '.sqlite' if not db_id.endswith('.sqlite') else db_id)
        try:
            result, report = self.exec_on_db(task_id, db_path, run_code, output_path)
        except TimeoutError as error:
            result = ""
            report = "Timeout Error"
        except Exception as e:
            result = ""
            report = str(e)

        return result, report