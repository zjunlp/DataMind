import json
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

class SQLInterpreter:
    def __init__(
        self,
        config, 
        timeout_length: int = 240,
    ) -> None:
        self.config = config
        self.timeout_length = timeout_length

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
        import pandas as pd

        save_dir = self.config.pred_csv_results_dir
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, task_id), exist_ok=True)

        query = self.replace_cur_year(query)
        conn = sqlite3.connect(sqlite_path)

        try:
            # Execute the SQL command and fetch the results
            if self.is_not_select_query(query):
                return "Error: Only SELECT statements are allowed.", "Error: Only SELECT statements are allowed."
            df = pd.read_sql_query(query, conn)
            
            # Check if the output should be saved to a CSV file or printed directly
            if output_path.lower().endswith(".csv"):
                save_path = os.path.join(save_dir, task_id, output_path)
                df.to_csv(save_path, index=False)

                if "sql FROM sqlite_master WHERE type='table'" in query:
                    database_schemas = "\n".join(df["sql"])
                    return f"Output has been saved to: {output_path}. The output content is {database_schemas}", "Done"
                else:
                    return f"Output has been saved to: {output_path}. The output content is {df.to_string()}", "Done"
            else:
                # print(df.to_string())
                return df.to_string(), "Done"
        # df.to_csv, not find dir error

        except Exception as e:
            if "Cannot save file into a non-existent directory" in str(e):
                return f"Error: you should save the file directly in the current directory", f"Error: {e}"
            return f"Error: {e}", f"Error: {e}"
        finally:
            # Close the connection to the database
            conn.close()

    def exec_on_db(
        self, task_id: str, sqlite_path: str, query: str, output_path: str, process_id: str = ""
    ) -> Tuple[str, Any]:
        try:
            # return await asyncio.wait_for(self.exec_sql(sqlite_path, query), self.timeout_length)
            return func_timeout(self.timeout_length, self.exec_sql, args=(task_id, sqlite_path, query, output_path))
        except FunctionTimedOut:
            return 'Timeout', TimeoutError
        except Exception as e:
            return f"Error: {e}", f"Error: {e}"

    # postprocess the model predictions to avoid execution errors
    # e.g. removing spaces between ">" and "="
    def postprocess(self, query: str) -> str:
        query = query.replace('> =', '>=').replace('< =', '<=').replace('! =', '!=')
        # Remove single-line comments
        query = re.sub(r"--.*", "", query)
        # Remove multi-line comments
        query = re.sub(r"/\*.*?\*/", "", query, flags=re.DOTALL)
        return query.strip()

    def view_db_schema(self, task_id: str):
        db_schema_data = json.load(open(self.config.db_schema_data_path))

        pattern = re.compile(r"_Attempt\d+")
        match = pattern.search(task_id)
        if match:
            attempt_identifier = match.group(0)
            task_id = task_id.replace(attempt_identifier, "")

        # print(f"Task id: {task_id}")
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