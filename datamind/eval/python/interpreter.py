import os
import io
import regex
import subprocess
import re
import pickle
import traceback
import copy
import datetime
import dateutil.relativedelta
import multiprocess
from multiprocess import Pool
from typing import Any, Dict, Optional
from pebble import ProcessPool
from tqdm import tqdm
from concurrent.futures import TimeoutError
from functools import partial
from timeout_decorator import timeout
from contextlib import redirect_stdout


class GenericRuntime:
    GLOBAL_DICT = {}
    LOCAL_DICT = None
    HEADERS = []

    def __init__(self):
        self._global_vars = copy.copy(self.GLOBAL_DICT)
        self._local_vars = copy.copy(self.LOCAL_DICT) if self.LOCAL_DICT else None

        for c in self.HEADERS:
            self.exec_code(c)

    def exec_code(self, code_piece: str) -> None:
        if regex.search(r"(\s|^)?input\(", code_piece) or regex.search(
                r"(\s|^)?os.system\(", code_piece
        ):
            raise RuntimeError()
        exec(code_piece, self._global_vars)

    def eval_code(self, expr: str) -> Any:
        return eval(expr, self._global_vars)

    def inject(self, var_dict: Dict[str, Any]) -> None:
        for k, v in var_dict.items():
            self._global_vars[k] = v

    @property
    def answer(self):
        return self._global_vars["answer"]


class DateRuntime(GenericRuntime):
    GLOBAL_DICT = {
        "datetime": datetime.datetime,
        "timedelta": dateutil.relativedelta.relativedelta,
        "relativedelta": dateutil.relativedelta.relativedelta,
    }


class CustomDict(dict):
    def __iter__(self):
        return list(super().__iter__()).__iter__()


class ColorObjectRuntime(GenericRuntime):
    GLOBAL_DICT = {"dict": CustomDict}


class Interpreter:
    def __init__(
            self,
            runtime: Optional[Any] = None,
            timeout_length: int = 20,
            batch_size: int = 256,
    ) -> None:
        self.runtime = runtime if runtime else GenericRuntime()
        self.timeout_length = timeout_length
        self.batch_size = batch_size
        self.codes = [[] for _ in range(self.batch_size)]
        self.pool = Pool(min(multiprocess.cpu_count(), batch_size))

    def process_generation_to_code(self, gens: str):
        codes = []
        for index, g in gens:
            g_split = g.split("\n")
            run_code = self.codes[index].copy()
            for c in g_split:
                run_code.append(c)
            codes.append((index, run_code))
        return codes

    @staticmethod
    def execute(
            code,
            runtime=None,
            timeout_length=20,
    ):
        try:
            program_io = io.StringIO()
            with redirect_stdout(program_io):
                timeout(timeout_length)(runtime.exec_code)("\n".join(code[1]))
            program_io.seek(0)
            result = program_io.read()
            report = "Done"
            str(result)
            pickle.dumps(result)  # serialization check
        except:
            result = ""
            report = traceback.format_exc().split("\n")[-2]

        if 'No module named' in report:
            match = re.search(r"No module named ['\"]([^'\"]+)['\"]", report)
            if match:
                missing_module = match.group(1)
                pip_res = subprocess.run(
                    ["pip", "install", missing_module],
                    capture_output=True
                )
                if pip_res.returncode != 0:
                    print(
                        f"[Interpreter pip error] an error occured when using pip to install missing package :{pip_res.stderr.decode('utf-8')}")
                else:
                    print(f"[Interpreter pip] successfully pip install module: {missing_module} restart the code")
                    try:
                        re_program_io = io.StringIO()
                        with redirect_stdout(re_program_io):
                            timeout(timeout_length)(runtime.exec_code)("\n".join(code[1]))
                        re_program_io.seek(0)
                        result = re_program_io.read()
                        report = "Done"
                        str(result)
                        pickle.dumps(result)
                    except:
                        result = ""
                        report = traceback.format_exc().split("\n")[-2]
        return result, report

    def apply(self, code):
        return self.run_code([code])[0]

    @staticmethod
    def truncate(s, max_length=2048):
        half = max_length // 2
        if len(s) > max_length:
            s = s[:half] + "..." + s[-half:]
        return s

    def run_code(self, batch_code):
        all_code_snippets = self.process_generation_to_code(batch_code)

        timeout_cnt = 0
        all_exec_results = []
        with ProcessPool(
                max_workers=min(len(all_code_snippets), os.cpu_count())
        ) as pool:
            executor = partial(
                self.execute,
                runtime=self.runtime,
                timeout_length=self.timeout_length,  # this timeout not work
            )
            future = pool.map(executor, all_code_snippets, timeout=self.timeout_length)
            iterator = future.result()

            if len(all_code_snippets) > 100:
                progress_bar = tqdm(total=len(all_code_snippets), desc="Execute")
            else:
                progress_bar = None

            while True:
                try:
                    result = next(iterator)
                    all_exec_results.append(result)
                except StopIteration:
                    break
                except TimeoutError as error:
                    all_exec_results.append(("", "Timeout Error"))
                    timeout_cnt += 1
                except Exception as error:
                    exit()
                if progress_bar is not None:
                    progress_bar.update(1)

            if progress_bar is not None:
                progress_bar.close()

        batch_results = []
        for (index, code), (res, report) in zip(all_code_snippets, all_exec_results):
            # post processing
            res, report = str(res).strip(), str(report).strip()
            res, report = self.truncate(res, max_length=2048), self.truncate(report, max_length=2048)
            batch_results.append((res, report))
            self.update_code(report, index, code)
        return batch_results

    def update_code(self, report, index, code):
        if report == 'Done':
            self.codes[index] = []
            for c in code:
                if not (c.strip().startswith("print") or "print(" in c):
                    self.codes[index].append(c)
                else:
                    indent = re.match(r"^\s*", c).group(0)
                    self.codes[index].append(f"{indent}pass")

    def clear_cache(self, active_masks):
        # clear locals for those task have done
        for i, mask in enumerate(active_masks):
            if mask == 0:
                self.codes[i] = []

    def reset(self):
        self.runtime = GenericRuntime()
        self.codes = [[] for _ in range(self.batch_size)]
        self.pool = Pool(min(multiprocess.cpu_count(), self.batch_size))
