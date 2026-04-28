import os
import subprocess
import re
import traceback
import tempfile
import gc
import ast

def run_task(code, workspace, timeout, tool_functions=None):
    if tool_functions is not None:
        running_code = "\n".join(tool_functions) + "\n\n" + "\n".join(code)
    else:
        running_code = "\n".join(code)

    def sanitize_paths(text: str, workspace: str, script_path: str = None) -> str:
        sanitized_text = text
        
        if script_path:
            sanitized_text = sanitized_text.replace(script_path, "<string>")

        real_workspace = os.path.realpath(workspace)
        sanitized_text = sanitized_text.replace(real_workspace, "<WORKSPACE>")
        
        return sanitized_text

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, dir=workspace) as tmp_file:
        tmp_file.write(running_code)
        tmp_file_path = tmp_file.name

    try:
        result = subprocess.run(
            [os.sys.executable, tmp_file_path],
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        if result.returncode == 0:
            return True, sanitize_paths(result.stdout.strip(), workspace, tmp_file_path)
        else:
            stderr_output = (result.stdout + result.stderr).strip()
            
            filtered_stderr = "\n".join(
                line for line in stderr_output.splitlines()
                if not line.strip().startswith("ERROR conda.cli.main_run")
            )
            # replace workspace path with placeholder
            filtered_stderr = sanitize_paths(filtered_stderr, workspace, tmp_file_path)

            return False, f"[Error]:\n{filtered_stderr}"
    except subprocess.TimeoutExpired as e:
        return False, f"[Error]:\nCode execution timed out after {timeout} seconds. {str(e)}"
    except FileNotFoundError as e:
        return False, f"[Error]:\nExecution failed because the workspace directory does not exist: <WORKSPACE>. {str(e)}"
    except Exception as e:
        return False, f"[Error]:\nAn unexpected error occurred during code execution: {str(sanitize_paths(str(e), workspace))}"
    finally:
        os.remove(tmp_file_path)

class StatementRemover(ast.NodeTransformer):
    def __init__(self, functions_to_remove):
        self.functions_to_remove = set(functions_to_remove)

    def visit_Expr(self, node):
        if isinstance(node.value, ast.Call):
            call_node = node.value
            if isinstance(call_node.func, ast.Name):
                if call_node.func.id in self.functions_to_remove:
                    pass_node = ast.Pass()
                    ast.copy_location(pass_node, node)
                    return pass_node
        
        return self.generic_visit(node)
    
class Interpreter:
    def __init__(
        self,
        timeout_length: int = 30,
    ) -> None:
        self.timeout_length = timeout_length
        self.codes = []

    def extra_code(self, code):
        match = re.search(r"```(?:python)?(.*?)```", code, re.DOTALL)
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

    def process_temp_code(self, gen):
        code = self.extra_code(gen)
        g_split = code.split("\n")
        temp_code = []
        for c in g_split:
            temp_code.append(c)

        return temp_code


    def execute(
        self,
        code,
        workspace,
        tool_functions=None,
        timeout_length=30,
    ):
        try:
            succ, exec_output = run_task(code, workspace, timeout_length, tool_functions)
        except Exception as e:
            succ = False
            exec_output = f"[Error]: {str(e)}"

        return succ, exec_output

    def apply(self, code, workspace, tool_functions=None):
        return self.run_code(code, workspace, tool_functions) 

    def run_code(self, code, workspace, tool_functions=None):
        all_code_snippets = self.process_generation_to_code(code)

        if tool_functions is not None:
            tool_code_snippets = self.process_temp_code(tool_functions)
        else:
            tool_code_snippets = None
        try:
            succ, exec_output = self.execute(all_code_snippets, workspace, tool_code_snippets, timeout_length=self.timeout_length)
        except Exception as error:
            succ = False
            exec_output = f"[Error]: {str(error)}"

        exec_output = str(exec_output).strip()

        self.update_code(succ, all_code_snippets)

        gc.collect()

        return exec_output
    
    def update_code(self, succ, code):
        if succ:
            self.codes=[]
            code_text = "\n".join(code)
            try:
                tree = ast.parse(code_text)
                transformer = StatementRemover(functions_to_remove=['print', 'query_document'])
                
                transformed_tree = transformer.visit(tree)
                ast.fix_missing_locations(transformed_tree)
                cleaned_code = ast.unparse(transformed_tree)
                cleaned_code_lines = cleaned_code.splitlines()
                for c in cleaned_code_lines:
                    if c.strip() != "":
                        self.codes.append(c)
            except Exception as e:
                print(f"Error during AST transformation: {e}")
                self.codes = self._fallback_update(code)
    
    def _fallback_update(self, code):
        self.codes=[]
        for c in code:
            if (c.strip().startswith("print") or "print(" in c):
                indent = re.match(r"^\s*", c).group(0)
                self.codes.append(f"{indent}pass")
            else:
                self.codes.append(c)
