#!/usr/bin/env python3
"""
Code MCP Server for Data Analysis
Python code execution server based on the official MCP Python SDK
"""

import argparse
import ast
import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import mcp.types as types
from base_mcp_server import BaseMCPServer, setup_logging

class InteractiveExpressionTransformer(ast.NodeTransformer):
    """
    AST transformer that converts standalone expressions to print statements,
    mimicking Jupyter notebook behavior.
    """
    
    def visit_Expr(self, node):
        """
        Transform standalone expression statements to print them.
        Only transform if the expression is likely to have a meaningful result.
        """
        # Get the expression
        expr = node.value
        
        # Skip certain types of expressions that shouldn't be auto-printed
        if self._should_skip_expression(expr):
            return node
        
        # Transform the expression to be auto-displayed
        # Instead of just 'expr', we make it '_auto_display(expr)'
        new_call = ast.Call(
            func=ast.Name(id='_auto_display', ctx=ast.Load()),
            args=[expr],
            keywords=[]
        )
        
        # Create a new expression statement with the wrapped call
        new_node = ast.Expr(value=new_call)
        
        # Copy location info
        return ast.copy_location(new_node, node)
    
    def _should_skip_expression(self, expr):
        """
        Determine if an expression should be skipped (not auto-printed).
        """
        # Skip function/method calls that are likely to be side-effect operations
        if isinstance(expr, ast.Call):
            # Get the function name if it's a simple name or attribute
            func_name = self._get_function_name(expr.func)
            
            # Skip common side-effect functions
            skip_functions = {
                'print', 'input', 'open', 'close', 'write', 'flush',
                'append', 'extend', 'insert', 'remove', 'pop', 'clear',
                'sort', 'reverse', 'update', 'add', 'discard',
                'mkdir', 'rmdir', 'unlink', 'rename', 'move',
                'plt.show', 'plt.savefig', 'plt.close',
                'fig.show', 'fig.savefig', 'fig.close'
            }
            
            if func_name in skip_functions:
                return True
            
            # Skip methods that end with common side-effect patterns
            if func_name and (
                func_name.endswith('_') or  # Methods ending with underscore (often in-place)
                'save' in func_name.lower() or
                'write' in func_name.lower() or
                'delete' in func_name.lower() or
                'remove' in func_name.lower() or
                'create' in func_name.lower()
            ):
                return True
        
        # Skip assignments
        if isinstance(expr, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            return True
        
        # Skip imports
        if isinstance(expr, (ast.Import, ast.ImportFrom)):
            return True
        
        # Skip control flow statements
        if isinstance(expr, (ast.If, ast.For, ast.While, ast.With, ast.Try)):
            return True
        
        # Skip function and class definitions
        if isinstance(expr, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
            return True
        
        # Skip standalone constants (numbers, strings, etc.) unless they might be meaningful
        if isinstance(expr, ast.Constant):
            # Allow meaningful constants like large numbers or long strings that might be results
            if isinstance(expr.value, (int, float)) and abs(expr.value) < 10:
                return True
            if isinstance(expr.value, str) and len(expr.value) < 10:
                return True
        
        return False
    
    def _get_function_name(self, func_node):
        """
        Extract function name from a function call node.
        """
        if isinstance(func_node, ast.Name):
            return func_node.id
        elif isinstance(func_node, ast.Attribute):
            # For method calls like obj.method()
            if isinstance(func_node.value, ast.Name):
                return f"{func_node.value.id}.{func_node.attr}"
            else:
                return func_node.attr
        return None

class CodeMCPServer(BaseMCPServer):
    """Code MCP Server for data analysis"""
    
    def __init__(self, base_path: str, enable_detailed_logging: bool = True):
        self.base_path = Path(base_path).resolve()
        super().__init__("code-mcp", enable_detailed_logging)
        
        
    def _setup_specific_handlers(self):
        """Setup code-specific MCP handlers"""
        pass  # Base class handles common handlers
    
    def _get_specific_resources(self) -> List[types.Resource]:
        """Get code-specific resources"""
        return [
            types.Resource(
                uri="code-mcp://base_path",
                name="Base Path Info",
                description="Information about the base working directory",
                mimeType="application/json"
            )
        ]
    
    async def _read_specific_resource(self, uri: str) -> str:
        """Read code-specific resource"""
        if uri == "code-mcp://base_path":
            path_info = {
                "base_path": str(self.base_path),
                "exists": self.base_path.exists(),
                "is_directory": self.base_path.is_dir() if self.base_path.exists() else False
            }
            return json.dumps(path_info, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"Unknown resource: {uri}")
    
    def _get_specific_tools(self) -> List[types.Tool]:
        """Get code-specific tools"""
        return [
            types.Tool(
                name="execute_code",
                description="Execute Python code for data analysis. Your code can only read files. YOU MUST NOT write/save/modify files or generate plots. You should focus on one aspect in each execution. You can write algorithms to model and capture complext patterns beyond some basic pandas operations. DO NOT WRITE ANY INSIGHTS OR REASONING IN THE CODE.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Python code to execute for data analysis. Use relative paths to access data files in the base directory. Try to get various complex statistics from data instead of listing the data since the data is too large to list. All you can get is information from stdout. DO NOT WRITE ANY INSIGHTS OR REASONING IN THE CODE."
                        },
                        "timeout": {
                            "type": "integer",
                            "description": "Execution timeout in seconds (default: 30, max: 300)",
                            "default": 30,
                            "maximum": 300
                        }
                    },
                    "required": ["code"]
                }
            ),
            types.Tool(
                name="list_files",
                description="List files you have access to with table descriptions for CSV files. For CSV files, this tool provides detailed table-level metadata including descriptions, categories, and data sources based on the GLOBEM dataset documentation.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Directory path to list (relative to base path, default: '.')",
                            "default": "."
                        },
                        "pattern": {
                            "type": "string",
                            "description": "File pattern to match (e.g., '*.csv', '*.json')"
                        },
                        "recursive": {
                            "type": "boolean",
                            "description": "Whether to list files recursively",
                            "default": False
                        }
                    },
                    "required": []
                }
            ),
            types.Tool(
                name="get_field_description",
                description="Get actual field name and field descriptions for given CSV data file to help you understand the data structure.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "data_file": {
                            "type": "string",
                            "description": "Path to the CSV data file (relative to base path, which means only one filename)"
                        }
                    },
                    "required": ["data_file"]
                }
            )
        ]
    
    async def _handle_specific_tool_call(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle code-specific tool calls"""
        if name == "execute_code":
            code_content = arguments.get("code", "")
            return self._execute_code(
                code_content,
                arguments.get("timeout", 30)
            )
        elif name == "list_files":
            return self._list_files(
                arguments.get("path", "."),
                arguments.get("pattern"),
                arguments.get("recursive", False)
            )
        elif name == "get_field_description":
            return self._get_field_description(arguments["data_file"])
        else:
            raise ValueError(f"Unknown tool: {name}")

    def _transform_code_for_interactive_output(self, code: str) -> str:
        """
        Transform code to automatically print results of standalone expressions,
        mimicking Jupyter notebook behavior.
        """
        try:
            # Parse the code into an AST
            tree = ast.parse(code)
            
            # Transform the AST
            transformer = InteractiveExpressionTransformer()
            transformed_tree = transformer.visit(tree)
            
            # Convert back to code
            try:
                # Try using ast.unparse (Python 3.9+)
                if hasattr(ast, 'unparse'):
                    transformed_code = ast.unparse(transformed_tree)
                else:
                    # Fallback: try using astor if available
                    import astor
                    transformed_code = astor.to_source(transformed_tree)
            except ImportError:
                # If astor is not available and ast.unparse is not available,
                # use a simple regex-based approach as fallback
                return self._simple_transform_interactive_expressions(code)
            
            return transformed_code
            
        except Exception as e:
            # If transformation fails, return original code with simple transformation
            return self._simple_transform_interactive_expressions(code)

    def _simple_transform_interactive_expressions(self, code: str) -> str:
        """
        Simple regex-based approach to transform common interactive expressions.
        This is a fallback when AST transformation is not available.
        """
        import re
        
        lines = code.split('\n')
        transformed_lines = []
        
        for line in lines:
            stripped = line.strip()
            
            # Skip empty lines, comments, and lines that already have print
            if not stripped or stripped.startswith('#') or 'print(' in line:
                transformed_lines.append(line)
                continue
            
            # Skip lines with assignments, imports, control flow
            if any(keyword in stripped for keyword in ['=', 'import ', 'from ', 'if ', 'for ', 'while ', 'def ', 'class ', 'with ', 'try:', 'except', 'finally']):
                transformed_lines.append(line)
                continue
            
            # Skip lines that are likely side-effect operations
            if any(func in stripped for func in ['print(', 'input(', '.save(', '.write(', '.close(', '.show(', '.savefig(']):
                transformed_lines.append(line)
                continue
            
            # Check if it looks like a standalone expression that should be printed
            # Common patterns: data.head(), df.describe(), variable_name, obj.method(), etc.
            if (re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*(\.[a-zA-Z_][a-zA-Z0-9_]*)*(\(\))?(\[.*\])?$', stripped) or
                re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*(\.[a-zA-Z_][a-zA-Z0-9_]*)*\(.*\)$', stripped)):
                
                # Get the indentation
                indent = line[:len(line) - len(line.lstrip())]
                transformed_lines.append(f"{indent}_auto_display({stripped})")
            else:
                transformed_lines.append(line)
        
        return '\n'.join(transformed_lines)

    def _check_file_write_operations(self, code: str) -> str:
        """Check for file writing operations and return warning if found"""
        import re
        
        # Patterns that indicate file writing operations
        write_patterns = [
            r'open\s*\([^)]*[\'"]w[\'"]',  # open(..., 'w')
            r'open\s*\([^)]*[\'"]a[\'"]',  # open(..., 'a')
            r'open\s*\([^)]*[\'"]x[\'"]',  # open(..., 'x')
            r'open\s*\([^)]*[\'"]wb[\'"]', # open(..., 'wb')
            r'open\s*\([^)]*[\'"]ab[\'"]', # open(..., 'ab')
            r'open\s*\([^)]*[\'"]xb[\'"]', # open(..., 'xb')
            r'\.write\s*\(',               # .write(
            r'\.writelines\s*\(',          # .writelines(
            r'\.save\s*\(',                # .save(
            r'\.savefig\s*\(',             # .savefig(
            r'\.to_csv\s*\(',              # .to_csv(
            r'\.to_excel\s*\(',            # .to_excel(
            r'\.to_json\s*\(',             # .to_json(
            r'\.to_pickle\s*\(',           # .to_pickle(
            r'\.dump\s*\(',                # .dump(
            r'\.dumps\s*\(',               # .dumps(
            r'pickle\.dump\s*\(',          # pickle.dump(
            r'json\.dump\s*\(',            # json.dump(
            r'np\.save\s*\(',              # np.save(
            r'np\.savez\s*\(',             # np.savez(
            r'np\.savetxt\s*\(',           # np.savetxt(
            r'plt\.savefig\s*\(',          # plt.savefig(
            r'fig\.savefig\s*\(',          # fig.savefig(
            r'\.to_file\s*\(',             # .to_file(
            r'\.export\s*\(',              # .export(
            r'Path\s*\([^)]*\)\.write_text\s*\(',  # Path(...).write_text(
            r'Path\s*\([^)]*\)\.write_bytes\s*\(', # Path(...).write_bytes(
            r'Path\s*\([^)]*\)\.touch\s*\(',       # Path(...).touch(
            r'Path\s*\([^)]*\)\.mkdir\s*\(',       # Path(...).mkdir(
            r'Path\s*\([^)]*\)\.makedirs\s*\(',    # Path(...).makedirs(
            r'os\.makedirs\s*\(',          # os.makedirs(
            r'os\.mkdir\s*\(',             # os.mkdir(
            r'os\.remove\s*\(',            # os.remove(
            r'os\.unlink\s*\(',            # os.unlink(
            r'os\.rename\s*\(',            # os.rename(
            r'os\.replace\s*\(',           # os.replace(
            r'shutil\.copy\s*\(',          # shutil.copy(
            r'shutil\.copy2\s*\(',         # shutil.copy2(
            r'shutil\.move\s*\(',          # shutil.move(
            r'shutil\.rmtree\s*\(',        # shutil.rmtree(
            r'with\s+open\s*\([^)]*[\'"]w[\'"]',  # with open(..., 'w')
            r'with\s+open\s*\([^)]*[\'"]a[\'"]',  # with open(..., 'a')
            r'with\s+open\s*\([^)]*[\'"]x[\'"]',  # with open(..., 'x')
        ]
        
        # Check for file writing patterns
        for pattern in write_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                return "Warning: File write operations detected. Code execution is prohibited. You can not do any write, save, create, or modify operations."
        
        return None

    def _execute_code(self, code: str, timeout: int = 30) -> Dict[str, Any]:
        """Execute Python code for data analysis"""
        start_time = time.time()
        
        # Add detailed debug log
        code_lines = len(code.split('\n'))
        
        # Check for file write operations first
        write_warning = self._check_file_write_operations(code)
        if write_warning:
            return {
                "success": False,
                "error": write_warning,
                "base_path": str(self.base_path),
                "code_length": len(code),
                "code_lines": len(code.split('\n'))
            }
        
        # Validate timeout
        if timeout > 300:
            timeout = 300
        
        # Basic preprocessing - only handle escaped newlines
        processed_code = code.strip()
        if '\\n' in code and '\n' not in code:
            processed_code = code.replace('\\n', '\n').replace('\\t', '\t').replace('\\r', '\r').strip()
        
        # Transform code to auto-print interactive expressions
        processed_code = self._transform_code_for_interactive_output(processed_code)
        
        # Calculate metrics once
        code_lines = len(processed_code.split('\n'))
        code_length = len(code)
        
        # Create temporary Python script
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as temp_file:
                # Prepare the code with base path and common imports
                full_code = f"""
import sys
import os
import pandas as pd
import numpy as np
import json
from pathlib import Path

# Configure pandas and numpy to show full information
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
np.set_printoptions(threshold=np.inf, linewidth=np.inf)

# Set base path for data access
BASE_PATH = Path(r"{self.base_path}")
os.chdir(BASE_PATH)

# Helper function to auto-display results like in Jupyter
def _auto_display(obj):
    '''Auto-display function that mimics Jupyter notebook behavior'''
    if obj is not None:
        print(obj)
    return obj

# User code starts here
{processed_code}
"""
                temp_file.write(full_code)
                temp_file_path = temp_file.name
            
            # Execute the code
            try:
                result = subprocess.run(
                    [sys.executable, temp_file_path],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=str(self.base_path)
                )
                
                execution_time = time.time() - start_time
                
                # Return simple result
                return {
                    "success": result.returncode == 0,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "returncode": result.returncode,
                    "execution_time": execution_time,
                    "base_path": str(self.base_path),
                    "code_length": code_length,
                    "code_lines": code_lines
                }
                
            except subprocess.TimeoutExpired:
                return {
                    "success": False,
                    "error": f"Code execution timed out after {timeout} seconds",
                    "timeout": True,
                    "execution_time": timeout,
                    "base_path": str(self.base_path),
                    "code_length": code_length,
                    "code_lines": code_lines
                }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to execute code: {str(e)}",
                "base_path": str(self.base_path),
                "code_length": code_length,
                "code_lines": code_lines
            }
        
        finally:
            # Clean up temporary file
            try:
                if 'temp_file_path' in locals():
                    os.unlink(temp_file_path)
            except:
                pass
    
    def _list_files(self, path: str = ".", pattern: str = None, recursive: bool = False) -> Dict[str, Any]:
        """List files in specified directory with table descriptions"""
        try:
            target_path = self.base_path / path
            if not target_path.exists():
                raise ValueError(f"Path not found: {target_path}")
            
            if not target_path.is_dir():
                raise ValueError(f"Path is not a directory: {target_path}")
            
            files = []
            
            if recursive:
                if pattern:
                    file_paths = target_path.rglob(pattern)
                else:
                    file_paths = target_path.rglob("*")
            else:
                if pattern:
                    file_paths = target_path.glob(pattern)
                else:
                    file_paths = target_path.iterdir()
            
            for file_path in sorted(file_paths):
                try:
                    # Filter out files ending with _fields.json
                    if file_path.is_file() and file_path.name.endswith('_fields.json'):
                        continue
                        
                    relative_path = file_path.relative_to(self.base_path)
                    stat = file_path.stat()
                    
                    file_info = {
                        "name": file_path.name,
                        "path": str(relative_path),
                        "absolute_path": str(file_path),
                        "type": "directory" if file_path.is_dir() else "file",
                        "size": stat.st_size if file_path.is_file() else None,
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                    }
                    
                    # Add table description for CSV files
                    if file_path.is_file() and file_path.suffix.lower() == '.csv':
                        table_description = self._get_table_description(file_path)
                        if table_description:
                            file_info["description"] = table_description
                    
                    files.append(file_info)
                except Exception as e:
                    continue
            
            return {
                "path": str(target_path.relative_to(self.base_path)),
                "absolute_path": str(target_path),
                "files": files,
                "count": len(files),
                "pattern": pattern,
                "recursive": recursive
            }
            
        except Exception as e:
            error_msg = f"Failed to list files: {str(e)}"
            raise ValueError(error_msg)
    
    def _get_table_description(self, csv_file_path: Path) -> str:
        """Get simple one-sentence table description for a CSV file from JSON"""
        try:
            # Look for corresponding JSON metadata file
            possible_json_files = [
                csv_file_path.with_suffix('.json'),
                csv_file_path.parent / f"{csv_file_path.stem}_fields.json",
                csv_file_path.parent / f"{csv_file_path.stem}_description.json",
            ]
            
            json_file_path = None
            for possible_path in possible_json_files:
                if possible_path.exists():
                    json_file_path = possible_path
                    break
            
            if not json_file_path:
                # Return generic description if no JSON file found
                base_name = csv_file_path.stem
                return f"Data table containing {base_name.replace('_', ' ')} information."
            
            # Read and parse JSON file
            with open(json_file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            # Check if JSON contains simple table description
            if isinstance(json_data, dict) and "_table_description" in json_data:
                return json_data["_table_description"]
            
            # If no table description, return generic one
            base_name = csv_file_path.stem
            return f"Data table containing {base_name.replace('_', ' ')} information."
            
        except Exception as e:
            return None
    
    def _get_field_description(self, data_file: str) -> Dict[str, Any]:
        """Get JSON field descriptions for a CSV data file"""
        try:
            data_file_path = self.base_path / data_file
            if not data_file_path.exists():
                raise ValueError(f"Data file not found: {data_file_path}")
            
            # Look for corresponding JSON field description file
            possible_json_files = [
                data_file_path.with_suffix('.json'),
                data_file_path.parent / f"{data_file_path.stem}_fields.json",
                data_file_path.parent / f"{data_file_path.stem}_description.json",
            ]
            
            json_file_path = None
            for possible_path in possible_json_files:
                if possible_path.exists():
                    json_file_path = possible_path
                    break
            
            if not json_file_path:
                # Try to find any JSON file in the same directory with similar name
                json_pattern = f"*{data_file_path.stem}*.json"
                json_files = list(data_file_path.parent.glob(json_pattern))
                if json_files:
                    json_file_path = json_files[0]
            
            if not json_file_path:
                raise ValueError(f"No field description JSON file found for {data_file}")
            
            # Read and parse JSON file
            with open(json_file_path, 'r', encoding='utf-8') as f:
                field_descriptions = json.load(f)
            
            return {
                "data_file": str(data_file_path.relative_to(self.base_path)),
                "field_descriptions": field_descriptions,
                "field_count": len(field_descriptions) if isinstance(field_descriptions, dict) else 0
            }
            
        except Exception as e:
            error_msg = f"Failed to get field descriptions: {str(e)}"
            raise ValueError(error_msg)


async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Code MCP Server for Data Analysis")
    parser.add_argument(
        "--data-path",
        default=None,
        help="Base path for data files (default: current directory)"
    )
    parser.add_argument(
        "--config",
        help="Path to config.yaml file"
    )
    parser.add_argument(
        "--scenario",
        help="Scenario name to load from config (e.g., globem)"
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )
    parser.add_argument(
        "--disable-detailed-logging",
        action="store_true",
        help="Disable detailed operation logging"
    )
    
    args = parser.parse_args()
    
    # Resolve data path
    data_path = args.data_path
    
    # Try to load from config if scenario is provided and data_path is missing
    if not data_path and args.scenario:
        try:
            # Add parent directory to path to find config module
            sys.path.append(str(Path(__file__).resolve().parent.parent))
            from config import get_config
            
            config = get_config(args.config)
            scenario_config = config.get_scenario(args.scenario)
            if scenario_config.data_path:
                data_path = scenario_config.data_path
                print(f"Loaded data_path from config for scenario '{args.scenario}': {data_path}")
        except Exception as e:
            print(f"Warning: Failed to load config: {e}")
            
    # Fallback to environment variable or default
    if not data_path:
        data_path = os.getenv('MCP_DATA_PATH', ".")
    
    try:
        server = CodeMCPServer(
            data_path, 
            enable_detailed_logging=not args.disable_detailed_logging
        )
        await server.run()
    except Exception as e:
        print(f"❌ Server failed to start: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())