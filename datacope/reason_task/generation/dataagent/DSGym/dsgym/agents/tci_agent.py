import os
import json
import litellm
from together import Together
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


class DataScienceAgent:
    def __init__(self, model: str = "together_ai/Qwen/Qwen3-235B-A22B-Instruct-2507-tput"):
        self.model = model
        self.client = Together()
        self.system_prompt = """You are an expert data scientist. Analyze the provided data to answer the question.

Rules:
- Use Python for all calculations and data analysis
- Code execution maintains state across calls
- No plotting libraries (use text summaries instead)
- Provide your final answer in the format: <answer>your answer</answer>
- For numerical answers, round as specified in the question"""
        
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "execute_python_code",
                    "description": "Execute Python code in a sandboxed environment. Can install packages with !pip install, create visualizations, perform data analysis, and maintain state across multiple calls in the same session.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "The Python code to execute",
                            }
                        },
                        "required": ["code"],
                    },
                },
            },
        ]
        
        self.available_functions = {
            "execute_python_code": self._execute_python_code,
        }
    
    def _execute_python_code(self, code: str, session_id: str = None, files: list = None) -> dict:
        response = self.client.code_interpreter.run(
            code=code,
            language="python",
            session_id=session_id,
            files=files if files else None,
        )
        
        result_parts = []
        for output in response.data.outputs:
            if output.type == "stdout" or output.type == "stderr":
                result_parts.append(output.data)
            elif output.type == "execute_result":
                if isinstance(output.data, dict) and "text/plain" in output.data:
                    result_parts.append(output.data["text/plain"])
                else:
                    result_parts.append(str(output.data))
            elif output.type == "display_data":
                if isinstance(output.data, dict) and "text/plain" in output.data:
                    result_parts.append(output.data["text/plain"])
        
        if response.data.errors:
            result_parts.append(f"ERROR: {response.data.errors}")
        
        output_str = "\n".join(result_parts) if result_parts else "Code executed successfully with no output"
        
        return {
            "output": output_str,
            "session_id": response.data.session_id,
        }
    
    def _prepare_data_files(self, sample: dict) -> list:
        data_files = []
        extra_info = sample["extra_info"]
        
        if "data_files" not in extra_info:
            return []
        
        absolute_paths = extra_info["data_files"]["absolute"]
        virtual_paths = extra_info["data_files"]["virtual"]
        
        for abs_path, virt_path in zip(absolute_paths, virtual_paths):
            with open(abs_path, "r") as f:
                content = f.read()
            
            data_files.append({
                "name": virt_path,
                "encoding": "string",
                "content": content,
            })
            logging.info(f"Prepared file: {virt_path}")
        
        return data_files
    
    def _upload_files(self, data_files: list) -> str:
        logging.info(f"Uploading {len(data_files)} file(s)")
        
        simple_files = []
        for i, file_info in enumerate(data_files):
            simple_files.append({
                "name": f"uploaded_file_{i}.tmp",
                "encoding": "string",
                "content": file_info["content"]
            })
        
        setup_code_lines = ["import os", "import shutil"]
        for i, file_info in enumerate(data_files):
            target_path = file_info["name"]
            dir_path = os.path.dirname(target_path)
            if dir_path:
                setup_code_lines.append(f"os.makedirs('{dir_path}', exist_ok=True)")
            setup_code_lines.append(f"shutil.move('uploaded_file_{i}.tmp', '{target_path}')")
        
        setup_code_lines.append("print('Files placed')")
        setup_code = "\n".join(setup_code_lines)
        
        upload_response = self.client.code_interpreter.run(
            code=setup_code,
            language="python",
            files=simple_files,
        )
        session_id = upload_response.data.session_id
        logging.info(f"Files uploaded, session_id: {session_id}")
        return session_id
    
    def run(self, user_message: str, data_files: list = None) -> str:
        messages = []
        messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": user_message})
        session_id = None
        max_iterations = 7
        iteration = 0
        
        if data_files:
            session_id = self._upload_files(data_files)
        
        while iteration < max_iterations:
            iteration += 1
            logging.info("Calling LLM")
            response = litellm.completion(
                model=self.model,
                messages=messages,
                tools=self.tools,
            )
            
            assistant_message = response.choices[0].message
            messages.append(assistant_message.model_dump())
            
            if not assistant_message.tool_calls:
                logging.info("No tool calls, returning final response")
                return assistant_message.content
            
            logging.info(f"Executing {len(assistant_message.tool_calls)} tool call(s)")
            for tool_call in assistant_message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                if function_name == "execute_python_code" and session_id:
                    function_args["session_id"] = session_id
                
                function_to_call = self.available_functions[function_name]
                function_result = function_to_call(**function_args)
                
                if function_name == "execute_python_code":
                    session_id = function_result["session_id"]
                    result_content = function_result["output"]
                else:
                    result_content = str(function_result)
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": result_content,
                })
        
        logging.info(f"Max iterations ({max_iterations}) reached")
        return messages[-1]["content"] if messages and messages[-1]["role"] == "assistant" else "Max iterations reached without final answer"
    
    def run_single_sample(self, sample: dict) -> dict:
        query = sample["prompt"][1]["content"]
        data_files = self._prepare_data_files(sample)
        result = self.run(query, data_files=data_files)
        logging.info(f"Agent result: {result}")
        return {"prediction": result}

