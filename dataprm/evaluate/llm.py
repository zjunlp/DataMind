import re
import io
import subprocess
import tempfile
import threading
import traceback
import contextlib
import os
import time
import requests
from openai import OpenAI
import math
from interpreter import Interpreter

class LLM:
    def __init__(
        self,
        model_name: str,
        base_url: str = "http://localhost:19007/v1/",
        api_key: str = "dummy",
        max_rounds: int = 30,
    ):
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key
        self.max_rounds = max_rounds
        self._execution_lock = threading.Lock()
        self.interpreter = Interpreter()

    def generate(
        self,
        prompt: str | list,
        workspace: str,
        temperature: float = 0.5,
        max_tokens: int = 8192,
        top_p: float = None,
        top_k: int = None,
        context_codes: list = None,
        execute_context_code: bool = False,
        obs_max_len: int = 8192,
        use_tool: bool = False,
        tool_functions: str = None,
    ) -> dict:
        client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key
        )
        reasoning = ""
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = prompt.copy()
        response_message = []

        total_prompt_tokens = 0
        total_completion_tokens = 0
        generate_start = time.time()

        if execute_context_code:
            if context_codes is None:
                raise ValueError("context_codes must be provided when execute_context_code is True.")
            if isinstance(context_codes, str):
                raise ValueError("prompt must be a list when execute_context_code is True.")
            else:
                # Execute all previous <code> blocks in the context
                for index, msg in enumerate(context_codes):
                    if msg["role"] == "assistant":
                        content = msg["content"]
                        code_match = re.search(r"<Code>(.*?)</Code>", content, re.DOTALL)
                        if code_match:
                            code_content = code_match.group(1).strip()
                            md_match = re.search(r"```(?:python)?(.*?)```", code_content, re.DOTALL)
                            code_str = md_match.group(1).strip() if md_match else code_content

                            # Execute code and append output
                            exe_output = self.interpreter.run_code(code_content, workspace)

        try:
            for round_idx in range(self.max_rounds):
                payload = {
                    "model": self.model_name,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stop": ["</code>"],
                    "extra_body": {}
                }
                if top_p is not None:
                    payload["top_p"] = top_p

                if top_k is not None:
                    payload["extra_body"]["top_k"] = top_k

                response = client.chat.completions.create(**payload)
                ans = response.choices[0].message.content

                # accumulate token usage
                if response.usage is not None:
                    total_prompt_tokens += response.usage.prompt_tokens
                    total_completion_tokens += response.usage.completion_tokens

                if ans.strip() == "" or ans is None:
                    print(ans)

                if response.choices[0].finish_reason == "stop" and "<code>" in ans and "</code>" not in ans:
                    ans += "</code>"

                response_message.append(ans)

                # Check for <code> block
                code_match = re.search(r"<code>(.*?)</code>", ans, re.DOTALL)
                if not code_match or "<score>" in ans:
                    messages.append({"role": "assistant", "content": ans})
                    break

                code_content = code_match.group(1).strip()

                # Execute code and append output
                if use_tool and tool_functions is not None:
                    exe_output = self.interpreter.run_code(code_content, workspace, tool_functions=tool_functions)
                else:
                    exe_output = self.interpreter.run_code(code_content, workspace)
                if len(exe_output) > obs_max_len:
                    exe_output = f"First {obs_max_len // 2} characters:\n" + exe_output[:obs_max_len // 2] + f"\n......\nLast {obs_max_len // 2} characters:\n" + exe_output[-(obs_max_len // 2):] + f"\n\nThe observation is too long, truncated to the first and last {obs_max_len // 2} characters. If you need more information, please try to write new code to get the information."

                exe_output = f"<interpreter>\n{exe_output}\n</interpreter>"
                response_message.append(exe_output)

                # Append messages for next round
                messages.append({"role": "assistant", "content": ans})
                messages.append({"role": "user", "content": exe_output})

            reasoning = "\n".join(response_message)

        except Exception as e:
            print(f"Exception: {str(e)}")
            reasoning = "\n".join(response_message) + f"\n[Error]: An unexpected error occurred during generation: {str(e)}\n{traceback.format_exc()}"

        generate_elapsed = time.time() - generate_start
        return {
            "reasoning": reasoning,
            "messages": messages,
            "token_usage": {
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
                "total_tokens": total_prompt_tokens + total_completion_tokens,
            },
            "elapsed_seconds": generate_elapsed,
        }