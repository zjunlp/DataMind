"""Minimal ReAct agent: chat history + a Python tool + a bounded tool loop."""

import json
import os
from pathlib import Path
import subprocess

from openai import OpenAI
from longds import save


class Agent:
    def __init__(self, model=None, python=None, max_steps=20):
        self.client = OpenAI()  # OPENAI_API_KEY and optional OPENAI_BASE_URL
        self.model = model or os.environ['REACT_MODEL']
        self.python = python or os.environ.get('REACT_PYTHON', '/usr/local/bin/python')
        self.max_steps = max_steps
        # Each task gets its own instance; history survives across respond() calls.
        self.messages = [{
            'role': 'system',
            'content': (
                'You are a data analysis agent. Use the python tool to inspect files and '
                'calculate answers from data. Solve the current question. '
                f'Input data (read-only): {Path.cwd() / "data"}. '
                f'Working directory for tools, scripts, and intermediate files: {Path.cwd()}. '
                f'Use {self.python} for data analysis; keep agent-specific dependencies '
                'in the agent environment. Read input data without modifying it. '
                'Do not access benchmark source tasks, reference answers, other tasks, '
                'or future questions. '
                'Save useful intermediate results in the working directory for later turns. '
                'Each python call starts a fresh process: variables do not persist, but files do. '
                'Print results you need to inspect. Follow the requested rounding and ordering. '
                'When finished, respond with the final answer to the current question.'
            ),
        }]

        self.tools = [{
            'type': 'function',
            'function': {
                'name': 'python',
                'description': 'Execute Python in the task workspace and return stdout/stderr.',
                'parameters': {
                    'type': 'object',
                    'properties': {'code': {'type': 'string'}},
                    'required': ['code'],
                    'additionalProperties': False,
                },
            },
        }]

    def run_python(self, code: str) -> str:
        save('python_code', {'code': code})
        print(f'[python]\n{code}', flush=True)
        try:
            result = subprocess.run(
                [self.python, '-'], input=code, capture_output=True, text=True, timeout=120,
            )
            output = f'Exit code: {result.returncode}\n{result.stdout}{result.stderr}'
        except subprocess.TimeoutExpired:
            output = 'Python execution timed out after 120 seconds.'
        save('python_output', {'output': output})
        output = output[:20000]
        print(f'[observation]\n{output}', flush=True)
        return output

    def respond(self, message: str) -> str:
        self.messages.append({'role': 'user', 'content': message})
        for _ in range(self.max_steps):
            save('model_request', {'model': self.model, 'messages': self.messages, 'tools': self.tools})
            try:
                response = self.client.chat.completions.create(model=self.model, messages=self.messages, tools=self.tools)
            except Exception as exc:
                save('model_error', {'error': str(exc)})
                raise
            save('model_response', response.model_dump(exclude_none=True))
            assistant = response.choices[0].message
            self.messages.append(assistant.model_dump(exclude_none=True))

            if not assistant.tool_calls:
                save('final_answer', {'answer': assistant.content or ''})
                return assistant.content or ''

            for call in assistant.tool_calls:
                code = json.loads(call.function.arguments)['code']
                observation = self.run_python(code)
                self.messages.append({
                    'role': 'tool', 'tool_call_id': call.id, 'content': observation,
                })

        error = f'ReAct reached {self.max_steps} model calls without a final answer'
        save('error', {'error': error})
        raise RuntimeError(error)
