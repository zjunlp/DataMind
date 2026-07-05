import os
import time
import asyncio
import traceback
import json
from typing import Dict, Any

from src.runtimes.base_agent import BaseAgent
from src.runtimes.registry import register_agent
from src.core.schema import AgentResult

AGENT_TYPE = "codex"

@register_agent(AGENT_TYPE)
class CodexAgent(BaseAgent):
    """Autonomous data science agent using OpenAI Codex SDK.

    Codex is a full agent with built-in bash/file tools. Instead of the manual
    backend + environment loop used by ReActDSAgent, this agent sends the task
    to Codex and lets it run autonomously.
    """

    def __init__(self, backend: str, model: str, **kwargs):
        super().__init__(AGENT_TYPE, model, backend, **kwargs)

        self.working_dir = kwargs.get('working_dir', '.')
        self.output_schema = kwargs.get('output_schema', {})
        self.api_key = kwargs.get('api_key') or os.environ.get('OPENAI_API_KEY', '')
        self.base_url = kwargs.get('base_url') or os.environ.get('OPENAI_API_BASE', 'https://api.openai.com/v1')
        self.timeout = kwargs.get('timeout', None)

    async def _run_codex(self, prompt: str, working_directory: str = ".") -> Dict[str, Any]:
        """Run a single Codex autonomous session and return the result."""
        from openai_codex_sdk import Codex

        codex = Codex({"api_key": self.api_key, "base_url": self.base_url})

        thread_opts: Dict[str, Any] = {
            "working_directory": working_directory,
        }
        if self.model:
            thread_opts["model"] = self.model

        thread = codex.start_thread(thread_opts)

        run_opts: Dict[str, Any] = {}
        if self.output_schema:
            run_opts["output_schema"] = self._make_openai_strict_schema(self.output_schema)

        try:
            turn = await thread.run(prompt, run_opts)
        except Exception as e:
            raise RuntimeError(f"Codex SDK error: {e}")

        return {
            "final_response": getattr(turn, "final_response", "") or "",
            "id": getattr(turn, "id", ""),
            "thread_id": getattr(turn, "thread_id", ""),
            "items": getattr(turn, "items", []),
        }

    def _run_codex_sync(self, prompt: str, working_directory: str = ".") -> Dict[str, Any]:
        async def runner():
            if self.timeout is None:
                return await self._run_codex(prompt, working_directory)
            return await asyncio.wait_for(
                self._run_codex(prompt, working_directory),
                timeout=self.timeout,
            )
        
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            outer_timeout = None if self.timeout is None else self.timeout + 30
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(lambda: asyncio.run(runner()))
                return future.result(timeout=outer_timeout)

        return asyncio.run(runner())

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def solve_task(self, prompt: str, system: str = "", **kwargs) -> AgentResult:
        start_time = time.time()
        actual_turns = 0
        env = None

        try:
            result = self._run_codex_sync(f"{system}\n\n{prompt}" if system else prompt, self.working_dir)
            items = result.get("items", [])
            
            agent_conversation = []
            for item in items:
                if item.type == "command_execution":
                    actual_turns += 1
                    agent_conversation.append({
                        "type": "command_execution",
                        "command": item.command,
                        "aggregated_output": item.aggregated_output,
                    })
                elif item.type == "agent_message":
                    agent_conversation.append({
                        "type": "agent_message",
                        "text": item.text,
                    })
                else:
                    agent_conversation.append({
                        "type": item.type,
                        "text": getattr(item, "text", "Cannot extract text"),
                    })

            response_text = result["final_response"]

            raw_conversation = [
                {
                    "type": "system_message",
                    "text": system,
                },
                {
                    "type": "user_message",
                    "text": prompt,
                }
            ] + agent_conversation

            execution_time = time.time() - start_time
            
            results_fields = {
                'conversation': raw_conversation,
                'response': self._parse_results(response_text) if response_text else {},
                'raw_response': response_text,
                'turns': actual_turns,
                'error': None,
                'metadata': {
                    'model': self.model,
                    'agent_type': 'codex',
                    'execution_time': execution_time,
                    'conversation_length': len(raw_conversation),
                    'codex_turn_id': result.get("id", ""),
                    'codex_thread_id': result.get("thread_id", ""),
                },
                'raw_result': {
                    'raw_conversation': raw_conversation,
                    "raw_response": response_text,
                    'turns': actual_turns,
                    'codex_items': result.get("items", []),
                },
            }

            return AgentResult(**results_fields)

        except Exception as e:
            execution_time = time.time() - start_time
            error_trace = traceback.format_exc()
            print(f"CodexAgent error: {e}")
            print(f"CodexAgent error trace: {error_trace}")
            results_fields = {
                'conversation': [],
                'response': '',
                'raw_response': '',
                'turns': actual_turns,
                'error': str(e),
                'metadata': {
                    'model': self.model,
                    'agent_type': 'codex',
                    'execution_time': execution_time,
                    'error_trace': error_trace,
                },
                'raw_result': None,
            }

            return AgentResult(**results_fields)
        finally:
            if env is not None:
                env.close()

    def _make_openai_strict_schema(self, schema: dict[str, Any]) -> dict[str, Any]:
        """Convert a Pydantic JSON schema to OpenAI strict structured output format.

        OpenAI's Responses API requires:
            - "additionalProperties": false at the top level
            - "required" must list ALL property keys (no optional fields)

        Pydantic's model_json_schema() only puts truly required fields in "required",
        but OpenAI demands every property is listed. Fields with defaults still work
        because the model will always produce them.
        """
        strict = {**schema, "additionalProperties": False}
        if "properties" in strict:
            strict["required"] = list(strict["properties"].keys())
        return strict

    def _parse_results(self, response_text: str) -> Dict[str, Any]:
        """Parse the raw result from Codex SDK into a structured format."""
        if response_text:
            try:
                parsed = json.loads(response_text)
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to parse Codex response as JSON: {e}")
        return parsed