# Minimal ReAct example

[`react_agent.py`](react_agent.py) defines an `Agent` class with `respond(message)` using an
OpenAI-compatible Chat Completions endpoint with function calling. It keeps the
full conversation across turns and gives the model one tool: execute Python.

The instance owns `client`, `model`, `messages`, and `tools`; `run_python()`
executes the tool and `respond()` runs the ReAct loop. The runner constructs one
instance per task and reuses it across all turns. No module-level agent instance
or forwarding function is needed.

The loop is: call model → execute requested Python → append tool output → call
model again, until it returns an answer without tool calls. There is a limit of
20 model calls per turn. Python errors are returned as tool observations so the
model can correct its code.

Set `model` in [`react_config.json`](react_config.json) to your API's model name:

```json
{
  "model": "<your-model>",
  "max_steps": 20
}
```

LongDS passes these values to `Agent(model=..., max_steps=...)`. Run from the
`longds/` root, forwarding the API credentials through environment variables:

```bash
export OPENAI_API_KEY="<your-key>"
export OPENAI_BASE_URL="<your-chat-completions-base-url>"

python runners/custom/run_custom_longds.py \
  --agent runners/custom/examples/react_agent.py:Agent \
  --agent-config runners/custom/examples/react_config.json \
  --requirements runners/custom/examples/react_requirements.txt \
  --env OPENAI_API_KEY --env OPENAI_BASE_URL \
  --task-limit 1 --turn-limit 2
```

For OpenAI's default endpoint, omit both the `OPENAI_BASE_URL` export and its
`--env` flag. This is a real model run and may incur charges. To evaluate all of
Lite, remove the two limits. Add `--run-parallel 4 --judge` after configuring
`JUDGE_API_KEY` and `JUDGE_BASE_URL` on the host.

The agent runs in Docker by default. Its Python tool uses `/usr/local/bin/python`
from the analysis image, reads `data/`, and writes intermediate files in the
current workspace. Each tool call uses a fresh Python process; variables do not
persist, but files and chat history do. For local runs, add `--local`, omit
`--requirements` and `--env` flags, install `openai` yourself, and add
`"python": "/path/to/analysis/python"` to the config for your local interpreter.

The example deliberately has no history compression, memory retrieval, or
framework abstraction. Tool output is capped at 20,000 characters, and each Python
call has a 120-second timeout. Code and observations appear in `agent.stderr.log`.

Every turn also saves `detail/turn_N/trajectory.jsonl`: the exact model requests
(including history and tools), raw API responses, executed Python, full captured
Python output before truncation, and final answer. Requests and responses are
recorded as they happen, so earlier steps survive failures. The raw response
preserves any reasoning fields and usage returned by the endpoint; it cannot
provide reasoning that the endpoint does not expose. `prompt.md` records the
message actually passed to `respond()`. No API key or HTTP headers are logged.

Recording uses the optional `from longds import save` helper provided by the
runner, so the example does not implement file handling. Its system prompt is
the first entry in `self.messages`, set in `Agent.__init__`; edit that entry to change it.

The config accepts `model`, `python`, and `max_steps`. The class is selected
explicitly by `:Agent`; there is no forwarding function or required base class.

[`agent.py`](agent.py) is the offline echo example for checking the runner without
calling a model.
