# Run your own agent on LongDS

Provide an Agent class with **`respond(message: str) -> str`**. LongDS handles
questions, a Docker analysis environment, sequential turns, saved answers,
and optional scoring.

Run commands from the `longds/` root with Python 3.10+ and a working local Linux
Docker daemon. [Download the dataset](../../README.md#1-download-the-dataset) first.

## 1. Connect your agent

Expose your agent in `my_agent.py`:

```python
from your_package import ExistingAgent


class MyAgent:
    def __init__(self, **config):
        self.agent = ExistingAgent(**config)

    def respond(self, message: str) -> str:
        return self.agent.run(message)  # Return the final answer as a string.
```

Adapt the call to your agent's API. If your class already implements `respond()`,
use it directly. No base class is required; other methods and attributes work
normally. See the [complete ReAct example](examples/README.md) for an agent with
conversation history and a Python tool.

- LongDS creates **one instance per task**, reusing it across that task's turns.
- Your agent owns its system prompt, history, memory, and tools. LongDS sends
  only the current context/question each turn, including the first. Put environment
  instructions and analysis guidance in your agent's own prompt, as in the ReAct example.
- Return the final answer string, including JSON text when requested. Finish
  tool execution and consume async/streaming results before returning.
- In Docker, read input from `/workspace/data` and write outputs to `/workspace`.
  Use `/usr/local/bin/python` for analysis with the supplied image.

## 2. Run a two-turn check

List your agent's extra packages in `requirements.txt` (for example, `openai`),
then forward the environment variables it needs. For an agent using an
OpenAI-compatible API:

```bash
export OPENAI_API_KEY="<agent-key>"
export OPENAI_BASE_URL="<your-api-base-url>"

python runners/custom/run_custom_longds.py \
  --agent my_agent.py:MyAgent \
  --requirements requirements.txt \
  --env OPENAI_API_KEY --env OPENAI_BASE_URL \
  --task-limit 1 --turn-limit 2
```

Configuration comes from your agent's own interface:

- **Environment variables:** `--env NAME` forwards a host variable into Docker.
  Your agent reads it with `os.environ`, or its SDK reads it automatically.
- **Constructor arguments:** optionally add `--agent-config config.json`.
  LongDS loads the JSON object and calls `MyAgent(**config)`; for example,
  `{"model": "your-model"}` passes `model="your-model"` if your agent accepts it.
- **Model selection:** use your agent's existing configuration. The bundled
  ReAct example accepts a `model` constructor argument, falling back to
  `REACT_MODEL` (export it and add `--env REACT_MODEL`). The runner's `--model`
  is only a reporting label, not an API model setting.

Use the variable and argument names your agent expects; LongDS does not impose them.
Omit the base URL export and flag to use your client's default endpoint, and omit
`--requirements` if no extra packages are needed.

Docker is the default. LongDS builds the analysis image if missing and caches
extra dependencies. The first build needs network access and may take time.

To check the runner without an API, use the offline echo agent:

```bash
python runners/custom/run_custom_longds.py \
  --agent runners/custom/examples/agent.py:Agent \
  --task-limit 1 --turn-limit 2
```

## 3. Multi-file projects

Mount your source directory with `--agent-dir`:

```text
my_agent/
├── entry.py          # Defines MyAgent
├── core/
├── tools/
├── configs/
└── requirements.txt
```

```bash
python runners/custom/run_custom_longds.py \
  --agent ./my_agent/entry.py:MyAgent \
  --agent-dir ./my_agent \
  --requirements ./my_agent/requirements.txt \
  --env OPENAI_API_KEY --env OPENAI_BASE_URL \
  --task-limit 1 --turn-limit 2
```

The source directory is mounted read-only at `/agent` and added to `PYTHONPATH`.
Resolve bundled configuration files relative to `__file__`; the working
directory is `/workspace`. Put caches and outputs there, outside the source
directory. Mount only your agent project, not the benchmark repository.

For system dependencies or packages requiring a local build, use
`--docker-build ./image-directory` or an existing `--docker-image IMAGE`.

## 4. Run and score the benchmark

Remove `--task-limit` and `--turn-limit` from your working command to run all of
**v1.1 Lite (24 tasks / 777 turns)**. Add `--run-parallel 4` for concurrent tasks,
or `--split full` for the 68-task Full split.

To score answers, install the judge dependency on the host and set its credentials:

```bash
python -m pip install openai
export JUDGE_API_KEY="<judge-key>"
export JUDGE_BASE_URL="<judge-base-url>"
```

Then add `--judge` to your working run command. Model and judge calls may incur
charges. Scores average fully judged task means; report completion counts too.

Results are saved under `results/longds_v1.1_lite/<run_name>/`:

- `summary.json`: completion counts and score.
- `<domain>/<dataset>/taskN/results.json`: answers, saved after each turn.
- Each task also saves `results_eval.json` when judged, `agent.stderr.log`,
  per-turn records in `detail/`, and agent files in `workspace/`.

A failed turn stops its task and preserves completed answers. Use a fresh
`--run-name` to rerun; saved answers do not restore an agent session.

For local execution, custom images, function/CLI interfaces, internal traces,
and judge retries, see the [reference](ADVANCED.md). Run
`python runners/custom/run_custom_longds.py --help` for all options.
