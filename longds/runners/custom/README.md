# Run your own agent on LongDS

Provide **your own Agent class with `respond(self, message: str) -> str`**.
LongDS supplies questions, a Docker analysis environment, sequential turns,
saved answers, and optional judging. No inheritance, wrapper Agent, lifecycle
hooks, or Dockerfile is required for the normal Python workflow.

Run from the `longds/` root on Linux with Python 3.10+ and a working local Docker
daemon. [Download the dataset](../../README.md#1-download-the-dataset) first.

## 1. Connect your agent

Define your class in `my_agent.py`:

```python
class MyAgent:
    def respond(self, message: str) -> str:
        # Run your analysis / model-and-tool loop here.
        answer = ...  # Your final answer string.
        return answer
```

This is the agent itself, not a mandatory wrapper around another Agent. Only
`respond()` is required. Add an ordinary `__init__()` if you need a model client,
message history, or other state. Existing frameworks with different APIs may
need a small forwarding method; agents implementing `respond()` need no extra
adapter. The [ReAct example](examples/README.md) shows a complete implementation.

Select the class explicitly with `--agent my_agent.py:MyAgent`. LongDS constructs
one instance per task, calls its `respond()` method sequentially for every turn,
and destroys the task process afterward. A new task gets a fresh instance and
workspace. The runner does **not** append previous questions or answers for you:
your agent owns conversation history, memory, planning, and tools.

The process already starts in `/workspace` inside Docker. You do not need to
pass `work_dir` in a constructor unless your tool framework uses a separately
configured execution directory. Optional constructor keyword arguments can be
supplied through `--agent-config config.json`.

Your agent also owns the **system prompt**. LongDS supplies a string to
`respond()`; the environment instructions in that string are not a model API
system message. Your agent decides how to place the supplied message into its
conversation and which system instructions to use.

The first message contains environment instructions, the current context, and
question. Later messages contain only that turn's context and question. Input
files are read-only at `/workspace/data`; tools, scripts, and intermediate files
belong in `/workspace`. The supplied analysis image has `/usr/local/bin/python`
for analysis. Point remote tools at the task container too; host paths are not
implicitly available to a remote agent service.

Return the final answer as a string, including JSON text if the question asks
for structured output. Wait for tool execution to finish before returning.
Generators, asynchronous coroutines, dictionaries, and background task IDs are
not final answer strings. Wrap an async framework with a persistent event loop
if necessary. Python, native-library, and subprocess stdout logs from Python
adapters are captured in `agent.stderr.log` automatically.

## 2. Run it

```bash
python runners/custom/run_custom_longds.py --agent my_agent.py:MyAgent
```

Docker is the default. On first use, LongDS builds the missing `executor-prebuilt`
analysis image from the repository; later runs reuse it. The first build needs
network access and can take time. Your agent file is mounted automatically:
editing it does not require rebuilding an image.

For a two-turn check before a full run:

```bash
python runners/custom/run_custom_longds.py \
  --agent my_agent.py:MyAgent --task-limit 1 --turn-limit 2
```

A complete **offline echo example** is already provided; no generator is needed:

```bash
python runners/custom/run_custom_longds.py \
  --agent runners/custom/examples/agent.py:Agent \
  --task-limit 1 --turn-limit 2
```

The echo example tests the integration and persistent state; it does not solve
the benchmark. Add `--dry-run` to validate dataset selection and write prompt
previews without importing agent code, building images, or starting Docker.

The [minimal ReAct agent](examples/README.md) is the main integration example:
one class, one Python tool, full chat history, and a short tool-calling loop.

### Extra dependencies

List the additional packages your agent imports in `requirements.txt`:

```bash
python runners/custom/run_custom_longds.py \
  --agent my_agent.py:MyAgent --requirements requirements.txt
```

LongDS builds and caches a dependency image keyed by the base image ID,
requirements contents, and installation recipe. The agent gets its own Python
virtual environment; the base analysis Python is preserved. Pin versions for
reproducible runs. Use a self-contained pip requirements file containing package
specifiers or URLs. Local packages, nested `-r` files, private build credentials,
and system packages belong in a custom image instead. Only the supplied
requirements file is copied into the automatic build context.

The cache does not refresh unpinned packages merely because new versions appear.
Change/pin the requirements or remove the printed dependency image to rebuild.

### Model credentials

Export your agent's usual variables, then forward only the ones it needs:

```bash
export OPENAI_API_KEY="<agent-key>"
python runners/custom/run_custom_longds.py \
  --agent my_agent.py:MyAgent --env OPENAI_API_KEY
```

Repeat `--env NAME` for other keys/endpoints. `--env-file agent.env` accepts
literal `NAME=value` lines, blank lines, and comments starting with `#`; it does
not perform shell expansion or quote removal. Explicit `--env` values override
values from env files. Judge variables are not forwarded. Do not put credentials
in agent source, requirements files, or images.

### Score the run

Install the host judge dependency and configure its endpoint:

```bash
python -m pip install openai
export JUDGE_API_KEY="<judge-key>"
export JUDGE_BASE_URL="<judge-base-url>"

python runners/custom/run_custom_longds.py \
  --agent my_agent.py:MyAgent \
  --env OPENAI_API_KEY \
  --run-parallel 4 --judge
```

Keep `--requirements` or other options your agent needs. This runs all of
**v1.1 Lite: 24 tasks / 777 turns**. Use `--split full` for 68 tasks.
Model and judge calls may incur charges. Set `JUDGE_MODEL` or `--judge-model` to
change the judge. `--model` is an optional reporting label; it does not configure
your agent's actual model.

Results appear in `results/longds_v1.1_lite/<run_name>/`:

```text
summary.json
<domain>/<dataset>/taskN/
    results.json                    # answers, saved after each turn
    results_with_ground_truth.json  # host judge input, after agent shutdown
    results_eval.json               # with --judge
    task_metadata.json
    agent.stderr.log
    judge.log                       # with --judge
    detail/turn_N/result.json
    workspace/                      # retained agent files
```

Each turn also saves `input.json` and its stderr log; `respond()` agents get
`prompt.md` containing the actual input message. The ReAct example additionally
saves `trajectory.jsonl` with model requests/responses and tool execution.
For another agent, `respond()` alone exposes only its input/output: internal
steps cannot be inferred automatically. Use the optional `save()` helper below
to record whichever internal events are useful. This does not change the
`respond()` interface.

### Optional: save internal traces

```python
from longds import save


class MyAgent:
    def respond(self, message: str) -> str:
        save("request", {"message": message})
        answer = ...  # Your agent's final answer string.
        save("response", {"answer": answer})
        return answer
```

`save(event, data)` appends a timestamped event to the current turn's
`trajectory.jsonl`. Event names are arbitrary strings; data can be any
JSON-serializable value. For SDK response objects, pass their dictionary/JSON
representation (for example, `response.model_dump(mode="json")`). Call it around
model requests, tool execution, memory updates, or any other steps you want to
inspect. It records only what you pass; do not include credentials.

The helper is supplied automatically by the Python worker, in both Docker and
local modes: no extra package installation or path configuration is required
when using the runner. The runner selects the turn directory and copies traces
into `detail/turn_N/`, including when a task fails. Calls made during module
initialization, before the first turn, go to `workspace/.longds/trajectory.jsonl`.
When using the helper outside the runner, make `runners/custom` importable;
the default output is `.longds/trajectory.jsonl` in the current directory.

Using `save()` is optional. Without it, the runner still records inputs, final
answers, stdout/stderr, timing, and errors. The ReAct example already uses it.

Scores are the equal mean of fully judged task means; incomplete/unjudged tasks
are excluded. Report completion counts with scores. A failed turn stops its task
and preserves completed answers; other tasks continue. Failures produce
`error.json` and a nonzero process exit code. Sessions cannot be reconstructed
from saved answers: rerun failed tasks with a fresh run name.

For a judge retry without rerunning the agent:

```bash
python runners/src/judge.py --results-root results/longds_v1.1_lite/RUN --overwrite
python runners/summarize_scores.py results/longds_v1.1_lite --run-name RUN
```

`--overwrite` rejudges existing evaluations too. Use `--run-dir` pointing at one
task to retry only that task. Separate judging updates task evaluation files,
but does not regenerate the original invocation's `summary.json`.

## Advanced options

These are optional; the normal interface is an Agent class with one `respond()` method.

| Need | Option |
| --- | --- |
| Agent imports sibling source files | `--agent-dir ./agent-source` mounts that explicit directory at `/agent` |
| Agent is an installed module | `--agent my_package.agent:MyAgent` loads the installed class |
| Existing image with dependencies/tools | `--docker-image my-agent:latest` |
| Custom Dockerfile | `--docker-build ./image-directory` builds once before tasks |
| Different agent Python | `--agent-python /opt/venv/bin/python` |
| Local execution without Docker | `--local` (install dependencies yourself) |
| Timeout | `--timeout 3600` per turn and initialization/finalization request |
| Limits | `--task-limit N`, `--turn-limit N`, `--start-index N` |
| Output location/name | `--output-dir PATH`, `--run-name NAME` (must be new) |

An existing local adapter file is mounted at `/longds/adapter.py`. Use
`--agent-dir` if filename-relative resources or sibling imports are needed; do
not mount the benchmark repository. Absolute adapter paths absent on the host
are interpreted inside the image, e.g. `--agent /opt/my_agent.py:MyAgent`.

Custom images must provide the Python your adapter needs; `--requirements` uses
`/usr/local/bin/python` to create its virtual environment. Arbitrary external
images are not automatically pulled. Existing ENTRYPOINT/CMD are replaced by
the worker or command. If your image needs services, arrange their startup in
your adapter. Custom images derived from `executor-prebuilt` need that image
built first if it is not yet present.

Local mode runs in the host Python environment and copies task data into the
workspace. Use `Path.cwd()` to configure local tool paths; they are not always
`/workspace`. Unlike Docker, local mode has no filesystem isolation. Docker is
now the default for class and command adapters too: add `--local` to old commands
that intentionally use the host environment.

### Compatibility with earlier interfaces

Module-level functions still work: `--agent my_agent.py:respond`, or
`--agent my_agent.py` when that file defines `respond(message)`. Without a suffix,
loading prefers a module-level `respond` and otherwise falls back to `Agent`.
Use an explicit class name for the normal class workflow to avoid ambiguity.
Functions are configured through their module/environment, not constructor args.

Older classes with `run_turn(context, question)` remain supported, with optional
`start_task(workspace, data_dir)` and `end_task()` hooks. They receive raw
context/question and build their own messages, and may return a string or a dict
with a string `answer` and optional `reasoning_summary`/`files_used`. If a class
also defines `respond(message)`, that method takes precedence and must return a
string. New agents do not need `run_turn()` or lifecycle hooks.

### Existing CLI / non-Python agents

A Python `respond()` can invoke your CLI with `subprocess`, reuse its native
session ID, and extract its final answer. Docker does not automatically infer
arbitrary CLI arguments or session semantics.

Alternatively, run a persistent executable directly:

```bash
python runners/custom/run_custom_longds.py \
  --docker-image my-cli-agent:latest \
  --agent-command 'node /opt/my-agent/server.js'
```

This advanced route requires the following JSON Lines protocol. Read stdin one
line at a time, write exactly one flushed response on stdout, and log to stderr.
The command is parsed without a shell; it receives all turns in one process.

| Request | Response |
| --- | --- |
| `{"type":"start","workspace":"/workspace","data_dir":"/workspace/data","config":{}}` | `{"type":"ready"}` |
| `{"type":"turn","turn_id":1,"context":"…","question":"…"}` | `{"type":"answer","answer":"…"}` |
| `{"type":"end"}` | `{"type":"done"}` |

## Execution boundary

Each task gets an independent container with a read-only input mount and writable
workspace. Only the selected adapter, optional explicit code directory, and
worker are mounted. Future turns, task source files, reference answers, other
results, the host home directory, and Docker socket are not mounted. Do not
bundle those files in a custom image or code directory. Judging stays on the host.

Artifacts survive container removal. The host workspace's `data/` is an empty
mount point after execution; original input remains in `dataset/`. Container
UID:GID matches the host user by default (`--docker-user` overrides it), with
`HOME=/workspace/.home`. Containers are removed on completion, failure, timeout,
and Ctrl+C. SIGKILL or host/daemon failure may require manual removal using the
container name in `task_metadata.json`.

Validated with a local Linux Docker daemon. Remote daemons, Docker Desktop, and
user-namespace remapping are not validated. Network access uses normal Docker
networking; localhost refers to the container, not the host. Docker administrators
can inspect forwarded credentials. Use trusted agent code/images and report the
execution environment, tool access, model/agent version, and benchmark split.
