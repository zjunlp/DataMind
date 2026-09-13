# Custom runner reference

For the normal integration workflow, start with the [README](README.md).

## Dependencies and environment variables

LongDS builds and caches a dependency image keyed by the base image ID,
requirements contents, and installation recipe. The agent gets its own Python
virtual environment; the base analysis Python is preserved. Pin versions for
reproducible runs. Use a self-contained pip requirements file containing package
specifiers or URLs. Local packages, nested `-r` files, private build credentials,
and system packages belong in a custom image instead. Only the supplied
requirements file is copied into the automatic build context.

The cache does not refresh unpinned packages merely because new versions appear.
Change/pin the requirements or remove the printed dependency image to rebuild.

Repeat `--env NAME` for other keys/endpoints. `--env-file agent.env` accepts
literal `NAME=value` lines, blank lines, and comments starting with `#`; it does
not perform shell expansion or quote removal. Explicit `--env` values override
values from env files. Judge variables are not forwarded. Do not put credentials
in agent source, requirements files, or images.

## Recording internal traces


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

## Scoring and failed runs

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
