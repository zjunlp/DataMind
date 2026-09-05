# Running LongDS with Claude Code in a Conda Environment

Results default to `./results/longds_<version>_<split>/` relative to the
current working directory. `--output-dir` overrides the base `./results`;
the dataset version/split group is appended automatically.

Task selection defaults to `--longds_version v1.1 --split lite`. Use `--split lite`
for the 24-task subset, or `--longds_version v1 --split full` for v1. All versions
share `dataset/data/longds`. See the [shared runner guide](../README.md) for path
overrides and versioned results.

This directory contains a direct Claude Code runner for LongDS-Bench. It does not use the DSGym Docker executor or LiteLLM. Claude Code runs inside each task workspace, uses its own shell/code tools, and keeps the same Claude session across turns in a task.

## Files

- `run_claude_longds.py`: runs LongDS tasks with `claude -p` and a fixed Claude session id per task.
- `judge.py`: scores Claude Code run outputs with the LongDS LLM judge.
- `prompt.py`: stores the first-turn prompt template and turn prompt formatting.
- `requirements-environment.txt`: Python packages for the local Claude Code LongDS environment.

## Create the Conda Environment

Create and activate a Python 3.12 conda environment:

```bash
cd /mnt/40t/xkw/LongMemDA/DataMind/longds/runners/claude_code

conda create -n longds python=3.12 -y
conda activate longds
```

Install the environment packages from this directory:

```bash
pip install --upgrade pip
pip install -r requirements-environment.txt
```

`requirements-environment.txt` matches the LongDS Docker executor Python package set and includes `openai` for judge/API calls.

## Quick Start

Make sure Claude Code is installed and authenticated:

```bash
claude --version
claude
```

Then run one LongDS turn from the activated conda environment:

```bash
cd /mnt/40t/xkw/LongMemDA/DataMind/longds/runners/claude_code

python run_claude_longds.py \
  --task-limit 1 \
  --turn-limit 1
```

`run_claude_longds.py` passes the current Python executable to Claude Code as `--analysis-python`, so when you run it from the activated `longds` conda environment, Claude Code is instructed to use that conda Python for analysis code.

To be explicit:

```bash
python run_claude_longds.py \
  --task-limit 1 \
  --turn-limit 1 \
  --analysis-python "$(python -c 'import sys; print(sys.executable)')"
```

## Docker Mode

Docker mode keeps Claude Code's normal tool set, but prevents it from seeing host files by running
one dedicated container per task. The runner copies only the current task's `data/` directory into
the container workspace, executes all turns with `docker exec`, then snapshots `/workspace` back to
the host run directory.

Build the base LongDS executor image first, then build the thin Claude Code image from it:

```bash
cd /mnt/40t/xkw/LongMemDA/DataMind/longds

docker build \
  -t executor-prebuilt \
  runners/DSGym/executors/container_images/longds_image

docker build \
  -t longds-claude-code:latest \
  --build-arg BASE_IMAGE=executor-prebuilt \
  --build-arg CLAUDE_CODE_VERSION=latest \
  runners/claude_code
```

Then run with Docker enabled:

```bash
python runners/claude_code/run_claude_longds.py \
  --use-docker \
  --task-limit 1 \
  --turn-limit 1
```

In Docker mode the default analysis Python is `/usr/local/bin/python`, the Python installed in the
container. For each task, the runner starts a container named from the run and task id, copies data
to `/workspace/data`, and runs each turn as:

```bash
docker exec -i <task-container> claude -p ...
```

The same container is reused for all turns in that task, so Claude Code session files, `/tmp`, and
intermediate analysis files persist across turns. By default the container is removed after the task
workspace is copied back to `results/longds_<version>_<split>/<run_name>/<domain>/<dataset>/<task_id>/workspace`; pass
`--keep-docker-container` to keep it for debugging or manual resume.

Authentication follows the same shape as Harbor: pass secret names into the agent environment, and
keep secret values outside runner metadata and Docker command arguments. The runner automatically
passes common Claude Code/API variables when they exist in the runner process environment:

```bash
export ANTHROPIC_API_KEY="<your_anthropic_api_key>"
export ANTHROPIC_BASE_URL="<optional_third_party_gateway_base_url>"

python runners/claude_code/run_claude_longds.py \
  --use-docker \
  --task-limit 1
```

You can also put secrets in a local `.env` file and load it for Docker only:

```bash
python runners/claude_code/run_claude_longds.py \
  --use-docker \
  --docker-env-file /path/to/claude.env \
  --task-limit 1
```

The `.env` file can contain `KEY=VALUE` or `export KEY=VALUE` lines. Values loaded this way are
placed in the Docker CLI process environment; Docker receives only `--env KEY`, not
`--env KEY=VALUE`.

Claude Code also accepts a settings JSON file. The runner passes
`runners/claude_code/settings.json` by default.

```json
{
  "model": "deepseek-v4-flash",
  "env": {
    "ANTHROPIC_API_KEY": "<your_api_key>",
    "ANTHROPIC_BASE_URL": "https://www.dmxapi.cn/v1"
  }
}
```

```bash
python runners/claude_code/run_claude_longds.py \
  --use-docker \
  --task-limit 1
```

Use `--claude-settings /path/to/claude-settings.json` to override the default settings file. When
the model is read from settings, the runner uses it for the run name and metadata but does not add a
separate `claude --model` argument. Pass `--claude-model NAME` when you want an explicit CLI model
argument to override the settings model.

In Docker mode, the runner copies this settings file to `/tmp/longds_claude_settings.json` inside
each task container and passes `claude --settings /tmp/longds_claude_settings.json ...`. The file is
not copied into `/workspace`, so it is not included in the final workspace snapshot. If the settings
file contains credentials, keep it outside this repository and prefer `apiKeyHelper` or environment
variables for secrets.

For compatibility with Harbor-style shared provider keys, the runner maps these variables when the
standard target variable is not already set:

```text
HARBOR_ANTHROPIC_KEY      -> ANTHROPIC_API_KEY
HARBOR_ANTHROPIC_BASE_URL -> ANTHROPIC_BASE_URL
```

Additional variables can be selected with repeatable `--docker-env KEY`. Avoid typing API keys as
`--docker-env KEY=VALUE` in shell history; prefer `export KEY=...` or `--docker-env-file` for
secrets. The runner does not print these values in run config, logs, or metadata.

Some gateways describe themselves as OpenAI-compatible because their raw HTTP API also supports
`/v1/chat/completions`. That is a gateway protocol detail, not the runner identity: this runner still
starts Claude Code and passes provider configuration through Claude Code's `ANTHROPIC_*` variables.

## Run More Tasks

Run one full task:

```bash
python run_claude_longds.py \
  --task-limit 1 \
  --timeout 7200
```

Run the LLM judge automatically after each Claude Code task finishes:

```bash
python run_claude_longds.py \
  --task-limit 1 \
  --timeout 7200 \
  --judge
```

`--judge` uses the existing `JUDGE_API_KEY`, `JUDGE_BASE_URL`, and optional `JUDGE_MODEL`
environment variables. Each task is evaluated immediately after all its turns finish, before the
worker takes another task.

Run all tasks:

```bash
python run_claude_longds.py \
  --run-parallel 4 \
  --timeout 7200
```

All remaining tasks after `--start-index` are selected by default. Pass `--task-limit N` to run
only the next `N` tasks. A failed task is recorded and does not stop the remaining tasks; after all
selected tasks finish, the runner exits with status `1` if any task failed.

If a selected task already has a directory with the same `run_name`, that task is skipped without
reading, overwriting, resuming, or judging the existing result. Skipped tasks are reported separately
as `skipped_tasks` in the final status. Pass `--overwrite` to delete each existing task run directory
before running that task again. Do not run concurrent processes with the same `run_name` when using
`--overwrite`.

`--run-parallel` controls task-level concurrency and defaults to `1`. Turns within the same task
always run sequentially in one Claude Code session. When `--judge` is enabled, each worker evaluates
its completed task before taking another task. Parallel terminal output from different tasks may be
interleaved; each task's raw and formatted logs remain isolated in its own run directory.

Useful Claude Code options:

```text
--claude-model NAME        Model passed to `claude --model`.
--claude-settings PATH     Settings JSON file passed to `claude --settings`. Defaults to settings.json.
--permission-mode MODE     Claude Code permission mode. Default: bypassPermissions.
--bare                     Reduce external context, hooks, plugins, and memory lookup.
--max-budget-usd VALUE     Optional per-turn Claude Code budget cap.
--use-docker               Run Claude Code in the LongDS Claude Code container.
```

## Outputs

Outputs are written under `results/longds_<version>_<split>/<run_name>/<domain>/<dataset>/<task_id>/`.
During each turn, Claude Code stdout and stderr are streamed to the terminal in real time. Raw Claude Code stream JSON and stderr are saved under that turn directory.

For each task run:

```text
results/longds_<version>_<split>/<run_name>/<domain>/<dataset>/<task_id>/
├── workspace/                    # copied data plus Claude Code temporary files
│   └── data/                     # copied released dataset files
├── claude_turn.schema.json
├── task_metadata.json
├── task_metadata_with_sources.json
├── results.json
├── results_with_ground_truth.json
└── detail/
    └── turn_1/
        ├── prompt.md
        ├── last_message.json
        ├── result.json
        ├── formatted_steps.json
        ├── claude_stdout.jsonl
        └── claude_stderr.txt
```

Claude Code is launched with `cwd` set to the task workspace:
`results/longds_<version>_<split>/<run_name>/<domain>/<dataset>/<task_id>/workspace/`. From inside Claude Code, benchmark files are available under `data/`, and temporary analysis files should be written outside `data/`.

The runner first copies only that task's released `data/` directory into `workspace/data/`. Claude Code is not given the original `dataset/task/...` path that contains `task.json`, `task.py`, `task.ipynb`, metadata, and gold answers.

With `--use-docker`, no host workspace is bind-mounted during normal execution. Claude Code tools are
not restricted, but the container only receives the copied task data and cannot see the host repo,
task JSON, gold answers, or other datasets.

## Disk Usage

The full LongDS data set is about 19 GB, so copying every task's data would leave ~20 GB behind per run. To keep that bounded, the runner deletes `workspace/data/` as soon as a task's turns finish and records the result in `task_metadata.json`:

```json
"data_cleanup": {"removed": true, "reason": "task_completed", "freed_bytes": 44969266}
```

A finished task leaves roughly 100 KB of results. Only the copied inputs are removed: helper scripts and intermediate artifacts Claude Code wrote into the workspace are kept as trajectory evidence, as is everything under `detail/`, so `judge.py` still works on a cleaned run.

Two cases keep the data:

- A task that fails keeps `workspace/data/` so the failure can be reproduced in place.
- `--keep-data` disables the cleanup entirely. Use it when you intend to resume the session with `manual_resume_command`, since a resumed session cannot re-read data that has been removed.

`--dry-run` never copies data at all; it still validates that each task's source data directory exists.

After a run finishes, you can reopen the Claude Code session from the task workspace. The session ID
and manual resume command are recorded in `task_metadata.json`.

Note: Claude Code permission mode controls tool approval behavior, not filesystem sandboxing. This runner isolates tasks by copying data into a workspace and launching Claude Code with that workspace as `cwd`.

## Run the LLM Judge

Set the judge endpoint first:

```bash
export JUDGE_API_KEY="<your_judge_api_key>"
export JUDGE_BASE_URL="<your_judge_base_url>"
```

Score one Claude Code run:

```bash
python judge.py \
  --run-dir results/longds_<version>_<split>/<run_name>/<domain>/<dataset>/<task_id>
```

Or score every completed run under `results/`:

```bash
python judge.py
```

The judge writes `results_eval.json` back to each run directory. Runs that already have
`results_eval.json` are skipped by default and reused in the printed summary. In all-runs mode,
no aggregate file is written unless `--out` is provided:

```bash
python judge.py --out results_eval.json
```

To force re-evaluation, pass `--overwrite`.
