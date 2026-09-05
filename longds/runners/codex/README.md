# Running LongDS with Codex in a Conda Environment

Results default to `./results/longds_<version>_<split>/` relative to the
current working directory. `--output-dir` overrides the base `./results`;
the dataset version/split group is appended automatically.

Task selection defaults to `--longds_version v1.1 --split lite`. Use `--split lite`
for the 24-task subset, or `--longds_version v1 --split full` for v1. All versions
share `dataset/data/longds`. See the [shared runner guide](../README.md) for path
overrides and versioned results.

This directory contains a direct Codex runner for LongDS-Bench. It can run Codex locally or in one
isolated Docker container per task. The Docker image extends the LongDS executor environment with
Codex CLI.

## Files

- `run_codex_longds.py`: runs LongDS tasks directly with `codex exec` and `codex exec resume`.
- `config.toml`: default local Codex configuration (ignored by Git).
- `config.example.toml`: configuration template without credentials.
- `config_oauth.toml`: OpenAI provider configuration for ChatGPT browser login.
- `judge.py`: scores Codex run outputs with the LongDS LLM judge.
- `prompt.py`: stores the first-turn prompt template and turn prompt formatting.
- `requirements-environment.txt`: Python packages for the local Codex LongDS environment.

## Create the Conda Environment

Create and activate a Python 3.12 conda environment:

```bash
cd /mnt/40t/xkw/LongMemDA/DataMind/longds/runners/codex

conda create -n longds python=3.12 -y
conda activate longds
```

Install the environment packages from this directory:

```bash
pip install --upgrade pip
pip install -r requirements-environment.txt
```

`requirements-environment.txt` matches the LongDS Docker executor Python package set and includes `openai` for judge/API calls. 

## Docker Mode

Docker mode starts one dedicated container per task. Only the current task's released `data/`
directory is copied to `/workspace/data`; every turn for that task runs in the same container and
resumes the same Codex thread. Parallel tasks therefore have independent containers, workspaces,
and Codex homes.

Build the LongDS executor base image, then the thin Codex image:

```bash
cd /mnt/40t/xkw/LongMemDA/DataMind/longds

docker build \
  -t executor-prebuilt \
  runners/DSGym/executors/container_images/longds_image

docker build \
  -t longds-codex:latest \
  --build-arg BASE_IMAGE=executor-prebuilt \
  --build-arg CODEX_VERSION=latest \
  runners/codex
```

Run a one-turn smoke test:

```bash
python runners/codex/run_codex_longds.py \
  --use-docker \
  --task-limit 1 \
  --turn-limit 1
```

The bundled `config.toml` selects the endpoint, model, reasoning effort, and bearer token. No Codex
authentication or model environment variables are needed. Common proxy variables are still
inherited. Use repeatable `--docker-env KEY` or `--docker-env-file PATH` only for unrelated runtime
variables.

Inside Docker, Codex tools are unrestricted because the task container is the isolation boundary.
Each turn is executed as `docker exec -i ... codex exec ...`; later turns add
`resume <thread_id>`. After the task, `/workspace` is copied back to the run directory and the
Codex session files are saved under `codex_home/` with `auth.json` removed. Codex shell snapshots
and the copied `config.toml` are also removed because they can contain inherited environment values
or custom headers. The container is then deleted unless `--keep-docker-container` is set.

By default the runner loads `config.toml` from this directory. Docker mode copies it to
`/codex-home/config.toml`; local mode copies it to the task's dedicated Codex home. Pass
`--codex-config PATH` to select another file. Explicit `--codex-model`, `--reasoning-effort`, and
`--codex-base-url` values override the file. The custom provider uses
`experimental_bearer_token`, so the API key is read directly from TOML rather than an environment
variable. Codex documents direct bearer tokens as discouraged; `config.toml` is therefore ignored
by Git and its copied task version is removed from saved session state.

For ChatGPT browser-login authentication, first run `codex login` on the host and verify that
`~/.codex/auth.json` exists. Then use the OpenAI provider configuration and explicitly pass the
credential file:

```bash
python runners/codex/run_codex_longds.py \
  --use-docker \
  --codex-config runners/codex/config_oauth.toml \
  --codex-auth ~/.codex/auth.json \
  --task-list-name task_list_lite.json
```

Each task receives an independent copy at `/codex-home/auth.json`. Both successful and failed tasks
remove `auth.json` from saved results. Do not combine ChatGPT OAuth credentials with a third-party
provider configuration such as DMX.


## Run a Codex Smoke Test

When using the normal Codex provider with a different configuration, authenticate the CLI first:

```bash
codex --version
codex login
```

To override only the endpoint for one run, use the runner argument:

```bash
python run_codex_longds.py \
  --codex-base-url "https://your-api.example.com/v1" \
  --task-limit 1
```

The override applies to the provider selected by `model_provider`; its bearer token is still read
from the selected TOML file. The configured endpoint must support the Responses API.

Override the model and reasoning effort for one run without changing `config.toml`:

```bash
python run_codex_longds.py \
  --codex-model gpt-5.6-sol \
  --reasoning-effort high \
  --task-limit 1
```

Supported values are `none`, `low`, `medium`, `high`, `xhigh`, and `max`. Omit the option to use
the Codex configuration default. The selected endpoint and model must support the requested value.

For a provider that requires Codex reasoning-summary metadata, enable it only for that run:

```bash
python run_codex_longds.py \
  --codex-model glm-5.2 \
  --model-supports-reasoning-summaries \
  --task-limit 1
```

This injects `-c model_supports_reasoning_summaries=true` without changing
`~/.codex/config.toml`. Omit the option to use the Codex default, or pass
`--no-model-supports-reasoning-summaries` to force-disable the capability.

Then run one LongDS turn from the activated conda environment:

```bash
cd /mnt/40t/xkw/LongMemDA/DataMind/longds/runners/codex

python run_codex_longds.py \
  --task-limit 1 \
  --turn-limit 1
```

`run_codex_longds.py` passes the current Python executable to Codex as `--analysis-python`, so when you run it from the activated `longds` conda environment, Codex is instructed to use that conda Python for analysis code.

To be explicit:

```bash
python run_codex_longds.py \
  --task-limit 1 \
  --turn-limit 1 \
  --analysis-python "$(python -c 'import sys; print(sys.executable)')"
```

## Run More Tasks

Run one full task:

```bash
python run_codex_longds.py \
  --task-limit 1 \
  --timeout 7200
```

Run the LLM judge automatically after each Codex task finishes:

```bash
python run_codex_longds.py \
  --task-limit 1 \
  --timeout 7200 \
  --judge
```

`--judge` uses the existing `JUDGE_API_KEY`, `JUDGE_BASE_URL`, and optional `JUDGE_MODEL`
environment variables. Each task is evaluated immediately after all its turns finish, before the
next task starts.

Run all tasks:

```bash
python run_codex_longds.py \
  --run-parallel 4 \
  --timeout 7200
```

All remaining tasks after `--start-index` are selected by default. Pass `--task-limit N` to run
only the next `N` tasks. A failed task is recorded and does not stop the remaining tasks; after all
selected tasks finish, the runner exits with status `1` if any task failed.

`--run-parallel` controls task-level concurrency and defaults to `1`. Turns within the same task
always run sequentially in one Codex thread. When `--judge` is enabled, each worker evaluates its
completed task before taking another task. Parallel terminal output from different tasks may be
interleaved; each task's raw and formatted logs remain isolated in its own run directory.

If a selected task already has a directory with the same `run_name`, it is skipped. Pass
`--overwrite` to delete that task directory and run it again. `--task-list-name` selects a JSON file
under `--task-root`; for example, `--task-list-name task_list_lite.json` runs the Lite subset.

Outputs are written under `results/longds_<version>_<split>/<run_name>/<domain>/<dataset>/<task_id>/`. By default,
`run_name` is `codex_<model>_<timestamp>`, for example
`codex_qwen3.7-plus_20260729_120000`. Passing `--run-name` overrides the complete directory name.
During each Codex turn, stdout and stderr are streamed to the terminal in real time with formatted, colorized step blocks. Raw Codex JSONL stdout and stderr are still saved under that turn directory.

For each task run:

```text
results/longds_<version>_<split>/<run_name>/<domain>/<dataset>/<task_id>/
├── workspace/                    # copied data plus Codex temporary files
│   └── data/                     # copied released dataset files
├── codex_home/                   # Docker session state; credentials and shell snapshots removed
├── codex_turn.schema.json
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
        ├── codex_stdout.jsonl
        └── codex_stderr.txt
```

The Codex CLI execution directory is always the task workspace:
`results/longds_<version>_<split>/<run_name>/<domain>/<dataset>/<task_id>/workspace/`. From inside Codex, benchmark files
are available under `data/`, and temporary analysis files should be written outside `data/`.
The first turn uses Codex `-C`, and resumed turns are also launched with the workspace as the
subprocess working directory so relative paths cannot fall back to `runners/codex/`.

The runner first copies only that task's released `data/` directory into `workspace/data/`.
Codex is not given the original `dataset/task/...` path that contains `task.json`, `task.py`,
`task.ipynb`, metadata, and gold answers.

## Disk Usage

The full LongDS data set is about 19 GB, so copying every task's data would leave ~20 GB behind per
run. To keep that bounded, the runner deletes `workspace/data/` as soon as a task's turns finish and
records the result in `task_metadata.json`:

```json
"data_cleanup": {"removed": true, "reason": "task_completed", "freed_bytes": 44969266}
```

Peak input-data disk usage therefore scales with `--run-parallel`, not with the number of tasks.
Docker runs also retain Codex session traces under `codex_home/`.

Only the copied inputs are removed. Helper scripts and intermediate artifacts Codex wrote into the
workspace are kept as trajectory evidence, as is everything under `detail/`, so `judge.py` still
works on a cleaned run.

Two cases keep the data:

- A task that fails keeps `workspace/data/` so the failure can be reproduced in place.
- `--keep-data` disables the cleanup entirely. Use it when you intend to resume the session with
  `manual_resume_command`, since a resumed session cannot re-read data that has been removed.

`--dry-run` never copies data at all; it still validates that each task's source data directory
exists.

During a task, `results.json` does not include ground truth. After the task finishes, the runner
writes `results_with_ground_truth.json` and `task_metadata_with_sources.json` for offline scoring
and debugging. If a task fails, its run directory contains `error.json`.

The session ID and manual resume command are recorded in `task_metadata.json`. In Docker mode the
manual command is usable while the task container still exists; pass `--keep-docker-container` to
retain it for interactive debugging.

## Run the LLM Judge

Set the judge endpoint first:

```bash
export JUDGE_API_KEY="<your_judge_api_key>"
export JUDGE_BASE_URL="<your_judge_base_url>"
```

Score one Codex run:

```bash
python judge.py \
  --run-dir results/longds_<version>_<split>/<run_name>/<domain>/<dataset>/<task_id>
```

Or score every completed run under `results/`:

```bash
python judge.py
```

The judge writes a DSGym-compatible `results_eval.json` list back to each run directory. Each
turn contains `turn_id`, `question`, `ground_truth`, `solution`, `success`, `steps`, `trajectory`,
and `judge`; the final list element contains `summary.correct`, `summary.incorrect`, and
`summary.avg_score`. Runs that already have `results_eval.json` are skipped by default and reused
in the printed summary. In all-runs mode, no aggregate file is written unless `--out` is provided:

```bash
python judge.py --out results_eval.json
```

To force re-evaluation, pass `--overwrite`:

```bash
python judge.py --overwrite
```
