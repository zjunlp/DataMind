# Running LongDS with Codex in a Conda Environment

This directory contains a direct Codex runner for LongDS-Bench. It does not use the DSGym Docker executor, but you can create a local conda environment that mirrors the Python packages from `runners/DSGym/executors/container_images/longds_image`.

## Files

- `run_codex_longds.py`: runs LongDS tasks directly with `codex exec` and `codex exec resume`.
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


## Run a Codex Smoke Test

When using the normal Codex provider, authenticate the CLI first:

```bash
codex --version
codex login
```

To use an OpenAI-compatible Responses API instead of the provider in the Codex configuration,
set the endpoint and API key as environment variables:

```bash
export CODEX_BASE_URL="https://your-api.example.com/v1"
export CODEX_API_KEY="<your_api_key>"
```

When `CODEX_BASE_URL` is set, the runner injects a temporary Codex provider that reads
`CODEX_API_KEY`. The API key is not written to commands, logs, metadata, or result files. The
configured endpoint must support the Responses API. If these variables are omitted, Codex uses
its normal authentication and provider configuration. The custom provider does not require a
separate `codex login`.

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
  --all-tasks \
  --run-parallel 4 \
  --timeout 7200 \
  --continue-on-error
```

`--run-parallel` controls task-level concurrency and defaults to `1`. Turns within the same task
always run sequentially in one Codex thread. When `--judge` is enabled, each worker evaluates its
completed task before taking another task. Parallel terminal output from different tasks may be
interleaved; each task's raw and formatted logs remain isolated in its own run directory.

Outputs are written under `results/<domain>/<dataset>/<task_id>/<run_name>/`. By default,
`run_name` is `codex_<model>_<timestamp>`, for example
`codex_qwen3.7-plus_20260729_120000`. Passing `--run-name` overrides the complete directory name.
During each Codex turn, stdout and stderr are streamed to the terminal in real time with formatted, colorized step blocks. Raw Codex JSONL stdout and stderr are still saved under that turn directory.

For each task run:

```text
results/<domain>/<dataset>/<task_id>/<run_name>/
├── workspace/                    # copied data plus Codex temporary files
│   └── data/                     # copied released dataset files
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
`results/<domain>/<dataset>/<task_id>/<run_name>/workspace/`. From inside Codex, benchmark files
are available under `data/`, and temporary analysis files should be written outside `data/`.
The first turn uses Codex `-C`, and resumed turns are also launched with the workspace as the
subprocess working directory so relative paths cannot fall back to `runners/codex/`.

The runner first copies only that task's released `data/` directory into `workspace/data/`.
Codex is not given the original `dataset/task/...` path that contains `task.json`, `task.py`,
`task.ipynb`, metadata, and gold answers.

During a task, `results.json` does not include ground truth. After the task finishes, the runner
writes `results_with_ground_truth.json` and `task_metadata_with_sources.json` for offline scoring
and debugging. If a task fails, its run directory contains `error.json`.

After a run finishes, you can reopen the Codex session from the task workspace. The session ID and
manual resume command are recorded in `task_metadata.json`.

## Run the LLM Judge

Set the judge endpoint first:

```bash
export JUDGE_API_KEY="<your_judge_api_key>"
export JUDGE_BASE_URL="<your_judge_base_url>"
```

Score one Codex run:

```bash
python judge.py \
  --run-dir results/<domain>/<dataset>/<task_id>/<run_name>
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
