# Running LongDS with Claude Code in a Conda Environment

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

## Run More Tasks

Run one full task:

```bash
python run_claude_longds.py \
  --task-limit 1 \
  --timeout 7200
```

Run all tasks:

```bash
python run_claude_longds.py \
  --all-tasks \
  --timeout 7200 \
  --continue-on-error
```

Useful Claude Code options:

```text
--claude-model NAME        Model passed to `claude --model`.
--permission-mode MODE     Claude Code permission mode. Default: bypassPermissions.
--bare                     Reduce external context, hooks, plugins, and memory lookup.
--max-budget-usd VALUE     Optional per-turn Claude Code budget cap.
```

## Outputs

Outputs are written under `results/<domain>/<dataset>/<task_id>/<run_name>/`.
During each turn, Claude Code stdout and stderr are streamed to the terminal in real time. Raw Claude Code stream JSON and stderr are saved under that turn directory.

For each task run:

```text
results/<domain>/<dataset>/<task_id>/<run_name>/
├── workspace/                    # copied data plus Claude Code temporary files
│   └── data/                     # copied released dataset files
├── claude_turn.schema.json
├── summary.json
├── task_metadata.json
├── task_metadata_with_sources.json
├── results.json
├── results_with_ground_truth.json
└── detail/
    └── turn_1/
        ├── prompt.md
        ├── last_message.json
        ├── result.json
        ├── claude_stdout.jsonl
        └── claude_stderr.txt
```

Claude Code is launched with `cwd` set to the task workspace:
`results/<domain>/<dataset>/<task_id>/<run_name>/workspace/`. From inside Claude Code, benchmark files are available under `data/`, and temporary analysis files should be written outside `data/`.

The runner first copies only that task's released `data/` directory into `workspace/data/`. Claude Code is not given the original `dataset/task/...` path that contains `task.json`, `task.py`, `task.ipynb`, metadata, and gold answers.

After a run finishes, you can reopen the claude code session from `runners/claude_code/results/<domain>/<dataset>/<task_id>/<run_name>/workspace`; the session ID is recorded in `runners/claude_code/results/<domain>/<dataset>/<task_id>/<run_name>/summary.json`.

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
  --run-dir results/<domain>/<dataset>/<task_id>/<run_name>
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
