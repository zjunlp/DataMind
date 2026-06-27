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

Make sure Codex CLI is authenticated first:

```bash
codex --version
codex login
```

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

Run all tasks:

```bash
python run_codex_longds.py \
  --all-tasks \
  --timeout 7200 \
  --continue-on-error
```

Outputs are written under `results/<domain>/<dataset>/<task_id>/<run_name>/`.
During each Codex turn, stdout and stderr are streamed to the terminal in real time with formatted, colorized step blocks. Raw Codex JSONL stdout and stderr are still saved under that turn directory.

For each task run:

```text
results/<domain>/<dataset>/<task_id>/<run_name>/
├── workspace/                    # copied data plus Codex temporary files
│   └── data/                     # copied released dataset files
├── codex_turn.schema.json
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
        ├── codex_stdout.jsonl
        └── codex_stderr.txt
```

The Codex CLI execution directory (`-C`) is the task workspace:
`results/<domain>/<dataset>/<task_id>/<run_name>/workspace/`. From inside Codex, benchmark files
are available under `data/`, and temporary analysis files should be written outside `data/`.

The runner first copies only that task's released `data/` directory into `workspace/data/`.
Codex is not given the original `dataset/task/...` path that contains `task.json`, `task.py`,
`task.ipynb`, metadata, and gold answers.

During a task, `results.json` does not include ground truth. After the task finishes, the runner
writes `results_with_ground_truth.json` and `task_metadata_with_sources.json` for offline scoring
and debugging. The run-level `summary.json` is saved in each task's run directory.

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

The judge writes `results_eval.json` back to each run directory. Runs that already have
`results_eval.json` are skipped by default and reused in the printed summary. In all-runs mode,
no aggregate file is written unless `--out` is provided:

```bash
python judge.py --out results_eval.json
```

To force re-evaluation, pass `--overwrite`:

```bash
python judge.py --overwrite
```
