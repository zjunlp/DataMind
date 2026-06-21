# Running LongDS with Codex in a Conda Environment

This directory contains a direct Codex runner for LongDS-Bench. It does not use the DSGym Docker executor, but you can create a local conda environment that mirrors the Python packages from `DSGym/executors/container_images/longds_image`.

## Files

- `run_codex_longds.py`: runs LongDS tasks directly with `codex exec` and `codex exec resume`.
- `prompt.py`: stores the first-turn prompt template and turn prompt formatting.
- `requirements-longds-docker.txt`: exact copy of the LongDS Docker image Python requirements.

## Create the Conda Environment

Run from the repository root:

```bash
cd /mnt/40t/xkw/LongMemDA/DataMind/longds

conda create -n longds python=3.12 -y
conda activate longds
```


Install the same Python packages used by the LongDS Docker executor:

```bash
pip install --upgrade pip
pip install -r scripts/requirements-longds-docker.txt
```



## Run a Codex Smoke Test

Make sure Codex CLI is authenticated first:

```bash
codex --version
codex login
```

Then run one LongDS turn from the activated conda environment:

```bash
cd /mnt/40t/xkw/LongMemDA/DataMind/longds/scripts

python run_codex_longds.py \
  --task-limit 1 \
  --turn-limit 1
```

`run_codex_longds.py` passes the current Python executable to Codex as `--analysis-python`, so when you run it from `longds-codex`, Codex is instructed to use that conda Python for analysis code.

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

Outputs are written under `scripts/results/<domain>/<dataset>/<task_id>/<run_name>/`.

For each task run:

```text
scripts/results/<domain>/<dataset>/<task_id>/<run_name>/
├── data/                         # copied released dataset files
├── workspace/                    # Codex temporary code, caches, and intermediate outputs
├── codex_turn.schema.json
├── summary.json
├── task_metadata.json
├── task_metadata_with_sources.json
├── results.json
├── results_with_ground_truth.json
└── turn_001/
    ├── prompt.md
    ├── last_message.json
    ├── result.json
    ├── codex_stdout.jsonl
    └── codex_stderr.txt
```

The Codex CLI execution directory (`-C`) is the task run directory itself:
`scripts/results/<domain>/<dataset>/<task_id>/<run_name>/`. Codex should put temporary analysis
files in `workspace/` and read benchmark files from `data/`.

The runner first copies only that task's released `data/` directory into the task run directory.
Codex is not given the original `DSGym/data/task/...` path that contains `task.json`, `task.py`,
`task.ipynb`, metadata, and gold answers.

During a task, `results.json` does not include ground truth. After the task finishes, the runner
writes `results_with_ground_truth.json` and `task_metadata_with_sources.json` for offline scoring
and debugging. The run-level `summary.json` is saved in each task's run directory.
