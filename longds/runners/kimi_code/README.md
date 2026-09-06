# Run LongDS with Kimi Code

Run all commands from the `longds/` root. The steps below use Docker and
evaluate **v1.1 Lite (24 tasks / 777 turns)** by default.
First [download the dataset](../../README.md#1-download-the-dataset).

## 1. Prepare the environment

You need Docker installed and running. Create a Python 3.12 environment,
or activate an existing one:

```bash
cd /path/to/DataMind/longds
conda create -n longds python=3.12 -y
conda activate longds
python -m pip install openai
```

Docker supplies the agent CLI and data-analysis packages; you do not need to
install them on the host for this workflow.

## 2. Configure Kimi Code

Create your local configuration without overwriting an existing file:

```bash
cp -n runners/kimi_code/config.example.toml runners/kimi_code/config.toml
```

Edit `runners/kimi_code/config.toml`:

- Set `providers.bailian.base_url` and `providers.bailian.api_key` to your
  OpenAI Chat Completions-compatible endpoint and key.
- Set `models."bailian/kimi-k3".model` to your endpoint's model name.
- Adjust context size, capabilities, and thinking settings to match that model.

`bailian/kimi-k3` is the template's local model alias, selected by
`default_model`. If you rename the alias or provider, update their references
in the same file.

The local configuration is Git-ignored. Keep credentials out of commits and
shared result files.

## 3. Build the Docker images

```bash
docker build -t executor-prebuilt runners/DSGym/executors/container_images/longds_image
docker build -t longds-kimi-code:latest runners/kimi_code
```

Skip the first build if `executor-prebuilt` is already available.
No DSGym executor pool is needed.

## 4. Configure the judge and check one turn

```bash
export JUDGE_API_KEY="<your_judge_api_key>"
export JUDGE_BASE_URL="<your_judge_base_url>"

python runners/kimi_code/run_kimi_longds.py \
  --use-docker \
  --task-limit 1 \
  --turn-limit 1 \
  --judge
```

The judge endpoint must support Chat Completions. Its default model is
`deepseek-v4-pro`; set `JUDGE_MODEL` to use another name.
This is a real, billable model and judge call. Check the printed
`summary.json` path for execution or judge errors before continuing.

## 5. Run all of v1.1 Lite

```bash
python runners/kimi_code/run_kimi_longds.py \
  --use-docker \
  --run-parallel 4 \
  --judge
```

Each invocation gets an automatic run name. Reduce `--run-parallel` if needed.
The default timeout is one hour **per turn**; change it with `--timeout SECONDS`.

For v1.1 Full, add `--split full`. For v1, add
`--longds_version v1 --split full`.

## 6. View results

Open the run directory printed by the launcher:

```text
results/longds_v1.1_lite/<run_name>/
├── summary.json
└── <domain>/<dataset>/taskN/
    ├── results.json
    ├── results_eval.json
    ├── detail/
    └── workspace/
```

`summary.json` contains completion counts and `task_avg_score`.
Task folders contain answers, judge scores, and execution traces.
No separate scoring command is needed when running with `--judge`.

## Optional: run without Docker

Install Node.js 22, then install the CLI and analysis dependencies in your
active Python environment:

```bash
npm install -g @moonshot-ai/kimi-code
conda activate longds
python -m pip install -r runners/kimi_code/requirements-environment.txt
kimi --version
```

Use the same configuration and run commands above, but omit `--use-docker`.
Local mode gives the agent access to the host environment; use Docker when
you need task isolation.

## Common options

Set thinking effort in `config.toml`; this runner has no dedicated effort flag.

| Option | Usage / default |
| --- | --- |
| `--kimi-model NAME` | Select a model alias defined in the Kimi configuration for this run. |
| `--use-docker` | Run each task in Docker. Omit for local execution. |
| `--judge` | Automatically score each completed task. Requires judge credentials. |
| `--run-parallel N` | Concurrent tasks. Default: `1`. |
| `--timeout SECONDS` | Time limit per turn. Default: `3600`. |
| `--task-limit N` | Run at most N tasks. Default: all selected tasks. |
| `--turn-limit N` | Run at most N turns per task. Default: all turns. |
| `--start-index N` | Start at index N in the task list, counting from `0`. |
| `--longds_version VERSION` | Task version: `v1.1` (default) or `v1`. Use `--split full` with v1. |
| `--split SPLIT` | `lite` (default) or `full`. |
| `--kimi-config PATH` | Use another configuration. Default: `runners/kimi_code/config.toml`. |
| `--output-dir PATH` | Output base directory. Default: `./results`; version/split and run name are appended automatically. |
| `--run-name NAME` | Set an experiment name instead of generating one. Use a unique name for each experiment. |

For the complete option list:

```bash
python runners/kimi_code/run_kimi_longds.py --help
```
