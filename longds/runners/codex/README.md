# Run LongDS with Codex

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

## 2. Configure Codex

Create your local configuration without overwriting an existing file:

```bash
cp -n runners/codex/config.example.toml runners/codex/config.toml
```

Edit `runners/codex/config.toml`:

- Set `model` and `model_reasoning_effort` to values supported by your endpoint.
- Set `model_providers.longds_env.base_url` and `experimental_bearer_token`.
- Keep `model_provider = "longds_env"` and `wire_api = "responses"`.

The endpoint must support the Responses API. API-key mode does not require host
Codex login. For ChatGPT login, see the optional section below.

The local configuration is Git-ignored. Keep credentials out of commits and
shared result files.

## 3. Build the Docker images

```bash
docker build -t executor-prebuilt runners/DSGym/executors/container_images/longds_image
docker build -t longds-codex:latest runners/codex
```

Skip the first build if `executor-prebuilt` is already available.
No DSGym executor pool is needed.

## 4. Configure the judge and check one turn

```bash
export JUDGE_API_KEY="<your_judge_api_key>"
export JUDGE_BASE_URL="<your_judge_base_url>"

python runners/codex/run_codex_longds.py \
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
python runners/codex/run_codex_longds.py \
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

## Optional: ChatGPT login

Install the host CLI with Node.js 22 and sign in:

```bash
npm install -g @openai/codex
codex login
cp -n runners/codex/config_oauth.example.toml runners/codex/config_oauth.toml
```

Choose the model in `config_oauth.toml`, then run:

```bash
python runners/codex/run_codex_longds.py \
  --use-docker \
  --codex-config runners/codex/config_oauth.toml \
  --codex-auth ~/.codex/auth.json \
  --run-parallel 4 \
  --judge
```

Use this OpenAI-provider configuration for login credentials, not the third-party
API-key template.

## Optional: run without Docker

Install Node.js 22, then install the CLI and analysis dependencies in your
active Python environment:

```bash
npm install -g @openai/codex
conda activate longds
python -m pip install -r runners/codex/requirements-environment.txt
codex --version
```

Use the same configuration and run commands above, but omit `--use-docker`.
Local mode gives the agent access to the host environment; use Docker when
you need task isolation.

## Common options

| Option | Usage / default |
| --- | --- |
| `--codex-model NAME` | Override the model in the configuration for this run. |
| `--reasoning-effort LEVEL` | Override reasoning effort; use a level supported by your model. |
| `--use-docker` | Run each task in Docker. Omit for local execution. |
| `--judge` | Automatically score each completed task. Requires judge credentials. |
| `--run-parallel N` | Concurrent tasks. Default: `1`. |
| `--timeout SECONDS` | Time limit per turn. Default: `3600`. |
| `--task-limit N` | Run at most N tasks. Default: all selected tasks. |
| `--turn-limit N` | Run at most N turns per task. Default: all turns. |
| `--start-index N` | Start at index N in the task list, counting from `0`. |
| `--longds_version VERSION` | Task version: `v1.1` (default) or `v1`. Use `--split full` with v1. |
| `--split SPLIT` | `lite` (default) or `full`. |
| `--codex-config PATH` | Use another configuration. Default: `runners/codex/config.toml`. |
| `--output-dir PATH` | Output base directory. Default: `./results`; version/split and run name are appended automatically. |
| `--run-name NAME` | Set an experiment name instead of generating one. Use a unique name for each experiment. |

For the complete option list:

```bash
python runners/codex/run_codex_longds.py --help
```
