# Running LongDS with Kimi Code

This runner executes LongDS directly with Kimi Code CLI. It follows the Claude Code runner's task lifecycle and result format: one isolated Docker container per task, one persistent agent session across all turns in that task, task-level parallelism, optional per-task judging, and the same `results_eval.json` format.

## Configure Kimi Code

The runner reads `config.toml` from this directory by default. Start from the example:

```bash
cp runners/kimi_code/config.example.toml runners/kimi_code/config.toml
```

Set `providers.bailian.api_key` in `config.toml`. The included provider uses Alibaba Bailian's OpenAI Chat Completions compatible endpoint because Kimi Code's `openai` provider preserves K3 reasoning and tool-call state correctly.

The model alias is `bailian/kimi-k3`; the actual API model is `kimi-k3`. To use another provider or model, edit `providers` and `models` according to Kimi Code's official configuration format, or pass another file with `--kimi-config`.

`config.toml` is ignored by git. The runner copies it into the task's private Kimi home before execution and removes it from saved results after the task, so the API key is not written to metadata, command arguments, or result JSON.

## Build Docker Image

Build the LongDS executor base image, then the thin Kimi Code image:

```bash
cd /mnt/40t/xkw/LongMemDA/DataMind/longds

docker build \
  -t executor-prebuilt \
  runners/DSGym/executors/container_images/longds_image

docker build \
  -t longds-kimi-code:latest \
  --build-arg BASE_IMAGE=executor-prebuilt \
  --build-arg KIMI_CODE_VERSION=latest \
  runners/kimi_code
```

The second image adds Node.js 22 and `@moonshot-ai/kimi-code`; the Python data-analysis environment still comes from `executor-prebuilt`.

## Run

Smoke test one turn:

```bash
python runners/kimi_code/run_kimi_longds.py \
  --use-docker \
  --task-limit 1 \
  --turn-limit 1
```

Run all tasks with four task workers and judge each completed task:

```bash
python runners/kimi_code/run_kimi_longds.py \
  --use-docker \
  --run-parallel 4 \
  --timeout 7200 \
  --judge
```

The runner defaults to all remaining tasks. `--task-limit N` limits the selected slice after `--start-index`. Turns within a task always remain sequential.

If a task already has the same run directory, it is skipped. Use `--overwrite` to remove that directory and rerun it:

```bash
python runners/kimi_code/run_kimi_longds.py \
  --use-docker \
  --run-name kimi_code_bailian_kimi-k3_experiment1 \
  --overwrite
```

## Kimi Invocation

The first turn starts a new Kimi session:

```text
kimi -p '<turn prompt>' --output-format stream-json
```

Later turns restore the session announced by Kimi's `session.resume_hint` event:

```text
kimi -p '<turn prompt>' --output-format stream-json --session <session_id>
```

In Docker mode this command is wrapped by:

```text
docker exec -i --user <uid:gid> --workdir /workspace <task-container> ...
```

Kimi's prompt mode always uses its non-interactive `auto` permission policy, so the runner does not add `--auto` or `--yolo`; Kimi rejects either flag when combined with `--prompt`.

Kimi Code currently has no JSON Schema output flag. The runner appends a strict JSON contract to each turn and validates the final Assistant message locally. If the final message is not valid contract JSON, the runner keeps the complete final Assistant text as `answer` and uses `"none"` / `["none"]` for `reasoning_summary` / `files_used` instead of failing the turn.

## Task Isolation

Docker mode starts one named container per LongDS task and reuses it for every turn in that task. Parallel tasks therefore have separate workspaces, Kimi homes, and session IDs. The container receives only the current task's `data/` directory under `/workspace/data`.

Kimi stores session state under `/tmp/longds_kimi_home` inside the container. At task completion the runner copies that session state to `kimi_home/`, removes `config.toml`, copies `/workspace` to the result directory, and removes the container. `--keep-docker-container` keeps it for debugging.

## Outputs

Each task writes:

```text
results/<domain>/<dataset>/<task_id>/<run_name>/
├── workspace/
├── kimi_home/                 # session trace, config.toml removed
├── kimi_turn.schema.json
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
        ├── kimi_stdout.jsonl
        └── kimi_stderr.txt
```

When `--judge` is enabled, `judge.py` also writes `results_eval.json` using the same LongDS judge contract as the Claude Code and Codex runners.

## Useful Options

```text
--kimi-config PATH           Kimi config.toml; defaults to this directory's config.toml.
--kimi-model ALIAS           Explicit Kimi model alias passed with --model.
--kimi-bin PATH              Kimi executable; defaults to kimi.
--kimi-arg ARG               Extra Kimi CLI argument; repeatable.
--use-docker                 Run one isolated container per task.
--run-parallel N             Number of tasks to run concurrently.
--task-limit N               Number of tasks after --start-index.
--turn-limit N               Maximum turns per task.
--timeout SECONDS            Wall-clock timeout per Kimi turn.
--judge                      Judge each successful task immediately.
--keep-data                  Preserve copied input data in results.
--keep-docker-container      Keep task containers after completion or failure.
```
