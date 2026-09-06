# Run LongDS with DSGym

Run all commands from the `longds/` root. You need Python 3.12, `uv`, Docker,
and Docker Compose. First [download the dataset](../../README.md#1-download-the-dataset).

## 1. Install dependencies and build images

```bash
cd /path/to/DataMind/longds
uv sync --project runners/DSGym
docker build -t executor-prebuilt runners/DSGym/executors/container_images/longds_image
docker build -t manager-prebuilt runners/DSGym/executors/manager
```

## 2. Start the executor pool

```bash
(
  cd runners/DSGym/executors
  uv run --project .. python generate_compose.py \
    -n 8 --types "executor-prebuilt:8" -m ../../../dataset/data
  docker compose -f docker-compose.yml up -d --build
)
```

Keep `--run-parallel` at or below the number of executors. If a proxy causes
manager-to-executor `502` errors, add Docker service names to `NO_PROXY`.

## 3. Configure model and judge access

For an OpenAI-compatible model endpoint:

```bash
export OPENAI_API_KEY="<your_model_api_key>"
export OPENAI_BASE_URL="<your_model_base_url>"
export JUDGE_API_KEY="<your_judge_api_key>"
export JUDGE_BASE_URL="<your_judge_base_url>"
```

The judge defaults to `deepseek-v4-pro`. Pass `--judge-model` to change it.

## 4. Run v1.1 Lite

Replace `<your_model_name>` with your endpoint's model name:

```bash
uv run --project runners/DSGym python runners/DSGym/scripts/longds.py \
  --dataset longds \
  --model "openai/<your_model_name>" \
  --backend litellm \
  --run-parallel 4
```

Judging runs automatically. For a small connectivity check, add
`--task-limit 1 --turn-limit 1`. To run v1.1 Full, add `--split full`;
for v1, add `--longds_version v1 --split full`.

Results are saved under `results/longds_v1.1_lite/<run_name>/`.
Open `summary.json` for the run overview and task-average score.

## 5. Stop the executor pool

```bash
docker compose -f runners/DSGym/executors/docker-compose.yml down
```

For more details, see the [executor guide](executors/README.md) and
[upstream DSGym documentation](README_DSGym.md).
