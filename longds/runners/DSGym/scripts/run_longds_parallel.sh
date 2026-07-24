#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-}"
PARALLEL="${2:-8}"

if [[ -z "$MODEL" ]]; then
  echo "Usage: bash run_longds_parallel.sh openai/<model_name> [parallel]"
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TASK_LIST="../../../dataset/task/longds/task_list.json"
TOTAL_TASKS="$(python3 - "$TASK_LIST" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    print(len(json.load(f)))
PY
)"

TASKS_PER_SHARD=$(((TOTAL_TASKS + PARALLEL - 1) / PARALLEL))
RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="./logs/longds_parallel_${RUN_ID}"
mkdir -p "$LOG_DIR"

PID_FILE="$LOG_DIR/pids.txt"
: > "$PID_FILE"

echo "Model: $MODEL"
echo "Parallel: $PARALLEL"
echo "Total tasks: $TOTAL_TASKS"
echo "Tasks per shard: $TASKS_PER_SHARD"
echo "Log dir: $LOG_DIR"
echo

for ((i = 0; i < PARALLEL; i++)); do
  start=$((i * TASKS_PER_SHARD))
  remaining=$((TOTAL_TASKS - start))
  if [[ "$remaining" -le 0 ]]; then
    continue
  fi

  limit="$TASKS_PER_SHARD"
  if [[ "$remaining" -lt "$limit" ]]; then
    limit="$remaining"
  fi

  log_file="$LOG_DIR/shard_${i}.log"

  nohup uv run python longds.py \
    --dataset longds \
    --model "$MODEL" \
    --backend litellm \
    --output-dir ./results_fix \
    --start-index "$start" \
    --task-limit "$limit" \
    --judge-model deepseek-v4-pro-guan \
    > "$log_file" 2>&1 &

  pid=$!
  echo "$pid shard_${i} start=$start limit=$limit log=$log_file" >> "$PID_FILE"
  echo "Started shard_${i}: start=$start limit=$limit pid=$pid"
done

echo
echo "All shards started in background."
echo "PID file: $PID_FILE"
echo "Example log:"
echo "  tail -f $LOG_DIR/shard_0.log"
