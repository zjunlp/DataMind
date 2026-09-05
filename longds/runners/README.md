# LongDS runner task selection

All runner entry points use `src/longds_dataset.py` to resolve task versions and
task subsets. The default is `--longds_version v1.1 --split lite`.

| Selection | Task list | Tasks |
| --- | --- | --- |
| `--longds_version v1 --split full` | `dataset/task/longds_v1/task_list_full.json` | 68 |
| `--longds_version v1.1 --split full` | `dataset/task/longds_v1.1/task_list_full.json` | 68 |
| `--longds_version v1.1 --split lite` | `dataset/task/longds_v1.1/task_list_lite.json` | 24 |

All versions share `dataset/data/longds/{domain}/{dataset}/{task_id}/data/`.
Lite selects complete tasks from full and reads the same `task.json` files.
Missing versions or splits raise an error; there is no fallback to another list.
Selection order is version/split, then `--start-index`, then `--task-limit`,
then `--turn-limit` within each task. Agent-agnostic preparation applies its
optional domain filter before the index and task limits.

From the repository root, with the runner's model configuration already set up:

```bash
python runners/codex/run_codex_longds.py --longds_version v1.1 --split lite
python runners/claude_code/run_claude_longds.py --longds_version v1.1 --split full
python runners/kimi_code/run_kimi_longds.py --longds_version v1 --split full
python runners/qoder_cli/run_qoder_longds.py --longds_version v1.1 --split lite
```

Add `--task-limit 1 --turn-limit 1 --dry-run` to a CLI runner to check task
selection and prompts without invoking an agent or copying input data.

For a shared launch directory, run `cd runners` and then
`python codex/run_codex_longds.py`. With the default selection, this writes to
`runners/results/longds_v1.1_lite/` in the repository. Input data and model
configuration defaults remain relative to the repository/script locations.

## Explicit path overrides

`--task-root`, `--task-list-name`, and `--data-root` override the corresponding
derived paths independently. For example:

```bash
python runners/codex/run_codex_longds.py \
  --task-root dataset/task/longds_v1 \
  --task-list-name task_list_full.json \
  --data-root dataset/data/longds
```

The effective version is inferred from an explicit `longds_<version>` directory;
other directory names use `custom`. An explicit `task_list_full.json` or
`task_list_lite.json` determines the effective split. Other list filenames use
their filename stem as the split label. The resolved paths are saved in metadata.

## Results and scoring

The four CLI runners and DSGym default to `results/` under the current working
directory. `--output-dir` overrides this base; relative paths are resolved from
the current working directory. All runners share a dataset version/split directory:

```text
<output-dir>/longds_<version>_<split>/<run_name>/<domain>/<dataset>/<task_id>/
```

Default run names include the runner, model, and timestamp. Explicit `--run-name`
values must be unique across runners within the same output group to avoid collisions.
Pass only the base to `--output-dir`; do not include the dataset group or run name.

One run directory contains all tasks from that invocation. Judges accept a task
leaf through `--run-dir`, or scan an entire run through `--results-root`.
The summarizer accepts either a selection group or one run directory. Both tools
continue to read legacy task-first layouts; existing results are not moved.

Each CLI runner and DSGym writes `<run_name>/summary.json` when the invocation
finishes, including when individual tasks fail or the selection is empty. It records
the model, dataset selection, task/turn limits, execution counts, and per-task status.
`completed_tasks` means execution returned successfully in this invocation, not that
the answer was correct; skipped tasks and dry runs are counted separately.
`judge_problem_tasks` records failed, invalid, or incomplete evaluations separately.
`task_avg_score` is a 0–1 equal-weight average of fully judged task means, or `null`
when none are available. Unjudged/invalid tasks are not treated as zero; consult
`judged_tasks` alongside the score. Valid saved evaluations of skipped tasks are included.
Only tasks selected by this invocation are summarized. Reusing a run name replaces
the overview with the latest invocation's selection, without deleting other task results.
Separate later judge runs do not refresh this snapshot. Forced termination may prevent
the final summary from being written.

`task_metadata.json` records `longds_version`, `split`, `task_root`, `task_list`,
and `data_root`. Existing results are not moved. Summarize one selection at a time:

```bash
python runners/summarize_scores.py results/longds_v1.1_lite
```

The summarizer rejects roots containing multiple version/split selections to
avoid merging scores. When filtering results with `--subset` or
`--task-list-name`, use `--longds_version` (default `v1.1`) or `--task-root` to
choose the matching task definitions.

## DSGym

`runners/DSGym/scripts/longds.py` accepts the same selection flags.
`--dataset-path` remains an alias for `--task-root`.

```bash
cd runners/DSGym
uv run python scripts/longds.py --dataset longds --model openai/MODEL \
  --longds_version v1.1 --split lite
```

Executors still read shared data through a read-only Docker mount. Mount the
repository's `dataset/data` directory at `/data`; the default
`--executor-data-root /data/longds` is the path used in prompts. `--data-root`
identifies the host data source and does not reconfigure existing Docker mounts.
If the executor mount differs, set `--executor-data-root` to its container path.

## Agent-agnostic preparation

```bash
python runners/agent_agnostic/longds_bench/scripts/prepare_dataset.py \
  --dataset-root dataset --longds_version v1.1 --split lite \
  --out-dir runners/agent_agnostic/results/v1.1/lite/example
```

Preparation keeps `--out-dir` as the exact workspace path so existing manifest,
answer, and judge workflows continue to use that path. Choose a separate output
directory per selection. `dataset_metadata.json` records the selected sources.
Preparation does not copy the shared input data. Do not use `--strip-source`
with the shared repository dataset: that option deletes answer-bearing sources.
