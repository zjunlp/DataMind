# DABStep Evaluation Pipeline

Run DABStep task evaluation with generated skills, collect model predictions, and
validate the final results from one YAML-driven entrypoint.

## What This Pipeline Does

The pipeline is orchestrated by `run_eval.py` and runs three ordered steps:

| Step | Name | Purpose |
| --- | --- | --- |
| 1 | `eval_run` | Run model inference for each target DABStep category using the matching skill directory. |
| 2 | `collect_results` | Merge per-category outputs into one prediction JSONL file. |
| 3 | `validate` | Score predictions against the DABStep ground-truth file. |

You can run the full pipeline, resume from any step, stop after any step, or print
the commands without executing them.

## Directory Layout

```text
eval/
|-- run_eval.py                    # Main pipeline entrypoint
|-- eval.yaml                      # Default pipeline configuration
|-- DSGym/
|   |-- data/task/DABStep/         # DABStep task data and category metadata
|   `-- examples/                  # Evaluation and result collection scripts
|-- skill/                         # Generated skills grouped by category
|-- val/dabstep_benchmark/         # Validation script and scorer
`-- results/                       # Collected prediction JSONL outputs
```

## Quick Start

### 1. Configure Credentials

Prefer environment variables for credentials:

```bash
export REASON_API_KEY="your_api_key"
export REASON_BASE_URL="https://your-api-base-url/v1"
```

`run_eval.py` resolves credentials in this order:

1. command-line arguments: `--api-key`, `--base-url`
2. environment variables: `REASON_API_KEY`, `REASON_BASE_URL`
3. values in `eval.yaml`

### 2. Review `eval.yaml`

Use `eval.yaml` to control the run range, model, worker count, paths, and target
categories:

```yaml
run:
  dry_run: false

model:
  model: qwen3.5-397b-a17b
  api_key: "your_api_key_here"
  base_url: "your_base_url_here"
  manager_url: http://localhost:5005
  max_workers: 30

paths:
  results_dir: DSGym/examples/results/dabstep_test
  skill_dir: skill/0/iter2
  category_file: DSGym/data/task/DABStep/all_query_category.json
  all_jsonl: DSGym/data/task/DABStep/all.jsonl
  gt_file: DSGym/data/task/DABStep/test.jsonl
  eval_script_dir: DSGym/examples
  val_script_dir: val/dabstep_benchmark
  output_jsonl: results/dabstep_test.jsonl

pipeline:
  target_categories:
    - Applicable_Fee_IDs
    - Average_Fee_Estimation
    - Average_Transaction_Value_Stats
    - Dataset_Metadata_and_Business_Rules
    - Fee_Delta_and_Impact_Simulation
    - Fraud_and_General_Macro_Analysis
    - Highest_Cost_Scenario_Identification
    - Routing_and_Cost_Optimization
    - Total_Fees_Calculation
```

Relative paths are resolved from the directory containing the config file.

### 3. Run the Pipeline

```bash
python run_eval.py
```

Use another config file when needed:

```bash
python run_eval.py --config /path/to/eval.yaml
```

### 4. DABStep Details
We provide the DABStep `all.jsonl` file with answer annotations under the `DSGym/data/task/DABStep` path, and split it into `explore.jsonl` and `test.jsonl`. Since the official DABStep benchmark does not provide gold-labeled samples, and its official leaderboard evaluation is slow and does not support concurrent testing, we use the answers from the Genesis Data Agent as the gold-standard annotations for our testing, which achieves 100% accuracy on the unvalidated leaderboard. This agent shows a Pearson correlation coefficient of 0.9994 and a Spearman correlation coefficient of 0.9972 with the official tests, making it a reliable proxy. We will improve the reporting of DataCOPE's results on the official DABStep tests in the future.

## Common Commands

### Full Evaluation

```bash
python run_eval.py --start-from eval_run --stop-after validate
```

### Inference Only

```bash
python run_eval.py --stop-after eval_run
```

### Validate an Existing Prediction File

```bash
python run_eval.py \
  --start-from validate \
  --stop-after validate \
  --output-jsonl results/dabstep_test.jsonl
```

### Preview Commands Without Running

```bash
python run_eval.py --dry-run
```

## Configuration Reference

### `run`

| Key | Description |
| --- | --- |
| `start_from` | First step to run. Must be one of `eval_run`, `collect_results`, `validate`. |
| `stop_after` | Last step to run. Must be one of `eval_run`, `collect_results`, `validate`. |
| `dry_run` | Print commands without executing them. |

### `model`

| Key | Description |
| --- | --- |
| `model` | Model name passed to `DSGym/examples/evaluate.py`. |
| `api_key` | API key fallback. Prefer `REASON_API_KEY` or `--api-key` for secrets. |
| `base_url` | OpenAI-compatible API base URL fallback. Prefer `REASON_BASE_URL` or `--base-url`. |
| `manager_url` | Manager service URL used by the evaluation backend. |
| `max_workers` | Number of parallel workers for evaluation. |

### `paths`

| Key | Description |
| --- | --- |
| `results_dir` | Base directory for per-category evaluation outputs. |
| `skill_dir` | Base skill directory. Each category is expected under this directory. |
| `category_file` | JSON file mapping queries to DABStep categories. |
| `all_jsonl` | Full DABStep query JSONL used during result collection. |
| `gt_file` | Ground-truth JSONL used by validation. |
| `eval_script_dir` | Directory containing `evaluate.py` and `get_result_jsonl.py`. |
| `val_script_dir` | Directory containing `val.py`. |
| `output_jsonl` | Collected prediction file written before validation. |

### `pipeline`

`target_categories` controls which DABStep categories are evaluated and collected.
The default set is:

```text
Applicable_Fee_IDs
Average_Fee_Estimation
Average_Transaction_Value_Stats
Dataset_Metadata_and_Business_Rules
Fee_Delta_and_Impact_Simulation
Fraud_and_General_Macro_Analysis
Highest_Cost_Scenario_Identification
Routing_and_Cost_Optimization
Total_Fees_Calculation
```

## Outputs

During `eval_run`, each category writes under:

```text
DSGym/examples/results/{results_dir}/{results_dir}_<CATEGORY>/
```

During `collect_results`, the merged prediction JSONL is written to the configured
`output_jsonl`, for example:

```text
results/dabstep_test.jsonl
```

During `validate`, `val.py` reads `output_jsonl` and `gt_file`, then prints the score report to stdout.
