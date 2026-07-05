# Reason Task Generation

Reason Task Generation is a pipeline for iteratively creating and refining
task-specific Skills for DABStep reasoning tasks. It runs model exploration,
organizes trajectories, generates Skills from those trajectories, and then
uses later exploration rounds to refine the Skills.

## Repository Layout

```text
generation/
|-- run_pipeline.py              # Pipeline entry point
|-- pipeline_config.yaml         # User-facing runtime configuration
|-- pipeline_config.py           # Configuration loader and defaults
|-- dataagent/DSGym/             # DABStep evaluation framework
|-- skill_manager/               # Skill generation and refinement logic
`-- verifier/                    # Trajectory organization and scoring utilities
```

## Configuration

Edit `pipeline_config.yaml` for normal runs. Relative paths are resolved from
the `generation/` directory.

Key settings:

| Section | Field | Description |
| --- | --- | --- |
| `task` | `name` | Dataset/task name. Defaults to `dabstep`. |
| `task` | `target_categories` | Categories to explore and generate Skills for. |
| `model` | `name` | Model name passed to trajectory generation. |
| `model` | `api_key` | API key fallback if `REASON_API_KEY` is not set. |
| `model` | `base_url` | API base URL fallback if `REASON_BASE_URL` is not set. |
| `model` | `manager_url` | Model manager endpoint used by evaluation. |
| `runtime` | `num_runs` | Number of exploration runs per setting. |
| `runtime` | `max_workers` | Parallel workers for DSGym exploration. |
| `runtime` | `max_concurrent_skill_tasks` | Concurrent Skill generation/refinement jobs. |
| `paths` | `results_run_name` | Output run name under `dataagent/DSGym/examples/results/`. |
| `paths` | `category_file` | Query category mapping file. |
| `paths` | `skill_save_dir` | Directory for saved Skill snapshots. |

API credentials can be provided through environment variables:

```bash
export REASON_API_KEY="your_api_key"
export REASON_BASE_URL="https://your-api-base-url"
```

Command-line values override both environment variables and YAML values:

```bash
python run_pipeline.py \
  --api-key "your_api_key" \
  --base-url "https://your-api-base-url" \
  --model "your-model-name"
```

## Quick Start

Preview the full pipeline without executing commands:

```bash
python run_pipeline.py --dry-run
```

Run the complete three-iteration pipeline:

```bash
python run_pipeline.py
```

## Pipeline Stages

The pipeline contains nine ordered steps:

| Step | Description |
| --- | --- |
| `iter1_explore` | Explore the task without Skills. |
| `iter1_organize` | Organize no-skill trajectories. |
| `iter1_create_skill` | Generate initial Skills from organized trajectories. |
| `iter2_explore` | Explore with the generated Skills. |
| `iter2_organize` | Pair and organize iteration 1 vs. iteration 2 trajectories. |
| `iter2_modify_skill` | Refine Skills from the first comparison. |
| `iter3_explore` | Explore with refined Skills. |
| `iter3_organize` | Pair and organize iteration 2 vs. iteration 3 trajectories. |
| `iter3_modify_skill` | Produce the final refined Skills. |

Resume or limit execution with `--start-from` and `--stop-after`:

```bash
# Resume from the second exploration round.
python run_pipeline.py --start-from iter2_explore

# Run only the first iteration.
python run_pipeline.py --stop-after iter1_create_skill

# Run only the second iteration.
python run_pipeline.py \
  --start-from iter2_explore \
  --stop-after iter2_modify_skill
```

## Outputs

The pipeline writes runtime outputs to these locations by default:

| Output | Path |
| --- | --- |
| DSGym exploration results | `dataagent/DSGym/examples/results/<results_run_name>/` |
| Organized trajectories | `verifier/dabstep/` |
| Current working Skills | `skill_manager/workspace/current_skills/` |
| Skill snapshots for each run | `skill_manager/skills/run_<MMDDHH>/` |
| Skill-manager copied data | `skill_manager/workspace/data/` |

When the pipeline starts from `iter1_explore`, it resets the Skill-manager
workspace directories before running. When `--start-from` is set to a later
step, the workspace reset is skipped so existing intermediate outputs can be
reused.