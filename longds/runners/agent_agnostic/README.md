# Agent-Agnostic Runner Guide

This directory integrates a portable [agent-agnostic skill](https://github.com/ldclabs/longds-bench/blob/main/README.md) into LongDS-Bench. LongDS-Bench evaluates long-horizon, multi-turn agentic data analysis; in this execution mode, **the agent itself is the runtime under test**.

The [LDC Labs team](https://github.com/ldclabs) developed the [longds-bench skill](https://github.com/ldclabs/longds-bench) independently of any specific agent framework. Its `SKILL.md` defines the multi-turn workflow and data-access constraints, while its scripts provide dataset preparation, persistent Python sessions, and LLM-based judging. Any agent with shell or code-execution capabilities can follow the skill to run LongDS-Bench.

Unlike the DSGym runner, this method does not use a Docker executor or invoke a model externally through LiteLLM. The agent reads `longds_bench/SKILL.md` directly, uses its own shell or code-execution tools and a persistent Python session, and completes all turns of each task in order.

## Components

| Path | Used by | Purpose |
| --- | --- | --- |
| `longds_bench/SKILL.md` | Agent | Defines data access, multi-turn state, and answer-writing rules. |
| `longds_bench/scripts/prepare_dataset.py` | Operator | Produces an answer-free `manifest/` and judge-only `gold/`. |
| `longds_bench/scripts/pysession.py` | Agent | Maintains a persistent IPython session across turns for each task. |
| `longds_bench/scripts/judge.py` | Operator | Applies the LongDS-Bench Judge Prompt to score each turn as 0 or 1. |

## 1. Create the Environment and Configure Paths

### Create the Environment

On first use, create a Python 3.12 Conda environment:

```bash
cd DataMind/longds/runners/agent_agnostic
conda create -n longds python=3.12 -y
conda activate longds
```

Install the LongDS Docker executor's Python dependencies plus the `openai` package required by the judge:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements-environment.txt
```

### Configure Paths

Set the runtime paths in the current shell:

```bash
cd DataMind/longds/runners/agent_agnostic
conda activate longds

export SKILL_DIR="$PWD/longds_bench"
export RUN="$PWD/results/example"
export VENV="$CONDA_PREFIX"

mkdir -p "$RUN"
```

`$VENV/bin/python` must point to the Python interpreter in the `longds` Conda environment:

```bash
"$VENV/bin/python" -c 'import sys; print(sys.executable)'
```

## 2. Prepare Pilot Data

Prepare three tasks with at most three turns per task:

```bash
python "$SKILL_DIR/scripts/prepare_dataset.py" \
  --dataset-root "$PWD/../../dataset" \
  --out-dir "$RUN" \
  --task-limit 3 \
  --turn-limit 3
```

Confirm that `$RUN/index.json` exists and that `data_dir_exists` is `true`.

**Important:** `--strip-source` modifies the shared dataset. Do not use this option with the shared `dataset/` directory.

## 3. Run the Agent

You do not need to install or register `longds_bench` as a named skill. Launch the agent under test from the same shell and instruct it to read `$SKILL_DIR/SKILL.md` directly.

To run a pilot and score it immediately:

```text
Use SKILL_DIR, RUN, and VENV from the environment. Read and follow $SKILL_DIR/SKILL.md to run the prepared LongDS tasks on yourself. Start with the one-task pilot, show me the score, then ask before the full run.
```

To answer tasks without scoring immediately:

```text
Use SKILL_DIR, RUN, and VENV from the environment. Read and follow $SKILL_DIR/SKILL.md to run the prepared LongDS tasks with your highest reasoning effort. Only answer the tasks; do not evaluate the score. For a full run, process 3-5 tasks concurrently when resources permit.
```

The agent must be launched from the same shell in which these variables were exported. Following `SKILL.md`, it starts one persistent Python session per task under `$RUN/workspace/<key>/` and processes turns strictly in order. The original `data_dir` is treated as read-only input; scripts, caches, and intermediate files are written to the task workspace. Final answers are written to:

```text
$RUN/answers/<domain>__<dataset>__<task_id>.json
```

While solving tasks, the agent may read only `manifest/` and the referenced data directory. It must not read `gold/` or the original `task.json`. A full run may take several hours. Prefer background execution, run tasks concurrently only when appropriate, and never process multiple turns of the same task concurrently.

### 3.1 Manual Scoring

If the agent only produced answers, run the judge afterward as the operator:

```bash
export JUDGE_API_KEY="<judge-api-key>"
export JUDGE_BASE_URL="<judge-base-url>"

python "$SKILL_DIR/scripts/judge.py" \
  --answers "$RUN/answers" \
  --gold "$RUN/gold" \
  --out "$RUN/results_eval.json" \
  --judge-model deepseek-v4-pro \
  --max-workers 8
```

Scores are saved to `$RUN/results_eval.json`. After the pilot succeeds, use a new `RUN` directory and prepare the full benchmark without `--task-limit` or `--turn-limit`.

## 4. Example Results: AndaBot + GPT-5.5

The [LDC Labs team](https://github.com/ldclabs) reported a full AndaBot + GPT-5.5 run in [longds-bench](https://github.com/ldclabs/longds-bench). The run took approximately seven hours, completed and scored all 2,225 turns, and achieved **39.37%** overall accuracy.

| Domain | Accuracy | Judged turns |
| --- | ---: | ---: |
| Overall | **0.3937** | 2,225 |
| business | 0.5766 | 411 |
| community | 0.6821 | 475 |
| education | 0.8889 | 216 |
| geoscience | 0.1814 | 678 |
| social_good | 0.0000 | 336 |
| sports | 0.0000 | 109 |

This result demonstrates that the agent-agnostic method can complete the full multi-turn benchmark and produce judgeable outputs. Because the runtime environment and agent scaffold differ, the score should not be compared directly with DSGym runner results. See the [longds-bench project documentation](https://github.com/ldclabs/longds-bench#example-self-run-andabot--gpt-55) for details.

## 5. Runtime Limitations and Result Comparability

Under the execution setup described in the LongDS paper and the standard DSGym workflow, an agent must not see future turns while processing the current turn. This method primarily relies on `SKILL.md` and prompt instructions rather than complete enforcement at the execution-environment level. It therefore has the following differences and risks:

1. Each task's `manifest` JSON contains all of its turns. Although the skill instructs the agent to process turns in order and not read ahead, the agent may still access future turns.
2. Reference answers are stored under `$RUN/gold/`. Access restrictions on `gold/` also rely on the skill and prompt. If the agent has permission to read these files, label leakage remains possible.
3. The shared dataset at `DataMind/longds/dataset/` is intended to be read-only input. If the agent's tools have write access, they may accidentally modify source files; without a backup, these changes cannot be recovered automatically.

We therefore treat this as an additional LongDS-Bench execution method. Because its visibility and isolation guarantees differ from the strict setup described in the paper, its results should not be compared directly with paper or DSGym runner results.

For a specific agent framework, a more reliable approach is to construct agent input dynamically at the harness layer. Pass only the current turn's `context` and `question`, prepend the outputs from previous turns, and do not expose the complete task to the agent.

## Acknowledgments

We thank [LDC Labs](https://github.com/ldclabs) for open-sourcing the agent-agnostic [longds-bench](https://github.com/ldclabs/longds-bench) skill. This directory integrates and adapts that project; its original code and associated copyright notices remain under the MIT License.
