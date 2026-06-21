# longds-bench

[English](README.md) | [简体中文](README_cn.md)

A portable, **agent-agnostic** skill for running **LongDS-Bench** ([zjunlp/DataMind](https://github.com/zjunlp/DataMind/tree/main/longds)) — the long-horizon, multi-turn agentic data-analysis benchmark — **with the agent itself as the runtime under test**.

Unlike the official harness, this skill does **not** use DSGym's ~12 GB Docker executor and does **not** drive a model through LiteLLM. Instead, the agent that loads this skill *is* the thing being measured: it reads the dataset, does the multi-turn analysis with its own tools and a persistent Python session, answers each turn, and is then scored by the official LLM-judge rule. Any harness with a shell / code-execution tool ([AndaBot](https://github.com/ldclabs/anda-bot), or others) can use it.

> **Heads up:** the LongDS dataset is **~19.5 GB**. You download and prepare it **once** as the operator; the agent never downloads anything. Scores from this setup are *indicative*, **not** officially leaderboard-comparable (different execution environment + agent scaffold than the paper).

---

## What it measures

68 tasks / 2,225 turns across six domains (Business, Community, Education, Geoscience, Social Good, Sports). Each task is one continuous multi-turn conversation in which analytical state evolves (state inheritance, update, counterfactual perturbation, rollback, multi-state composition). Each turn is judged **0/1**; the mean over all turns is the accuracy.

Published reference (official runtime): best ≈ **48.45** (Gemini-3.1-Pro), GPT-5.4 **43.50**, Claude-4.6-Sonnet **41.56**.

## Example self-run: AndaBot + GPT-5.5

[AndaBot](https://github.com/ldclabs/anda-bot) completed a full LongDS-Bench answer run with GPT-5.5 in a little over **7 hours**. All **2,225 / 2,225** turns were judged, with **39.37%** overall accuracy:

| Domain | Accuracy | Judged turns |
| ------ | -------: | -----------: |
| Overall | **0.3937** | 2,225 |
| business | 0.5766 | 411 |
| community | 0.6821 | 475 |
| education | 0.8889 | 216 |
| geoscience | 0.1814 | 678 |
| social_good | 0.0000 | 336 |
| sports | 0.0000 | 109 |

This result is not an official leaderboard number, but it is a useful systems result: AndaBot can run the whole benchmark through this `SKILL.md`, maintain the answer-file contract across thousands of turns, and produce fully judgeable output. The sharp domain spread also shows what LongDS-Bench is good at exposing: real accuracy requires per-task computation discipline, not just schema-complete answers.

## What's in this skill

| File                         | Run by    | Purpose                                                                                                        |
| ---------------------------- | --------- | -------------------------------------------------------------------------------------------------------------- |
| `SKILL.md`                   | the agent | The rules + the turn-by-turn loop the agent follows on itself.                                                 |
| `scripts/prepare_dataset.py` | operator  | Splits a downloaded dataset into an answer-stripped agent **manifest** + held-out **gold**. Does not download. |
| `scripts/pysession.py`       | the agent | One persistent IPython session per task (continuous state across steps & turns).                               |
| `scripts/judge.py`           | operator  | Official binary LLM-judge (verbatim `JUDGE_PROMPT`); reports accuracy overall + per domain.                    |

---

## Setup (operator, once)

Clone this repository and pick a working area. The repository directory itself is
`SKILL_DIR`; you do not need to install or register it as a named skill.

```bash
mkdir -p "$HOME/github"
git clone https://github.com/ldclabs/longds-bench.git "$HOME/github/longds-bench"

export SKILL_DIR="$HOME/github/longds-bench"          # directory containing SKILL.md + scripts/
export WORK="$HOME/longds-bench-work"
export STAGING="$WORK/dataset"                        # raw download (~19.5 GB)
export LONGDS_RUN="$WORK/run"                          # prepared workspace the agent uses
export LONGDS_VENV="$WORK/venv"
mkdir -p "$WORK"
```

### 1. Python env (replaces DSGym's Docker image)

```bash
uv venv "$LONGDS_VENV" --python 3.12
. "$LONGDS_VENV/bin/activate"
uv pip install ipykernel jupyter_client openai huggingface_hub \
  pandas numpy scipy scikit-learn statsmodels python-dateutil pyarrow openpyxl
# Some tasks import more libraries; install them on demand when a turn raises ModuleNotFoundError.
```

### 2. Download the dataset (you do this — the agent never does)

**Full benchmark (~19.5 GB):**

```bash
hf download zjunlp/LongDS --repo-type dataset --local-dir "$STAGING"
```

**Or a subset for a quick try** (one domain — `*` matches across `/`):

```bash
hf download zjunlp/LongDS --repo-type dataset --local-dir "$STAGING" \
  --include "task/longds/task_list.json" "task/longds/business/*" "data/longds/business/*"
```

### 3. Prepare = strip answers + build the agent workspace

`task.json` carries the reference `answer` and gold `code` inline. This step extracts them into a **held-out gold** file and gives the agent an answer-free **manifest**.

```bash
# Smoke-test slice first:
python "$SKILL_DIR/scripts/prepare_dataset.py" \
  --dataset-root "$STAGING" --out-dir "$LONGDS_RUN" \
  --task-limit 1 --turn-limit 3

# Full (optionally per-domain). --strip-source also deletes task.json/task.py/task.ipynb
# from the download tree so the only copy of answers is $LONGDS_RUN/gold:
python "$SKILL_DIR/scripts/prepare_dataset.py" \
  --dataset-root "$STAGING" --out-dir "$LONGDS_RUN" --strip-source

cat "$LONGDS_RUN/index.json"   # check: tasks, turn counts, and data_dir_exists == true
```

This produces:

```
$LONGDS_RUN/
├── index.json          # task list
├── manifest/<key>.json  # AGENT reads: turn_id / context / question / data_dir  (NO answers)
├── gold/<key>.json      # JUDGE only: question + ground_truth  (held out)
└── answers/             # the agent WRITES its answers here, one <key>.json per task
```

---

## Run the agent on it

Tell your agent to read and follow `SKILL.md` from this repository, pointing it
at the prepared workspace. No `longds-bench` skill installation is required.
For example:

> `SKILL_DIR=/path/to/longds-bench`, `LONGDS_RUN=/path/to/run`, `LONGDS_VENV=/path/to/venv`. Please follow `/path/to/longds-bench/SKILL.md` to run the LongDS-Bench tasks on yourself. Start with the one-task pilot, show me the score, then ask before the full run.

If you only need the agent to answer tasks and do not want it to score the run
immediately, say that explicitly:

> `SKILL_DIR=/path/to/longds-bench`, `LONGDS_RUN=/path/to/run`, `LONGDS_VENV=/path/to/venv`. Please follow `/path/to/longds-bench/SKILL.md` to run the test tasks with your highest reasoning effort. Only answer the tasks; do not evaluate the score. To save time, run 3-5 tasks concurrently.

The agent then follows `SKILL.md`: for each task it opens a persistent Python session at the task's `data_dir`, solves each turn in order with continuous code execution (no plotting; exact numeric answers), and appends its final answer per turn to `answers/<key>.json`. **Integrity:** the agent must read only `manifest/`, never `gold/` or the raw `task.json`.

For a full run, expect thousands of reasoning steps over hours — prefer a harness that can run it in the background and one worker/subagent per task.

## Score it

Use a judge endpoint (OpenAI-compatible). **Use a judge model different from the agent's model** to avoid self-judging bias.

```bash
. "$LONGDS_VENV/bin/activate"
export JUDGE_API_KEY="<key>"; export JUDGE_BASE_URL="https://api.deepseek.com"
python "$SKILL_DIR/scripts/judge.py" \
  --answers "$LONGDS_RUN/answers" --gold "$LONGDS_RUN/gold" \
  --out "$LONGDS_RUN/results_eval.json" --judge-model "deepseek-v4-pro" --max-workers 8
```

Output (`results_eval.json` + console): overall accuracy, per-domain accuracy, per-task accuracy, and the per-turn scores/reasoning. Turns with an empty answer score 0; turns the judge can't parse after 3 retries are excluded and reported as unjudged.

---

## Caveats

- **Indicative, not official.** Local venv ≠ DSGym's pinned Docker image; the agent's own loop ≠ the benchmark's fixed ReAct scaffold. Report the number as the agent's ability, not a leaderboard score.
- **Judge bias.** Self-judging inflates scores; prefer a separate judge model and say which you used.
- **Cost/time.** Full run = thousands of paid steps + 2,225 judge calls. Always pilot first.

## Credits & licensing

- **This repository's own code** (the scripts and docs authored here) — MIT, © 2026 LDC Labs (see [LICENSE](LICENSE)).
- **LongDS-Bench** (benchmark, multi-turn task protocol, and `JUDGE_PROMPT`) — by Xu et al. (2026), [zjunlp/DataMind](https://github.com/zjunlp/DataMind), **Apache-2.0**. `scripts/judge.py` reproduces their `JUDGE_PROMPT` **verbatim** and `SKILL.md` follows their task protocol so scores match the official rule. That material stays © its authors under Apache-2.0 — see [NOTICE](NOTICE).
- **DSGym** — [fannie1208/DSGym](https://github.com/fannie1208/DSGym) ([paper](https://arxiv.org/abs/2601.16344)). Referenced in design only; not vendored. This adapter intentionally does **not** use DSGym's Docker runtime.
- **LongDS dataset** — [zjunlp/LongDS](https://huggingface.co/datasets/zjunlp/LongDS), license **`other`**. Not redistributed here; you download it from Hugging Face and must comply with its terms.

If you use LongDS-Bench, please cite the authors' paper (BibTeX in [NOTICE](NOTICE)).
