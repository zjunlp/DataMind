# Running LongDS with Qoder

Run commands from the `longds/` root; edit model configuration in `runners/qoder_cli/`.
Default configuration paths are resolved relative to the runner script, not the launch directory.

Results default to `./results/longds_<version>_<split>/` relative to the
current working directory. `--output-dir` overrides the base `./results`;
the dataset version/split group is appended automatically.

Task selection defaults to `--longds_version v1.1 --split lite`. Use `--split lite`
for the 24-task subset, or `--longds_version v1 --split full` for v1. All versions
share `dataset/data/longds`. See the [Quick start](../../README.md#quick-start)
for dataset download and standard run commands.

This directory contains a direct Qoder runner for LongDS-Bench. It supports a local conda
environment and a task-isolated Docker mode based on the LongDS executor image.

Each LongDS task becomes one Qoder CLI session: the first turn starts a headless session in the task
workspace, and every later turn resumes that same session so the agent keeps its context, helper
scripts, and intermediate artifacts.

## Files

- `run_qoder_longds.py`: runs LongDS tasks with `qoder --print --output-format stream-json` and resumes the session per turn.
- `settings.json`: runner-owned Qoder model and runtime defaults, passed through `qoder --settings`.
- `Dockerfile`: extends `executor-prebuilt` with Node.js and Qoder.
- `runners/src/judge.py`: scores Qoder CLI run outputs with the LongDS LLM judge.
- `prompt.py`: stores the first-turn prompt template, the JSON output contract, and turn prompt formatting.
- `requirements-environment.txt`: Python packages for the local LongDS analysis environment.

## Create the Conda Environment

Create and activate a Python 3.12 conda environment:

```bash
cd /mnt/40t/xkw/LongMemDA/DataMind/longds

conda create -n longds python=3.12 -y
conda activate longds
```

Install the environment packages from the LongDS root:

```bash
pip install --upgrade pip
pip install -r runners/qoder_cli/requirements-environment.txt
```

`requirements-environment.txt` matches the LongDS Docker executor Python package set and includes
`openai` for judge/API calls.

## Install and Authenticate Qoder CLI

Install the CLI (Node.js 20+ for the npm route):

```bash
curl -fsSL https://qoder.com/install | bash
# or
npm install -g @qoder-ai/qodercli

qoder --version
```

Authenticate once. For interactive use, start `qoder` and run `/login`. For unattended runs,
export a Personal Access Token from <https://qoder.com/account/integrations>:

```bash
export QODER_PERSONAL_ACCESS_TOKEN="<your_personal_access_token>"
```

The runner does not read or log the token; it only inherits the environment. When
`QODER_PERSONAL_ACCESS_TOKEN` is set it takes priority over credentials saved by `/login`.

Authentication is deliberately separate from `settings.json`. Qoder platform authentication comes
from `/login` state or `QODER_PERSONAL_ACCESS_TOKEN`; BYOK providers are configured with Qoder's
interactive `/model` flow rather than by writing API keys into this runner's settings file.

## Docker Mode

Build the image after `executor-prebuilt` is available:

```bash
cd /mnt/40t/xkw/LongMemDA/DataMind/longds
docker build -t longds-qoder:latest runners/qoder_cli
```

Run one task in Docker:

```bash
python runners/qoder_cli/run_qoder_longds.py \
  --use-docker \
  --qoder-model Qwen3.8-Max \
  --task-limit 1 \
  --judge
```

The runner creates one long-lived container per task. Every turn uses `docker exec` in that same
container and resumes the previous Qoder session ID, so task files and conversation state remain
available across turns. Parallel tasks receive deterministic, distinct container names. The
workspace is copied back to the result directory before the container is removed.

Docker authentication works in either of these ways:

- Export `QODER_PERSONAL_ACCESS_TOKEN`; the runner injects it when the task container starts.
- Use the existing host login under `~/.qoder`. The runner copies only `.auth`, `.models`,
  `settings.json`, `state.json`, and `installation_id` into the task-private config directory. It
  does not copy host caches, logs, old sessions, or projects, and it does not copy credentials back
  into the results.

The runner-owned `settings.json` is copied separately to `/tmp/longds_qoder_settings.json` and every
turn receives `--settings /tmp/longds_qoder_settings.json`. The host `~/.qoder` copy supplies login
state; it does not override the runner's explicit settings file.

Use `--qoder-config-dir` when the host login is stored somewhere other than `~/.qoder`; local mode
also forwards this path through `qoder --config-dir`. Standard
`HTTP_PROXY`, `HTTPS_PROXY`, and `NO_PROXY` variables are inherited automatically. Additional values
can be supplied with repeatable `--docker-env KEY_OR_KEY=VALUE` or `--docker-env-file PATH` options.

Useful Docker controls:

```text
--docker-image IMAGE          Default: longds-qoder:latest
--docker-network MODE         Default: host
--docker-memory 8g
--docker-cpus 4
--run-parallel N              One independent task container per worker
--keep-docker-container       Keep containers for interactive debugging
```

The outer Docker mode is the filesystem boundary. Leave Qoder's own `--sandbox` disabled in this
mode to avoid nesting a second container sandbox.

## Run a Qoder CLI Smoke Test

Preview the prompts and directory layout without calling the model:

```bash
python runners/qoder_cli/run_qoder_longds.py \
  --task-limit 1 \
  --turn-limit 1 \
  --dry-run
```

Then run one LongDS turn from the activated conda environment:

```bash
cd /mnt/40t/xkw/LongMemDA/DataMind/longds

python runners/qoder_cli/run_qoder_longds.py \
  --task-limit 1 \
  --turn-limit 1
```

`run_qoder_longds.py` passes the current Python executable to Qoder CLI as `--analysis-python`, so
when you run it from the activated `longds` conda environment, Qoder CLI is instructed to use that
conda Python for analysis code.

To be explicit:

```bash
python runners/qoder_cli/run_qoder_longds.py \
  --task-limit 1 \
  --turn-limit 1 \
  --analysis-python "$(python -c 'import sys; print(sys.executable)')"
```

By default, the runner passes [settings.json](settings.json) through `qoder --settings`. Its current
defaults are `Qwen3.8-Max`, `xhigh` reasoning effort, a 1,000,000-token context window, unlimited
session turns, and disabled CLI auto-update. Use `--qoder-settings PATH` to select another complete
settings file.

Explicit `--qoder-model`, `--reasoning-effort`, and `--context-window` arguments map to Qoder CLI
flags and take precedence over the settings file. Run `qoder --list-models` to see what your account
offers; model values can be aliases such as `Auto`, `Lite`, and `Performance`, a concrete name such
as `Qwen3.8-Max`, or a configured BYOK model ID:

```bash
python runners/qoder_cli/run_qoder_longds.py \
  --qoder-model Qwen3.8-Max \
  --reasoning-effort high \
  --task-limit 1
```

Pass a concrete model rather than `Auto` for benchmark runs. `Auto` lets Qoder pick, and the init
event then reports only `Auto`, so the model that actually answered is unrecoverable afterwards.

### Which Model Actually Ran

`model_slug` in `task_metadata.json` and the default run name use the effective model from either the
CLI override or `settings.json`. The runner also records what Qoder itself reported in its
`system/init` event, which is the authoritative runtime value:

| field | where | meaning |
| --- | --- | --- |
| `requested_model` | `task_metadata.json` | exactly what `--qoder-model` was given, or `null` when settings supplied it |
| `qoder_model` / `qoder_model_source` | `task_metadata.json` | effective model and whether it came from `cli`, `settings`, or Qoder's own default |
| `cli_model` | `task_metadata.json`, `results.json`, `detail/turn_*/result.json` | model the CLI reported for that session |
| `cli_model_by_turn` | `task_metadata.json` | per-turn map, so a mid-task change stays visible |
| `cli_version` | `task_metadata.json`, `detail/turn_*/result.json` | Qoder version that produced the run |

A task normally keeps one model across all its turns. If the turns disagree, `cli_model` is set to
`null`, `cli_model_by_turn` holds the details, and the runner prints a warning — those turns are not
comparable with each other.

## Permissions and Sandbox

LongDS turns must run shell commands and Python unattended, so the runner defaults to
`--permission-mode bypass_permissions`. This mode skips every permission check, so **the workspace is
not a filesystem boundary** — nothing stops the agent from reading or writing outside it, and only
the prompt rules in `prompt.py` keep it in place. That matches the Claude Code runner and leaves the
agent's behaviour unconstrained, which is the point when measuring a model on the benchmark.

Measured behaviour of each mode against Qoder 1.1.16, using a probe that tried three things —
run the conda Python, write a file inside the workspace, write a file one level above it:

| `--permission-mode` | external Python via Bash | write inside workspace | write outside workspace |
| --- | --- | --- | --- |
| `bypass_permissions` (default) | allowed | allowed | allowed |
| `auto` | allowed | allowed | **denied** |
| `dont_ask` | denied | denied | denied |

If you do want a write boundary, `auto` is the only mode that provides one while still running
unattended: it rejects an escape with `Error: Auto mode: operation outside workspace boundary is not
allowed`, and it still permits the external conda interpreter.

It is not the default because that boundary misfires. In a full LongDS turn, `auto` denied a
`cat > <file> << EOF` heredoc whose target was **inside** the workspace, reporting it as an
outside-boundary violation; `auto` appears to treat shell redirection as an escape regardless of the
path. The agent recovered by switching to the `Write` tool, but that turn took 888 s over 49 steps
against 156 s over 11 steps for the same task and turn under `bypass_permissions`. Those two runs
used different models and are a single sample each, so the slowdown is not attributable to the mode
alone, but the false denial and the retry it caused are real.

So choose `auto` when containment matters more than fidelity, and check
[Blocked Tool Calls](#blocked-tool-calls) afterwards to see what it cost. `dont_ask` denies anything
not pre-approved, including writes in the workspace, so it is unusable without an explicit rule set:

```bash
python runners/qoder_cli/run_qoder_longds.py \
  --permission-mode dont_ask \
  --allowed-tools 'Read,Grep,Glob,Bash(/path/to/conda/envs/longds/bin/python:*)'
```

`--sandbox` forwards Qoder CLI's sandbox switch, whose backend comes from `QODER_SANDBOX` and falls
back to auto-detecting Docker or Podman. It is off by default and should stay off: the container
backend mounts only the workspace, so the external conda interpreter would not exist inside the
sandbox and every analysis command would fail. `--permission-mode auto` is the cheaper way to get a
write boundary without a container.

Anything Qoder CLI does not expose through these options can be appended with the repeatable
passthrough flag:

```bash
python runners/qoder_cli/run_qoder_longds.py --qoder-arg --debug
```

### Verified CLI Surface

The runner was built against Qoder 1.1.16 and rechecked with 1.1.34. The CLI differs from older
published docs
in a few places worth knowing when upgrading:

- `qoder --help` lists `--permission-mode` choices as `default`, `accept_edits`,
  `bypass_permissions`, `dont_ask`, and `auto`. `plan` is not offered and is not useful here, so the
  runner does not expose it.
- `--max-turns`, `--sandbox`, and `--yolo` are accepted but hidden from `--help`. Unknown flags do
  fail fast with `error: unknown option`, so a future removal surfaces immediately rather than being
  silently ignored.
- `-o stream-json` needs no extra verbosity flag, unlike Claude Code's `--verbose`.
- A failed turn can exit non-zero *and* report `"subtype": "success"` with `"is_error": true`, for
  example on `authentication_failed`. The runner treats `is_error` as a hard turn failure on its own
  so a bad run cannot be recorded as a valid answer.
- A turn whose tool calls were *denied* reports the opposite: `"subtype": "success"` with
  `"is_error": false` and an empty `permission_denials`. See [Blocked Tool Calls](#blocked-tool-calls).
- The CLI writes shell scratch files to a hardcoded `/tmp/qoder-cli-<uid>/` and ignores `TMPDIR`, so
  it cannot run where `/tmp` is read-only.

## Retrying a Failed Turn

A turn can fail for reasons that have nothing to do with the model. Two full 68-task runs produced
four such failures: three reported `Connection interrupted. Progress saved.` and one reported
`Qoder API error: BAD_REQUEST` with `error_code` 500. Because every turn of a task shares one
session, a single dropped connection used to discard the whole task, including the 35 turns that had
already succeeded.

The runner now retries a turn that failed on an infrastructure fault, controlled by `--turn-retries`
(default 2 extra attempts) and `--retry-backoff` (default 30 s, doubled per attempt):

```bash
python runners/qoder_cli/run_qoder_longds.py --turn-retries 3 --retry-backoff 20
python runners/qoder_cli/run_qoder_longds.py --turn-retries 0     # disable
```

**Only infrastructure faults are retried.** Retrying a turn that the model simply answered badly
would hand it extra attempts at the benchmark task and inflate the score, so the classifier is an
allowlist rather than a list of exclusions. A failure is retried when `cli_error_code` is 429 or any
5xx, or when `cli_error` matches a known transient condition such as a dropped connection, a DNS
failure, a gateway error, or rate limiting. Everything else fails the turn immediately: an expired
login, a denied tool call, a missing final answer, or a 4xx response.

Each retry resumes the session the turn *started* from, not the session the failed attempt reported,
so a partially answered turn is never stacked on top of itself.

`result.json` records the whole sequence:

```json
"attempt": 2,
"attempts": [
  {"attempt": 1, "returncode": 1, "cli_error": "Connection interrupted.",
   "cli_error_code": null, "failure": "Qoder CLI exited with code 1"},
  {"attempt": 2, "returncode": 0, "cli_error": null, "failure": null}
]
```

The raw logs of a failed attempt are kept beside the successful one as
`attempt_<n>_qoder_stdout.jsonl`, `attempt_<n>_qoder_stderr.txt`, and
`attempt_<n>_formatted_steps.json`, so a recovered turn is still fully auditable.

A turn that exhausts its attempts still fails the task, and the run still exits non-zero. Retrying
does not replace resuming: a task that dies late has to be rerun from turn 1, since the runner has no
checkpoint mechanism.

## Run More Tasks

By default the runner reads `task_list_lite.json` and executes every task after `--start-index`. Pass
`--task-list-name task_list_lite.json` to run the Lite subset. Pass `--task-limit` to cap how many
tasks run, which is what the smoke tests above rely on.

Run one full task:

```bash
python runners/qoder_cli/run_qoder_longds.py \
  --task-list-name task_list_lite.json \
  --task-limit 1
```

Run the LLM judge automatically after each task finishes:

```bash
python runners/qoder_cli/run_qoder_longds.py \
  --task-limit 1 \
  --judge
```

`--judge` uses the existing `JUDGE_API_KEY`, `JUDGE_BASE_URL`, and optional `JUDGE_MODEL`
environment variables. Each task is evaluated immediately after all its turns finish, before the
next task starts.

Run all tasks:

```bash
python runners/qoder_cli/run_qoder_longds.py \
  --run-parallel 4
```

`--run-parallel` controls task-level concurrency and defaults to `1`. Turns within the same task
always run sequentially in one Qoder CLI session. When `--judge` is enabled, each worker evaluates
its completed task before taking another task. Parallel terminal output from different tasks may be
interleaved; each task's raw and formatted logs remain isolated in its own run directory.

When a task's run directory already exists, the runner skips it without starting Qoder or Docker.
Pass `--overwrite` to delete that task run directory and execute it again.

A failing task never aborts the run: the error is written to that task's `error.json`, the remaining
tasks keep going, and the process exits non-zero at the end if anything failed.

## Disk Usage

Each task copies its released data into its own workspace, and the full LongDS data set is about
19 GB, so a naive full run would leave ~20 GB behind per run. To keep that bounded, the runner
deletes `workspace/data/` as soon as a task's turns finish and records the result in
`task_metadata.json`:

```json
"data_cleanup": {"removed": true, "reason": "task_completed", "freed_bytes": 44969266}
```

Peak disk usage therefore scales with `--run-parallel`, not with the number of tasks: one
sequential run of all 68 tasks peaks at the largest single task (about 3.9 GB) and leaves roughly
100 KB of results per task.

Only the copied inputs are removed. Helper scripts, caches, and intermediate artifacts the agent
wrote into the workspace are kept as trajectory evidence, as are all files under `detail/`, so
`runners/src/judge.py` still works on a cleaned run.

Two cases keep the data:

- A task that fails keeps `workspace/data/` so the failure can be reproduced in place.
- `--keep-data` disables the cleanup entirely. Use it when you intend to reopen the session with
  `manual_resume_command`, since a resumed session cannot re-read data that has been removed.

`--dry-run` never copies data at all; it still validates that each task's source data directory
exists, so a dry run over all 68 tasks costs a few megabytes of prompts and metadata.

## Output Layout

Outputs are written under `results/longds_<version>_<split>/<run_name>/<domain>/<dataset>/<task_id>/`. By default, `run_name`
is `qoder_<model>_<timestamp>`, for example `qoder_performance_20260807_120000`. Passing
`--run-name` overrides the complete directory name. During each turn, stdout and stderr are streamed
to the terminal in real time with formatted, colorized step blocks. Raw Qoder CLI stream-json stdout
and stderr are still saved under that turn directory.

For each task run:

```text
results/longds_<version>_<split>/<run_name>/<domain>/<dataset>/<task_id>/
├── workspace/                    # copied data plus Qoder CLI temporary files
│   └── data/                     # copied released dataset files
├── qoder_turn.schema.json
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
        ├── qoder_stdout.jsonl
        └── qoder_stderr.txt
```

The Qoder CLI execution directory is always the task workspace:
`results/longds_<version>_<split>/<run_name>/<domain>/<dataset>/<task_id>/workspace/`. The runner passes it both as
`qoder --cwd` and as the subprocess working directory, so relative paths cannot fall back to
`runners/qoder_cli/`. From inside Qoder CLI, benchmark files are available under `data/`, and
temporary analysis files should be written outside `data/`.

The runner first copies only that task's released `data/` directory into `workspace/data/`. Qoder
CLI is not given the original `dataset/task/...` path that contains `task.json`, `task.py`,
`task.ipynb`, metadata, and gold answers. Once the task finishes, that copy is deleted again; see
[Disk Usage](#disk-usage).

During a task, `results.json` does not include ground truth. After the task finishes, the runner
writes `results_with_ground_truth.json` and `task_metadata_with_sources.json` for offline scoring
and debugging. If a task fails, its run directory contains `error.json`.

After a run finishes, you can reopen the Qoder CLI session from the task workspace. The session ID
and manual resume command are recorded in `task_metadata.json`. Pass `--keep-data` on the original
run if you want the data still there when you resume.

## How the Final Answer Is Captured

Unlike Codex CLI (`--output-schema`) and Claude Code (`--json-schema`), Qoder CLI has no
server-side output schema flag. `qoder_turn.schema.json` is therefore enforced through the prompt:
every turn appends the required-keys contract, and the runner recovers the payload from the
stream-json events in this order.

1. A `structured_output` field or `StructuredOutput` tool call, if a future CLI version emits one.
2. The `result` event text, parsed as raw JSON or JSON inside a Markdown fence.
3. The last assistant text message, parsed the same way.
4. The raw text itself, used as `answer` when no JSON can be recovered.

Each turn records which path was taken in `answer_source` in `result.json` and `results.json`, for
example `result_json` or `result_raw_text`. A high share of `*_raw_text` means the model is ignoring
the output contract, and the answers were still scored from free-form text.

`result.json` also keeps the CLI diagnostics from the `result` event: `agent_turns`,
`terminal_reason`, `usage`, `total_cost_usd`, `total_credits`, and `cli_error`.

### Blocked Tool Calls

Qoder CLI 1.1.16 leaves the `result` event's own `permission_denials` array **empty even when a tool
was blocked**, and still reports `"subtype": "success"` with `"is_error": false` for that turn. A
turn whose tools were all denied would otherwise look like a clean success with an empty answer.

The runner therefore detects denials from the failed `tool_result` blocks and records them itself:

- `permission_denials`: one entry per blocked call, with the `tool`, the `target` path or command,
  and the CLI's message. Also printed in red after the turn finishes.
- `tool_error_count`: every failed tool call, including ordinary ones such as a Python traceback.
- `cli_permission_denials`: whatever the `result` event reported, kept as a passthrough so a future
  CLI version that populates it is not silently ignored.

Detection matches the CLI's own wording, for example `Auto mode: operation outside workspace
boundary is not allowed` or `the 'Don't ask' permission mode ... automatically denied`, so a normal
failing analysis command is counted in `tool_error_count` but not reported as a denial.

Under `auto` a denial usually means the agent tried to leave its workspace, but it can also be the
false positive on in-workspace shell redirection described above, so read the `target` before
concluding the agent misbehaved. Under the default `bypass_permissions` nothing is ever denied and
both lists stay empty.

`--session-flag` selects how later turns rejoin the session; the default `resume` maps to
`qoder --resume <id>`. Switch to `--session-flag session-id` if a CLI version handles
`--session-id` better. Turn-level session state is printed live and recorded in
`formatted_steps.json` as `new`, `same_as_previous_turn`, or `changed_from_previous_turn`; anything
other than `same_as_previous_turn` after turn 1 means the multi-turn context was lost.

## Run the LLM Judge

Run these commands from `longds/`. Set the judge endpoint first:

```bash
export JUDGE_API_KEY="<your_judge_api_key>"
export JUDGE_BASE_URL="<your_judge_base_url>"
```

Score one run:

```bash
python runners/src/judge.py \
  --run-dir results/longds_<version>_<split>/<run_name>/<domain>/<dataset>/<task_id>
```

Or score every completed run under `results/`:

```bash
python runners/src/judge.py
```

The judge writes a DSGym-compatible `results_eval.json` list back to each run directory. Each turn
contains `turn_id`, `question`, `ground_truth`, `solution`, `success`, `steps`, `trajectory`, and
`judge`; the final list element contains `summary.correct`, `summary.incorrect`, and
`summary.avg_score`. Runs that already have `results_eval.json` are skipped by default and reused in
the printed summary. In all-runs mode, no aggregate file is written unless `--out` is provided:

```bash
python runners/src/judge.py --out results_eval.json
```

To force re-evaluation, pass `--overwrite`:

```bash
python runners/src/judge.py --overwrite
```
