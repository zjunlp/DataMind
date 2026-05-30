<h1 align="center"> LongDS-Bench </h1>

<p align="center">
  <a href="https://github.com/zjunlp/DataMind">💻 GitHub</a> •
  <a href="https://huggingface.co/datasets/zjunlp/LongDS">🤗 Hugging Face</a> •
  <a href="https://huggingface.co/collections/zjunlp/datamind">📚 DataMind Collection</a>
</p>

## Table of Contents

- 👀 [Overview](#overview)
- 🔧 [Installation](#installation)
- 📦 [Data](#data)
- 🐳 [Execution Environment](#execution-environment)
- 💻 [Running LongDS](#running-longds)
- 📁 [Outputs](#outputs)
- 🙏 [Acknowledgements](#acknowledgements)
- 📖 [Citation](#citation)

---

## 👀 Overview

LongDS-Bench is a benchmark for evaluating long-horizon, multi-turn agentic data analysis. Real-world analysis is rarely a sequence of independent questions: filters, metric definitions, assumptions, intermediate tables, and branch-specific results evolve over many turns. LongDS tests whether agents can maintain and apply these evolving analytical states correctly.

LongDS contains **68 tasks** constructed from real-world Kaggle notebooks and datasets, spanning **2,225 turns** across six domains: Business, Community, Education, Geoscience, Social Good, and Sports. The tasks cover representative state-evolution patterns, including:

- initial analytical state construction;
- state inheritance;
- state update;
- counterfactual perturbation;
- rollback to earlier states;
- multi-state composition.


Experiments can be run with [DSGym](https://arxiv.org/abs/2601.16344), which provides isolated Docker execution environments for code-based data analysis.

## 🔧 Installation

Follow the [DSGym](https://github.com/fannie1208/DSGym) setup instructions to configure the evaluation environment.
### 📋 Prerequisites

- Python 3.12
- Docker and Docker Compose
- `uv`

### ⚙️ Environment Setup

```bash
cd /path/to/DataMind/longds/DSGym

# Install main dependencies (includes litellm by default)
uv sync
```

### 🐳 Execution Environment

LongDS uses DSGym's container manager to allocate isolated Python execution environments.

Build and start the LongDS executor pool:

```bash
cd /path/to/DataMind/longds/DSGym/executors

docker build -t executor-prebuilt ./container_images/longds_image
docker build -t manager-prebuilt ./manager

python generate_compose.py \
  -n 8 \
  --types "executor-prebuilt:8" \
  -m ../data/data

docker compose -f docker-compose.yml up -d --build
```

Stop the executor pool:

```bash
cd /path/to/DataMind/longds/DSGym/executors
docker compose -f docker-compose.yml down
```

## 📦 Data

LongDS expects the following layout under `DSGym/data`:

```text
DSGym/data/
├── data/
│   └── longds/
│       └── {domain}/{dataset}/taskN/data/...
└── task/
    └── longds/
        ├── task_list.json
        └── {domain}/{dataset}/taskN/
            ├── task.ipynb
            ├── task.py
            ├── task.json
            └── metadata.json
```

You can download the released data from [Hugging Face](https://huggingface.co/datasets/zjunlp/LongDS):

```bash
cd /path/to/DataMind/longds/DSGym
hf download zjunlp/LongDS \
  --repo-type dataset \
  --local-dir data
```

## 💻 Running LongDS

Run LongDS using `examples/longds.py`.

### 1. Configure Model Access

For API-based models via LiteLLM, set the required environment variables for your provider.

OpenAI-compatible example:

```bash
export OPENAI_API_KEY="<your_openai_compatible_api_key>"
export OPENAI_BASE_URL="<your_openai_compatible_base_url>"
```

Anthropic example:

```bash
export ANTHROPIC_API_KEY="<your_anthropic_api_key>"
export ANTHROPIC_BASE_URL="<your_anthropic_base_url>"
```

LongDS also runs an LLM-as-judge evaluation after each task. Configure the judge endpoint as follows:

```bash
export JUDGE_API_KEY="<your_judge_api_key>"
export JUDGE_BASE_URL="<your_judge_base_url>"
```

The default judge model is set in `examples/longds.py`. Modify that file if your endpoint uses a different model name.

### 2. Run Evaluation

Evaluate all LongDS tasks:

```bash
cd /path/to/DataMind/longds/DSGym/examples

uv run python longds.py \
  --dataset longds \
  --model openai/deepseek-v4-pro-guan \
  --backend litellm \
  --output-dir ./results
```

Useful options:

```text
--task-limit N        Evaluate at most N task directories.
--start-index N       Start from the task directory at index N in task_list.json.
--turn-limit N        Evaluate at most N turns per task.
--max-steps N         Maximum agent steps per turn. Default: 40.
```

## 📁 Outputs

For each task, LongDS writes results under:

```text
{output_dir}/longds/{domain}/{dataset}/taskN/{model_name}_{timestamp}/
├── traj.json          # full multi-turn conversation and solutions
├── results.json       # per-turn trajectories before judging
├── results_eval.json  # per-turn trajectories with LLM judge scores
├── code.py            # extracted Python code from model responses
└── bak/               # intermediate execution records
```

The main evaluation score is stored in `results_eval.json`, where each turn receives a judge score and the final element contains a summary with the average score.

## 🙏 Acknowledgements

We thank the [DSGym](https://github.com/fannie1208/DSGym) team for their open-source evaluation framework. We adapted DSGym's evaluation pipeline to support long-horizon, multi-turn data analysis tasks and use DSGym's Docker-based execution infrastructure. For more details about DSGym, please refer to their [paper](https://arxiv.org/abs/2601.16344).

## 📖 Citation

If you use LongDS, please cite:

```bibtex
@misc{xu2026longdsbench,
  title = {LongDS-Bench: On the Failure of Long-Horizon Agentic Data Analysis},
  author = {Xu, Kewei and Lu, Xiaoben and Qiao, Shuofei and Ding, Zihan and Xu, Haoming and Liang, Lei and Zhang, Ningyu},
  year = {2026},
  howpublished = {\url{https://github.com/zjunlp/DataMind}}
}
```
