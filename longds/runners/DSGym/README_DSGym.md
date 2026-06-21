# DSGym: A Holistic Framework for Advancing Data Science Agents

DSGym is a comprehensive framework for evaluating and training Large Language Model (LLM) agents on real-world data science tasks. Building upon the research presented in our [paper](https://arxiv.org/pdf/2601.16344), DSGym provides a unified evaluation platform with isolated execution environments, multiple datasets spanning diverse data science domains, and robust evaluation metrics.

## Overview

DSGym addresses the critical need for systematic evaluation of AI agents' data science capabilities. Unlike existing benchmarks that focus on narrow tasks, DSGym provides a holistic assessment across the full data science workflow - from data exploration and preprocessing to modeling and interpretation.

## Features

- **Unified Benchmark Framework**: Evaluate LLMs across multiple data science datasets
- **Isolated Execution**: Docker-based container system for safe code execution
- **Multiple Backend Support**: LiteLLM (API-based), vLLM, and SGLang inference backends
- **Comprehensive Metrics**: Various evaluation metrics including exact match, semantic similarity, and domain-specific scoring
- **Multi-Dataset Support**: DAEval, DiscoveryBench, DABStep, QRData, DSBio, and DSPredict integration
- **Trajectory Generation**: Generate multiple trajectories per sample for synthetic data creation and pass@k evaluation

## Project Structure

```
DSGym/
├── dsgym/                    # Core framework
│   ├── agents/              # LLM agents and backends
│   ├── datasets/            # Dataset loaders and prompts
│   ├── eval/                # Evaluation system and metrics
│   ├── synth/               # Synthetic data generation and trajectory tools
│   ├── train/               # Model training and fine-tuning (coming soon)
│   └── cli/                 # Command-line interface
├── executors/               # Docker-based execution system
├── examples/                # Ready-to-use evaluation scripts
└── data/                    # Dataset storage
```

## Quick Start

### 1. Installation

```bash
# Install main dependencies (includes litellm by default)
uv sync

# Install with optional extras
uv sync --extra dev        # Development tools
uv sync --extra vllm       # vLLM inference backend
uv sync --extra sglang     # SGLang inference backend
uv sync --extra metrics    # Additional evaluation metrics
uv sync --extra synth      # Synthetic data generation
```

### 2. Setup Docker Execution Environment

```bash
cd executors

# Standard DS env
python generate_compose.py -n 64 --types "executor-prebuilt:64" -m ../data/data

sudo docker build -t executor-prebuilt ./container_images/instance
sudo docker build -t manager-prebuilt ./manager
sudo docker compose -f docker-compose.yml up -d --build

sudo docker compose -f docker-compose.yml down

# Bio-specific DS env
python generate_compose.py -n 64 --types "executor-bio:64" -m ../data/data --output docker-compose-bio.yml -c container_config_bio.json

sudo docker build -t executor-bio ./container_images/bio_image
sudo docker build -t manager-prebuilt ./manager
sudo docker compose -f docker-compose-bio.yml up -d --build

sudo docker compose -f docker-compose-bio.yml down

# DSPredict-specific env
python generate_compose.py -n 8 --types "executor-kaggle:8" -m ../data/data --output docker-dspredict-hard.yml -c container_config_dspredict_hard.json \
-g 0,1,2,3,4,5,6,7 -s ../submissions -e "EXECUTION_TIMEOUT=3600,MEM_LIMIT=24G,CPUS=8,MEM_RESERVATION=100G" \

sudo docker build -t executor-kaggle ./container_images/kaggle_image
sudo docker build -t manager-prebuilt ./manager
sudo docker compose -f docker-dspredict-hard.yml up -d --build

sudo docker compose -f docker-dspredict-hard.yml down
```

### 3. Run Evaluations

#### Using CLI

```bash
# Set API key
export OPENAI_API_KEY=<your-api-key>
# or
export TOGETHER_API_KEY=<your-api-key>

# Evaluate on DSBio
dsgym eval --model gpt-4o --dataset dsbio --limit 10

# Evaluate on DAEval
dsgym eval --model together_ai/Qwen/Qwen3-235B-A22B-Instruct-2507-tput --dataset daeval --backend litellm --limit 10


# Get help
dsgym --help
dsgym eval --help
```
To run DSPredict, ensure your Kaggle API credentials are correctly configured. Generate a kaggle api key and:

```bash
export KAGGLE_API_TOKEN=$YOUR_KEY
```

See the [Kaggle API documentation](https://www.kaggle.com/docs/api) for more details. Note that for online assessment, you will have to manually enroll in the competitions (on your account) in order to submit. This will take a bit of time at the beginning. Alternatively, you can use the offline leaderboard for evaluation, which compares submissions against cached leaderboard data. However, this will still require manual enrollment into the competition. (you will see a 403 error if you don't enroll).


#### Using Example Scripts

Alternatively, use the example scripts for different datasets:

#### DSBio
```bash
export TOGETHER_API_KEY=<your-api-key>

cd examples
python evaluate.py \
    --dataset dsbio \
    --model together_ai/Qwen/Qwen3-235B-A22B-Instruct-2507-tput \
    --backend litellm \
    --limit 10 \
    --output-dir ./results/dsbio_test
```

#### QRData
```bash
cd examples
python evaluate.py \
    --dataset qrdata \
    --model gpt-4o \
    --backend litellm \
    --limit 10 \
    --output-dir ./results/qrdata_test
```

### 4. CLI Commands

DSGym provides a unified CLI interface:

#### Available Commands

- **`dsgym eval`**: Evaluate models on datasets
- **`dsgym generate`**: Generate trajectories for synthetic data (coming soon)
- **`dsgym train`**: Train models (coming soon)

#### CLI Examples

```bash
# Evaluate with custom output directory
dsgym eval \
    --model gpt-4o \
    --dataset qrdata \
    --limit 10 \
    --output-dir ./my_results \
    --temperature 0.1


# Use different backend
dsgym eval \
    --model gpt-4o \
    --dataset daeval \
    --backend vllm \
    --max-workers 1
```

#### Output Structure

```
trajectory_results/
├── prediction_0_traj_0.json      # Sample 0, trajectory 0 (complete conversation)
├── prediction_0_traj_1.json      # Sample 0, trajectory 1
├── ...
├── predictions/
│   └── trajectory_..._all.json    # Combined analysis file
└── metrics/
    └── trajectory_..._metrics.json # Pass@K and average metrics
```

## Supported Datasets

- **DAEval**: Data analysis evaluation with list matching metrics
- **DiscoveryBench**: Scientific discovery tasks with LLM and HMS scoring  
- **DABStep**: Step-by-step data analysis tasks
- **QRData**: Question-reasoning evaluation
- **DSBio**: Bioinformatics and computational biology tasks (90 tasks from academic literature)
- **DSPredict**: Competition integration and submission validation

All datasets support trajectory generation for synthetic data creation and pass@k evaluation.

## Key Innovations

### Holistic Evaluation Framework
- **Real-World Tasks**: Derived from actual data science competitions and academic research
- **End-to-End Workflows**: Evaluates entire data science pipelines, not isolated components

### Robust Execution Environment
- **Isolated Containers**: Docker-based execution prevents interference and ensures reproducibility
- **Scalable Architecture**: Supports parallel evaluation across multiple containers
- **Safety First**: Sandboxed execution environment for secure code evaluation

### Dataset Coverage
- **Diverse Domains**: From general/applied ds to scientific ds tasks, covering various data science applications
- **Varied Complexity**: Tasks ranging from basic data analysis to complex scientific discovery

## Resources

- 📄 **Paper**: [DSGym: A Holistic Framework for Evaluating and Training Data Science Agents](https://arxiv.org/pdf/2601.16344)
- 🤗 **Hugging Face**: [DSGym Repository](https://huggingface.co/DSGym)
- 💻 **GitHub**: [Source Code](https://github.com/fannie1208/DSGym)

## Development

### Linting and Formatting
```bash
uv run ruff check .              # Lint check
uv run ruff check --fix .        # Auto-fix linting issues
uv run black .                   # Format code
```

### Testing
```bash
# Executor system tests
cd executors/tests
pytest

# Main system tests
uv run pytest
```

## Configuration

### Environment Variables
- `TOGETHER_API_KEY`: For Together AI models
- `OPENAI_API_KEY`: For OpenAI models
- `ANTHROPIC_API_KEY`: For Anthropic models

### Docker Configuration
- Container specifications in `executors/container_config.json`
- Docker compose generation via `executors/generate_compose.py`

## Adding New Datasets

1. Create dataset loader in `dsgym/datasets/loaders/your_dataset.py`
2. Implement `get_metrics()` and `get_metric_configs()` methods
3. Register dataset using `@register_dataset("your_dataset")` decorator
4. Create example evaluation script in `examples/evaluate_your_dataset.py`
5. Dataset automatically supports trajectory generation

## Adding New Metrics

1. Create metric class in `dsgym/eval/metrics/`
2. Inherit from `BaseMetric` and implement required methods
3. Register metric in `dsgym/eval/metric_registry.py`
4. Update dataset loaders to use the new metric
5. Metrics automatically support pass@k and average calculations

## Trajectory Generation

Generate multiple trajectories per sample for synthetic data creation and pass@k evaluation. Supports configurable temperature, parallel processing, and automatic metric computation including both pass@k (maximum) and average scores across trajectories.

## Requirements

- Python 3.12+
- Docker and Docker Compose
- GPU support recommended for local inference backends

## Citation

If you use DSGym in your research, please cite our paper:

```bibtex
@misc{nie2026dsgym,
      title={DSGym: A Holistic Framework for Evaluating and Training Data Science Agents}, 
      author={Fan Nie and Junlin Wang and Harper Hua and Federico Bianchi and Yongchan Kwon and Zhenting Qi and Owen Queen and Shang Zhu and James Zou},
      year={2026},
      eprint={2601.16344},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2601.16344}, 
}
```


## Contributing

We welcome contributions to DSGym! Please see our contributing guidelines for more information on how to:
- Add new datasets
- Add new tasks
- Add new agent scaffolds
- Implement new evaluation metrics
- Improve the framework
- Report bugs and suggest features

Detailed guidelines will be posted soon!
