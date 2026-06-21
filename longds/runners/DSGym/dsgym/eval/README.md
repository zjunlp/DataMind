# DSGym Evaluation Framework

A comprehensive, modular evaluation system for data science tasks with pluggable metrics, flexible protocols, and reproducible evaluation workflows.

## Features

- **Modular Architecture**: Pluggable metrics and evaluation protocols
- **Multiple Metrics**: Exact match, LLM-based evaluation, semantic similarity, code execution, and domain-specific metrics
- **Flexible Protocols**: Single-turn and multi-turn evaluation
- **Batch Processing**: Efficient batch evaluation with parallel processing support
- **Domain-Specific**: Specialized metrics for DiscoveryBench, DABStep, QRData, and Kaggle
- **Dataset Integration**: Automatic metric selection based on dataset type
- **Easy Integration**: Compatible with DSGym agents and datasets

## Quick Start

```python
from dsgym.eval import Evaluator, EvaluationConfig
from dsgym.datasets import DatasetRegistry
from dsgym.agents import ReActDSAgent

# Load dataset (automatically gets appropriate metrics)
dataset = DatasetRegistry.load("discoverybench")
samples = dataset.load(split="test", limit=10)

# Initialize agent
agent = ReActDSAgent(backend="litellm", model="gpt-4o")

# Create evaluator with dataset-specific metrics
evaluator = Evaluator(
    protocol="multi_turn",
    dataset=dataset  # Auto-uses dataset metrics: ["llm_score", "hms_score"]
)

# Configure evaluation
config = EvaluationConfig(
    model_name="gpt-4o",
    dataset_name="discoverybench",
    output_dir="./results"
)

# Run evaluation
results = evaluator.evaluate(agent, samples, config=config, save_results=True)
```

## Available Metrics

### Basic Metrics
- `exact_match`: Exact string matching with numeric tolerance
- `list_match`: DAEval-style list format matching (e.g., `@key[value]`)

### Code Evaluation
- `code_execution`: Evaluates if code runs successfully without errors

### LLM-Based Evaluation
- `llm_judge`: Uses LLM to evaluate answer correctness
- `llm_score`: DiscoveryBench LLM-based discrete scoring

### Semantic Similarity
- `semantic_similarity`: Embedding-based similarity scoring using sentence transformers

### Domain-Specific Metrics
- `hms_score`: DiscoveryBench HMS (Hypothesis Matching Score) evaluation
- `dabstep`: DABStep task evaluation (future use)
- `kaggle`: Kaggle submission format validation (future use)

### Dataset-Specific Metrics (Auto-Selected)
- **DAEval**: `list_match`
- **DiscoveryBench**: `llm_score`, `hms_score`
- **QRData**: `exact_match`
- **Bio**: `exact_match`

## Evaluation Protocols

### Single Turn
```python
evaluator = Evaluator(protocol="single_turn")
```
Agent receives query and provides one response without environment interaction.

### Multi Turn (Default)
```python
evaluator = Evaluator(protocol="multi_turn", max_turns=15)
```
Agent can interact with code execution environment over multiple turns. This is the standard protocol for DSGym data science tasks.

## Custom Metrics

Create custom metrics by extending `BaseMetric`:

```python
from dsgym.eval.metrics.base import BaseMetric, MetricResult
from dsgym.eval.metric_registry import register_metric

class MyCustomMetric(BaseMetric):
    @property
    def name(self) -> str:
        return "my_metric"
    
    def evaluate(self, prediction: str, ground_truth: str = None, **kwargs) -> MetricResult:
        # Your evaluation logic here
        score = your_scoring_function(prediction, ground_truth)
        return MetricResult(
            metric_name=self.name,
            score=score,
            details={"custom_info": "value"}
        )

# Register and use
register_metric("my_metric", MyCustomMetric)
evaluator = Evaluator(metrics=["my_metric"])
```

## Real-World Usage Examples

The evaluation framework integrates with DSGym's complete workflow:

### Example 1: DiscoveryBench Evaluation
```python
from dsgym.datasets import DatasetRegistry
from dsgym.agents import ReActDSAgent
from dsgym.eval import Evaluator, EvaluationConfig

# Load DiscoveryBench dataset
dataset = DatasetRegistry.load("discoverybench")
samples = dataset.load(split="test", limit=10)

# Initialize agent with specific model
agent = ReActDSAgent(
    backend="litellm",
    model="together_ai/Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8",
    manager_url="http://localhost:5000"
)

# Create evaluator with dataset-specific metrics
evaluator = Evaluator(
    protocol="multi_turn",
    dataset=dataset,  # Auto-uses: ["llm_score", "hms_score"]
    parallel_workers=24
)

# Configure and run evaluation
config = EvaluationConfig(
    model_name=agent.model,
    backend_type=agent.backend,
    dataset_name="discoverybench", 
    output_dir="./results/discovery_test",
    run_name="coder480b_discovery_eval"
)

results = evaluator.evaluate(agent, samples, config=config, save_results=True)
```

### Example 2: DAEval Evaluation
```python
# Load DAEval dataset
dataset = DatasetRegistry.load("daeval")
samples = dataset.load(split="test", limit=50)

# Create evaluator (auto-uses list_match metric)
evaluator = Evaluator(protocol="multi_turn", dataset=dataset)

results = evaluator.evaluate(agent, samples, config=config, save_results=True)
```

## Configuration

### Manual Metric Selection
```python
# Override dataset defaults with custom metrics
evaluator = Evaluator(
    protocol="multi_turn",
    metrics=["exact_match", "llm_score", "semantic_similarity"],
    metric_configs={
        "exact_match": {"numeric_tolerance": 0.05},
        "llm_score": {"model": "gpt-4o-mini"},
        "hms_score": {"model": "gpt-4o"}
    }
)
```

### Parallel Processing
```python
# Optimal settings by backend type
evaluator = Evaluator(
    parallel_workers=24  # For API-based models (litellm)
    # parallel_workers=1   # For local models (vllm/sglang)
)
```

## Output and Results

The framework provides comprehensive results including:

- **Individual sample results** with agent trajectories and tool usage
- **Aggregated metrics** across all samples with statistical summaries
- **Error tracking** for failed predictions and exceptions
- **Execution timing** for performance analysis
- **Automatic file saving** in JSON format

### Result Structure
```python
# Results object returned by evaluator.evaluate()
{
    "sample_results": [
        {
            "sample_index": 0,
            "query": "What is the correlation...",
            "prediction": "The correlation is 0.85",
            "ground_truth": "0.8543",
            "trajectory": [...],  # Agent's step-by-step actions
            "metric_results": {
                "llm_score": 1.0,
                "hms_score": 0.75
            },
            "execution_time": 45.2,
            "success": True
        },
        ...
    ],
    "aggregated_metrics": {
        "llm_score": {"mean_score": 0.78, "total_samples": 10},
        "hms_score": {"mean_score": 0.65, "total_samples": 10}
    },
    "metadata": {
        "dataset_name": "discoverybench",
        "model_name": "gpt-4o",
        "total_execution_time": 420.5
    }
}
```

### File Outputs
When `save_results=True`:
- `evaluation_results.json`: Complete results with all details
- `metrics_summary.json`: Aggregated metrics only
- `sample_predictions/`: Individual prediction files

## Running Evaluations

Use the provided example scripts to run complete evaluations:

```bash
cd examples

# DiscoveryBench evaluation
python evaluate_discoverybench.py \
    --model gpt-4o \
    --backend litellm \
    --limit 10 \
    --output-dir ./results/discovery_test

# DAEval evaluation  
python evaluate_daeval.py \
    --model together_ai/Qwen/Qwen3-235B-A22B-Instruct-2507-tput \
    --backend litellm \
    --limit 20 \
    --output-dir ./results/daeval_test

# QRData evaluation
python evaluate_qrdata.py \
    --model gpt-4o \
    --backend litellm \
    --limit 15 \
    --output-dir ./results/qrdata_test

# Bio evaluation
python evaluate_bio.py \
    --model gpt-4o \
    --backend litellm \
    --split hard \
    --limit 10 \
    --output-dir ./results/bio_test
```

## Ground Truth Handling

The framework gracefully handles cases where ground truth is missing:

- Metrics that require ground truth will be skipped automatically
- Results track which samples have ground truth available
- Metrics like `code_execution` can run without ground truth
- Aggregated statistics separate samples with/without ground truth

## Dependencies

The evaluation framework is included with DSGym's main dependencies:

- **Core**: `pandas`, `numpy`, `tqdm`
- **LLM metrics**: `litellm` (included by default)
- **Semantic similarity**: `sentence-transformers` (optional extra)
- **Code execution**: Uses DSGym's container execution system
- **Agents**: DSGym agents and environment system

Install with specific extras if needed:
```bash
uv sync --extra metrics    # Additional evaluation metrics
```

## Architecture

```
dsgym/eval/
├── __init__.py              # Main exports (Evaluator, EvaluationConfig)
├── evaluator.py             # Main evaluation orchestrator
├── protocols.py             # Single-turn and multi-turn protocols
├── metric_registry.py       # Automatic metric discovery and management
├── utils.py                 # Data structures and result handling
├── dataset_integration.py   # Dataset-specific metric integration
├── metrics/                 # Metric implementations
│   ├── __init__.py         # Metric exports
│   ├── base.py             # BaseMetric interface and MetricResult
│   ├── exact_match.py      # String matching (exact_match, list_match)
│   ├── code_execution.py   # Code evaluation metrics
│   ├── equivalence_by_llm.py # LLM judge metrics
│   ├── semantic_similarity.py # Embedding-based metrics
│   └── domain_specific.py  # HMS, LLM score, dataset-specific metrics
└── README.md               # This documentation
```

## Key Features

1. **Automatic Dataset Integration**: Datasets specify their own metrics via `get_metrics()` and `get_metric_configs()`
2. **Parallel Evaluation**: Efficient batch processing with configurable worker counts
3. **Comprehensive Results**: Detailed trajectories, timing, and error tracking
4. **Extensible**: Easy to add new metrics and evaluation protocols
5. **Production Ready**: Used in DSGym's evaluation scripts and benchmarks