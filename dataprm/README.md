

<h1 align="center"> DataPRM </h1>

<p align="center">
  <a href="https://arxiv.org/abs/2604.24198">📄arXiv</a> •
  <a href="https://huggingface.co/collections/zjunlp/datamind-687d90047c58bb1e3d901dd8">🤗HuggingFace</a>
</p>


## Table of Contents

- 👀 [Overview](#overview)
- 🔧 [Installation](#installation)
- 💻 [Training](#Training)
- 🧐 [Evaluation](#evaluation)
- ✍️ [Citation](#citation)
---

## 👀 Overview

Process Reward Models (PRMs) have achieved remarkable success in enhancing the reasoning capabilities of Large Language Models (LLMs) in static domains such as mathematics. However, their potential in dynamic data analysis tasks remains largely underexplored.

In this work, we first present an empirical study revealing that general-domain PRMs struggle to supervise data analysis agents. Specifically, they fail to detect **silent errors** — logical flaws that produce incorrect results without triggering interpreter exceptions — and erroneously penalize exploratory actions by mistaking necessary trial-and-error behavior for grounding failures.

To address this gap, we introduce **DataPRM**, a novel environment-aware generative process reward model that:
1. Acts as an **active verifier**, autonomously interacting with the execution environment to probe intermediate states and uncover silent errors.
2. Employs a **reflection-aware ternary reward strategy** that distinguishes between correctable grounding errors and irrecoverable mistakes.

We design a scalable pipeline to construct over **8K high-quality training instances** for **DataPRM** via diversity-driven trajectory generation and knowledge-augmented step-level annotation.

Experimental results demonstrate that **DataPRM** improves downstream policy LLMs by **7.21%** on ScienceAgentBench and **11.28%** on DABStep using Best-of-N inference.

Notably, with only **4B parameters**, **DataPRM** outperforms strong baselines and exhibits robust generalizability across diverse Test-Time Scaling strategies. Furthermore, integrating **DataPRM** into Reinforcement Learning yields substantial gains over outcome-reward baselines, achieving **78.73%** on DABench and **64.84%** on TableBench, validating the effectiveness of process reward supervision.


## 🔧 Installation

We recommend using separate conda environments for different components to avoid dependency conflicts.

### Prerequisites

- Anaconda
- GPU with CUDA support (recommended: CUDA 12.6)

### Environment Setup

**SFT Training**

We use the **[ms-swift](https://github.com/modelscope/ms-swift)** (v3.11.0.dev0) framework for supervised fine-tuning.

```bash
cd ms-swift
pip install -e .
pip install deepspeed
pip install liger-kernel
```

**Deployment**

We recommend deploying DataPRM with vLLM.

```bash
pip install vllm==0.13.0
```

**Evaluation**

```bash
cd evaluate
pip install -r requirements.txt

# For ScienceAgentBench evaluation, we recommend creating a separate environment
pip install -r requirements-sab.txt
conda install -c conda-forge gdal
```


## 💻 Training

### SFT Training
Our model training was completed using the powerful and user-friendly **[ms-swift](https://github.com/modelscope/ms-swift)** (3.11.0.dev0), which provided us with an efficient fine-tuning workflow.

#### 1. Training Data

The training datasets are available in huggingface [dataprm-dabstep](https://huggingface.co/datasets/zjunlp/DataPRM-DABStep), [dataprm-scienceagentbench](https://huggingface.co/datasets/zjunlp/DataPRM-ScienceAgentBench). Download them and place the file at `ms-swift/data`.

#### 2. Launch Training

Use the following command to start training.

```bash
cd ms-swift
bash train.sh
```

## 🧐 Evaluation

**Step 1: Start the model server**

DataPRM models are available in huggingface [dataprm](https://huggingface.co/collections/zjunlp/datamind-687d90047c58bb1e3d901dd8). Modify `model.sh` to match your environment, then launch the vLLM server:

```bash
bash model.sh
```

**Step 2: Configure tool dependencies**

Set the required environment variables for document and image query tools (see `evaluate/tools/query_document.py` and `evaluate/tools/query_image.py`):

```bash
export DOC_OPENAI_API_KEY="<your_document_model_api_key>"
export DOC_OPENAI_BASE_URL="<your_document_model_base_url>"
export DOC_MODEL_NAME="<your_document_model_name>"

export IMG_OPENAI_API_KEY="<your_image_model_api_key>"
export IMG_OPENAI_BASE_URL="<your_image_model_base_url>"
export IMG_MODEL_NAME="<your_image_model_name>"
```

**Step 3: Run evaluation**

```bash
# DABStep
python run_eval.py dabstep \
    --input_files /path/to/qwen3_235b_dabstep_all_results_{}_bs_32_maxlen_10.json \
    --number 0 --skip 16 \
    --output_dir /path/to/output/dab \
    --data_file_dir /path/to/DABStep/data/context_fix \
    --workspace_root /path/to/workspace \
    --final_step_index 10 \
    --base_url <your_base_url> \
    --api_key <your_api_key> \
    --model_name <your_model_name>

# ScienceAgentBench
python run_eval.py scienceagentbench \
    --input_files /path/to/qwen3_235b_sab_all_results_{}_bs_32_maxlen_15.json \
    --number 0 --skip 16 \
    --output_dir /path/to/output/sab \
    --data_file_dir /path/to/ScienceAgentBench/benchmark/datasets \
    --workspace_root /path/to/workspace \
    --final_step_index 15 \
    --base_url <your_base_url> \
    --api_key <your_api_key> \
    --model_name <your_model_name>
```

### Datasets

- **DABStep**: Download from [DABStep-Data](https://huggingface.co/datasets/adyen/DABstep/tree/main/data). We also provide a fixed file version at `evaluate/datasets/dabstep/context_fix`.
- **ScienceAgentBench**: Download from [ScienceAgentBench](https://github.com/OSU-NLP-Group/ScienceAgentBench). The task list used in our experiments is provided at `evaluate/datasets/scienceagentbench/science_agent_bench_tasks_with_file_info.json`.

An example input file is provided in `evaluate/input_files_examples`; refer to it to format your own input files for DataPRM.

## ✍️ Citation

If you find our work helpful, please use the following citations.

```
@article{qiu2026rewarding,
  title={Rewarding the scientific process: Process-level reward modeling for agentic data analysis},
  author={Qiu, Zhisong and Qiao, Shuofei and Xu, Kewei and Zhu, Yuqi and Du, Lun and Zhang, Ningyu and Chen, Huajun},
  journal={arXiv preprint arXiv:2604.24198},
  year={2026}
}
```
