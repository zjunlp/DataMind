

<h1 align="center"> DataMind </h1>

<p align="center">
  <a href="https://arxiv.org/abs/2509.25084">📄arXiv</a> •
  <a href="https://huggingface.co/collections/zjunlp/datamind-687d90047c58bb1e3d901dd8">🤗HuggingFace</a>
</p>

<div align="center">

</div>


## Table of Contents

- 👀 [Overview](#overview)
- 🔧 [Installation](#installation)
- 💻 [Training](#training)
- 🧐 [Evaluation](#evaluation)
- ✍️ [Citation](#citation)


---

## 👀 Overview

Data-analytic agents are emerging as a key catalyst for automated scientific discovery and for the vision of Innovating AI. Current approaches, however, rely heavily on prompt engineering or multi-agent scaffolds over proprietary models, while open-source models still struggle with diverse-format, large-scale data files and long-horizon, multi-step reasoning that real-world analytics demands. This paper introduces **DataMind**, a scalable data synthesis and agent training recipe designed to construct generalist data-analytic agents. **DataMind** tackles three key challenges in building open-source data-analytic agents, including insufficient data resources, improper training strategy, and unstable code-based multi-turn rollout. 

Concretely, **DataMind** applies
- A fine-grained task taxonomy and a recursive easy-to-hard task composition mechanism to increase the diversity and difficulty of synthesized queries; 
- A knowledge-augmented trajectory sampling strategy followed by model-based and rule-based filtering; 
- A dynamically adjustable training objective combining both SFT and RL losses;
- A memory-frugal and stable code-based multi-turn rollout framework. 

Built on **DataMind**, we curate **DataMind-12K**, a high-quality trajectory set spanning diverse domains, task categories, and data file formats for data-analytic tasks. Trained on DataMind-12K, our DataMind-14B achieves state-of-the-art with an average score of 71.16\% on multiple data analysis benchmarks, outperforming the strongest proprietary baselines DeepSeek-V3.1 and GPT-5. Our DataMind-7B also performs best among all open-source models with a score of 68.10\%. We also list some empirical insights gained from our exploratory trials in the analysis experiments, aiming to provide actionable insights about agent training for the community. We will release DataMind-12K and DataMind-7B,14B for the community's future research.

<!-- Large Language Models (LLMs) hold promise in automating data analysis tasks, yet opensource models face significant limitations in these kinds of reasoning-intensive scenarios. In this work, we investigate strategies to enhance the data analysis capabilities of open-source LLMs. By curating a seed dataset of diverse, realistic scenarios, we evaluate models across three dimensions: data understanding, code generation, and strategic planning. Our analysis reveals three key findings: *(1) Strategic planning quality serves as the primary determinant of model performance*; *(2) Interaction design and task complexity significantly influence reasoning capabilities*; *(3) Data quality demonstrates a greater impact than diversity in achieving optimal performance.* We leverage these insights to develop a data synthesis methodology, demonstrating significant improvements in open-source LLMs’ analytical reasoning capabilities. -->

## 🔧Installation
#### Manual Environment Configuration

Conda virtual environments offer a light and flexible setup. For different projects, we recommend using separate conda environments for management.

#### Prerequisites

- Anaconda Installation
- GPU support (recommended CUDA version: 12.6)

#### Detail

- SFT training

    For SFT training, we use **[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)** (0.9.4.dev0) framework. 
    ```bash
    cd train/SFT/LLaMA-Factory
    pip install -e ".[torch,metrics]" --no-build-isolation
    ```

- RL training

    For RL training, we use **[verl](https://github.com/volcengine/verl)** (v0.4.0) framework.
    ```bash
    cd train/RL/verl
    USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
    pip install -e .[vllm]
    pip install -e .[sglang]
    apt install sqlite3
    ```

- Eval
    ```bash
    cd eval/Datamind
    pip install -r requirements.txt
    apt install sqlite3
    ```

<!-- 
## 🔧 Installation

#### 🔩Manual Environment Configuration

Conda virtual environments offer a light and flexible setup.

**Prerequisites**

- Anaconda Installation
- GPU support (recommended CUDA version: 12.4)

**Configure Steps**

1. Clone the repository:

```bash
git clone https://github.com/zjunlp/DataMind.git
```

2. Enter the working directory, and all subsequent commands should be executed in this directory.

```bash
cd DataMind/eval
```

3. Create a virtual environment using `Anaconda`.

```bash
conda create -n DataMind python=3.10
conda activate DataMind
```

4. Install all required Python packages.

```bash
pip install -r requirements.txt
``` -->



## 💻  Training

### SFT training
Our model training was completed using the powerful and user-friendly **[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)** framework (0.9.4.dev0), which provided us with an efficient fine-tuning workflow.

#### 1. Training Data

The training dataset `datamind_12k` is available in huggingface [datamind-12k](https://huggingface.co/datasets/zjunlp/DataMind-12K/tree/main). You can download it and put it in `train/SFT/LLaMA-Factory/data/datamind/datamind_12k.json`.

#### 2. Training Configuration

We provide our configuration for full-parameter fine-tuning using DeepSpeed ZeRO-3 in yaml file. You can find it in `train/SFT/LLaMA-Factory/examples/train_full/datamind_12k_full_sft.yaml`.

#### 3. Launch Training
You can use the following command to start training. Here we take `datamind_12k_full_sft.yaml` as an example. Or you can use the shell script `train/SFT/LLaMA-Factory/train.sh`.
```
CUDA_VISIBLE_DEVICES=0,1,2,3 llamafactory-cli train examples/train_full/datamind_12k_full_sft.yaml
```

### RL training
Our RL training framework is modified from the [verl](https://github.com/volcengine/verl) (v0.4.0) framework, which is a flexible, efficient and production-ready RL training library for large language models (LLMs).

#### 1. Data
The data in *Scaling Generalist Data-Analytic Agents* is available in huggingface [DataMind-Data](https://huggingface.co/datasets/zjunlp/DataMind-Data). You can find the RL data in the `rl` directory.
- train.parquet: training data for RL.
- test.parquet: evaluation data for RL.
- gold_csv_results: the directory storing the golden CSV answers for SQL data.
- train_files: the directory storing the csv, excel and sqlite files.
- db_schema.json: the file recording the database schema for SQL data.

#### 2. Training Configuration
You should modify the following configurations to ensure the stable operation of the framework.
- train/RL/verl/multi.sh: modify the environment variables and paths within the sh file.
- train/RL/verl/agent/async_interpreter.py: modify the conda path to your own conda path (we use another independent conda environment with Python 3.10 to run the code, and the relevant packages can be found in the run_code_requirements.txt file).

#### 3. Launch Training
You can run the `multi.sh` script to start training.
```bash
bash multi.sh
```

## 🧐 Evaluation
### 1. Evaluation Data
The data is available in huggingface [DataMind-Data](https://huggingface.co/datasets/zjunlp/DataMind-Data). You can find the Evaluation data in the `eval` directory.
You should unzip the zip files and place them in the corresponding folders.
```
├── model.sh
├── requirements.txt
├── python
│   ├── compute_pass3.py
│   ├── da-dev-tables
│   ├── eval_python.py
│   ├── eval.sh
│   ├── interpreter.py
│   ├── tablebench_csv
│   └── test_file
│       ├── daeval_test.parquet
│       └── tablebench_test.parquet
└── sql
    ├── bird
    │   ├── bird_dev_csv_results
    │   ├── dev_sqlite_files
    │   ├── bird_dev_omni_ddl.json
    │   └── test_file
    │       └── bird_dev.parquet
    ├── compute_pass3.py
    ├── eval_bird.py
    ├── eval.sh
    └── interpreter.py
```

### 2. Evaluation
We use vLLM to launch a local model server. You can modify the `model.sh` to adapt to your own environment and run it to start the model server.
```sh
bash model.sh
```

#### For Python Evaluation
You can modify the eval/python/eval.sh and run it to start Python evaluation. Notice that you should modify the `base_url` and `api_key` for judge model in `eval/python/eval_python.py`.
```sh
PORT=19007
export OPENAI_BASE_URL=http://0.0.0.0:${PORT}/v1
export OPENAI_API_KEY=placeholder_key

python eval_python.py \
    --model datamind \
    --temperature 0.7 \
    --top_p 0.95 \
    --bs 5 \
    --test_bench dabench \
    --test_file test_file/daeval_test.parquet \
    --csv_or_db_folder da-dev-tables \
```

#### For SQL Evaluation
You can modify the eval/sql/eval.sh and run it to start SQL evaluation.
```sh
PORT=19008
export OPENAI_BASE_URL=http://0.0.0.0:${PORT}/v1
export OPENAI_API_KEY=placeholder_key

python eval_bird.py \
    --model datamind \
    --temperature 0.7 \
    --top_p 0.95 \
    --bs 5 \
    --test_bench bird \
    --test_file bird/test_file/bird_dev.parquet \
    --csv_or_db_folder bird/dev_sqlite_files \
    --gold_csv_results_dir bird/bird_dev_csv_results \
    --db_schema_data_path bird/bird_dev_omni_ddl.json
```

## ✍️ Citation

If you find our work helpful, please use the following citations.

```
@article{qiao2025scaling,
  title={Scaling Generalist Data-Analytic Agents},
  author={Qiao, Shuofei and Zhao, Yanqiu and Qiu, Zhisong and Wang, Xiaobin and Zhang, Jintian and Bin, Zhao and Zhang, Ningyu and Jiang, Yong and Xie, Pengjun and Huang, Fei and others},
  journal={arXiv preprint arXiv:2509.25084},
  year={2025}
}
```
