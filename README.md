

<h1 align="center"> DataMind </h1>

<div align="center">
 
[![Awesome](https://awesome.re/badge.svg)](https://github.com/zjunlp/DataMind) 
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
![](https://img.shields.io/github/last-commit/zjunlp/DataMind?color=green) 
 
</div>


## Table of Contents

- 🔔 [News](#news)
- 👀 [Overview](#overview)
- 🔧 [Installation](#installation)
- 💻  [Training](#training)
- 🧐 [Evaluation](#evaluation)
- ✍️[Citation](#citation)



---

## 🔔 News

- **[2025-06]** We release a new paper: "[Why Do Open-Source LLMs Struggle with Data Analysis? A Systematic Empirical Study](https://arxiv.org/pdf/2506.19794)".



## 👀 Overview

Large Language Models (LLMs) hold promise in automating data analysis tasks, yet opensource models face significant limitations in these kinds of reasoning-intensive scenarios. In this work, we investigate strategies to enhance the data analysis capabilities of open-source LLMs. By curating a seed dataset of diverse, realistic scenarios, we evaluate models across three dimensions: data understanding, code generation, and strategic planning. Our analysis reveals three key findings: *(1) Strategic planning quality serves as the primary determinant of model performance*; *(2) Interaction design and task complexity significantly influence reasoning capabilities*; *(3) Data quality demonstrates a greater impact than diversity in achieving optimal performance.* We leverage these insights to develop a data synthesis methodology, demonstrating significant improvements in open-source LLMs’ analytical reasoning capabilities.



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
```



## 💻  Training

Our model training was completed using the powerful and user-friendly **[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)** framework, which provided us with an efficient fine-tuning workflow.

##### 1. Training Data

Our training dataset is available in `train/datamind-da-dataset.json`

##### 2. Training Configuration

The following is an example configuration for full-parameter fine-tuning using DeepSpeed ZeRO-3. You can save it as a YAML file (e.g., `datamind_sft.yaml`).

```
### model
model_name_or_path:  Qwen/Qwen2.5-7B-Instruct # Or Qwen/Qwen2.5-14B-Instruct

### method
stage: sft
do_train: true
finetuning_type: full
deepspeed: examples/deepspeed/ds_z3_config.json
flash_attn: fa2


### dataset
dataset: datamind-da-dataset
template: qwen
cutoff_len: 8192
overwrite_cache: true
preprocessing_num_workers: 16

### output
output_dir: checkpoints/your-model-name
logging_steps: 1
save_strategy: epoch
plot_loss: true
overwrite_output_dir: true
report_to: none

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 4
learning_rate: 1.0e-5
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000
```

##### 3. Launch Training

```
CUDA_VISIBLE_DEVICES=0,1,2,3 llama-factory-cli train datamind_sft.yaml
```



## 🧐 Evaluation

> Note:
>
> - **Ensure** that your working directory is set to the **`eval`** folder in a virtual environment.
> - If you have more questions, feel free to open an issue with us.
> - If you need to use local model, you need to deploy it according to **(Optional)`local_model.sh`**.

**Step 1: Download the evaluation datasets and our sft models**
The evaluation datasets we used are in [QRData](https://github.com/xxxiaol/QRData) and [DiscoveryBench](https://github.com/allenai/discoverybench).  The script expects data to be at `data/QRData/benchmark/data/*.csv` and `data/DiscoveryBench/*.csv`.

 You can also download our sft models directly from Hugging Face:  [DataMind-Qwen2.5-7B](https://huggingface.co/zjunlp/DataMind-Qwen2.5-7B) ,[DataMind-Qwen2.5-14B ](https://huggingface.co/zjunlp/DataMind-Qwen2.5-14B).

You can use the following `bash` script to download the dataset:
```bash
bash download_eval_data.sh
```

**Step 2: Prepare the parameter configuration**

Here is the example:
**`config.yaml`**

```yaml
api_key: your_api_key # your API key for the model with API service. No need for open-source models.
data_root: /path/to/your/project/DataMind/eval/data # Root directory for data. (absolute path !!!)
```

**`run_eval.sh`**

```bash
python do_generate.py \
  --model_name DataMind-Qwen2.5-7B \  # Model name to use.
  --check_model gpt-4o-mini \  # Check model to use.
  --output results \  # Output directory path.
  --dataset_name QRData \  # Dataset name to use, chosen from QRData, DiscoveryBench.
  --max_round 25 \  # Maximum number of steps.
  --api_port 8000 \  # API port number, it is necessary if the local model is used.
  --bidx 0 \  # Begin index (inclusive), `None` indicates that there is no restriction.
  --eidx None \  # End index (exclusive), `None` indicates that there is no restriction.
  --temperature 0.0 \  # Temperature for sampling.
  --top_p 1 \  # Top p for sampling.
  --add_random False \  # Whether to add random files.
```

**(Optional)`local_model.sh`**

```bash
CUDA_VISIBLE_DEVICES=$i python -m vllm.entrypoints.openai.api_server \
  --model $MODEL_PATH \ # Local model path.
  --served-model-name $MODEL_NAME \ # The model name specified by you.
  --tensor-parallel-size $i \ # Set the size of tensor parallel processing.
  --port $port # API port number, which is consistent with the `api_port` above.
```

**Step 3: Run the shell script**

**(Optional)** Deploy the local model if you need.

```bash
bash local_model.sh
```

Run the shell script to start the process.

```bash
bash run_eval.sh
```



## 🎉Contributors

<a href="https://github.com/zjunlp/DataMind/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=zjunlp/DataMind" /></a>


We deeply appreciate the collaborative efforts of everyone involved. We will continue to enhance and maintain this repository over the long term. If you encounter any issues, feel free to submit them to us!



## ✍️ Citation

If you find our work helpful, please use the following citations.

```
@article{zhu2025open,
  title={Why Do Open-Source LLMs Struggle with Data Analysis? A Systematic Empirical Study},
  author={Zhu, Yuqi and Zhong, Yi and Zhang, Jintian and Zhang, Ziheng and Qiao, Shuofei and Luo, Yujie and Du, Lun and Zheng, Da and Chen, Huajun and Zhang, Ningyu},
  journal={arXiv preprint arXiv:2506.19794},
  year={2025}
}
```
