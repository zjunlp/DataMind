# Reason Task

The code for this project is the reason task (DABStep) from the [DataCOPE](https://arxiv.org/pdf/2606.06416) paper.

## Environment

Our framework primarily depends on DSGym and Claude Code SDK. We recommend Python 3.12 or later.

1. DSGym Env
   1. Install DSGym Environment
      ```bash
      pip install -r requirements.txt
      ```
   2. Launch servers. We recommend using a different environment to deploy the server.
      ```bash
      pip install -r env_requirements.txt

      cd generation/dataagent/DSGym/executors/container_images/instance
      bash start.sh

      cd ../../
      python manager/main.py  # default using 5005 port
      ```

2. Claude Code Env
   1. Install Claude Code
      ```bash
      curl -fsSL https://claude.ai/install.sh | bash
      ```

## Data
The DABStep Data can be found in `https://github.com/fannie1208/DSGym/tree/main/data/data/dabstep`. You can download it and put it in `generation/dataagent/DSGym/data/data/dabstep`,  `generation/skill_manager/workspace/data/dabstep` and `eval/DSGym/data/data/dabstep`. We also directly provide the data files in the project.

## Overview

The project consists of two main stages:

### 1. Generation (`generation/`)

Generates and iteratively refines data-analytic skills through a three-iteration loop on the explore split.

```bash
cd generation
python run_pipeline.py --dry-run          # preview all steps
python run_pipeline.py                    # full run
python run_pipeline.py --start-from iter2_explore  # resume from a step
```

See [`generation/README.md`](generation/README.md) for more details.

### 2. Evaluation (`eval/`)

Evaluates skill quality on the test split

```bash
cd eval
python run_eval.py --skill-dir ./skill/2/iter2 --dry-run   # preview
python run_eval.py --skill-dir ./skill/2/iter2              # full eval
```

See [`eval/README.md`](eval/README.md) for more details.


## Quick Start

We provide pre-generated skills under `eval/skill/`, organized as:

```
eval/skill/
├── 0/                          # Experiment run 0
│   ├── iter0/                  
│   ├── iter1/                  
│   └── iter2/                  
├── 1/                          # Experiment run 1
│   ├── iter0/
│   ├── iter1/
│   └── iter2/
└── 2/                          # Experiment run 2
    ├── iter0/
    ├── iter1/
    └── iter2/
```

Each `iter*/` directory contains one skill file per category (9 categories total). You can directly use this skill to reproduce the results of the paper：

```bash
cd eval
python run_eval.py --skill-dir ./skill/2/iter2
```

## Acknowledgements
Our framework references [DSGym](https://github.com/fannie1208/DSGym), and we are grateful for their contributions!