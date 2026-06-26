# Documentation
<a href="https://verl.readthedocs.io/en/latest/index.html"><b>Documentation</b></a>

# Quickstart:
[Installation](https://verl.readthedocs.io/en/latest/start/install.html)

## step 1:
Make sure your CUDA version satisfied:
- CUDA: Version >= 12.4

- cuDNN: Version >= 9.8.0

- Apex

CUDA above 12.4 is recommended to use as the docker image, please refer to [NVIDIA’s official website](https://developer.nvidia.com/cuda-toolkit-archive) for other version of CUDA.

```bash
# change directory to anywher you like, in verl source code directory is not recommended
wget https://developer.download.nvidia.com/compute/cuda/12.4.1/local_installers/cuda-repo-ubuntu2204-12-4-local_12.4.1-550.54.15-1_amd64.deb
dpkg -i cuda-repo-ubuntu2204-12-4-local_12.4.1-550.54.15-1_amd64.deb
cp /var/cuda-repo-ubuntu2204-12-4-local/cuda-*-keyring.gpg /usr/share/keyrings/
apt-get update
apt-get -y install cuda-toolkit-12-4
update-alternatives --set cuda /usr/local/cuda-12.4
```

cuDNN can be installed via the following command, please refer to  [NVIDIA’s official website](https://developer.nvidia.com/cuda-toolkit-archive) for other version of cuDNN.

```bash
# change directory to anywher you like, in verl source code directory is not recommended
wget https://developer.download.nvidia.com/compute/cudnn/9.8.0/local_installers/cudnn-local-repo-ubuntu2204-9.8.0_1.0-1_amd64.deb
dpkg -i cudnn-local-repo-ubuntu2204-9.8.0_1.0-1_amd64.deb
cp /var/cudnn-local-repo-ubuntu2204-9.8.0/cudnn-*-keyring.gpg /usr/share/keyrings/
apt-get update
apt-get -y install cudnn-cuda-12
```

## step 2:
Install dependencies
```bash
conda create -n verl python==3.10
conda activate verl
# Make sure you have activated verl conda env
# If you need to run with megatron
bash scripts/install_vllm_sglang_mcore.sh
# Or if you simply need to run with FSDP
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
```

## step 3:
post installation
use `pip install` to install packages below:
- pebble
- timeout_decorator

if you are using sglang, use `conda install` to install packages below:
- CUDAtoolkit

## step 4:
Please make sure that the installed packages are not overridden during the installation of other packages.
- torch and torch series

- vLLM

- SGLang

- pyarrow

- tensordict

- nvidia-cudnn-cu12: For Magetron backend

# Reinforcement Learning Training Entrance File
- To train DAPO : use `recipe/dapo/main_dapo.py`
- To train GRPO : use `verl/trainer/main_ppo.py`