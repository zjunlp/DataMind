# Configuration Presets

Configuration presets for different research scenarios.

## Available Configurations

### `default.env` - Default Settings (3 minutes)
```bash
EXECUTION_TIMEOUT=180
```
Standard timeout for most experiments.

### `quick-tests.env` - Quick Testing (1 minute)
```bash
EXECUTION_TIMEOUT=60
```
For rapid iteration and debugging.

### `long-running.env` - Long Experiments (1 hour)
```bash
EXECUTION_TIMEOUT=3600
```
For training models or long-running data processing.

## Usage

### Option 1: Direct Command Line
```bash
python generate_compose.py \
  --num-containers 8 \
  --types executor-prebuilt-kaggle-gpu:8 \
  --gpu-ids 0,1,2,3,4,5,6,7 \
  --env "EXECUTION_TIMEOUT=3600" \
  -o docker-compose.yml
```

### Option 2: Using Config Files
```bash
python generate_compose.py \
  --num-containers 8 \
  --types executor-prebuilt-kaggle-gpu:8 \
  --gpu-ids 0,1,2,3,4,5,6,7 \
  --env "EXECUTION_TIMEOUT=3600" \
  -o docker-compose.yml
```

### Option 3: Custom Timeout Per Experiment
Set different timeouts for different experiments:

```bash
# 2 hours for deep learning training
python generate_compose.py --env "EXECUTION_TIMEOUT=7200" ...

# 30 seconds for unit tests
python generate_compose.py --env "EXECUTION_TIMEOUT=30" ...

# 10 minutes for data preprocessing
python generate_compose.py --env "EXECUTION_TIMEOUT=600" ...
```

## Adding More Configurations

You can add custom environment variables:

```bash
python generate_compose.py \
  --env "EXECUTION_TIMEOUT=3600,MY_CUSTOM_VAR=value,ANOTHER_VAR=value2" \
  ...
```

All environment variables are passed to the containers.

