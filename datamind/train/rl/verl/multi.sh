#!/usr/bin/env bash

# export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=1
export SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK=True
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_USE_V1=0
export WANDB_API_KEY="<your wandb api key>"
export SWANLAB_API_KEY="<your swanlab api key>"
export OPENAI_BASE_URL="<your openai api url>"
export OPENAI_API_KEY="<your openai api key>"

set -xeuo pipefail

# wandb
project_name='datamind'
exp_name='rl'

# actor update
use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0
clip_ratio_low=0.2
clip_ratio_high=0.28

# data
max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 8))

# loss
loss_agg_mode="token-mean"

# dapo
enable_filter_groups=True
filter_groups_metric=seq_final_reward
max_num_gen_batches=10
train_prompt_bsz=16
gen_prompt_bsz=$((train_prompt_bsz * 2)) ## batch size for generation
max_start_length=2048
max_obs_length=2048
rollout_n=4
n_sample=1 # grpo group size
train_prompt_mini_bsz=2

# rollout
rollout_mode=sync
rollout_engine_name=sglang

# reward
use_template_reward=true
use_execution_reward=False
answer_ratio=0.9
template_ratio=0.1
execution_ratio=0
enable_overlong_buffer=True
overlong_buffer_len=$((1024 * 4))
overlong_penalty_factor=1.0

# Ray
RAY_ADDRESS=${RAY_ADDRESS:-"http://localhost:8265"}
WORKING_DIR=${WORKING_DIR:-"your_verl_dir"} # the path of verl, such as /xxx/yyy/DataMind/train/RL/verl
RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/verl/trainer/runtime_env.yaml"}
NNODES=${NNODES:-16}

# Paths
HOME_DIR="your_home_dir" # the path of RL code, such as /xxx/yyy/DataMind/train/RL
RAY_DATA_HOME="${HOME_DIR}/verl"
MODEL_PATH="your_model_path"
CKPTS_DIR="your_ckpt_path"
TRAIN_FILE="the_train_file_path" # train.parquet path
TEST_FILE="the_test_file_path" # test.parquet path

# Rollout config
working_path='your_workspace_dir' # such as /xxx/yyy/workspace
working_tmp_path='your_workspace_dir/tmp' # such as /xxx/yyy/workspace/tmp
pred_csv_result_dir_parent="${working_path}/data/results"
gold_csv_results_dir="the_path_of_gold_csv_results" # such as /xxx/yyy/gold_csv_results
db_schema_data_path="the_path_of_db_schema.json" # such as /xxx/yyy/db_schema.json
csv_folder="the_path_of_csv_and_db" # train_files path, such as /xxx/yyy/train_files

# rollout path
validation_data_dir="${HOME_DIR}/verl/rollout/${exp_name}/val"
rollout_data_dir="${HOME_DIR}/verl/rollout/${exp_name}/train"

# sft params
sft_loss_type=mu
mu_warmup_steps=0
mu_decay_steps=350
mu_peak=0.9
mu_valley=0.05

# Algorithm
temperature=0.7
top_p=1
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout
val_top_p=0.95
val_temperature=0.7
val_batch_size=64
adv_estimator=grpo

# Performance Related Parameter
sp_size=4
use_dynamic_bsz=True
actor_ppo_max_token_len=$((max_prompt_length + max_response_length))
infer_ppo_max_token_len=$((max_prompt_length + max_response_length))
offload=False
gen_tp=1

# execute code
do_execute=False
max_turns=10
no_think_rl=False

# training config
n_gpus=8
epoch=1
test_freq=20
save_freq=1000
val_before_train=true
val_only=false

# multi_turn
CONFIG_PATH="${HOME_DIR}/verl/examples/sglang_multiturn/config"

ray stop
sleep 5
ray start --head --node-ip-address 0.0.0.0 --num-gpus ${n_gpus} --ray-debugger-external --port 6378

PYTHONUNBUFFERED=1 python3 -m recipe.dapo.main_dapo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.return_raw_chat=True \
    data.max_start_length=${max_start_length} \
    data.max_obs_length=${max_obs_length} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.gen_batch_size=${gen_prompt_bsz} \
    data.train_batch_size=${train_prompt_bsz} \
    data.val_batch_size=${val_batch_size} \
    data.working_dir=${working_path} \
    data.csv_folder=${csv_folder} \
    data.max_turns=${max_turns} \
    data.pred_csv_result_dir_parent=${pred_csv_result_dir_parent} \
    data.gold_csv_results_dir=${gold_csv_results_dir} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    algorithm.filter_groups.enable=${enable_filter_groups} \
    algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    algorithm.filter_groups.metric=${filter_groups_metric} \
    algorithm.no_think_rl=${no_think_rl} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=20 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    actor_rollout_ref.actor.sft_loss_type=${sft_loss_type} \
    actor_rollout_ref.actor.mu_warmup_steps=${mu_warmup_steps} \
    actor_rollout_ref.actor.mu_decay_steps=${mu_decay_steps} \
    actor_rollout_ref.actor.mu_peak=${mu_peak} \
    actor_rollout_ref.actor.mu_valley=${mu_valley} \
    actor_rollout_ref.rollout.n=${rollout_n} \
    actor_rollout_ref.rollout.n_sample=${n_sample} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k="${top_k}" \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${val_temperature} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.mode=${rollout_mode} \
    actor_rollout_ref.rollout.name=${rollout_engine_name} \
    actor_rollout_ref.rollout.db_schema_data_path=${db_schema_data_path} \
    actor_rollout_ref.rollout.working_path=${working_path} \
    actor_rollout_ref.rollout.working_tmp_path=${working_tmp_path} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    reward_model.reward_manager=dapo \
    reward_model.overlong_buffer.enable=${enable_overlong_buffer} \
    reward_model.overlong_buffer.len=${overlong_buffer_len} \
    reward_model.overlong_buffer.penalty_factor=${overlong_penalty_factor} \
    trainer.logger=['console','swanlab'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=${n_gpus} \
    trainer.nnodes=1 \
    trainer.val_before_train=${val_before_train} \
    trainer.val_only=${val_only} \
    trainer.test_freq=${test_freq} \
    trainer.save_freq=${save_freq} \
    trainer.total_epochs=${epoch} \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto \
    trainer.rollout_data_dir=${rollout_data_dir} \
    trainer.validation_data_dir=${validation_data_dir} \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_turns=${max_turns} \
    +actor_rollout_ref.rollout.multi_turn.tokenization_sanity_check_mode="off" \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="${HOME_DIR}/verl/examples/sglang_multiturn/config/tool_config/sandbox_fusion_tool_config.yaml" \
    do_execute=${do_execute} $@ 2>&1 | tee $exp_name.log