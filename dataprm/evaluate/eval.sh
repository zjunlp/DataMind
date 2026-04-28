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