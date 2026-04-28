export CUDA_VISIBLE_DEVICES=0,1,2,3
vllm serve 'path_to_dataprm' \
    --served_model_name 'dataprm' \
    --port 19002 \
    --gpu_memory_utilization 0.95 \
    --tensor-parallel-size 4