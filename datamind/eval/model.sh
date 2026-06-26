export CUDA_VISIBLE_DEVICES=0,1
export VLLM_USE_V1=0

python3 -m vllm.entrypoints.openai.api_server \
    --model <your model path> \
    --host 0.0.0.0 \
    --port 33038 \
    --dtype bfloat16 \
    --trust-remote-code \
    --tensor-parallel-size 2 \
    --served_model_name 'datamind' \