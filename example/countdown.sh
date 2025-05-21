#!/usr/bin/env bash
model_name="nvidia/Llama-3_3-Nemotron-Super-49B-v1"
dataset_name="yzhuang/tinyzero-Countdown-Tasks-4"

# hyper-parameters
temperature=0.8
top_p=0.8
max_new_tokens=32768

out_dir="./results"

export MIXINPUTS_BETA=2.0

MIXINPUTS_LOGGING_LEVEL=INFO \
VLLM_USE_V1=1 \
TOKENIZERS_PARALLELISM=false \
PYTHONPATH="./:${PYTHONPATH:-}" \
python ./tinyzero_moi.py \
    --model_name "${model_name}" \
    --dataset_name "${dataset_name}" \
    --max_new_tokens "${max_new_tokens}" \
    --temperature "${temperature}" \
    --top_p "${top_p}" \
    --eager true \
    --output_dir "${out_dir}" \
    --debug
