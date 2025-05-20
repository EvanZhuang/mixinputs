#!/usr/bin/env bash
model_name="Qwen/QwQ-32B"
dataset_name=""AI-MO/aimo-validation-aime""

# hyper-parameters
temperature=0.8
top_p=0.8
max_new_tokens=32768

out_dir="./results"

MIXINPUTS_LOGGING_LEVEL=INFO \
VLLM_USE_V1=1 \
TOKENIZERS_PARALLELISM=false \
PYTHONPATH="./:${PYTHONPATH:-}" \
python ./aime_moi.py \
    --model_name "${model_name}" \
    --dataset_name "${dataset_name}" \
    --max_new_tokens "${max_new_tokens}" \
    --temperature "${temperature}" \
    --top_p "${top_p}" \
    --eager true \
    --output_dir "${out_dir}" \
    --debug
