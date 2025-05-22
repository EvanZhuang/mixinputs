export VLLM_USE_V1=1
export MIXINPUTS_BETA=2.0

vllm serve Qwen/QwQ-32B --tensor-parallel-size 8 --enforce-eager --max-seq-len-to-capture 32768 --max-num-seqs 64