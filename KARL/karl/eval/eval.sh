export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_DISABLE_USAGE_STATS=1


# KVG-Bench
GPU_IDs="0,1,2,3,4,5,6,7"
DATA_PATH=./data/KVG-Bench/kvg-bench.parquet

CKPT=/path/to/your/checkpoint/Qwen3-VL-8B-Instruct

# KVG-Bench
OUT_DIR=/path/to/your/eval_output/directory

# Evaluate DeepPerception
# To ensure precise reproduction of the experimental results of KVG-Bench presented in the paper, please strictly adhere to the package versions specified in the requirements.txt file and DO NOT use the vllm.

python evaluate.py \
    --data_path $DATA_PATH \
    --ckpt_path $CKPT \
    --gpu_ids $GPU_IDs \
    --output_path $OUT_DIR \
    --infer_model qwen3_vl \
    --vllm \
    --seed 20 \
    --prompt r1




