GPU_IDs=$1
DATA_PATH=$2
CKPT=$3

export CUDA_VISIBLE_DEVICES=$GPU_IDs

DATA_PATH=/home/maxinyu/data/KVG-Bench/kvg-bench.parquet
CKPT=/home/maxinyu/exp/R1-V/Qwen2-VL-7B-Instruct/oven-all-cot-grpo/checkpoint-800
OUT_DIR=$CKPT/kvg-bench-eval

# Evaluate DeepPerception on KVG-Bench
# It is recommended to use --vllm for faster inference

python evaluate.py \
    --data_path $DATASET \
    --ckpt_path $CKPT \
    --gpu_ids $GPU_IDs \
    --output_dir_path $OUT_DIR \
    --vllm \
    --prompt r1

# Evaluate Qwen2-VL on KVG-Bench
# DO NOT use --prompt r1
\
# python evaluate.py \
#     --data_path $DATASET \
#     --ckpt_path $CKPT \
#     --gpu_ids $GPU_IDs \
#     --output_dir_path $OUT_DIR \
#     --vllm