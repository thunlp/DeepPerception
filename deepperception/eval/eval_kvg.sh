export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
GPU_IDs="0,1,2,3,4,5,6,7"

DATA_PATH=/home/maxinyu/data/KVG-Bench/kvg-bench.parquet
CKPT=/home/maxinyu/exp/R1-V/Qwen2-VL-7B-Instruct/oven-all-cot-grpo/checkpoint-800
OUT_DIR=$CKPT/kvg-bench-eval

python evaluate.py \
    --data_path $DATASET \
    --ckpt_path $CKPT \
    --gpu_ids $GPU_IDs \
    --output_dir_path $OUT_DIR \
    --vllm \
    --prompt r1