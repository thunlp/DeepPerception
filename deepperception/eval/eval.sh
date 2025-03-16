GPU_IDs=$1
DATA_PATH=$2
CKPT=$3


if [[ $DATA_PATH == *"kvg-bench"* ]]; then 
# KVG-Bench
OUT_DIR=$CKPT/kvg-bench-eval

# Evaluate DeepPerception
# To ensure precise reproduction of the experimental results of KVG-Bench presented in the paper, please strictly adhere to the package versions specified in the requirements.txt file and DO NOT use the vllm.

python evaluate.py \
    --data_path $DATA_PATH \
    --ckpt_path $CKPT \
    --gpu_ids $GPU_IDs \
    --output_path $OUT_DIR \
    --prompt r1 

# Evaluate Qwen2-VL
# DO NOT use --prompt r1, which requires model to first output the thinking process

# python evaluate.py \
#     --data_path $DATASET \
#     --ckpt_path $CKPT \
#     --gpu_ids $GPU_IDs \
#     --output_path $OUT_DIR 
else 
# TODO
# FGVR
OUT_DIR=$CKPT/fgvr-eval

# Evaluate DeepPerception-FGVR

python evaluate.py \
    --data_path $DATA_PATH \
    --ckpt_path $CKPT \
    --gpu_ids $GPU_IDs \
    --output_path $OUT_DIR \
    --vllm \
    --prompt r1

# Evaluate Qwen2-VL

# python evaluate.py \
#     --data_path $DATA_PATH \
#     --ckpt_path $CKPT \
#     --gpu_ids $GPU_IDs \
#     --output_path $OUT_DIR \
#     --vllm
fi