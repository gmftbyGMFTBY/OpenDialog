#!/bin/bash

# ./run_self_play.sh <cuda_id>
CUDA_VISIBLE_DEVICES=$1 python self-play.py \
    --model bertretrievalkg \
    --retrieval_model bertretrieval \
    --method greedy \
    --max_step 20 \
    --seed 20 \
    --multi_gpu $1 \
    --lang zh \
    --mode test \
    --history_length 5