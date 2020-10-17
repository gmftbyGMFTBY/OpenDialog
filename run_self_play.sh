#!/bin/bash

# ./run_self_play.sh <cuda_id>
CUDA_VISIBLE_DEVICES=$1 python self-play.py \
    --retrieval_model bertretrieval \
    --method greedy \
    --max_step 10 \
    --seed 20 \
    --multi_gpu $1 \
    --lang zh \
    --mode test \
    --history_length 3 \
    --talk_samples 64 | tee rest/self_play.txt