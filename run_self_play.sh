#!/bin/bash

# ./run_self_play.sh <cuda_id>
CUDA_VISIBLE_DEVICES=$1 python self_play.py \
    --retrieval_model bertretrieval \
    --method clustergreedy \
    --max_step 30 \
    --seed 50 \
    --multi_gpu $1 \
    --lang zh \
    --mode test \
    --history_length 3 \
    --recoder rest/self_play.txt \
    --talk_samples 256