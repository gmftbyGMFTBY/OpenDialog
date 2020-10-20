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
    --history_length 2 \
    --min_topic_length 6 \
    --max_topic_length 7 \
    --talk_samples 128 | tee rest/self_play.txt