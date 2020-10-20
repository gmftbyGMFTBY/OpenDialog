#!/bin/bash

CUDA_VISIBLE_DEVICES=$2 python api.py \
    --model $1 \
    --gpu_id $2 \
    --chat_mode 0 \
    --multi_turn_size 5 \
    --verbose
