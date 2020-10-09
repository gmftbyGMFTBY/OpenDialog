#!/bin/bash
# python api.py <model_name> <cuda_id>
CUDA_VISIBLE_DEVICES=$2 python api.py $1 $2
