#!/bin/bash
# export FLASK_APP=api.py
# export FLASK_ENV=development
# CUDA_VISIBLE_DEVICES=$1 flask run --host=0.0.0.0 --port 8080

# python api.py <model_name> <cuda_id>
CUDA_VISIBLE_DEVICES=$2 python api.py $1 $2
