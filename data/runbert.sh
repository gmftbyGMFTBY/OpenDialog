#!/bin/bash

# use cuda 9.0 for bert-as-service
LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64/:$LD_LIBRARY_PATH
echo "[!] config: $LD_LIBRARY_PATH"

if [ $1 = "" ]; then
    lang="zh"
else
    lang=$1
fi

if [ $lang == "zh" ]; then
    echo "Start chinese bert model"
    bert-serving-start -model_dir bert_model/chinese_L-12_H-768_A-12 \
        -num_worker=3 \
        -device_map 3 4 7 \
        -max_seq_len 200 \
        -pooling_strategy REDUCE_MEAN
else
    echo "Start english bert model"
    bert-serving-start -model_dir bert_model/uncased_L-12_H-768_A-12 \
        -num_worker=2 \
        -device_map 0 1 \
        -max_seq_len 200 \
        -pooling_strategy REDUCE_MEAN
fi
