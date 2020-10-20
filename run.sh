#!/bin/bash

# ========== How to run this script ========== #
# ./run.sh <train/test> <dataset_name> <model_name> <cuda_ids>
# for example: ./run/sh train train_generative gpt2 0,1,2,3

mode=$1
dataset=$2
model=$3
cuda=$4 

if [ $mode = 'init' ]; then
    models=(topicprediction seq2seq transformer bert_na bertirbi uni lccc lcccir bertmcf bertmc DualLSTM pone pfgpt2 kwgpt2 when2talk gpt2retrieval decouple_gpt2gan gpt2_mmi gpt2 gpt2v2 gpt2v2rl bertretrieval_multiview bertretrieval_cl bertretrieval bertretrieval_dis bertlogic gpt2gan gpt2lm)
    datasets=(qingyun LCCC STC douban300w when2talk empchat dstc7 personachat dailydialog cornell xiaohuangji tencent LM zh50w train_retrieval mutual decouple_rl train_generative train_generative_rl)
    mkdir bak ckpt rest
    for m in ${models[@]}
    do
        for d in ${datasets[@]}
        do
            mkdir -p ckpt/$d/$m
            mkdir -p rest/$d/$m
            mkdir -p bak/$d/$m
        done
    done
    if [ ! -d "data/train_generative" ]; then
        mkdir -p data/train_generative
    fi
    if [ ! -d "data/train_retrieval" ]; then
        mkdir -p data/train_retrieval
    fi
    # two necessary folder of multiview module
    mkdir -p ckpt/NIDF_TF
    mkdir -p ckpt/fasttext
elif [ $mode = 'backup' ]; then
    # rm bak/$dataset/$model/*
    cp ckpt/$dataset/$model/param.txt bak/$dataset/$model/
    cp ckpt/$dataset/$model/best.pt bak/$dataset/$model/
    cp rest/$dataset/$model/rest.txt bak/$dataset/$model/
    cp rest/$dataset/$model/events* bak/$dataset/$model/
elif [ $mode = 'irdata' ]; then
    python utils.py \
        --dataset $dataset \
        --mode irdata \
        --batch_size 512
elif [ $mode = 'train' ]; then
    ./run.sh backup $dataset $model
    rm ckpt/$dataset/$model/*
    rm rest/$dataset/$model/events*    # clear the tensorboard cache

    english_datasets=(mutual dstc7 empchat dailydialog personachat cornell)
    if [[ ${english_datasets[@]} =~ $dataset ]]; then
        lang='en'
    else
        lang='zh'
    fi
    
    gpu_ids=(${cuda//,/ })
    CUDA_VISIBLE_DEVICES=$cuda python -m torch.distributed.launch --nproc_per_node=${#gpu_ids[@]} --master_addr 127.0.0.1 --master_port 29500 main.py \
        --dataset $dataset \
        --model $model \
        --mode train \
        --batch_size 64 \
        --n_vocab 80000 \
        --epoch 5 \
        --seed 50 \
        --src_len_size 256 \
        --tgt_len_size 50 \
        --multi_gpu $cuda \
        --lang $lang
elif [ $mode = 'test' ]; then
    one_batch_model=(kwgpt2 pfgpt2 gpt2gan lccc multigpt2 when2talk bertirbi)
    if [[ ${one_batch_model[@]} =~ $model ]]; then
        batch_size=1
    else
        batch_size=32
    fi
    
    english_datasets=(mutual dstc7 empchat dailydialog personachat)
    if [[ ${english_datasets[@]} =~ $dataset ]]; then
        lang='en'
    else
        lang='zh'
    fi

    CUDA_VISIBLE_DEVICES=$cuda python main.py \
        --dataset $dataset \
        --model $model \
        --mode test \
        --n_vocab 70000 \
        --batch_size $batch_size \
        --src_len_size 256 \
        --tgt_len_size 50 \
        --seed 30 \
        --multi_gpu $cuda \
        --lang $lang
elif [ $mode = 'eval' ]; then
    python evalp.py \
        --dataset $dataset \
        --model $model
else
    echo "[!] mode needs to be train/test/eval, but got $mode"
fi
