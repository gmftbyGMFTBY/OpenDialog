#!/bin/bash

mode=$1

if [ $mode = 'init_es' ]; then
    # init the ElasticSearch and restore the retrieval database
    python process_data.py --mode insert
elif [ $mode = 'init_gen' ]; then
    # init and create the whole generative dataset
    python process_data.py --mode generative
elif [ $mode = 'init_ir' ]; then
    # init and create the retrieval dataset
    python process_data.py --mode retrieval
elif [ $mode = 'topic' ]; then
    # init and generate the topic model for SMP-MCC 2020
    python process_data.py --mode topic
elif [ $mode = 'eda' ]; then
    # use EDA data augmentation techiques for the special dataset
    python process_data.py --mode EDA --dataset zh50w 
elif [ $mode = 'keywords' ]; then
    # generate the keywords graph
    python process_data.py --mode keywords --data zh50w
else
    echo '[!] run.sh script get unkown mode [ $mode ] for running'
fi
