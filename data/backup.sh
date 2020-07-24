#!/bin/bash

# only backup the text file
rm -rf backup
mkdir -p backup
folder=(zh50w zhihu MTDialogRewritten ECG zhidao kgdialog kdconv doubangroup xiaohuangji weibo400w ptt qingyun11w douban300w topic)

for path in ${folder[@]}
do
    echo "[!] backup $path folder"
    mkdir -p backup/$path
    # copy file
    if [ $path = 'zh50w' ]; then
        cp $path/train.txt backup/$path/
        cp $path/train_.txt backup/$path/
        cp $path/test.txt backup/$path/
    elif [ $path = 'zhihu' ]; then
        cp $path/train.txt backup/$path/
        cp $path/train_.txt backup/$path/
    elif [ $path = 'MTDialogRewritten' ]; then
        cp $path/train.txt backup/$path/
        cp $path/corpus.txt backup/$path
    elif [ $path = 'ECG' ]; then
        cp $path/train.txt backup/$path
        cp $path/ecg_train_data.json backup/$path
        cp $path/ecg_test_data.xlsx backup/$path
    elif [ $path = 'zhidao' ]; then
        cp $path/train.txt backup/$path
        cp $path/filter_zhidao.txt backup/$path
    elif [ $path = 'kgdialog' ]; then
        cp $path/dev.txt backup/$path
        cp $path/test.txt backup/$path
        cp $path/train.txt backup/$path
        cp $path/train_.txt backup/$path
    elif [ $path = 'kdconv' ]; then
        cp $path/*.json backup/$path
        cp $path/train.txt backup/$path
    elif [ $path = 'doubangroup' ]; then
        cp $path/train.txt backup/$path
    elif [ $path = 'xiaohuangji' ]; then
        cp $path/test.txt backup/$path
        cp $path/train.txt backup/$path
    elif [ $path = 'weibo400w' ]; then
        cp $path/stc* backup/$path
        cp $path/train.txt backup/$path
    elif [ $path = 'ptt' ]; then
        cp $path/ptt.txt backup/$path
        cp $path/train.txt backup/$path
    elif [ $path = 'qingyun11w' ]; then
        cp $path/train.txt backup/$path
        cp $path/qingyun.csv backup/$path
    elif [ $path = 'douban300w' ]; then
        cp $path/dev.txt backup/$path
        cp $path/test.txt backup/$path
        cp $path/train.txt backup/$path
        cp $path/train_.txt backup/$path
    elif [ $path = 'topic' ]; then
        cp $path/train.txt backup/$path
    else
        echo "[!] got unknow dataset $path"
    fi
done
