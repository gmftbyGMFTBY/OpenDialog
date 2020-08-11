import sys
import argparse
from tqdm import tqdm
sys.path.append('..')
from models import *
import numpy as np
import random
import jieba
import json
import os
import ipdb
from utils import *

'''
1. Retrieval dialog 
Collect all the training samples, store it into the ElasticSearch database

2. Generative dialog
Collect the high-quality training samples for GPT2 Language Model

Elasticsearch save the retrieval and generative responses, q-a match and q-q match
* single-turn dialog: q-a and q-q match
* multi-turn dialog: q-a match
'''

retrieval_datasets = ['ECG', 'zhihu', 'zhidao', 'kgdialog', 'kdconv', 'zh50w', 'doubangroup']
# do not use zhihu dataset for generative dialog, which leads to the long but nonsense generated results
generative_datasets = ['kgdialog', 'kdconv', 'zh50w', 'xiaohuangji', 'doubangroup']

single_turn = ['ECG', 'zhihu', 'xiaohuangji', 'zhidao', 'weibo400w', 'qingyun11w', 'ptt', 'doubangroup']
multi_turn = ['kgdialog', 'kdconv', 'zh50w', 'douban300w']

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='insert')
    parser.add_argument('--dataset', type=str, default='train_retrieval')
    return parser.parse_args()

def read_dataset(name):
    path = f'{name}/train.txt'
    with open(path) as f:
        data = f.read()
        # data = data.replace('[SEP]', ' [SEP] ')
        data = data.split('\n\n')
        dialogs = [i.split('\n') for i in data if i.strip()]
        # make sure the [SEP] have the spaces around them
    print(f'[!] read dataset {name} over, get {len(dialogs)} dialogs')
    return dialogs

def read_dataset_(name):
    path = f'{name}/train_.txt'
    with open(path) as f:
        data = f.read()
        data = data.split('\n\n')
        utterances = []
        for i in data:
            if i.strip():
                utterances.extend(i.split('\n'))
    utterances = list(set(utterances))
    print(f'[!] read dataset {name} over, get {len(utterances)} utterances')
    return utterances

def filter_useless(pairs):
    words = ['图片评论', '如图']
    rest = []
    for pair in pairs:
        if len(pair[0]) > 300 or len(pair[1]) > 300:
            continue
        if len(pair[0]) < 3 or len(pair[1]) < 3:
            continue
        for word in words:
            if word in pair[1] or word in pair[0]:
                break
        else:
            rest.append(pair)
    return rest

def make_pairs(dialogs, qa=True):
    if qa:
        pairs = []
        for dialog in dialogs:
            pairs.append((' [SEP] '.join(dialog[:-1]), dialog[-1]))
    else:
        pairs = []
        for dialog in dialogs:
            for utterance in dialog:
                pairs.append((utterance, utterance))
    return pairs

def collect_samples_qq(index_name):
    '''
    if insert new dataset, just set parameters `create_index` as False
    '''
    tool = ESUtils(index_name, create_index=True)
    print(f'[!] {index_name} elasticsearch database created')
    def insert_data(pairs):
        tool.insert_pairs(pairs)
        print(f'{tool.es.count(index=index_name)["count"]} utterances in database')
    for i in single_turn:
        insert_data(filter_useless(make_pairs(read_dataset(i), qa=True)))
        print(f'[!] finish insert the {i} dataset')
    for i in multi_turn:
        insert_data(filter_useless(make_pairs(read_dataset(i), qa=True)))
        print(f'[!] finish insert the {i} dataset')

# ======== only insert the zh50w dataset into collect_sample_qq ========== #
def collect_samples_qq_zh50w(index_name):
    '''
    if insert new dataset, just set parameters `create_index` as False
    '''
    tool = ESUtils(index_name, create_index=True)
    print(f'[!] {index_name} elasticsearch database created')
    def insert_data(pairs):
        tool.insert_pairs(pairs)
        print(f'{tool.es.count(index=index_name)["count"]} utterances in database')
    i = 'zh50w'
    insert_data(filter_useless(make_pairs(read_dataset(i), qa=True)))
    print(f'[!] finish insert the {i} dataset')

def collect_samples_qa(index_name):
    '''
    QA pairs use the sentences from the zhihu
    '''
    tool = ESUtils(index_name, create_index=False)
    def insert_data(pairs):
        tool.insert_pairs(pairs)
        print(f'{tool.es.count(index=index_name)["count"]} utterances in database')

if __name__ == "__main__":
    args = parse_args()
    args = vars(args)
    if args['mode'] == 'insert':
        # collect_samples_qq('retrieval_database')
        collect_samples_qq('retrieval_database')
        # collect_samples_qq_zh50w('zh50w_database')
    elif args['mode'] == 'generative':
        # train, test mode (99:1), without dev
        # prepare the generative dataset
        pairs = []
        for dname in tqdm(generative_datasets):
            pairs.extend(filter_useless(make_pairs(read_dataset(dname), qa=True)))
        random.shuffle(pairs)
        train_pairs, test_pairs = pairs[:-5000], pairs[-5000:]
        # write train
        with open(f'train_generative/train.txt', 'w') as f:
            stat = {'context': [], 'response': []}
            for pair in train_pairs:
                if len(pair[0]) <= 1:
                    continue
                f.write(f'{pair[0]}\n')
                f.write(f'{pair[1]}\n\n')
                stat['context'].append(len(pair[0]))
                stat['response'].append(len(pair[1]))
            print(f'[!] train average context length: {np.mean(stat["context"])}')
            print(f'[!] train max context length: {np.max(stat["context"])}')
            print(f'[!] train min context length: {np.min(stat["context"])}')
            print(f'[!] train average response length: {np.mean(stat["response"])}')
            print(f'[!] train max response length: {np.max(stat["response"])}')
            print(f'[!] train min response length: {np.min(stat["response"])}')
            print(f'[!] train dataset size: {len(train_pairs)}')
        # write test
        with open(f'train_generative/test.txt', 'w') as f:
            stat = {'context': [], 'response': []}
            for pair in test_pairs:
                if len(pair[0]) <= 1:
                    continue
                f.write(f'{pair[0]}\n')
                f.write(f'{pair[1]}\n\n')
                if len(pair[0]) == 0:
                    ipdb.set_trace()
                stat['context'].append(len(pair[0]))
                stat['response'].append(len(pair[1]))
            print(f'[!] test average context length: {np.mean(stat["context"])}')
            print(f'[!] test max context length: {np.max(stat["context"])}')
            print(f'[!] test min context length: {np.min(stat["context"])}')
            print(f'[!] test average response length: {np.mean(stat["response"])}')
            print(f'[!] test max response length: {np.max(stat["response"])}')
            print(f'[!] test min response length: {np.min(stat["response"])}')
            print(f'[!] test dataset size: {len(test_pairs)}')
    elif args['mode'] == 'retrieval':
        # prepare the retrieval dataset
        # do not need all the dialogs trained, high-quality is enough 
        pairs = []
        for dname in tqdm(retrieval_datasets):
            pairs.extend(filter_useless(make_pairs(read_dataset(dname), qa=True)))
        random.shuffle(pairs)
        train_pairs, test_pairs = pairs[:-5000], pairs[-5000:]
        # write train
        with open(f'train_retrieval/train.txt', 'w') as f:
            stat = {'context': [], 'response': []}
            for pair in train_pairs:
                if len(pair[0]) <= 1:
                    continue
                f.write(f'{pair[0]}\n')
                f.write(f'{pair[1]}\n\n')
                stat['context'].append(len(pair[0]))
                stat['response'].append(len(pair[1]))
            print(f'[!] train average context length: {np.mean(stat["context"])}')
            print(f'[!] train max context length: {np.max(stat["context"])}')
            print(f'[!] train min context length: {np.min(stat["context"])}')
            print(f'[!] train average response length: {np.mean(stat["response"])}')
            print(f'[!] train max response length: {np.max(stat["response"])}')
            print(f'[!] train min response length: {np.min(stat["response"])}')
            print(f'[!] train dataset size: {len(train_pairs)}')
        # write test 
        with open(f'train_retrieval/test.txt', 'w') as f:
            stat = {'context': [], 'response': []}
            for pair in test_pairs:
                if len(pair[0]) <= 1:
                    continue
                f.write(f'{pair[0]}\n')
                f.write(f'{pair[1]}\n\n')
                stat['context'].append(len(pair[0]))
                stat['response'].append(len(pair[1]))
            print(f'[!] test average context length: {np.mean(stat["context"])}')
            print(f'[!] test max context length: {np.max(stat["context"])}')
            print(f'[!] test min context length: {np.min(stat["context"])}')
            print(f'[!] test average response length: {np.mean(stat["response"])}')
            print(f'[!] test max response length: {np.max(stat["response"])}')
            print(f'[!] test min response length: {np.min(stat["response"])}')
            print(f'[!] test dataset size: {len(test_pairs)}')
    elif args['mode'] == 'topic':
        # use the doubangroup dataset generate the topic dataset for short text classification
        # context and response are all used
        data = []
        for topic in ['electric', 'food', 'movie', 'music', 'sport']:
            path = f"doubangroup/{topic}"
            for file in tqdm(os.listdir(path)):
                fp = f'{path}/{file}'
                with open(fp) as f:
                    item = json.load(f)
                    utterances = [item['topic']['content']] + [i['content'] for i in item['replys']]
                    for i in utterances:
                        i = list(jieba.cut(i)) + [f'__label__{topic}']
                        data.append(' '.join(i))

        print(f'[!] obtain {len(data)} training sampls')
        # write file
        with open('topic/train.txt', 'w') as f:
            for i in data:
                f.write(f'{i}\n')
    elif args['mode'] == 'foresee':
        # agent = BERTRetrievalAgent('5', kb=False)
        test_agent = TestAgent(kb=False)
        # agent.load_model('ckpt/zh50w/bertretrieval/best.pt')
        pairs = filter_useless(make_pairs(read_dataset(f'data/kdconv'), qa=True))
        pairs.extend(filter_useless(make_pairs(read_dataset(f'data/kgdialog'), qa=True)))
        data_p, data_n = [], []
        # positive sample
        for dialog in tqdm(pairs):
            res = test_agent.talk(None, f'{dialog[0]} [SEP] {dialog[1]}')
            data_p.append((dialog[0], dialog[1], res))
        # negative sample
        for dialog in tqdm(pairs):
            # replace the response with the retrieval results
            response = test_agent.talk(None, dialog[0])
            res = test_agent.talk(None, f'{dialog[0]} [SEP] {response}')
            data_n.append((dialog[0], response, res))
    elif args['mode'] == 'keywords':
        # generate the keywords of the dataset
        data = read_dataset(args['dataset'])
        graph = KeywordsCollector()
        graph.process_dataset(data)
        graph.save(f'{args["dataset"]}/graph.pkl')
    elif args['mode'] == 'stat':
        # statistical the information of the dataset (dailydialog, dstc7, empchat, personachat)
        dataset_names = ['dailydialog', 'dstc7', 'empchat', 'personachat']
        for name in dataset_names:
            data = read_dataset(name)
            length = []
            for dialog in data:
                l = 0
                for s in dialog:
                    l += len(s.split())
                length.append(l)
            print(f'[!] the average length of the {name} is {round(np.mean(length), 4)}')
    elif args['mode'] == 'EDA':
        contexts, responses = read_dataset_eda(args['dataset'])
        print(f'[!] begin to apply EDA data augmentation for {args["dataset"]} dataset')
        with open(f'{args["dataset"]}/train_eda.txt', 'w') as f:
            for c, r in tqdm(list(zip(contexts, responses))):
                aug_sentences = gen_eda(r, alpha=0.5)
                c = [c] + [r] + aug_sentences
                for item in c:
                    f.write(f'{item}\n')
                f.write('\n')
