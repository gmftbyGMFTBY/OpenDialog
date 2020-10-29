from tqdm import tqdm
import json
import random
import ipdb
from itertools import chain
from elasticsearch import Elasticsearch
import argparse

'''
expand more high-quality negative samples for testing retrieval dialog systems (not only the original 10 negative samples)
'''

def parser_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--num', default=100, type=int)
    parser.add_argument('--bsz', default=256, type=int)
    parser.add_argument('--dataset', default='ecommerce', type=str)
    return parser.parse_args()

class ESUtils:
    
    def __init__(self, index_name):
        self.es = Elasticsearch(http_auth=('elastic', 'elastic123'), timeout=30)
        self.index = index_name

    def multi_search(self, msgs, samples=10):
        search_arr = []
        for msg in msgs:
            search_arr.append({'index': self.index})
            # https://elasticsearch.cn/article/132
            search_arr.append({
                'query': {
                    'bool': {
                        'should': [{'match': {'utterance': {'query': msg.replace('[SEP]', '')}}}],
                    }
                },
                'size': samples,
            })
        request = ''
        for each in search_arr:
            request += f'{json.dumps(each)} \n'
        rest = self.es.msearch(body=request)
        return rest

def read_test():
    path = f'{args["dataset"]}/test.txt'
    with open(path) as f:
        lines = []
        for line in f.readlines():
            line = line.strip().split('\t')
            label, ctx, res = int(line[0]), line[1:-1], line[-1]
            ctx = ' [SEP] '.join([''.join(i.split()) for i in ctx])
            res = ''.join(res.split())
            lines.append((label, ctx, res))
    dataset = []
    for idx in range(0, len(lines), 10):
        dataset.append(lines[idx:idx+10])
    return dataset

def expand(dialogs):
    msgs = [dialog[0][1] for dialog in dialogs]
    responses = [[i[2] for i in dialog] for dialog in dialogs]
    rest = searcher.multi_search(msgs, samples=args['num']*5)['responses']
    
    new_dialogs, step = [], 0
    for m, r, r_ in tqdm(list(zip(msgs, responses, rest))):
        m = m.replace(' [SEP] ', '\t')
        r_ = r_['hits']['hits']
        p = []
        for idx, i in enumerate(r):
            if idx == 0:
                p.append((1, m, i))
            else:
                p.append((0, m, i))
        utterance_r = list(set([i['_source']['utterance'] for i in r_]) - set(r))
        left = args['num'] - len(utterance_r)
        if left > 0:
            # random sample
            item = list(set(chain(*responses)) - set(utterance_r) - set(r))
            utterance_r.extend(random.sample(item, left))
        else:
            utterance_r = random.sample(utterance_r, args['num'])
        p.extend([(0, m, i) for i in utterance_r])
        new_dialogs.append(p)
        step += 1
    return new_dialogs

if __name__ == "__main__":
    args = parser_args()
    args = vars(args)
    print(args)
    
    searcher = ESUtils('retrieval_database')
    dataset = read_test()
    
    with open(f'{args["dataset"]}/test_{args["num"]+10}.txt', 'w') as f:
        for idx in tqdm(range(0, len(dataset), args['bsz'])):
            dialogs = dataset[idx:idx+args['bsz']]
            new_dialogs = expand(dialogs)
            for dialog in new_dialogs:
                for item in dialog:
                    f.write(f'{item[0]}\t{item[1]}\t{item[2]}\n')
    print(f'[!] expand {args["num"]} negative samples into file {args["dataset"]}/test_{args["num"]+10}.txt')
