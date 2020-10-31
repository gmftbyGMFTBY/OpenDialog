from tqdm import tqdm
import json
import random
import ipdb
from itertools import chain
from elasticsearch import Elasticsearch
import argparse

'''
Transform this multi-turn dialog corpus into the same format of the douban300w and E-commerce dataset
'''

def parser_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--mode', default='train', type=str)
    parser.add_argument('--bsz', default=256, type=int)
    parser.add_argument('--samples', default=512, type=int)
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
                "collapse": {
                    "field": "keyword",     
                },
                'size': samples,
            })
        request = ''
        for each in search_arr:
            request += f'{json.dumps(each)} \n'
        rest = self.es.msearch(body=request)
        return rest
    
def read_text(path):
    with open(path) as f:
        dataset = []
        for i in f.read().split('\n\n'):
            if not i.strip():
                continue
            ctx, res = i.split('\n')
            ctx = [i.strip() for i in ctx.split('[SEP]')]
            dataset.append(ctx + [res])
    print(f'[!] collect {len(dataset)} samples for {args["mode"]} mode')
    return dataset

def write_file(dataset, path):
    with open(path, 'w') as f:
        responses = [i[-1] for i in dataset]
        for idx in tqdm(range(0, len(dataset), args['bsz'])):
            item = dataset[idx:idx+args['bsz']]
            msgs = [' [SEP] '.join(i[:-1]) for i in item]
            res = [i[-1] for i in item]
            
            write_data = []
            
            if args['mode'] == 'train':
                rest = [random.choice(responses) for _ in range(len(item))]
                for m, r, r_ in zip(msgs, res, rest):
                    m = m.replace(' [SEP] ', '\t')
                    write_data.append([f'1\t{m}\t{r}', f'0\t{m}\t{r_}'])
            else:   
                rest = chatter.multi_search(msgs, samples=args['samples'])['responses']
                for m, r, r_ in zip(msgs, res, rest):
                    ipdb.set_trace()
                    r_ = r_['hits']['hits']
                    neg = list(set([i['_source']['utterance'] for i in r_]) - set([r]))[:args['num_neg']]
                    try:
                        assert len(neg) == args['num_neg'], f'[!] cannot retrieval enough samples'
                    except:
                        ipdb.set_trace()
                    m = m.replace(' [SEP] ', '\t')
                    positive = f'1\t{m}\t{r}'
                    negative = [f'0\t{m}\t{i}' for i in neg]
                    write_data.append([positive] + negative)
            
            for dialog in write_data:
                for string in dialog:
                    f.write(f'{string}\n')

if __name__ == "__main__":
    args = parser_args()
    args = vars(args)
    print(args)
    
    args['num_neg'] = 1 if args['mode'] == 'train' else 10
    chatter = ESUtils('retrieval_database')
    
    dataset = read_text(f'{args["mode"]}.txt')
    write_file(dataset, f'{args["mode"]}_ir.txt')
    