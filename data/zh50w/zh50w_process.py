import csv
import random
from tqdm import tqdm
import ipdb
import sys
import pickle
sys.path.append('..')
from utils import read_stop_words
from collections import Counter
from gensim.summarization import bm25
from elasticsearch import Elasticsearch

'''
TODO
1. adding the reesponses into elasticsearch for q-r match
'''


class ESUtils:

    def __init__(self, index_name, create_index=False):
        self.es = Elasticsearch()
        self.index = index_name
        if create_index:
            mapping = {
                'properties': {
                    'context': {
                        'type': 'text',
                        'analyzer': 'ik_max_word',
                        'search_analyzer': 'ik_max_word'
                    }
                }
            }
            if self.es.indices.exists(index=self.index):
                self.es.indices.delete(index=self.index)
            rest = self.es.indices.create(index=self.index)
            print(rest)
            rest = self.es.indices.put_mapping(body=mapping, index=self.index)
            print(rest)

    def insert_pairs(self, pairs):
        count = self.es.count(index=self.index)['count']
        print(f'[!] begin of the idx: {count}')
        for i, qa in enumerate(tqdm(pairs)):
            data = {
                'context': qa[0],
                'response': qa[1]
            }
            self.es.index(index=self.index, body=data)
        print(f'[!] insert data over, whole size: {self.es.count(index=self.index)["count"]}')

class ESChat:

    def __init__(self, index_name):
        self.es = Elasticsearch()
        self.index = index_name

    def search(self, query, samples=10):
        dsl = {
            'query': {
                'match': {
                    'context': query
                }
            }
        }
        hits = self.es.search(index=self.index, body=dsl, size=samples)['hits']['hits']
        rest = []
        for h in hits:
            rest.append({'score': h['_score'], 'context': h['_source']['context'],
                'response': h['_source']['response']
            })
        return rest

    def chat(self):
        sentence = input('You are speaking: ').strip()
        while sentence:
            if sentence == 'exit':
                break
            rest = self.search(sentence)
            for idx, i in enumerate(rest):
                print(f'ESChat({idx}/{len(rest)}): {i["response"]}')
            sentence = input('You are speaking: ').strip()

def read_file(path):
    with open(path) as f:
        data = f.read()
    dialogs = data.split('\n\n')
    dialogs = [dialog.split('\n') for dialog in dialogs if dialog.strip()]
    random.shuffle(dialogs)
    return dialogs

def write_file(dialogs, mode='train', samples=10):
    chatbot = ESChat('retrieval_chatbot')
    with open(f'{mode}.csv', 'w') as f:
        f = csv.writer(f)
        f.writerow(['Context', 'Response'] + [f'Retrieval_{i+1}' for i in range(samples)])
        # f.writerow(['Context', 'Response'])
        error_counter = 0
        responses = [i[1] for i in dialogs]
        for dialog in tqdm(dialogs):
            rest = [i['response'] for i in chatbot.search(dialog[0], samples=samples+1)]
            if dialog[1] in rest:
                rest.remove(dialog[1])
            dialog = list(dialog) + rest
            if len(dialog) != samples + 2:
                error_counter += 1
            dialog.extend(random.sample(responses, samples+3-len(dialog)))
            # assert len(dialog) == samples + 2, f'{len(dialog)} retrieval utterances are obtained'
            f.writerow(dialog[:samples+2])
    print(f'[!] finish writing the file {mode}.csv, error counter: {error_counter}')

def process_data(dialogs, samples=10, max_len=10, max_utter_len=50):
    data = []
    for dialog in tqdm(dialogs):
        # dialog = [' '.join(list(jieba.cut(i))) for i in dialog]
        context, response = dialog[-(max_len+1):-1], dialog[-1]
        context = [i[-max_utter_len:] for i in context]
        context = ' <eou> '.join(context)
        data.append((context, response))
    return data

def retrieval_model():
    chatbot = ESChat('retrieval_chatbot')
    print(f'[!] load retrieval model from ElasticSearch, default 10 replys.')
    return chatbot


if __name__ == "__main__":
    import sys
    if sys.argv[1] == 'process':
        data = read_file('train.txt')
        whole_size = len(data)
        train_size = (0, int(0.95 * whole_size))
        dev_size = (train_size[1], train_size[1] + int(0.025 * whole_size))
        test_size = (dev_size[1], whole_size)
        print(f'data size: train({train_size[1]-train_size[0]}); dev({dev_size[1]-dev_size[0]}); test({test_size[1]-test_size[0]})')

        train_data = data[train_size[0]:train_size[1]]
        dev_data = data[dev_size[0]:dev_size[1]]
        test_data = data[test_size[0]:test_size[1]]
        
        train_data = process_data(train_data)
        dev_data = process_data(dev_data)
        test_data = process_data(test_data)
        # write file
        write_file(train_data, mode='train')
        write_file(dev_data, mode='dev')
        write_file(test_data, mode='test')
    else:
        # test elasticsearch
        # data = read_file('zh50w/train.txt')
        # pairs = [(' . '.join(i[:-1]), i[-1]) for i in data]
        # ut = ESUtils('retrieval_chatbot', create_index=True)
        # ut.insert_pairs(pairs)
        chatbot = ESChat('retrieval_chatbot')
        chatbot.chat()
