import random
import numpy as np
import torch
from tqdm import tqdm
import ipdb
import json
import numpy as np
from gensim.summarization import bm25
from bert_serving.client import BertClient
from .fluency_perturbation import *
import jieba

def generate_negative_samples_fluency(r, responses, samples=10, ratio=0.2, vocab=None):
    '''
    Retrieval the samples from responses and make the perturbation on them
    to generate the unfluency negative samples for training the bert_multiview model

    Fluency perturbation:
    1. shuffle
    2. random drop
    3. duplication
    4. replace
    '''
    negative = generate_negative_samples(r, responses, samples=samples)    # samples
    contains = []
    for i in negative:
        p = [
                shuffle(i), drop(i, ratio=ratio), 
                duplication(i, ratio=ratio), replace(vocab, i, ratio=ratio)
            ]
        contains.append(random.choice(p))
    return contains

def generate_negative_samples_diversity(r, responses, samples=10, nidf=None, temp=0.05):
    '''
    1. Distinct: prefer short response
    2. Length: prefer long response
    3. NIDF
    '''
    negative = random.sample(responses, 512)    # sampling pool size
    if not nidf:
        raise Exception(f'[!] need the NIDF diversity model')
    scores = nidf.scores(negative, topk=5)
    # apply the softmax with the temperature
    scores = torch.tensor(scores) / temp 
    scores = torch.nn.functional.softmax(scores, dim=0).numpy()
    # renorm to obtain the 1 probability
    scores /= sum(scores)
    # select the lowest diversity scores as the diversity negative samples
    index = np.random.choice(range(len(scores)), size=samples, p=scores)
    negative = [negative[i] for i in index]
    return negative

def generate_negative_samples(r, responses, samples=10):
    negative = random.sample(responses, samples)
    while r in negative:
        negative = random.sample(responses, samples)
    return negative

def generate_negative_samples_bm25(query, response, strategy='topk', pool_size=256, samples=10, lang='zh', bm25Model=None):
    '''
    Query-Answer matching: find the responses that have the similar topic with the query
    '''
    rest = bm25Model.search(None, response, samples=pool_size)
    # ipdb.set_trace()
    rest = [i['response'] for i in rest]
    if response in rest:
        rest.remove(response)

    # strategy 1: random sampling
    if strategy == 'random':
        negative = random.sample(rest, samples)
    elif strategy == 'topk':
        negative = rest[:samples]
    elif strategy == 'embedding':
        pass
    else:
        raise Exception(f'[!] unknow strategy {strategy}')
    return negative

def generate_logic_negative_samples(r, es, index_name, samples=10):
    '''
    Elasticsearch is used.
    Use the conversation context to search the near samples as the logic negative samples:
    topic or semantics are similar but not coherent with the conversation context.
    It should be noted that it's not the perfect way to collect the logic negative samples,
    and better way will be researched in the future.
    `Dialog Logic or Natural Language Interface is very important`

    Use `.msearch` call to speed up

    :r: is a batch of query
    '''
    search_arr = []
    for i in r:
        search_arr.append({"index": index_name})
        # dsl query
        search_arr.append({"query": {"match": {"context": i}}, 'size': samples+5})
    request = ''
    for each in search_arr:
        request += f'{json.dumps(each)} \n'
    rest = es.msearch(body=request)['responses']
    negative_samples = []
    for idx, each in enumerate(rest):
        ne = [i['_source']['response'] for i in each['hits']['hits']]
        if r[idx] in ne:
            ne.remove(r[idx])
            ne = ne[:samples]
        else:
            ne = ne[:samples]
        negative_samples.append(ne)
    return negative_samples

if __name__ == "__main__":
    pass
