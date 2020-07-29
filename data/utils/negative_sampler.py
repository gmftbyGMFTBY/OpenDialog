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

def generate_negative_samples_fluency(r, samples=10, ratio=0.2, vocab=None, duplicated=5):
    '''
    Retrieval the samples from responses and make the perturbation on them
    to generate the unfluency negative samples for training the bert_multiview model

    Fluency perturbation:
    1. shuffle
    2. random drop
    3. duplication
    4. replace
    '''
    # negative = generate_negative_samples(r, responses, samples=samples)    # samples
    contains = []
    for _ in range(duplicated):
        contains.extend([
                shuffle(r), 
                drop(r, ratio=ratio), 
                duplication(r, ratio=ratio), 
                replace(vocab, r, ratio=ratio)
            ])
    contains = list(set(contains))
    if len(contains) <= samples:
        pass
    else:
        contains = random.sample(contains, samples)
    return contains

def generate_negative_samples_diversity(r, responses, samples=10):
    '''
    1. Distinct: prefer short response
    2. Length: prefer long response
    3. NIDF
    
    negative = random.sample(responses, 128)    # sampling pool size
    if r in negative:
        negative.remove(r)
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
    '''
    return generate_negative_samples(r, responses, samples=samples)

def generate_negative_samples(r, responses, samples=10):
    negative = random.sample(responses, samples)
    while r in negative:
        negative = random.sample(responses, samples)
    return negative

def generate_negative_samples_naturalness(responses, pool_size=64, samples=10, lang='zh', bm25Model=None):
    '''
    Query-Answer matching: find the responses that have the similar topic with the query
    '''
    rest = bm25Model.multi_search(responses, samples=pool_size)
    rest_ = []    # [batch_size, 64]
    for i, r in zip(rest['responses'], responses):
        p = [j['_source']['response'] for j in i['hits']['hits']]
        if r in p:
            p.remove(r)
        rest_.append(p[:samples])
    return rest_

def generate_negative_samples_relatedness(responses, samples=10, pool_size=64, w2v=None, bm25Model=None, embedding_function=None):
    '''
    obtain the word embeddings by GloVe
    '''
    rest = bm25Model.multi_search(responses, samples=pool_size)
    rest_ = []
    for i, r in zip(rest['responses'], responses):
        p = [j['_source']['response'] for j in i['hits']['hits']]
        if r in p:
            p.remove(r)
        # embedding chosen
        rest_emb = embedding_function(w2v, p + [r])    # [pool_size+1, 300]
        if len(rest_emb) <= 1:
            rest_.append([])
            continue
        x, y = rest_emb[-1], rest_emb[:-1]
        y = np.stack(y)    # [pool_size, 300]
        cosine_similarity = np.dot(x, y.T) / np.linalg.norm(x) / np.linalg.norm(y)    # [pool_size]
        index = np.argsort(cosine_similarity)[:samples]
        rest_.append([p[i] for i in index])
    return rest_

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
