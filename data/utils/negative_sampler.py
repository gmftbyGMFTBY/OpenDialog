import random
from tqdm import tqdm
import ipdb
import json
import numpy as np
from gensim.summarization import bm25
from bert_serving.client import BertClient
import jieba

def generate_negative_samples(r, responses, samples=10):
    negative = random.sample(responses, samples)
    while r in negative:
        negative = random.sample(responses, samples)
    return negative

def generate_negative_samples_bm25(responses, samples=10, lang='zh', bert=False):
    if bert:
        print(f'[!] Make sure the bert-as-service is running; language is {lang}')
        bc = BertClient()
    # init the bm25 agent
    query = []
    for r in tqdm(responses):
        if lang == 'zh':
            query.append(list(jieba.cut(r)))
        else:
            query.append(r.split())
    bm25Model = bm25.BM25(query)
    # search the similar
    rest = []
    for q in tqdm(query):
        scores = bm25Model.get_scores(q)
        scores = np.array(scores)
        idx = np.argpartition(scores, -200)[-200:]
        texts = [query[i] for i in idx if query[i] != q]
        if bert:
            # use bert to choose the most half similar and half not similar
            sep_token = '' if lang == 'zh' else ' '
            convert_texts = [sep_token.join(q)] + [sep_token.join(i) for i in texts]
            convert_texts = convert_texts[:50]
            embeddings = bc.encode(convert_texts)    # [batch, 768]
            g_e = embeddings[0]
            n_e = embeddings[1:]
            scores = np.dot(g_e, n_e.T) / np.linalg.norm(g_e) / np.linalg.norm(n_e, axis=1)
            idx = np.argpartition(scores, -samples)[-samples:]
            convert_texts = convert_texts[1:]
            texts = [convert_texts[i] for i in idx]
            rest.append(texts)
        else:
            rest.append(texts[:samples])
    # rest: [dataset_size, samples] (list)
    data = []
    sep_token = '' if lang == 'zh' else ' '
    for i in rest:
        data.append([sep_token.join(j) for j in i])
    return data

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
