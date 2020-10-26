from gensim.models.keyedvectors import KeyedVectors
from tqdm import tqdm
import pickle, re, ipdb, os, argparse
import requests
import time, random
import sys
import networkx as nx
from LAC import LAC
import json
from collections import Counter
from itertools import chain
from utils import *
from elasticsearch import Elasticsearch

'''
1. load the word embedding
2. filter word embedding
    2.1 no PER word, no LAC
    2.2 cannot be splited by LAC tool
    2.3 allowPOS ['n', nw, 'v', 'a', nz, 'ORG']
3. write new word embedding
4. load new word embedding
5. construct the word network (entity-similairy-entity)

TODO:
1. 完成从STC中收集高质量的wordlist
2. 和词向量中的高质量wordlist合并，对于词向量中的word在STC是低频的过滤
3. 更大的词向量文件补充
4. 对于低相似度的，删除边
5. 扫描语料库，对于在语料库中的关键词中的transition关系的边，补充到wordnet中?感觉不合理，很容易引入噪声
6. use the knowledge to add the potential edges?
7. PMI create the edges ?

词向量相似读建边不合适，使用 PMI 可以更好的刻画上下文的转移规律，
PMI = log_2 [P(x, y) / P(x) * P(y)]

衡量 x, y 两个 term 共线，不一定是一个句子里，还可以上上下相邻的两个句子
'''

class ESChat:

    def __init__(self, index_name, kb=True):
        self.es = Elasticsearch(http_auth=('elastic', 'elastic123'))
        self.index = index_name

    def multi_search(self, topics, samples=10):
        # limit the querys length
        search_arr = []
        for topic in topics:
            search_arr.append({'index': self.index})
            search_arr.append({'query': {'bool': {'should': [{'match': {'utterance': {'query': topic}}}]}}, 'size': samples})
        request = ''
        for each in search_arr:
            request += f'{json.dumps(each)} \n'
        rest = self.es.msearch(body=request)
        return rest

    def multi_search_edge(self, topics, samples=10):
        # limit the querys length
        search_arr = []
        for topic1, topic2 in topics:
            search_arr.append({'index': self.index})
            search_arr.append({'query': {'bool': {'must': [{'match': {'utterance': {'query': topic1}}}, {'match': {'utterance': {'query': topic2}}}]}}, 'size': samples})
        request = ''
        for each in search_arr:
            request += f'{json.dumps(each)} \n'
        rest = self.es.msearch(body=request)
        return rest

def parser_args():
    parser = argparse.ArgumentParser(description='wordnet parameters')
    parser.add_argument('--weight_threshold', type=float, default=0.6)
    parser.add_argument('--topn', type=int, default=200)
    parser.add_argument('--mode', type=str, default='graph')
    return parser.parse_args()

def load_stopwords():
    with open('stopwords.txt') as f:
        data = f.read().split('\n')
        data = [i for i in data if i.strip()]
    return data

def collect_wordlist_from_corpus(path, topn=50000):
    '''only save the tokens that length from 2 to 4'''
    cutter = LAC(mode='lac')
    with open(path) as f:
        data = f.read().split('\n\n')
        data = [i.split('\n') for i in data if i.strip()]
        data = random.sample(data, 1000000)
        print(f'[!] load the dataset from {path}({len(data)}) over ...')
    batch_size, words_collector = 512, Counter()
    pbar = tqdm(range(0, len(data), batch_size))
    for idx in pbar:
        dialogs = data[idx:idx+batch_size]
        dialogs = [' '.join(i) for i in dialogs]
        rest = cutter.run(dialogs)
        collector = []
        for words, tags in rest:
            for word, tag in zip(words, tags):
                if filter(word, tag):
                    collector.append(word)
        words_collector.update(collector)
        pbar.set_description(f'[!] collect words: {len(words_collector)}')
    words = [w for w, _ in words_collector.most_common(topn)]
    print(f'[!] {len(words_collector)} -> {len(words)}')
    return words
        
def filter(word, tag):
    def isChinese():
        for ch in word:
            if not '\u4e00' <= ch <= '\u9fff':
                return False
        return True
    def HaveDigital():
        if bool(re.search(r'\d', word)):
            return False
        else:
            return True
    def Length():
        if 1 < len(word) < 5:
            return True
        else:
            return False
    def HaveAlpha():
        for ch in word:
            if ch.encode().isalpha():
                return False
        return True
    def Special():
        for ch in word:
            if ch in set('一二三四五六七八九十月日周年区东西南北。，|；“”‘’！~·、：=-+#￥%……&*（）【】@？.,?[]{}()!$^`";:'):
                return False
        return True
    def CheckTag():
        if tag in set(['n', 'nz', 'nw', 'v', 'vn', 'a', 'ad', 'an', 'ORG', 'PER', 'LOC']):
            return True
        else:
            return False
    def InW2V():
        if word in w2v:
            return True
        else:
            return False
    return isChinese() and HaveDigital() and Length() and HaveAlpha() and Special() and CheckTag() and InW2V()

def write_new_w2v(words, path):
    with open(path, 'w') as f:
        f.write(f'{len(words)} 300\n')
        for word in tqdm(words):
            vec = w2v[word].tolist()
            string = f'{word} {" ".join(map(str, vec))}\n'
            f.write(string)

def retrieval_edge(word_pairs, samples=64):
    rest = chatter.multi_search_edge(word_pairs, samples=samples)['responses']
    flag = []
    for (word1, word2), pair_rest in zip(word_pairs, rest):
        counter = 0
        pair_rest = pair_rest['hits']['hits']
        for utterance in pair_rest:
            utterance = utterance['_source']['utterance']
            if word1 in utterance and word2 in utterance:
                counter += 1
        if counter >= 0.5 * samples:
            flag.append(True)
        else:
            flag.append(False)
    return flag
 
def retrieval_filter(words, samples=64):
    rest = chatter.multi_search(words, samples=samples)
    rest = rest['responses']
    flag = []
    for word, word_rest in zip(words, rest):
        counter = 0
        word_rest = word_rest['hits']['hits']
        if len(word_rest) != samples:
            # less samples in the database
            flag.append(False)
            continue
        for utterance in word_rest:
            if word in utterance['_source']['utterance']:
                counter += 1
        if counter >= 0.5 * samples:
            flag.append(True)
        else:
            flag.append(False)
    return flag

if __name__ == "__main__":
    args = parser_args()
    args = vars(args)
    
    if args['mode'] == 'word':
        # default use the LCCC dataset
        print(f'[!] make sure you already run the graph mode')
        w2v = KeyedVectors.load_word2vec_format('chinese_w2v_base.txt', binary=False)
        words = collect_wordlist_from_corpus('LCCC/train.txt', topn=20000)
        with open('topic_words.pkl', 'wb') as f:
            pickle.dump(words, f)
    elif args['mode'] == 'graph':
        lac = LAC(mode='lac')
        chatter = ESChat('retrieval_database')
        stopwords = load_stopwords()

        if not os.path.exists('chinese_w2v_base.txt'):
            # 1)
            w2v = KeyedVectors.load_word2vec_format('chinese_w2v.txt', binary=False)
            print(f'[!] load the word2vec from chinese_w2v.txt')
            # 2)
            wordlist = w2v.index2word
            new_wordlist = [word for word in tqdm(wordlist) if filter(word)]
            print(f'[!] squeeze the wordlist from {len(wordlist)} to {len(new_wordlist)}')
            # stop words remove
            new_wordlist_ = list(set(new_wordlist) - set(stopwords))
            print(f'[!] squeeze the wordlist from {len(new_wordlist)} to {len(new_wordlist_)}')
            # retrieval check and remove
            new_wordlist_2, batch_size = [], 256
            for idx in tqdm(range(0, len(new_wordlist_), batch_size)):
                words = new_wordlist_[idx:idx+batch_size]
                for word, rest in zip(words, retrieval_filter(words)):
                    if rest:
                        new_wordlist_2.append(word)
            print(f'[!] squeeze the wordlist from {len(new_wordlist_)} to {len(new_wordlist_2)}')
            # 3)
            write_new_w2v(new_wordlist_2, 'chinese_w2v_base.txt')
            print(f'[!] write the new w2v into chinese_w2v_base.txt')
        # 4)
        w2v = KeyedVectors.load_word2vec_format('chinese_w2v_base.txt', binary=False)
        print(f'[!] load the new word2vec from chinese_w2v_base.txt')
        # 5)
        if not os.path.exists('wordnet.pkl'):
            graph = nx.Graph()
            graph.add_nodes_from(w2v.index2word)
            # batch_size = 64 
            # for idx in tqdm(range(0, len(w2v.index2word), batch_size)):
            #     words = w2v.index2word[idx:idx+batch_size]
            #     neighbors = []
            #     for word in words:
            #         neighbors.extend([(word, n, w) for n, w in w2v.most_similar(word, topn=args['topn'])])
            #     flag = retrieval_edge([(word, n) for word, n, w in neighbors])
            #     neighbors = [(word, n, w) for flag_, (word, n, w) in zip(flag, neighbors) if flag_]
            #     graph.add_weighted_edges_from([(word, n, 1-w) for word, n, w in neighbors])
            for word in tqdm(w2v.index2word):
                neighbors = w2v.most_similar(word, topn=args['topn'])
                flag = retrieval_edge([(word, i) for i, _ in neighbors])
                neighbors = [(n, w) for flag_, (n, w) in zip(flag, neighbors) if flag_]
                graph.add_weighted_edges_from([(word, n, 1 - w) for n, w in neighbors])
            with open('wordnet.pkl', 'wb') as f:
                pickle.dump(graph, f)
            print(f'[!] save the word net into wordnet.pkl')
        else:
            with open('wordnet.pkl', 'rb') as f:
                graph = pickle.load(f)
