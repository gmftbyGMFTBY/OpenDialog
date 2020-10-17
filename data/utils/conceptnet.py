import requests
from zhconv import convert
import ipdb
import time
import pprint
import networkx as nx
import pickle
from tqdm import tqdm
import sys
import re

class ConceptNetCrawl:
    
    def __init__(self, max_size=50000):
        self.url = 'http://api.conceptnet.io'
        self.graph = nx.Graph()
        self.time = 1.
        self.max_size = max_size
        
    def filter(self, word):
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
        return isChinese() and HaveDigital() and Length() and HaveAlpha()
        
    def get(self, word_):
        '''first verion: only collect the nodes in the first page;
        return: [(relation string, nodes string), ...]'''
        word = convert(word_, 'zh-tw')
        url = f'{self.url}/c/zh/{word}'
        obj = requests.get(url).json()
        time.sleep(self.time)
        
        if 'error' in obj:
            return False
        else:
            return True
        
        # essential information
        # word_id = obj['@id']
        # next_words = []
        # for item in obj['edges']:
        #     start, end = item['start'], item['end']
        #     start_label, end_label = convert(start['label'], 'zh-cn'), convert(end['label'], 'zh-cn')
        #     if start_label == end_label:
        #         continue
        #     else:
        #         src, tgt = start_label, end_label
        #     if self.filter(src) and self.filter(tgt):
        #         next_words.append(frozenset([src, tgt]))
        # next_words = [(i, j) for i, j in list(set(next_words))]
        # time.sleep(self.time)
        # return next_words
    
    def update_graph(self, edges):
        self.graph.add_edges_from(edges)
        
    def scan_seed_words(self, seed):
        step, collector, batch_size, pbar = 0, [], 128, tqdm(seed)
        for word in pbar:
            edges = self.get(word)
            step += 1
            if step % batch_size == 0:
                self.update_graph(collector)
                collector = []
            else:
                collector.extend(edges)
            pbar.set_description(f'[!] nodes: {len(g.nodes)}; edges: {len(g.edges)}')
        if collector:
            self.update_graph(collector)
            

if __name__ == "__main__":
    agent = ConceptNetCrawl()
    data = agent.get(sys.argv[1])
    print(data)