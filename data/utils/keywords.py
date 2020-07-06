import jieba
import pickle
from tqdm import tqdm
import ipdb
import jieba.analyse
import networkx as nx

class KeywordsCollector:

    '''
    ::corpus:: the list that contains the context and response of each dialog
    '''

    def __init__(self):
        self.allowpos = [
                'n', 'nr', 'nt', 'nw']
        self.topk = 5
        self.g = nx.Graph()

    def process_dataset(self, corpus):
        pbar = tqdm(corpus)
        counter = 0
        for context, response in pbar:
            c_words = jieba.analyse.extract_tags(context, topK=self.topk, allowPOS=self.allowpos)
            r_words = jieba.analyse.extract_tags(response, topK=self.topk, allowPOS=self.allowpos)
            if len(c_words) == 0 or len(c_words) == 0:
                counter += 1
                continue
            self.g.add_nodes_from(c_words)
            self.g.add_nodes_from(r_words)
            # edge
            for c_term in c_words:
                for r_term in r_words:
                    self.g.add_edge(c_term, r_term)
            pbar.set_description(f'ignore: {counter}; nodes: {len(self.g.nodes())}; edges: {len(self.g.edges())}')

        # delete the self-edge
        for i, j in self.g.edges:
            if i == j:
                self.g.remove_edge(i, i)

    def obtain_neighboors(self, node):
        ipdb.set_trace()
        data = self.g[node]

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.g, f)
        print(f'[!] save graph into {path}')

    def load(self, path):
        with open(path, 'rb') as f:
            self.g = pickle.load(f)
        print(f'[!] load graph from {path}')

if __name__ == "__main__":
    kw = KeywordsCollector()
    kw.load('../zh50w/graph.pkl')
    kw.obtain_neighboors('失眠')
