from annoy import AnnoyIndex
import pickle
from gensim.models.keyedvectors import KeyedVectors
import argparse
from itertools import chain
import numpy as np
from tqdm import tqdm
import ipdb
from LAC import LAC

'''
ANN Search with the word embeddings:
1. word embeddings
2. bert embeddings
'''

class ANNSearcher():
    
    '''ANN Searcher rather than Term or Span search by Elasticsearch'''
    
    def __init__(self, dim, distance, tree):
        self.db = AnnoyIndex(dim, distance)
        self.tree, self.origin_data = tree, {}
        
    def save(self, index_path, data_path):
        self.db.build(self.tree)
        self.db.save(index_path)
        with open(data_path, 'wb') as f:
            pickle.dump(self.origin_data, f)
        print(f'[!] write index into {index_path}; write data into {data_path}')
        
    def load(self, index_path, data_path):
        self.db.load(path)
        with open(data_path, 'rb') as f:
            self.origin_data = pickle.load(f)
        print(f'[!] load index from {index_path}; load data from {data_path}')
        
    def init(self, dataset):
        '''dataset: [(utterance, vector_represent)]'''
        for idx, (data, vector) in tqdm(enumerate(dataset)):
            self.origin_data[idx] = data
            self.db.add_item(idx, vector)
        print(f'[!] init the ANN database over ...')
        
    def search(self, vector, samples=10):
        rest = self.db.get_nns_by_vector(vector, samples)
        rest = [self.origin_data[i] for i in rest]
        return rest
    
def parser_args():
    parser = argparse.ArgumentParser(description='parameters')
    parser.add_argument('--dataset', type=str, default='zh50w')
    parser.add_argument('--distance', type=str, default='dot')
    parser.add_argument('--dimension', type=int, default=300)
    parser.add_argument('--tree', type=int, default=100)
    return parser.parse_args()
    
def load_dataset(path):
    with open(path) as f:
        data = [i.split('\n') for i in f.read().split('\n\n') if i.strip()]
        data = list(chain(*data))
    # obtain w2v
    dataset = []
    for utterance in tqdm(data):
        words = [w2v[word] for word in cutter.run(utterance) if word in w2v]
        if words:
            vector = np.array(words).mean(axis=0)    # [300]
            dataset.append((utterance, vector))
    print(f'[!] obtain {len(dataset)} samples')
    return dataset
    
if __name__ == "__main__":
    args = parser_args()
    args = vars(args)
    
    cutter = LAC(mode='seg')
    agent = ANNSearcher(args['dimension'], args['distance'], args['tree'])
    w2v = KeyedVectors.load_word2vec_format('chinese_w2v.txt', binary=False)
    dataset = load_dataset(f'{args["dataset"]}/train_.txt')
    agent.init(dataset)
    agent.save('ann_index.ann', 'ann_data.pkl')