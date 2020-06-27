import os
import jieba
import ipdb
import pickle
from tqdm import tqdm
import numpy as np

def load_w2v(path):
    '''
    Load the `text` embedding file, such as glove and tencent chinese word embedding
    '''
    if os.path.exists(f'{path}.pkl'):
        with open(f'{path}.pkl', 'rb') as f:
            w2v = pickle.load(f)
            print(f'[!] load w2v from {path}.pkl, size: {len(w2v)}')
            return w2v
    with open(f'{path}.txt') as f:
        w2v = {}
        for line in tqdm(f.readlines()):
            line = line.split()
            word, vec = line[0], line[1:]
            vec = np.array(list(map(float, vec)))
            assert len(vec) == 300, f'[!] except the vector 300, got {len(vec)}'
            w2v[word] = vec
    with open(f'{path}.pkl', 'wb') as f:
        pickle.dump(w2v, f)
    print(f'[!] write w2v into {path}.pkl, size: {len(w2v)}')
    return w2v

def convert_text_embedding(w2v, texts):
    '''
    w2v is the word embedding dict obtained by `load_w2v`
    return the 300 dimension embedding of the texts, the [UNK] tokens are not considered
    The embedding is much faster than the Bert-as-service
    '''
    e = []
    for text in texts:
        words = list(jieba.cut(text))
        vectors = []
        for w in words:
            if w in w2v:
                vectors.append(w2v[w])
        if not vectors:
            vectors.append(np.random.randn(300))
        vectors = np.stack(vectors).mean(axis=0)    # [words, 300] -> [300]
        e.append(vectors)
    return e    # [batch, 300]

if __name__ == "__main__":
    w2v = load_w2v('data/chinese_w2v')
    e = convert_text_embedding(w2v, ['我非常喜欢看电影', '我不认为电影很好看'])
    ipdb.set_trace()

