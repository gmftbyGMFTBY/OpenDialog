from gensim.models.keyedvectors import KeyedVectors
from tqdm import tqdm
import pickle, re, ipdb, os
import networkx as nx
from LAC import LAC
import json
from collections import Counter

'''
1. load the word embedding
2. filter word embedding
    2.1 no PER word
    2.2 cannot be splited by LAC tool
    2.3 allowPOS ['n', 'nz', 'LOC', 'ORG']
3. write new word embedding
4. load new word embedding
5. construct the word network (entity-similairy-entity)
'''

def filter(word):
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
        # ipdb.set_trace()
        word_, pos = lac.run(word)
        if len(word_) != 1:
            return False
        if pos[0] in ['n', 'nz', 'LOC', 'ORG']:
            return True
        else:
            return False
    return isChinese() and HaveDigital() and Length() and HaveAlpha() and Special() and CheckTag()

def write_new_w2v(words, path):
    with open(path, 'w') as f:
        f.write(f'{len(words)} 300\n')
        for word in tqdm(words):
            vec = w2v[word].tolist()
            string = f'{word} {" ".join(map(str, vec))}\n'
            f.write(string)

if __name__ == "__main__":
    if not os.path.exists('chinese_w2v_base.txt'):
        # 1)
        w2v = KeyedVectors.load_word2vec_format('chinese_w2v.txt', binary=False)
        print(f'[!] load the word2vec from chinese_w2v.txt')
        # 2)
        wordlist = w2v.index2word
        lac = LAC(mode='lac')
        new_wordlist = [word for word in tqdm(wordlist) if filter(word)]
        print(f'[!] squeeze the wordlist from {len(wordlist)} to {len(new_wordlist)}')
        # 3)
        write_new_w2v(new_wordlist, 'chinese_w2v_base.txt')
        print(f'[!] write the new w2v into chinese_w2v_base.txt')
    # 4)
    w2v = KeyedVectors.load_word2vec_format('chinese_w2v_base.txt', binary=False)
    print(f'[!] load the new word2vec from chinese_w2v_base.txt')
    # 5)
    graph = nx.Graph()
    graph.add_nodes_from(w2v.index2word)
    for word in tqdm(w2v.index2word):
        neighbors = w2v.most_similar(word, topn=10)
        # low similarity drop ?
        graph.add_weighted_edges_from([(word, n, w) for n, w in neighbors])
    
    with open('wordnet.pkl', 'wb') as f:
        pickle.dump(graph, f)
    print(f'[!] save the word net into wordnet.pkl')