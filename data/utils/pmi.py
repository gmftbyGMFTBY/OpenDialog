import numpy as np
from LAC import LAC
from collections import Counter
from itertools import combinations, product
from tqdm import tqdm
import re, ipdb
import jieba
import jieba.posseg as pseg

'''
PMI(x, y) = log2 [P(x, y) / P(x) * P(y)]
P(x, y)代表一个对话的session中，x和y出现在一个句子中或者上下两个相邻句子中
P(x)代表出现了x的对话session的个数

n	普通名词	f	方位名词	s	处所名词	t	时间
nr	人名	ns	地名	nt	机构名	nw	作品名
nz	其他专名	v	普通动词	vd	动副词	vn	名动词
a	形容词	ad	副形词	an	名形词	d	副词
m	数量词	q	量词	r	代词	p	介词
c	连词	u	助词	xc	其他虚词	w	标点符号
PER	人名	LOC	地名	ORG	机构名	TIME	时间

n, nt, nw, nz, v, a
'''

class PMIUtil:
    
    def __init__(self):
        self.session_num = 0
        self.word_counter = Counter()
        self.word_pair_counter = Counter()
        self.sw = self.load_stopwords()
        
    def load_stopwords(self):
        with open('stopwords.txt') as f:
            data = f.read().split('\n')
            data = [i for i in data if i.strip()]
        return data
        
    def filter(self, word, tag):
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
        def IsStopword():
            if word in self.sw:
                return False
            else:
                return True
        def CheckTag():
            if tag in set(['n', 'nt', 'nw', 'nz', 'v', 'a']):
                return True
            else:
                return False
        return Length() and isChinese() and IsStopword() and HaveDigital() and HaveAlpha() and Special() and CheckTag()
        
    def collect_keywords_one_dialog(self, dialog):
        overallwords = set()
        new_utterances = []
        for utterance in dialog:
            words = [i for i, j in jieba.posseg.cut(utterance) if self.filter(i, j)]
            new_utterances.append(words)
            overallwords |= set(words)
        self.word_counter.update(list(overallwords))
        return new_utterances
        
    def collect_co_occur_one_dialog(self, dialog):
        def ff(x):
            return [x_word for x_word in x if x_word in self.words]
        pair_set = set()
        pair_set |= set([i for i in map(frozenset, combinations(ff(dialog[0]), 2)) if len(i) == 2])
        for i in range(1, len(dialog)):
            up, down = ff(dialog[i-1]), ff(dialog[i])
            pair_set |= set([i for i in map(frozenset, product(up, down)) if len(i) == 2])
            pair_set |= set([i for i in map(frozenset, combinations(down, 2)) if len(i) == 2])
        self.word_pair_counter.update(list(pair_set))
            
    def process_docs(self, data, topn=50000, min_cut=2):
        pbar = tqdm(data)
        step, self.session_num = 0, len(data)
        # collect keywords
        processed_data = []
        for dialog in pbar:
            processed_data.append(
                self.collect_keywords_one_dialog(dialog),
            )
            step += 1
            if step % 1000 == 0:
                pbar.set_description(f'[!] words: {len(self.word_counter)}')
        self.words = {i: j for i, j in self.word_counter.most_common(topn)}
        print(f'[!] squeeze words from {len(self.word_counter)} to {len(self.words)}')
        
        # filter the word pair based on the words
        step = 0
        pbar = tqdm(processed_data)
        for dialog in pbar:
            self.collect_co_occur_one_dialog(dialog)
            step += 1
            if step % 1000 == 0:
                pbar.set_description(f'[!] words pairs: {len(self.word_pair_counter)}')
        self.word_pairs = {i:j for i, j in dict(self.word_pair_counter).items() if j > min_cut}
        
    def calculate_pmi(self, src, tgt):
        if src in self.words and tgt in self.words:
            p_xy = self.word_pairs[frozenset([src, tgt])] / self.session_num
            p_x, p_y = self.words[src] / self.session_num, self.words[tgt] / self.session_num
            return np.log2(p_xy / p_x / p_y)
        else:
            print(f'[!] src ot tgt are invalid')
            return -1

if __name__ == "__main__":
    pass