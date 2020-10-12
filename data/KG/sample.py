import gensim
import random
import argparse, pickle
from gensim.models.keyedvectors import KeyedVectors
import re, os, ipdb
from jieba import analyse
import jieba.posseg
from tqdm import tqdm
from collections import Counter
from LAC import LAC

def parser_args():
    parser = argparse.ArgumentParser(description='construct datasets')
    parser.add_argument('--num_train_entity', type=int, default=10000)
    parser.add_argument('--num_test_entity', type=int, default=1000)
    parser.add_argument('--each_used', type=int, default=5)
    parser.add_argument('--max_depth', type=int, default=7)
    parser.add_argument('--min_depth', type=int, default=4)
    parser.add_argument('--topn', type=int, default=10)
    parser.add_argument('--seed', type=float, default=30)
    parser.add_argument('--mode', type=str, default='train')
    return parser.parse_args()

def extract_important_word_list(path):
    '''
    n	普通名词	f	方位名词	s	处所名词	nw	作品名
    nz	其他专名	v	普通动词	vd	动副词	vn	名动词
    a	形容词	ad	副形词	an	名形词	d	副词
    m	数量词	q	量词	r	代词	p	介词
    c	连词	u	助词	xc	其他虚词	w	标点符号
    PER	人名	LOC	地名	ORG	机构名	TIME	时间
    '''
    lac = LAC(mode='lac')
    with open(path) as f:
        data = f.read().split('\n\n')
        data = [i.split('\n') for i in data]
        data = [i for i in data if len(i) == 2]
    print(f'[!] load the data from {path} over')
    words = Counter()
    allowPOS = ['n', 'nw', 'nz', 'a', 'TIME', 'PER', 'ORG', 'LOC', '']
    batch_size = 512
    for idx in tqdm(range(0, len(data), batch_size)):
        text = [i + ' ' + j for i, j in data[idx:idx+batch_size]]
        rest = lac.run(text)
        p = []
        for sample in rest:
            for word, pos in zip(*sample):
                if pos in allowPOS:
                    p.append(word)
        words.update(p)
    words = [i[0] for i in words.most_common(2 * args['num_train_entity'])]
    return words

def filter(entity):
    for ch in entity:
        if not '\u4e00' <= ch <= '\u9fff':
            return False
        
    if bool(re.search(r'\d', entity)):
        return False
        
    if len(entity) > 5 or len(entity) < 2:
        return False
    
    if entity.encode().isalpha():
        return False
    
    special_tokens = set('1234567890一二三四五六七八九十月日周年区东西南北。，|；“”‘’！~·、：=-+#￥%……&*（）【】@？.,?[]{}()!$^`";:')
    for char in entity:
        if char in special_tokens:
            return False
    return True      

def sample_one_word(word, f):
    '''return 10 samples'''
    rest = []
    for _ in range(args['each_used']):
        p, depth = [word], random.randint(args['min_depth'], args['max_depth'])
        for _ in range(depth):
            candidates = [i[0] for i in word_vectors.most_similar(word, topn=args['topn']) if filter(i[0])]
            try:
                candidate, counter = p[-1], 0
                while candidate in p:
                    candidate = random.choice(candidates)
                    counter += 1
                    if counter > args['topn']:
                        raise Exception()
                p.append(candidate)
            except:
                p = None
                break
        if p:
            rest.append(p)
    for sample in rest:
        sample = '\t'.join(sample)
        f.write(f'{sample}\n')
    
if __name__ == "__main__":
    args = vars(parser_args())
    word_vectors = gensim.models.KeyedVectors.load_word2vec_format(
        '../chinese_w2v.txt', binary=False)
    ipdb.set_trace()
    
    if os.path.exists(f'word_list_{args["mode"]}.pkl'):
        word_list = pickle.load(open(f'word_list_{args["mode"]}.pkl', 'rb'))
    else:
        word_list = [w for w in word_vectors.index2word if filter(w)]
        word_list_ = extract_important_word_list(f'../STC/{args["mode"]}.txt')
        print(f'[!] load {len(word_list_)} words from ../STC/{args["mode"]}.txt')
        word_list = list(set(word_list_) & set(word_list))
        print(f'[!] final word list size {len(word_list)}')

        with open(f'word_list_{args["mode"]}.pkl', 'wb') as f:
            pickle.dump(word_list, f)
    
    sample_size = args['num_train_entity'] if args['mode'] == 'train' else args['num_test_entity']
    word_list = random.sample(word_list, sample_size)
    print(f'[!] obtain {len(word_list)} for {args["mode"]}')
    
    with open(f'{args["mode"]}.txt', 'w') as f:
        for word in tqdm(word_list):
            sample_one_word(word, f)