import jieba
from itertools import combinations
from tqdm import tqdm
import random

def make_vocabs(responses):
    vocabs = set()
    for r in tqdm(responses):
        r = list(jieba.cut(r))
        vocabs |= set(r)
    vocabs = list(vocabs)
    print(f'[!] obtain {len(vocabs)} words')
    return vocabs

def clean(x):
    x = x.replace('[SEP]', '').replace('[UNK]', '')
    return x

def shuffle(x):
    x = clean(x)
    x = list(jieba.cut(x))
    random.shuffle(x)
    return ''.join(x)

def drop(x, ratio=0.1):
    x = clean(x)
    x = list(jieba.cut(x))
    x_ = []
    for i in x:
        if random.random() >= ratio:
            x_.append(i)
    if len(x_) == len(x):
        index = random.randint(0, len(x)-1)
        x_ = x_[0:index] + x[index+1:]
    return ''.join(x_)

def duplication(x, ratio=0.2):
    x = clean(x)
    x = list(jieba.cut(x))
    duplicate_numbers = [2, 3, 4]
    x_ = []
    # duplicate words
    for i in x:
        if random.random() >= ratio:
            x_.append(i)
        else:
            x_.extend([i] * random.choice(duplicate_numbers))
    if len(x) == len(x_):
        index = random.randint(0, len(x_)-1)
        x_ = x_[0:index] + [x_[index]] + x_[index:]
    # duplicate sub-sequence
    b, e = random.choice(list(combinations(range(len(x_)), 2)))
    x_ = x_[:e+1] + x_[b:e+1] + x_[e+1:]
    return ''.join(x_)

def replace(vocabs, x, ratio=0.2):
    x = clean(x)
    x = list(jieba.cut(x))
    x_, flag = [], False
    for i in x:
        if random.random() >= ratio:
            x_.append(i)
        else:
            x_.append(random.choice(vocabs))
            flag = True
    if not flag:
        x_[random.randint(0, len(x_)-1)] = random.choice(vocabs)
    return ''.join(x_)

if __name__ == "__main__":
    pass
