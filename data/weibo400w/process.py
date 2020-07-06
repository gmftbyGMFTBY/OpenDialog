import re
from tqdm import tqdm
from collections import Counter

ignore_pattern = ['艹', 'O', '*', '图', '贴', 'via', 'Via', '【', '】', '/', '_', ':', '3', '▽', '﹏']
pattern = ['alink', '——', '-', '…', '·', '\\', '""', '√', "'''", ':)', '="")', '：）', '^_^', '，，', '、', '=']
re_pattern = ['\(.*\)', '（.*\)', '\(.*）', '（.*）', '「.*」', '#.*#']

def filter_(msg):
    # repetitions over 3 words
    counter = Counter(msg)
    for key, value in counter.items():
        if value > 3:
            return None
    for i in ignore_pattern:
        if i in msg:
            return None
    # judge the english character
    for i in list('abcdefghijklmnopqrstuvwxyz'):
        if i in msg:
            return None
    for i in pattern:
        msg = msg.replace(i, '')
    for i in re_pattern:
        msg = re.sub(i, '', msg)
    return msg.strip() 

def read_file():
    src_path = 'stc_weibo_train_post'
    tgt_path = 'stc_weibo_train_response'
    src, tgt = [], []
    with open(src_path) as f:
        for line in tqdm(f.readlines()):
            line = ''.join(line.strip().split())
            src.append(line)
    with open(tgt_path) as f:
        for line in tqdm(f.readlines()):
            line = ''.join(line.strip().split())
            tgt.append(line)
    assert len(src) == len(tgt), f'[!] src: {len(src)}; tgt: {len(tgt)}'
    return src, tgt

def write_file(src, tgt):
    with open('train.txt', 'w') as f:
        for s, t in tqdm(list(zip(src, tgt))):
            s, t = filter_(s), filter_(t)
            if (s and t) and (t not in s) and (s not in t):
                f.write(f'{s}\n')
                f.write(f'{t}\n\n')

if __name__ == "__main__":
    src, tgt = read_file()
    write_file(src, tgt)
