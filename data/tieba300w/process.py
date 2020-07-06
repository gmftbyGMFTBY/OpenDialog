from tqdm import tqdm
import re
import ipdb
from collections import Counter

ignore_pattern = ['艹', 'O', '*', '图', '贴', 'via', 'Via', '【', '】', '/', '_', ':', '3', '▽', '﹏', '+', '@']
pattern = ['alink', '——', '-', '…', '·', '\\', '""', '√', "'''", ':)', '="")', '：）', '^_^', '，', '、', '=']
re_pattern = ['【.*】', '\(.*\)', '（.*\)', '\(.*）', '（.*）', '「.*」', '#.*#']

def filter_(msg):
    # length
    if len(msg) < 3:
        return None
    if len(msg) > 150:
        return None
    # repetitions over 3 words
    counter = Counter(msg)
    for key, value in counter.items():
        if value > 3:
            return None
    # for i in ignore_pattern:
    #     if i in msg:
    #         return None
    # judge the english character
    # for i in list('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'):
    #     if i in msg:
    #         return None
    for i in pattern:
        msg = msg.replace(i, '')
    for i in re_pattern:
        msg = re.sub(i, '', msg)
    return msg.strip()

def read_file():
    data = []
    with open('tieba.dialogues') as f:
        dialog = []
        cache = None
        for line in tqdm(f.readlines()):
            line = line.strip()
            src, tgt = line.split('\t')
            if cache == src:
                dialog.append(tgt)
            else:
                if dialog:
                    data.append(dialog)
                dialog = [src, tgt]
            cache = tgt
            # src, tgt = filter_(src), filter_(tgt)
            # if (src and tgt) and (src not in tgt) and (tgt not in src):
            #     data.append((src, tgt))
    return data

def write_file(data):
    with open('train.txt', 'w') as f:
        for i in data:
            for j in i:
                f.write(f'{j}\n')
            f.write(f'\n')

if __name__ == '__main__':
    data = read_file()
    write_file(data)
