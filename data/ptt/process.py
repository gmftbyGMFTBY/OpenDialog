from langconv import *
from tqdm import tqdm
import ipdb
from utils import *

def translate2simplified(s):
    s = Converter('zh-hans').convert(s)
    return s

def read_file():
    with open('ptt.txt') as f:
        data = []
        for i in tqdm(f.readlines()):
            i = i.strip()
            try:
                src, tgt = i.split('\t')
            except:
                continue
            src = ''.join(src.split())
            tgt = ''.join(tgt.split())
            src, tgt = translate2simplified(src), translate2simplified(tgt)
            src, tgt = filter_(src), filter_(tgt)
            if (src and tgt) and (src not in tgt) and (tgt not in src):
                data.append((src, tgt))
    return data

def write_file(data):
    with open('train.txt', 'w') as f:
        for i in tqdm(data):
            f.write(f'{i[0]}\n')
            f.write(f'{i[1]}\n\n')

if __name__ == "__main__":
    data = read_file()
    write_file(data)
