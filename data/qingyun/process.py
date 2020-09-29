from tqdm import tqdm
import csv
import ipdb
from utils import *

def read_file():
    with open('qingyun.csv') as f:
        f = csv.reader(f, delimiter='|')
        data = []
        for i in f:
            try:
                src, tgt = i
                src, tgt = src.strip(), tgt.strip()
            except:
                continue
            # src, tgt = filter_(src), filter_(tgt)
            # if (src and tgt) and (src not in tgt) and (tgt not in src):
            #     data.append((src, tgt))
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
