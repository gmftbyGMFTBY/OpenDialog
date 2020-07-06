import os
from tqdm import tqdm

def read_file(path):
    with open(path) as f:
        data = f.readlines()
        data = [i.strip() for i in data if i.strip()]
    return data

def write_file(src, tgt):
    with open('train_pone.txt', 'w') as f:
        for s, t in tqdm(list(zip(src, tgt))):
            f.write(f'{s}\n{t}\n\n')

if __name__ == "__main__":
    src, tgt = read_file('src-train.txt'), read_file('tgt-train.txt')
    print(f'[!] read over')
    write_file(src, tgt)
