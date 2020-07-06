import os
from collections import Counter
import pprint
import numpy as np
from tqdm import tqdm

'''
File to process the raw `wiki_{num}_string.txt_new.txt`
'''

def collect_files():
    rest = []
    for file in os.listdir('.'):
        if 'wiki_' in file:
           rest.append(file) 
    return rest

def process_one_file(path, min_len=10, max_len=512):
    # longer than max_len, will be cut
    with open(path) as f:
        data = f.read()
        data = data.split('\n')
        data = [i for i in data if i.strip()]
        nd = []
        for i in data:
            if len(i) <= max_len:
                nd.append(i)
            else:
                start = 0
                while start < len(i):
                    nd.append(i[start:start+max_len])
                    start += max_len
        nd = [i for i in nd if len(i) > min_len]
    print(f'[!] collect {len(nd)} string in {path}')
    return nd 

def write_file(data):
    with open('train.txt', 'w') as f:
        for i in data:
            f.write(f'{i}\n')

if __name__ == "__main__":
    files = collect_files()
    data = []
    for i in tqdm(files):
        data.extend(process_one_file(i, min_len=10, max_len=300))
    write_file(data)
    # state
    data_size = len(data)
    lengths = [len(i) for i in data]
    pprint.pprint(Counter(lengths))
    print(f'[!] data size: {data_size}')
    print(f'[!] mean lengths: {np.mean(lengths)}')
    print(f'[!] min lengths: {np.min(lengths)}')
    print(f'[!] max lengths: {np.max(lengths)}')
