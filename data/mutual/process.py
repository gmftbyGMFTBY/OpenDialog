from tqdm import tqdm
import os
import json

def read_file(mode):
    data = []
    an2idx = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    for file in os.listdir(mode):
        path = f'{mode}/{file}'
        with open(path) as f:
            item = json.load(f)
            response = item['options'][an2idx[item['answers']]]
            data.append((item['article'], response))
    print(f'[!] collect {len(data)} samples')
    return data 

def write_file(data, mode):
    with open(f'{mode}.txt', 'w') as f:
        for item in data:
            f.write(f'{item[0]}\n{item[1]}\n\n')

if __name__ == "__main__":
    data = read_file('test')
    write_file(data, 'test')

