import os
import random

def read_file(path):
    with open(path) as f:
        data = []
        for line in f.readlines():
            src, tgt, _ = line.split('\t')
            src, tgt = ''.join(src.split()), ''.join(tgt.split())
            data.append(f'{src}\n{tgt}')
    random.shuffle(data)
    train_data, test_data = data[:-100], data[-100:]
    return train_data, test_data

def write_file(data, path):
    with open(path, 'w') as f:
        for i in data:
            f.write(f'{i}\n\n')

if __name__ == "__main__":
    train_data, test_data = read_file('train.txt')
    write_file(train_data, 'train_pone.txt')
    write_file(test_data, 'test_pone.txt')
