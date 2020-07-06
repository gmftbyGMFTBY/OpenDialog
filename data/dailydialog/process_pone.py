import os
import random

def read_file(path):
    with open(path) as f:
        data = f.read().split('\n\n')
        data = [i.strip() for i in data if i.strip()]
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
