from tqdm import tqdm
import ipdb
import ijson
import json

def read_file(path, mode='train'):
    with open(path) as f:
        data = ijson.items(f, 'train')
        ipdb.set_trace()
        for i in tqdm(data):
            ipdb.set_trace()
    return data

def write_file(dataset, path):
    with open(path, 'w') as f:
        for data in dataset:
            f.write(f'{data[0]}\n{data[1]}\n\n')

if __name__ == '__main__':
    dataset = read_file('LCCC-base.json', mode='train')
    write_file(dataset, 'train.txt')
