from tqdm import tqdm
import ipdb
import json

def read_file(path, mode='train'):
    with open(path) as f:
        data = json.load(f)['train']
        dialogs = []
        for i in data:
            i = [''.join(j.split()) for j in i]
            dialogs.append(i)
    return dialogs

def write_file(dataset, path):
    with open(path, 'w') as f:
        for data in tqdm(dataset):
            for utterance in data:
                f.write(f'{utterance}\n')
            f.write('\n')

if __name__ == '__main__':
    dataset = read_file('LCCC-base.json', mode='train')
    write_file(dataset, 'train.txt')
