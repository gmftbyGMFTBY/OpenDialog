import json
import ipdb
from tqdm import tqdm
import numpy as np

def read_file(path):
    dialogs = []
    with open(path) as f:
        data = json.load(f)
        for i in data:
            i = i['messages']
            i = [item['message'] for item in i]
            dialogs.append(i)
    return dialogs

def write_file(data):
    with open('train.txt', 'w') as f:
        for dialog in tqdm(data):
            for utterance in dialog:
                f.write(f'{utterance}\n')
            f.write('\n')

if __name__ == "__main__":
    data = []
    data.extend(read_file('music.json'))
    data.extend(read_file('movie.json'))
    data.extend(read_file('travel.json'))
    print(f'utterance size: {np.sum([len(i) for i in data])}')
    print(f'average turn size: {round(np.mean([len(i) for i in data]), 4)}')
    word_len = []
    for dialog in data:
        word_len.append(np.sum([len(i) for i in dialog]))
    print(f'word length: {round(np.mean(word_len))}')
    print(f'dialog size: {len(data)}')
    write_file(data)
