import numpy as np
import ipdb

def read_file():
    with open('corpus.txt') as f:
        data = []
        for dialog in f.readlines():
            dialog = dialog.strip().split('\t')
            dialog = [i for i in dialog if i]
            data.append(dialog[:-1])
    avg_turn = [len(i) for i in data]
    print(f'[!] find {len(data)} multi-turn dialog, avg turn: {np.mean(avg_turn)}')
    return data

def write_data(data):
    with open('train.txt', 'w') as f:
        for dialog in data:
            for u in dialog:
                if u:
                    f.write(f'{u}\n')
            f.write('\n')

if __name__ == "__main__":
    data = read_file()
    write_data(data)
