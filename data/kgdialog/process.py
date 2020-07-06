from tqdm import tqdm
import json
import ipdb

def read_file(path):
    data = []
    with open(path) as f:
        for i in tqdm(f.readlines()):
            i = i.strip()
            i = json.loads(i)
            try:
                utterances = [''.join(j.split()) for j in i['conversation']]
            except:
                utterances = [''.join(j.split()) for j in i['history']]
            data.append(utterances)
    return data

def write_file(data):
    with open('train.txt', 'w') as f:
        for i in data:
            for u in i:
                f.write(f'{u}\n')
            f.write('\n')

if __name__ == "__main__":
    data = []
    data.extend(read_file('train_.txt'))
    data.extend(read_file('test.txt'))
    data.extend(read_file('dev.txt'))

    write_file(data)
