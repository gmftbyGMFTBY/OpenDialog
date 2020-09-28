import json
import random
import ipdb

def read_file(path, mode='test'):
    with open(path) as f:
        data = json.load(f)
        data = [(''.join(i[0].split()), ''.join(i[1].split())) for i in data[mode]]
    return data

def write_dialog(path, data):
    with open(path, 'w') as f:
        for i in data:
            f.write(f'{i[0]}\n{i[1]}\n\n')

if __name__ == '__main__':
    # data = read_file('STC_test.json')
    # write_dialog('test.txt', data)
    data = read_file('STC.json', mode='train')
    data = random.sample(data, 500000)
    write_dialog('train.txt', data)
