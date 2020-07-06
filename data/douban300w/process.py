import re
from tqdm import tqdm
import csv
import ipdb

pattern = ['alink', '——', '-', '…', '·', '\\', '""', '√', "'''", ':)', '="")', '：）', '^_^', '，，', '、', '=']
re_pattern = ['\(.*\)', '（.*\)', '\(.*）', '（.*）', '「.*」', '#.*#']

def filter_(msg):
    for i in pattern:
        msg = msg.replace(i, '')
    for i in re_pattern:
        msg = re.sub(i, '', msg)
    return msg

def read_file():
    data = []
    def load(path):
        data_ = []
        with open(path) as f:
            f = csv.reader(f, delimiter='\t')
            for line in tqdm(f):
                if line[0] == '1':
                    c = [filter_(''.join(i.split())) for i in line[1:]]
                    # filter
                    filter_flag = False
                    for i in c:
                        if len(i) > 50:
                            filter_flag = True
                            break
                    if filter_flag:
                        continue
                    data_.append(c[-10:])
        return data_
    data.extend(load('train_.txt'))
    data.extend(load('test.txt'))
    data.extend(load('dev.txt'))
    return data

def write_file(data):
    print(f'[!] begin write file train.txt')
    with open('train.txt', 'w') as f:
        for i in tqdm(data):
            for j in i:
                f.write(f'{j}\n')
            f.write('\n')

if __name__ == "__main__":
    data = read_file()
    write_file(data)
