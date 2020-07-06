from tqdm import tqdm
import ipdb

'''
preprocess the zh50w dataset into single-turn mode
'''

def read_file():
    with open('train_.txt') as f:
        data = f.read().split('\n\n')
        data = [i.split('\n') for i in data]
        data = [(' [SEP] '.join(i[:-1]), i[-1])for i in data]
        return data

def write_file(data):
    train_data, test_data = data[:-3000], data[-3000:]
    with open('train.txt', 'w') as f:
        for item in train_data:
            f.write(f'{item[0]}\n{item[1]}\n\n')
    with open('test.txt', 'w') as f:
        for item in test_data:
            f.write(f'{item[0]}\n{item[1]}\n\n')

if __name__ == "__main__":
    data = read_file()
    write_file(data)
