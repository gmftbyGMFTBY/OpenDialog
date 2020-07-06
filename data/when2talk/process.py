from tqdm import tqdm
import numpy as np
import os
import random
import ipdb
import re

split_pattern = '(；|。|…|……|？|！|～)'
split_pattern_list = ['；', '。', '…', '……', '？', '！', '～']

def read_file(path):
    with open(path) as f:
        data = f.read().split('\n\n')
        data = [i.split('\n') for i in data if i.strip()]
    print(f'[!] collect {len(data)} samples')
    return data

def process_one_dialog(context, response, length_threshold=10):
    length_threshold += 9    # consider the prefix [USER1]/[USER2]
    def cut_sentence(string, prefix):
        sentences = re.split(split_pattern, string)
        sentences = [i for i in sentences if i.strip()]
        rest, cache = [], None
        # filter the special operators
        for sentence in sentences:
            if len(rest) > 0 and (sentence in split_pattern_list or len(rest[-1]) < length_threshold):
                rest[-1] = f'{rest[-1]}{sentence}'
            else:
                rest.append(f'{prefix} {sentence}')
        return rest
    contexts = cut_sentence(context, '[USER1]')
    responses = cut_sentence(response, '[USER2]')
    contexts.extend(responses)
    return contexts

def stat(dialogs):
    '''
    statistic the average length of the responses
    '''
    avg = []
    for dialog in dialogs:
        dialog = ' '.join(dialog)
        response = dialog[dialog.index('[USER2]'):].replace('[USER2]', '').replace('[STP]', '').replace('[SPK]', '').replace(' ', '')
        avg.append(len(response))
    return round(np.mean(avg), 4)

def process_datasets(data):
    dialogs = []
    for dialog in tqdm(data):
        try:
            assert len(dialog) == 2, f'[!] dialog must contain the context and response, but got {len(dialog)}'
        except:
            continue
        rest = process_one_dialog(dialog[0], dialog[1])
        if len(rest) == 2:
            continue
        dialogs.append(rest)
    print(f'[!] processed dataset size: {len(dialogs)}')
    random.shuffle(dialogs)
    return dialogs

def write_file(data, path):
    with open(path, 'w') as f:
        for dialog in data:
            for sentence in dialog:
                f.write(f'{sentence}\n')
            f.write('\n')

if __name__ == "__main__":
    data = process_datasets(read_file('train_.txt'))
    train_data_size = int(len(data) * 0.95)
    train_data = data[:train_data_size]
    test_data = data[train_data_size:]
    print(f'[!] data size(train/test): {len(train_data)}/{len(test_data)}')
    print(f'[!] average length of the responses (train): {stat(train_data)}')
    print(f'[!] average length of the responses (test): {stat(test_data)}')
    write_file(train_data, 'train.txt')
    write_file(test_data, 'test.txt')
