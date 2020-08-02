import json
from tqdm import tqdm
import jsonlines
import ipdb
from copy import deepcopy

def read_file(path):
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    return data

def read_jsonlines(path):
    with open(path, "r+", encoding='utf-8') as f:
        data = []
        for item in jsonlines.Reader(f):
            data.append(item)
    return data

def collect_dialogs(item):
    key, value = item
    if value is None:
        return [[key]]
    subdialog = []
    for i in value.items():
        rest = collect_dialogs(i)
        subdialog.extend([[key] + i for i in rest])
    return subdialog

def write_file(dialogs, utterances):
    with open('train.txt', 'w') as f:
        for dialog in tqdm(dialogs):
            dialog = list(map(int, dialog))
            dialog = [utterances[i]['message'] for i in dialog]
            for u in dialog:
                f.write(f'{u}\n')
            f.write('\n')

if __name__ == "__main__":
    relation = read_file('trees.json')
    utterances = read_jsonlines('corpus.jsonl')
    dialogs = []
    for item in tqdm(relation):
        key, value = list(item.items())[0]
        dialogs.extend(collect_dialogs((key, value)))
    write_file(dialogs, utterances)
