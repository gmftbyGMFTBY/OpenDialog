import json
from tqdm import tqdm

'''
See the README.md for the training samples details
Train: 
    1. cnsd_snli_v1.0.train.jsonl
    2. cnsd_multil_train.jsonl
Dev:
    1. cnsd_snli_v1.0.dev.jsonl
    2. cnsd_multi_dev_matched.jsonl
    3. cnsd_multi_dev.mismatched.jsonl
Test:
    cnsd_snli_v1.0.test.jsonl
'''

def read_file(path):
    data = []
    with open(path) as f:
        for line in tqdm(f.readlines()):
            line = line.strip()
            line = json.loads(line)
            data.append(line)
    print(f'[!] {len(data)} samples')
    return data

def write_file(path, data):
    with open(path, 'w') as f:
        for item in data:
            item = json.dumps(item)
            f.write(f'{item}\n')

if __name__ == "__main__":
    data = []
    data.extend(read_file('cnsd_snli_v1.0.train.jsonl'))
    data.extend(read_file('cnsd_multil_train.jsonl'))
    write_file('train.jsonl', data)
    
    data = []
    data.extend(read_file('cnsd_snli_v1.0.dev.jsonl'))
    data.extend(read_file('cnsd_multil_dev_matched.jsonl'))
    data.extend(read_file('cnsd_multil_dev_mismatched.jsonl'))
    write_file('dev.jsonl', data)
    
    data = []
    data.extend(read_file('cnsd_snli_v1.0.test.jsonl'))
    write_file('test.jsonl', data)
