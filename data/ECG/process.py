import json
import numpy as np
import ipdb

def read_json(path):
    with open(path) as f:
        data = json.load(f)
    # processed
    nd, stat = [], []
    for post, resp in data:
        post = ''.join(post[0].strip().split())
        resp = ''.join(resp[0].strip().split())
        stat.append((len(post), len(resp)))
        nd.append((post, resp))
    # stat
    print(f'[!] collect {len(nd)} dialogs')
    l_src, l_tgt = [i[0] for i in stat], [i[1] for i in stat]
    avg_l_src, avg_l_tgt = round(np.mean(l_src), 4), round(np.mean(l_tgt), 4)
    print(f'[!] avg length of the src and tgt: {avg_l_src}/{avg_l_tgt}')
    return nd

def write_from_json_to_text(data, path):
    with open(path, 'w') as f:
        for dialog in data:
            f.write(f'{dialog[0]}\n{dialog[1]}\n\n')

if __name__ == "__main__":
    data = read_json('ecg_train_data.json')
    write_from_json_to_text(data, 'train.txt')
