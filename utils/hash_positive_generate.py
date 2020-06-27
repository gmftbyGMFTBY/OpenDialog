from models import BERTRetrievalAgent
from .utils import *
from tqdm import tqdm
import random
import ipdb, pudb
import os
import pickle
import argparse
from multiprocessing import Process, Pool

def run_the_result(gpu_id, ctx, ctx_, res, output_file):
    # init the bert retrieval agent
    agent = BERTRetrievalAgent(gpu_id, kb=False)
    agent.load_model(f'ckpt/zh50w/bertretrieval/best.pt')
    # revser search
    with open(output_file, 'w') as f:
        pbar = tqdm(list(zip(ctx, ctx_, res)))
        idx = 0
        threshold = 100 
        try:
            for c, c_, r in pbar:
                n_c = agent.reverse_search(c, c_, r)
                f.write(f'CTX: {c}\nHTX: {n_c}\nTGT: {r}\n\n')
                if idx % threshold == 0:
                    pbar.set_description(f'')
                idx += 1
        except:
            print(f'{output_file} access the error')

def check_sub_work(path, ctx, res):
    if not os.path.exists(path):
        print(f'[!] the sub work {path} is not finished')
        return False 
    with open(path) as f:
        data = f.read().split('\n\n')
        data = [i for i in data if i.strip()]
        nd = []
        for session in data:
            ctx, htx, tgt = session.split('\n')
            ctx, htx, tgt = ctx[5:], htx[5:], tgt[5:]
            nd.append((ctx, htx, tgt))
    # check
    flag = len(ctx) == len(nd)
    if flag:
        print(f'[!] the sub work {path} is already finished')
    else:
        print(f'[!] the sub work {path} is not finished')
    return flag

def obtain_length(ctx, max_length):
    utterances = ctx.split('[SEP]')
    l = 0
    chose_utterances = []
    for u in reversed(utterances):
        u = u.strip()
        l += len(u)
        if l < max_length:
            chose_utterances.append(u)
        else:
            break
    if not chose_utterances:
        chose_utterances.append(utterances[-1][-max_length:])
    chose_utterances = list(reversed(chose_utterances))
    u = ' [SEP] '.join(chose_utterances)
    return u

def check_finish(dataset):
    path = f'data/{dataset}'
    files = []
    for file in os.listdir(path):
        if 'hash' in file:
            fp = f'{path}/{file}'
            files.append(fp)
    data = read_text_data(f'data/{dataset}/train.txt')
    ctx_d, res_d = [i[0] for i in data], [i[1] for i in data]
    # read hash file
    hash_data = []
    for file in files:
        with open(file) as f:
            data = f.read().split('\n\n')
            data = [i for i in data if i.strip()]
            for item in data:
                ctx, htx, tgt = item.split('\n')
                ctx, htx, tgt = ctx[5:], htx[5:], tgt[5:]
                hash_data.append((ctx, htx, tgt))
    assert len(hash_data) == len(ctx_d), f'[!] except hash_data have length {len(ctx_d)}, but got {len(hash_data)}'
    # reconstruct the final dataset for hash
    # query by the tgt
    tgt_hash = [i[2] for i in hash_data]
    ctx_hash = [i[0] for i in hash_data]
    htx_hash = [i[1] for i in hash_data]
    final_hash = []
    for c, r in tqdm(list(zip(ctx_d, res_d))):
        # hash ctx
        index = tgt_hash.index(r)
        hh = htx_hash[index]
        # random ctx as negative
        rh = random.choice(ctx_d)
        while rh == hh:
            rh = random.choice(ctx_d)
        item = [c, hh, rh, r]
        final_hash.append(item)
    # write into file
    print(f'[!] write data into the file data/{dataset}/hash.txt')
    with open(f'data/{dataset}/hash.txt', 'w') as f:
        for item in final_hash:
            f.write(f'CTX: {item[0]}\nHTX: {item[1]}\nRTX: {item[2]}\nTGT: {item[3]}\n\n')

if __name__ == "__main__":
    # parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', default='7', type=str)
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--current_worker', default=0, type=int)
    parser.add_argument('--dataset', default='zh50w', type=str)
    parser.add_argument('--model', default='bertretrieval', type=str)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--output', default='data/zh50w/hash.txt', type=str)
    parser.add_argument('--max_query', default=100, type=int)
    args = vars(parser.parse_args())
    print(args)

    check_finish(args['dataset'])
    exit()

    # read the dataset
    data = read_text_data(f'data/{args["dataset"]}/train.txt')
    sub_size = int(len(data) / args['workers'])
    sub_sizes, counter = [], 0
    for i in range(args['workers']):
        if i == args['workers'] - 1:
            sub_sizes.append(data[-(len(data) - counter):])
        else:
            sub_sizes.append(data[counter:counter+sub_size])
        counter += sub_size
    
    i = args['current_worker']
    # check whether finish this subwork
    ctx_, res = [i[0] for i in sub_sizes[i]], [i[1] for i in sub_sizes[i]]
    # ctx max query
    ctx = []
    for i_ in ctx_:
        i_n = obtain_length(i_, args['max_query'])
        ctx.append(i_n)
    flag = check_sub_work(f'{args["output"]}_{i}.txt', ctx, res)
    if flag:
        print(f'[!] this subwork has finished')
    else:
        print(f'[!] this subwork has not finished')
        run_the_result(args['gpu_id'], ctx, ctx_, res, f'{args["output"]}_{i}.txt')
