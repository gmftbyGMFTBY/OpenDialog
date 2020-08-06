from header import *
from utils import *
from dataloader import *

def load_seq2seq_dataset(args):
    path = f'data/{args["dataset"]}/{args["mode"]}.csv'
    if args['mode'] == 'train':
        data = DialogDataset(path, mode=args['mode'], n_vocab=args['n_vocab'], src_len_size=args['src_len_size'], tgt_len_size=args['tgt_len_size'])
        args['vocab_size'] = len(data.vocab)
        args['vocab'] = data.vocab
    else:
        data = DialogDataset(path, mode=args['mode'], src_len_size=args['src_len_size'], tgt_len_size=args['tgt_len_size'], vocab=args['vocab'])
    iter_ = DataLoader(data, shuffle=True, batch_size=args['batch_size'], collate_fn=dialog_collate_fn)
    return iter_

def load_gpt2rl_dataset(args):
    path = f'data/{args["dataset"]}/{args["mode"]}.txt'
    data = GPT2RLDataset(path, src_len_size=args['src_len_size'], tgt_len_size=args['tgt_len_size'])
    iter_ = GPT2RLDataLoader(data, shuffle=True, batch_size=args['batch_size'])
    if not os.path.exists(data.pp_path):
        data.save_pickle()
    return iter_

def load_gpt2lm_dataset(args):
    path = f'data/{args["dataset"]}/{args["mode"]}.txt'
    data = GPT2LMDataset(path)
    args['total_steps'] = len(data) * args['epoch'] / args['batch_size']
    iter_ = DataLoader(data, shuffle=False, batch_size=args['batch_size'], collate_fn=gpt2_lm_collate_fn)
    if not os.path.exists(data.pp_path):
        data.save_pickle()
    return iter_

def load_pfgpt2_dataset(args):
    path = f'data/{args["dataset"]}/{args["mode"]}.txt'
    if args['mode'] in ['train', 'train_trs', 'dev']:
        data = GPT2Dataset(path, mode=args['mode'], src_len_size=args['src_len_size'], tgt_len_size=args['tgt_len_size'], lang=args['lang'])
        args['total_steps'] = len(data) * args['epoch'] / args['batch_size']
        iter_ = DataLoader(data, shuffle=False, batch_size=args['batch_size'], collate_fn=gpt2_train_collate_fn)
    else:
        data = GPT2Dataset(path, mode=args['mode'], src_len_size=args['src_len_size'], tgt_len_size=args['tgt_len_size'], lang=args['lang'])
        iter_ = DataLoader(data, shuffle=True, batch_size=args['batch_size'], collate_fn=gpt2_test_collate_fn)
    if not os.path.exists(data.pp_path):
        data.save_pickle()
    return iter_

def load_gpt2retrieval_dataset(args):
    path = f'data/{args["dataset"]}/{args["mode"]}.txt'
    if args['mode'] in ['train', 'dev']:
        data = GPT2Dataset(path, mode=args['mode'], src_len_size=args['src_len_size'], tgt_len_size=args['tgt_len_size'], lang=args['lang'], ensemble=True, candidates_k=2)
        args['total_steps'] = len(data) * args['epoch'] / args['batch_size']
        iter_ = DataLoader(data, shuffle=True, batch_size=args['batch_size'], collate_fn=gpt2retrieval_train_collate_fn)
    else:
        data = GPT2Dataset(path, mode=args['mode'], src_len_size=args['src_len_size'], tgt_len_size=args['tgt_len_size'], lang=args['lang'], ensemble=True, candidates_k=2)
        iter_ = DataLoader(data, shuffle=True, batch_size=args['batch_size'], collate_fn=gpt2retrieval_test_collate_fn)
    if not os.path.exists(data.pp_path):
        data.save_pickle()
    return iter_

def load_when2talk_dataset(args):
    path = f'data/{args["dataset"]}/{args["mode"]}.txt'
    if args['mode'] in ['train', 'dev']:
        data = When2talkDataset(path, mode=args['mode'], src_len_size=args['src_len_size'], tgt_len_size=args['tgt_len_size'], lang=args['lang'])
        args['total_steps'] = len(data) * args['epoch'] / args['batch_size']
        iter_ = DataLoader(data, shuffle=False, batch_size=args['batch_size'], collate_fn=gpt2_train_collate_fn)
    else:
        data = When2talkDataset(path, mode=args['mode'], src_len_size=args['src_len_size'], tgt_len_size=args['tgt_len_size'], lang=args['lang'])
        iter_ = DataLoader(data, shuffle=False, batch_size=args['batch_size'], collate_fn=gpt2_test_collate_fn)
    return iter_

def load_gpt2_dataset(args):
    path = f'data/{args["dataset"]}/{args["mode"]}.txt'
    if args['mode'] in ['train', 'dev']:
        data = GPT2Dataset(path, mode=args['mode'], src_len_size=args['src_len_size'], tgt_len_size=args['tgt_len_size'], lang=args['lang'], reversed=args['mmi'])
        args['total_steps'] = len(data) * args['epoch'] / args['batch_size']
        iter_ = DataLoader(data, shuffle=False, batch_size=args['batch_size'], collate_fn=gpt2_train_collate_fn)
    else:
        data = GPT2Dataset(path, mode=args['mode'], src_len_size=args['src_len_size'], tgt_len_size=args['tgt_len_size'], lang=args['lang'], reversed=args['mmi'])
        iter_ = DataLoader(data, shuffle=False, batch_size=args['batch_size'], collate_fn=gpt2_test_collate_fn)
    if not os.path.exists(data.pp_path):
        data.save_pickle()
    return iter_

def load_kwgpt2_dataset(args):
    path = f'data/{args["dataset"]}/{args["mode"]}.txt'
    if args['mode'] in ['train', 'dev']:
        data = KWGPT2Dataset(path, mode=args['mode'], src_len_size=args['src_len_size'], tgt_len_size=args['tgt_len_size'], lang=args['lang'])
        args['total_steps'] = len(data) * args['epoch'] / args['batch_size']
        iter_ = DataLoader(data, shuffle=False, batch_size=args['batch_size'], collate_fn=gpt2_train_collate_fn)
    else:
        data = KWGPT2Dataset(path, mode=args['mode'], src_len_size=args['src_len_size'], tgt_len_size=args['tgt_len_size'], lang=args['lang'])
        iter_ = DataLoader(data, shuffle=True, batch_size=args['batch_size'], collate_fn=gpt2_test_collate_fn)
    if not os.path.exists(data.pp_path):
        data.save_pickle()
    return iter_

def load_multigpt2_dataset(args):
    path = f'data/{args["dataset"]}/{args["mode"]}.csv'
    if args['mode'] in ['train', 'dev']:
        data = MultiGPT2Dataset(path, mode=args['mode'], src_len_size=args['src_len_size'], tgt_len_size=args['tgt_len_size'])
        args['total_steps'] = len(data) * args['epoch'] / args['batch_size']
        iter_ = DataLoader(data, shuffle=True, batch_size=args['batch_size'], collate_fn=multigpt2_train_collate_fn)
    else:
        data = MultiGPT2Dataset(path, mode=args['mode'], src_len_size=args['src_len_size'], tgt_len_size=args['tgt_len_size'])
        iter_ = DataLoader(data, shuffle=True, batch_size=args['batch_size'], collate_fn=multigpt2_test_collate_fn)
    return iter_

def load_ir_dataset(args):
    path = f'data/{args["dataset"]}/{args["mode"]}.csv'
    pickle_path = f'data/{args["dataset"]}/{args["mode"]}.pkl'
    print(f'[!] load dataset from {path} and {pickle_path}')
    data = IRDataset(path, pickle_path, mode=args['mode'])
    iter_ = DataLoader(data, shuffle=True, batch_size=args['batch_size'], collate_fn=ir_collate_fn)
    return iter_

def load_bert_ir_multi_dataset(args):
    path = f'data/{args["dataset"]}/{args["mode"]}.txt'
    data = BERTIRMultiDataset(path, max_len=512, mode=args['mode'])
    iter_ = BERTIRMultiDataLoader(data, shuffle=True, batch_size=args['batch_size'])
    if not os.path.exists(data.pp_path):
        data.save_pickle()
    return iter_

def load_bert_ir_cl_dataset(args):
    path = f'data/{args["dataset"]}/{args["mode"]}.txt'
    args['curriculum'] = True
    if args['mode'] in ['train', 'dev']:
        data = BERTIRCLDataset(path, mode=args['mode'], samples=1)
        T = int(len(data) * args['epoch'] / args['batch_size']) + 1
        iter_ = BERTIRCLDataLoader(data, T, batch_size=args['batch_size'])
    else:
        data = BERTIRCLDataset(path, mode=args['mode'], samples=9)
        iter_ = DataLoader(data, shuffle=True, batch_size=args['batch_size'], collate_fn=bert_ir_test_collate_fn)
    if not os.path.exists(data.pp_path):
        data.save_pickle()
    return iter_

def load_bert_ir_dataset(args):
    path = f'data/{args["dataset"]}/{args["mode"]}.txt'
    if args['mode'] in ['train', 'dev']:
        data = BERTIRDataset(path, mode=args['mode'], samples=1, negative_aspect='overall')
        iter_ = DataLoader(data, shuffle=True, batch_size=args['batch_size'], collate_fn=bert_ir_train_collate_fn)
    else:
        data = BERTIRDataset(path, mode=args['mode'], samples=9, negative_aspect='overall')
        iter_ = DataLoader(data, shuffle=True, batch_size=args['batch_size'], collate_fn=bert_ir_test_collate_fn)
    if not os.path.exists(data.pp_path):
        data.save_pickle()
    return iter_

def load_bert_ir_multiview_dataset(args):
    '''
    batch size is different:
    (1) for fluency, diversity, naturalness, relatedness, coherence, batch_size is 64
    (2) for overall aspect: half of the batch_size (32)
    '''
    if args['mode'] in ['train', 'dev']:
        iters = []
        for aspect in ['coherence', 'fluency', 'diversity', 'naturalness', 'relatedness', 'overall']:
            path = f'data/{args["dataset"]}/{args["mode"]}.txt'
            # samples = 1 if aspect in ['diversity', 'overall'] else 5
            samples = 5
            if args['mode'] in ['train', 'dev']:
                data = BERTIRDataset(path, mode=args['mode'], samples=samples, negative_aspect=aspect, reduce=True, reduce_num=50000)
                iter_ = DataLoader(data, shuffle=True, batch_size=args['batch_size'], collate_fn=bert_ir_train_collate_fn)
                iters.append(iter_)
            if not os.path.exists(data.pp_path):
                data.save_pickle()
            print(f'[!] process the negative aspect {aspect} over')
        return iters
    else:
        path = f'data/{args["dataset"]}/{args["mode"]}.txt'
        data = BERTIRDataset(path, mode=args['mode'], samples=9, negative_aspect='overall')
        iter_ = DataLoader(data, shuffle=True, batch_size=args['batch_size'], collate_fn=bert_ir_test_collate_fn)
        if not os.path.exists(data.pp_path):
            data.save_pickle()
        return iter_

def load_pone_dataset(args):
    path = f'data/{args["dataset"]}/{args["mode"]}_pone.txt'
    if args['mode'] in ['train', 'dev']:
        data = PONEDataset(path, mode=args['mode'], lang=args['lang'], samples=10, bert=False)
        iter_ = DataLoader(data, shuffle=True, batch_size=args['batch_size'], collate_fn=pone_train_collate_fn)
        if not os.path.exists(data.pp_path):
            data.save_pickle()
    else:
        paths = [f'data/annotator/{args["dataset"]}/sample-100.txt',
                 f'data/annotator/{args["dataset"]}/sample-100-tgt.txt',
                 f'data/annotator/{args["dataset"]}/pred.txt']
        human_annotations = [
                f'data/annotator/{args["dataset"]}/1/annotate.csv',
                f'data/annotator/{args["dataset"]}/2/annotate.csv',
                f'data/annotator/{args["dataset"]}/3/annotate.csv',
                ]
        data = PONEDataset(
                paths,
                mode=args['mode'], lang=args['lang'], bert=False, 
                human_annotations=human_annotations)
        iter_ = DataLoader(data, shuffle=False, batch_size=args['batch_size'], collate_fn=pone_test_collate_fn)
    return iter_

def load_bert_logic_dataset(args):
    path = f'data/{args["dataset"]}/{args["mode"]}.txt'
    data = BERTLOGICDataset(path, mode=args['mode'], samples=9)
    if args['mode'] in ['train', 'dev']:
        iter_ = DataLoader(data, shuffle=True, batch_size=args['batch_size'], collate_fn=bert_ir_train_collate_fn)
    else:
        iter_ = DataLoader(data, shuffle=True, batch_size=args['batch_size'], collate_fn=bert_ir_test_collate_fn)
    if not os.path.exists(data.pp_path):
        data.save_pickle()
    return iter_

def load_bert_nli_dataset(args):
    path = f'data/{args["dataset"]}/{args["mode"]}.jsonl'
    data = BERTNLIDataset(path)
    # save preprocessed file
    if not os.path.exists(data.pp_path):
        data.save_pickle()
    iter_ = DataLoader(data, shuffle=True, batch_size=args['batch_size'], collate_fn=nli_collate_fn)
    return iter_
