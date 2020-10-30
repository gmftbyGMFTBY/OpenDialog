from header import *
from utils import *
from dataloader import *

def load_prediction_greedy_dataset(args):
    path = f'data/{args["dataset"]}/{args["mode"]}.txt'
    data = TopicPredictDataset(path, mode=args['mode'], max_len=args['src_len_size'])
    train_sampler = torch.utils.data.distributed.DistributedSampler(data)
    iter_ = DataLoader(data, sampler=train_sampler, shuffle=False, batch_size=args['batch_size'], collate_fn=data.collate)
    if not os.path.exists(data.pp_path):
        data.save_pickle()
    return iter_

def load_seq2seq_trs_dataset(args):
    zh_tokenizer = False
    path = f'data/{args["dataset"]}/{args["mode"]}.txt'
    data = TransformerDataset(path, mode=args['mode'], lang=args['lang'], max_length=args['src_len_size'], n_vocab=args['n_vocab'], zh_tokenizer=zh_tokenizer)
    args['total_steps'] = len(data) * args['epoch'] / args['batch_size']
    if zh_tokenizer is True:
        args['vocab'] = data.vocab
    else:
        args['vocab'] = None
    if args['mode'] == 'train':
        train_sampler = torch.utils.data.distributed.DistributedSampler(data)
        iter_ = DataLoader(data, sampler=train_sampler, shuffle=False, batch_size=args['batch_size'], collate_fn=data.collate)
        return iter_
    else:
        iter_ = DataLoader(data, shuffle=False, batch_size=args['batch_size'], collate_fn=data.collate)
        return iter_

def load_seq2seq_dataset(args):
    path = f'data/{args["dataset"]}/{args["mode"]}.txt'
    data = Seq2SeqDataset(path, mode=args['mode'], lang=args['lang'], n_vocab=args['n_vocab'])
    args['vocab'] = data.vocab
    if args['mode'] == 'train':
        train_sampler = torch.utils.data.distributed.DistributedSampler(data)
        iter_ = DataLoader(data, sampler=train_sampler, shuffle=False, batch_size=args['batch_size'], collate_fn=data.collate)
        return iter_
    else:
        iter_ = DataLoader(data, shuffle=False, batch_size=args['batch_size'], collate_fn=data.collate)
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

def load_lccc_ir_dataset(args):
    path = f'data/{args["dataset"]}/{args["mode"]}.txt'
    if args['mode'] in ['train']:
        data = WBDataset('/home/lt/data/LCCD_GPT', path, samples=1)
        iter_ = DataLoader(data, shuffle=True, batch_size=args['batch_size'], collate_fn=data.collate)
    else:
        # NOTE: TEST PROCEDURE IS ERROR, WAIT TO REWRITE
        data = WBDataset('/home/lt/data/LCCD_GPT', path, samples=9)
        iter_ = DataLoader(data, shuffle=False, batch_size=args['batch_size'], collate_fn=data.collate)
    return iter_

def load_bert_na_dataset(args):
    path = f'data/{args["dataset"]}/{args["mode"]}.txt'
    data = BERTNADataset(path, mode=args['mode'], max_size=16)
    iter_ = DataLoader(data, shuffle=True, batch_size=args['batch_size'], collate_fn=data.collate)
    return iter_

def load_uni_dataset(args):
    if args['mode'] in ['train']:
        data = UNIDataset('/data/lantian/data/LCCD_GPT', f'data/{args["dataset"]}/LCCC-base.json', samples=1)
        args['total_steps'] = len(data) * args['epoch'] / args['batch_size']
        train_sampler = torch.utils.data.distributed.DistributedSampler(data)
        iter_ = DataLoader(data, sampler=train_sampler, batch_size=args['batch_size'], collate_fn=data.collate)
    else:
        data = UNIDataset('/home/lt/data/LCCD_GPT', f'data/{args["dataset"]}/LCCC-base_test.json', samples=9)
        iter_ = DataLoader(data, shuffle=True, batch_size=args['batch_size'], collate_fn=data.collate)
    return iter_

def load_lccc_dataset(args):
    path = f'data/{args["dataset"]}/{args["mode"]}.txt'
    data = FTWBDataset('/home/lt/data/LCCD_GPT', args['mode'], path)
    iter_ = DataLoader(data, shuffle=True, batch_size=args['batch_size'], collate_fn=data.collate)
    return iter_

def load_gpt2_dataset(args):
    path = f'data/{args["dataset"]}/{args["mode"]}.txt'
    if args['mode'] in ['train', 'dev']:
        data = GPT2Dataset(path, mode=args['mode'], src_len_size=args['src_len_size'], tgt_len_size=args['tgt_len_size'], lang=args['lang'])
        args['total_steps'] = len(data) * args['epoch'] / args['batch_size']
        train_sampler = torch.utils.data.distributed.DistributedSampler(data)
        iter_ = DataLoader(data, sampler=train_sampler, shuffle=False, batch_size=args['batch_size'], collate_fn=data.collate)
    else:
        data = GPT2Dataset(path, mode=args['mode'], src_len_size=args['src_len_size'], tgt_len_size=args['tgt_len_size'], lang=args['lang'])
        args['total_steps'] = 100
        iter_ = DataLoader(data, shuffle=False, batch_size=args['batch_size'], collate_fn=data.collate)
    if not os.path.exists(data.pp_path):
        data.save_pickle()
    return iter_

def load_gpt2v2rl_dataset(args):
    path = f'data/{args["dataset"]}/{args["mode"]}.txt'
    if args['mode'] in ['train', 'dev']:
        data = GPT2V2RLDataset(path, mode=args['mode'], src_len_size=args['src_len_size'], tgt_len_size=args['tgt_len_size'], lang=args['lang'], candidate=5)
        train_sampler = torch.utils.data.distributed.DistributedSampler(data)
        iter_ = DataLoader(data, sampler=train_sampler, shuffle=False, batch_size=args['batch_size'], collate_fn=data.collate)
    else:
        data = GPT2V2RLDataset(path, mode=args['mode'], src_len_size=args['src_len_size'], tgt_len_size=args['tgt_len_size'], lang=args['lang'], candidate=5)
        iter_ = DataLoader(data, shuffle=False, batch_size=args['batch_size'], collate_fn=data.collate)
    if not os.path.exists(data.pp_path):
        data.save_pickle()
    return iter_

def load_gpt2v2_dataset(args):
    path = f'data/{args["dataset"]}/{args["mode"]}.txt'
    if args['mode'] in ['train', 'dev']:
        data = GPT2V2Dataset(path, mode=args['mode'], src_len_size=args['src_len_size'], tgt_len_size=args['tgt_len_size'], lang=args['lang'], candidate=5)
        args['total_steps'] = len(data) * args['epoch'] / args['batch_size']
        train_sampler = torch.utils.data.distributed.DistributedSampler(data)
        iter_ = DataLoader(data, sampler=train_sampler, shuffle=False, batch_size=args['batch_size'], collate_fn=data.collate)
    else:
        data = GPT2V2Dataset(path, mode=args['mode'], src_len_size=args['src_len_size'], tgt_len_size=args['tgt_len_size'], lang=args['lang'], candidate=5)
        args['total_steps'] = 100
        iter_ = DataLoader(data, shuffle=False, batch_size=args['batch_size'], collate_fn=data.collate)
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
    path = f'data/{args["dataset"]}/{args["mode"]}.txt'
    if args['mode'] in ['train', 'dev']:
        data = BERTIRDataset(path, mode=args['mode'], samples=1, max_len=512, negative_aspect='overall')
        iter_ = DataLoader(data, shuffle=True, batch_size=args['batch_size'], collate_fn=bert_ir_train_collate_fn)
    else:
        data = BERTIRDataset(path, mode=args['mode'], samples=1, max_len=512, negative_aspect='overall')
        iter_ = DataLoader(data, shuffle=True, batch_size=args['batch_size'], collate_fn=bert_ir_test_collate_fn)
    if not os.path.exists(data.pp_path):
        data.save_pickle()
    return iter_

def load_bert_ir_dis_dataset(args):
    path = f'data/{args["dataset"]}/{args["mode"]}.txt'
    if args['mode'] in ['train', 'dev']:
        data = BERTIRDISDataset(path, mode=args['mode'], samples=1, max_len=512)
        iter_ = DataLoader(data, shuffle=True, batch_size=args['batch_size'], collate_fn=bert_ir_dis_train_collate_fn)
    else:
        data = BERTIRDISDataset(path, mode=args['mode'], samples=9, max_len=512)
        iter_ = DataLoader(data, shuffle=True, batch_size=args['batch_size'], collate_fn=bert_ir_test_collate_fn)
    if not os.path.exists(data.pp_path):
        data.save_pickle()
    return iter_

def load_bert_ir_mc_dataset(args):
    path = f'data/{args["dataset"]}/{args["mode"]}.txt'
    samples = 1 if args['mode'] == 'train' else 9
    data = BERTMCDataset(path, mode=args['mode'], samples=samples, max_len=512, harder=False)
    if args['mode'] in ['train', 'dev']:
        iter_ = DataLoader(data, shuffle=True, batch_size=args['batch_size'], collate_fn=bert_ir_mc_collate_fn)
    else:
        iter_ = DataLoader(data, shuffle=True, batch_size=args['batch_size'], collate_fn=bert_ir_mc_test_collate_fn)
    if not os.path.exists(data.pp_path):
        data.save_pickle()
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
        data = BERTIRDataset(path, mode=args['mode'], samples=9, negative_aspect='overall')
        iter_ = DataLoader(data, shuffle=True, batch_size=args['batch_size'], collate_fn=bert_ir_test_collate_fn)
    if not os.path.exists(data.pp_path):
        data.save_pickle()
    return iter_

def load_rubert_irbi_dataset(args):
    path = f'data/{args["dataset"]}/{args["mode"]}.txt'
    data = RURetrievalDataset(path, mode=args['mode'], max_len=50, max_turn_size=args['max_turn_size'])
    if args['mode'] in ['train', 'dev']:
        train_sampler = torch.utils.data.distributed.DistributedSampler(data)
        iter_ = DataLoader(
            data, shuffle=False, batch_size=args['batch_size'], collate_fn=data.collate, 
            sampler=train_sampler
        )
    else:
        iter_ = DataLoader(
            data, shuffle=False, batch_size=args['batch_size'], collate_fn=data.collate,
        )
    if not os.path.exists(data.pp_path):
        data.save_pickle()
    args['total_steps'] = len(data) * args['epoch'] / args['batch_size']
    args['bimodel'] = 'ru-no-compare'
    return iter_

# ================================================================================ #
def load_bert_irbi_dataset(args):
    path = f'data/{args["dataset"]}/{args["mode"]}.txt'
    # data = BERTIRBIDataset(path, mode=args['mode'], max_len=args['src_len_size'])
    data = RetrievalDataset(path, mode=args['mode'], max_len=args['src_len_size'])
    if args['mode'] in ['train', 'dev']:
        train_sampler = torch.utils.data.distributed.DistributedSampler(data)
        iter_ = DataLoader(
            data, shuffle=False, batch_size=args['batch_size'], collate_fn=data.collate, 
            sampler=train_sampler
        )
    else:
        iter_ = DataLoader(
            data, shuffle=False, batch_size=args['batch_size'], collate_fn=data.collate,
        )
    if not os.path.exists(data.pp_path):
        data.save_pickle()
    args['total_steps'] = len(data) * args['epoch'] / args['batch_size']
    args['bimodel'] = args['model']
    return iter_

def load_bert_irbicomp_dataset(args):
    path = f'data/{args["dataset"]}/{args["mode"]}.txt'
    data = RetrievalDataset(path, mode=args['mode'], max_len=args['src_len_size'])
    if args['mode'] in ['train', 'dev']:
        train_sampler = torch.utils.data.distributed.DistributedSampler(data)
        iter_ = DataLoader(
            data, shuffle=False, batch_size=args['batch_size'], collate_fn=data.collate, 
            sampler=train_sampler
        )
    else:
        iter_ = DataLoader(
            data, shuffle=False, batch_size=args['batch_size'], collate_fn=data.collate,
        )
    if not os.path.exists(data.pp_path):
        data.save_pickle()
    args['total_steps'] = len(data) * args['epoch'] / args['batch_size']
    args['bimodel'] = args['model']
    return iter_
# ================================================================================ #

def load_bert_ir_dataset(args):
    path = f'data/{args["dataset"]}/{args["mode"]}.txt'
    # path = f'data/{args["dataset"]}/LCCC-base.json'
    if args['mode'] in ['train', 'dev']:
        data = BERTIRDataset(path, mode=args['mode'], samples=1, max_len=args['src_len_size'], negative_aspect='coherence')
        train_sampler = torch.utils.data.distributed.DistributedSampler(data)
        iter_ = DataLoader(data, shuffle=False, batch_size=args['batch_size'], collate_fn=data.collate, sampler=train_sampler)
    else:
        data = BERTIRDataset(path, mode=args['mode'], samples=9, max_len=args['src_len_size'], negative_aspect='coherence')
        iter_ = DataLoader(data, shuffle=True, batch_size=args['batch_size'], collate_fn=data.collate)
    if not os.path.exists(data.pp_path):
        data.save_pickle()
    return iter_

def load_bert_ir_multiview_dataset(args):
    path = f'data/{args["dataset"]}/{args["mode"]}.txt'
    if args['mode'] in ['train', 'dev']:
        data = BERTIRDataset(path, mode=args['mode'], samples=1, negative_aspect='overall')
        iter_ = DataLoader(data, shuffle=True, batch_size=args['batch_size'], collate_fn=bert_ir_train_collate_fn)
    else:
        data = BERTIRDataset(path, mode=args['mode'], samples=9, negative_aspect='hard')
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
