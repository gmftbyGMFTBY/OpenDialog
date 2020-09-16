from header import *
from utils import *
from models import *
from eval import *
from dataloader import *
from dataset_init import * 

def parser_args():
    parser = argparse.ArgumentParser(description='train parameters')
    parser.add_argument('--dataset', default='zh50w', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--n_vocab', type=int, default=50000)
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--lang', type=str, default='zh')
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--seed', type=float, default=30)
    parser.add_argument('--src_len_size', type=int, default=300)
    parser.add_argument('--tgt_len_size', type=int, default=50)
    parser.add_argument('--multi_gpu', type=str, default=None)
    parser.add_argument('--curriculum', dest='curriculum', action='store_true')
    parser.add_argument('--no-curriculum', dest='curriculum', action='store_false')
    parser.add_argument('--local_rank', type=int)
    return parser.parse_args()

def load_dataset(args):
    if args['model'] == 'DualLSTM':
        return load_ir_dataset(args)
    elif args['model'] == 'seq2seq':
        return load_seq2seq_dataset(args)
    elif args['model'] == 'kwgpt2':
        return load_kwgpt2_dataset(args)
    elif args['model'] == 'gpt2':
        args['mmi'] = False
        return load_gpt2_dataset(args)
    elif args['model'] == 'pone':
        return load_pone_dataset(args)
    elif args['model'] == 'pfgpt2':
        return load_pfgpt2_dataset(args)
    elif args['model'] == 'gpt2_mmi':
        args['mmi'] = True
        return load_gpt2_dataset(args)
    elif args['model'] == 'when2talk':
        return load_when2talk_dataset(args)
    elif args['model'] == 'gpt2retrieval':
        return load_gpt2retrieval_dataset(args)
    elif args['model'] == 'gpt2lm':
        return load_gpt2lm_dataset(args)
    elif args['model'] in ['gpt2gan', 'gpt2gan_v2']:
        return load_gpt2rl_dataset(args)
    elif args['model'] == 'multigpt2':
        return load_multigpt2_dataset(args)
    elif args['model'] == 'bertretrieval':
        return load_bert_ir_dataset(args)
    elif args['model'] == 'lcccir':
        return load_lccc_ir_dataset(args)
    elif args['model'] == 'lccc':
        return load_lccc_dataset(args)
    elif args['model'] == 'bertretrieval_multiview':
        return load_bert_ir_multiview_dataset(args)
    elif args['model'] == 'bertretrieval_cl':
        return load_bert_ir_cl_dataset(args)
    elif args['model'] in ['bertmc', 'bertmcf']:
        return load_bert_ir_mc_dataset(args)
    elif args['model'] == 'bertretrieval_dis':
        return load_bert_ir_dis_dataset(args)
    elif args['model'] == 'bertnli':
        return load_bert_nli_dataset(args)
    elif args['model'] == 'bertlogic':
        return load_bert_logic_dataset(args)
    elif args['model'] == 'uni':
        return load_uni_dataset(args)
    elif args['model'] == 'bert_na':
        return load_bert_na_dataset(args)
    elif args['model'] == 'transformer':
        return load_seq2seq_trs_dataset(args)
    else:
        raise Exception(f'[!] got unknow model: {args["model"]}')

def main(**args):
    # speed up
    # torch.cuda.set_device(args['local_rank'])
    # torch.distributed.init_process_group(backend='nccl', init_method='env://')
    
    # backup_mode = args['mode']
    # args['mode'] = 'train'
    # train_iter = load_dataset(args)
    # args['mode'] = backup_mode

    agent_map = {
        'DualLSTM': DualLSTMAgent, 
        'seq2seq': Seq2SeqAgent,
        'gpt2': GPT2Agent,
        'gpt2retrieval': GPT2RetrievalAgent,
        'pfgpt2': PFGPT2Agent,
        'kwgpt2': KWGPT2Agent,
        'gpt2_mmi': GPT2Agent,
        'when2talk': When2TalkAgent,
        'gpt2lm': GPT2Agent,
        'multigpt2': MultiGPT2Dataset,
        'bertretrieval': BERTRetrievalAgent,
        'bertretrieval_multiview': BERTRetrievalAgent,
        'bertretrieval_cl': BERTRetrievalCLAgent,
        'bertlogic': BERTRetrievalAgent,
        'bertnli': BERTNLIAgent,
        'gpt2gan': GPT2RLAgent,
        'gpt2gan_v2': GPT2RLAgent_V2,
        'pone': PONEAgent,
        'bertmc': BERTMCAgent,
        'bertmcf': BERTMCAgent,
        'lccc': LCCCFTAgent,
        'bertretrieval_dis': BERTRetrievalDISAgent,
        'lcccir': LCCCIRAgent,
        'uni': UNIAgent,
        'bert_na': BERTNAAgent,
        'transformer': TransformerAgent,
    }

    if args['mode'] == 'train':
        train_iter = load_dataset(args)
        parameter_map, parameter_key = collect_parameter_4_model(args)
        agent = agent_map[args['model']](*parameter_map, **parameter_key)
        
        sum_writer = SummaryWriter(log_dir=f'rest/{args["dataset"]}/{args["model"]}')
        if args['curriculum']:
            # 1. collect the loss for resetting the order (bertretrieval model)
            train_iter.forLoss = True
            loss_path = f'rest/{args["dataset"]}/{args["model"]}/loss.pkl'
            if os.path.exists(loss_path):
                with open(loss_path, 'rb') as f:
                    print(f'[!] load the losses priority from {loss_path}')
                    losses = pickle.load(f)
            else:
                agent_ = agent_map['bertretrieval'](*parameter_map, **parameter_key)
                agent_.load_model(f'ckpt/{args["dataset"]}/bertretrieval/best.pt')
                losses = agent_.predict(train_iter, loss_path)
                del agent_    # delete the agent_, which is already useless
            train_iter.reset_order(losses)
            # 2. curriculum learning
            train_iter.forLoss = False
            agent.train_model(train_iter, mode='train', recoder=sum_writer)
            agent.save_model(f'ckpt/{args["dataset"]}/{args["model"]}/best.pt')
        else:
            for i in tqdm(range(args['epoch'])):
                train_loss = agent.train_model(
                    train_iter, 
                    mode='train',
                    recoder=sum_writer,
                    idx_=i,
                )
                agent.save_model(f'ckpt/{args["dataset"]}/{args["model"]}/best.pt')
        sum_writer.close()
    else:
        test_iter = load_dataset(args)
        parameter_map, parameter_key = collect_parameter_4_model(args)
        agent = agent_map[args['model']](*parameter_map, **parameter_key)
        agent.load_model(f'ckpt/{args["dataset"]}/{args["model"]}/best.pt')
        rest_path = f'rest/{args["dataset"]}/{args["model"]}/rest.txt'
        test_loss = agent.test_model(test_iter, rest_path)

if __name__ == "__main__":
    args = parser_args()
    args = vars(args)
    print('[!] parameters:')
    print(args)
    print(args, file=open(f'ckpt/{args["dataset"]}/{args["model"]}/param.txt', 'w'))

    random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args['seed'])

    main(**args)
