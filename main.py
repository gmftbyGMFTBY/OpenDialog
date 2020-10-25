from header import *
from utils import *
from eval import *
from config import *

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
    parser.add_argument('--max_turn_size', type=int, default=10)
    parser.add_argument('--multi_gpu', type=str, default=None)
    parser.add_argument('--local_rank', type=int)
    return parser.parse_args()

def main(**args):
    if args['mode'] == 'train':
        torch.cuda.set_device(args['local_rank'])
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        
        train_iter = load_dataset(args)
        parameter_map, parameter_key = collect_parameter_4_model(args)
        agent = agent_map[args['model']](*parameter_map, **parameter_key)
        
        sum_writer = SummaryWriter(log_dir=f'rest/{args["dataset"]}/{args["model"]}')
        for i in tqdm(range(args['epoch'])):
            train_loss = agent.train_model(
                train_iter, 
                mode='train',
                recoder=sum_writer,
                idx_=i,
            )
            # only one process save the checkpoint
            if args['local_rank'] == 0:
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
