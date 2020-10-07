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
    parser.add_argument('--multi_gpu', type=str, default=None)
    parser.add_argument('--curriculum', dest='curriculum', action='store_true')
    parser.add_argument('--no-curriculum', dest='curriculum', action='store_false')
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
        test_loss = agent.test_model_batch(test_iter, rest_path)

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
