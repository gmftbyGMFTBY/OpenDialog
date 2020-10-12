from header import *
from config import *

'''self-play for testing knowledge graph topic-driven open-domain dialoge systems'''

def parser_args():
    parser = argparse.ArgumentParser(description='train parameters')
    parser.add_argument('--multi_gpu', type=str, default=None)
    parser.add_argument('--model', type=str, default='bertretrievalkg')
    parser.add_argument('--retrieval_model', type=str, default='bertretrieval')
    parser.add_argument('--method', type=str, default='greedy')
    parser.add_argument('--lang', type=str, default='zh')
    parser.add_argument('--mode', type=str, default='test')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--max_step', type=int, default=20)
    parser.add_argument('--seed', type=float, default=30)
    return parser.parse_args()

def load_agent_model():
    parameter_map, parameter_key = collect_parameter_4_model(args)
    agent = agent_map[args['model']](*parameter_map, **parameter_key)
    agent.load_model(f'ckpt/zh50w/{args["retrieval_model"]}/best.pt')
    return agent

def load_human_model():
    args['model'] = 'bertretrieval'
    parameter_map, parameter_key = collect_parameter_4_model(args)
    agent = agent_map[args['model']](*parameter_map, **parameter_key)
    agent.load_model(f'ckpt/zh50w/{args["retrieval_model"]}/best.pt')
    return agent

def main(**args):
    '''interaction between two agents'''
    ipdb.set_trace()
    agent = load_agent_model()
    human = load_human_model()
    print(f'[!] finish loading the agent and human model for interaction')
    
    # choose the target and init node
    target = input('Target Node: ')
    init_node = input('Init Node: ')
    agent.reset(target, init_node, args['method'])
    
    step, status, data = 0, 'Success', {'msgs': []}
    for i in range(args['max_step']):
        context, done = agent.get_res(data)
        data['msgs'].append({'msg': context})
        if done:
            break
        reply = human.get_res(data)
        data['msgs'].append({'msg': reply})
        step += 1
    else:
        status = 'Failed'
    print(f'[!] interaction from {init_node} to {target} over, status: {status}')
    print('Dialog History:\n===========')
    for idx, (i, j) in enumerate(zip(agent.history, agent.topic)):
        user = 'Agent' if idx % 2 == 0 else 'Human'
        print(f'{user}-[{j}]: {i}')

if __name__ == "__main__":
    args = parser_args()
    args = vars(args)
    print('[!] parameters:')
    print(args)
    
    random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args['seed'])
        
    main(**args)
        