from header import *
from config import *

'''self-play for testing knowledge graph topic-driven open-domain dialoge systems'''

def parser_args():
    parser = argparse.ArgumentParser(description='train parameters')
    parser.add_argument('--multi_gpu', type=str, default=None)
    parser.add_argument('--retrieval_model', type=str, default='bertretrieval')
    parser.add_argument('--method', type=str, default='greedy')
    parser.add_argument('--lang', type=str, default='zh')
    parser.add_argument('--mode', type=str, default='test')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--max_step', type=int, default=20)
    parser.add_argument('--seed', type=float, default=30)
    parser.add_argument('--history_length', type=int, default=5)
    parser.add_argument('--talk_samples', type=int, default=128)
    return parser.parse_args()

def load_agent_model():
    parameter_map, parameter_key = collect_parameter_4_model(args)
    agent = agent_map[args['model']](*parameter_map, **parameter_key)
    agent.load_model(f'ckpt/zh50w/{args["retrieval_model"]}/best.pt')
    return agent

def load_human_model():
    args['model'] = 'bertretrievalenv'
    parameter_map, parameter_key = collect_parameter_4_model(args)
    agent = agent_map[args['model']](*parameter_map, **parameter_key)
    agent.load_model(f'ckpt/zh50w/{args["retrieval_model"]}/best.pt')
    return agent

def neighborhood(G, node, n, size=10):
    '''https://stackoverflow.com/questions/22742754/finding-the-n-degree-neighborhood-of-a-node'''
    path_lengths = nx.single_source_shortest_path_length(G, node, cutoff=n)
    nodes = [node for node, length in path_lengths.items() if length == n]
    return random.sample(nodes, size)

def isEnd(utterance, target):
    if target in utterance:
        return True
    else:
        return False

def main(source, target, **args):
    '''interaction between two agents'''
    path = nx.dijkstra_path(args['wordnet'], source=source, target=target)
    print(path)
    agent.reset(target, source, path)
    
    step, status, data, conversation = 0, 'Success', {'msgs': []}, []
    while True:
        context = agent.get_res(data)
        data['msgs'].append({'msg': context})
        conversation.append(('Agent', context))
        data['msgs'] = data['msgs'][-args['history_length']:]
        
        done = isEnd(context, target)
        if done:
            status = 'Success'
            break
        
        reply = human.get_res(data)
        data['msgs'].append({'msg': reply})
        conversation.append(('Human', reply))
        data['msgs'] = data['msgs'][-args['history_length']:]
        
        step += 1
        if step >= args['max_step']:
            status = 'Failed'
            break
    
    string = f'[!] interaction from {source} to {target} over, status: {status}'
    print(string)
    
    string = '========== Dialog History ==========='
    print(string)
    for idx, (speaker, utterance) in enumerate(conversation):
        string = f'{speaker}: {utterance}'
        print(string)

if __name__ == "__main__":
    args = parser_args()
    args = vars(args)
    
    random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args['seed'])
    
    # load word net
    with open('data/wordnet.pkl', 'rb') as f:
        wordnet = pickle.load(f)
    args['wordnet'] = wordnet
    
    print('[!] parameters:')
    print(args)
    
    # load the model
    if args['method'] == 'greedy':
        args['model'] = 'bertretrievalkggreedy'
    else:
        pass
    agent = load_agent_model()
    human = load_human_model()
    print(f'[!] finish loading the agent and human model for interaction')
    
    # src and tgt
    source = '啤酒'
    targets = neighborhood(wordnet, source, 4, size=10)
    
    for target in tqdm(targets):
        main(source, target, **args)
        