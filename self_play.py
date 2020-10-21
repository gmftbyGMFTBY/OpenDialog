from header import *
from config import *

'''self-play for testing topic-driven open-domain dialoge systems'''

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
    parser.add_argument('--recoder', type=str, default='rest/self_play.txt')
    parser.add_argument('--talk_samples', type=int, default=256)
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

def neighborhood(G, start_node, n, size=10):
    '''https://stackoverflow.com/questions/22742754/finding-the-n-degree-neighborhood-of-a-node'''
    n -= 1
    path_lengths = nx.single_source_shortest_path_length(G, start_node, cutoff=n)
    nodes = [node for node, length in path_lengths.items() if length == n]
    # choose the nodes that has the high similarity with the given node
    similairty = [(w2v.similarity(start_node, node), node) for node in nodes]
    similairty = sorted(similairty, key=lambda x: x[0], reverse=True)[:size]
    print(similairty)
    return [i[1] for i in similairty]

def isEnd(utterance, target):
    if target in utterance:
        return True
    else:
        return False

def main(source, target, **args):
    '''interaction between two agents'''
    agent.reset(target, source)
    human.reset()
    recoder = args['recoder']
    step, status, data, conversation = 0, 'Success', {'msgs': []}, []
    while True:
        context = agent.get_res(data)
        data['msgs'].append({'msg': context})
        conversation.append(('Agent', context, agent.topic_history[-1]))
        data['msgs'] = data['msgs'][-args['history_length']:]
        step += 1
        
        done = isEnd(context, target)
        if done:
            status = 'Success'
            break
        
        reply = human.get_res(data)
        data['msgs'].append({'msg': reply})
        conversation.append(('Human', reply, None))
        data['msgs'] = data['msgs'][-args['history_length']:]
        step += 1
        
        if step >= args['max_step']:
            status = 'Failed'
            break
            
    if status == 'Failed':
        return False, None
    
    print(
        f'========== Dialog History {source} -> {target} ===========', 
        file=recoder,
        flush=True,
    )
    for idx, (speaker, utterance, topic) in enumerate(conversation):
        if topic:
            string = f'{speaker}-{topic}: {utterance}'
        else:
            string = f'{speaker}: {utterance}'
        print(string, file=recoder, flush=True)
    return True, step

if __name__ == "__main__":
    args = parser_args()
    args = vars(args)
    
    # disable the fixed random seed can obtain the rich results
    random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args['seed'])
    
    # load word net and word2vec gensim
    with open('data/wordnet.pkl', 'rb') as f:
        wordnet = pickle.load(f)
    args['wordnet'] = wordnet
    w2v = gensim.models.KeyedVectors.load_word2vec_format('data/chinese_w2v_base.txt', binary=False)
    with open('data/topic_words.pkl', 'rb') as f:
        topic_words = pickle.load(f)
    
    print('[!] parameters:')
    print(args)
    
    # load the model
    if args['method'] == 'greedy':
        args['model'] = 'bertretrievalkggreedy'
    elif args['method'] == 'clustergreedy':
        args['model'] = 'bertretrievalclustergreedy'
    else:
        pass
    agent = load_agent_model()
    human = load_human_model()
    print(f'[!] finish loading the agent and human model for interaction')
    
    # src and tgt
    sources = random.sample(topic_words, 300)
    targets = random.sample(topic_words, 300)
    
    counter, avg_step = 0, []
    with open(args['recoder'], 'w') as f:
        args['recoder'] = f
        for source, target in tqdm(list(zip(sources, targets))):
            try:
                done, step = main(source, target, **args)
            except:
                continue
            if done:
                counter += 1
                avg_step.append(step)
    print(f'[!] {"=" * 10} success ratio: {round(counter/len(targets), 4)} average success step: {round(np.mean(avg_step), 4)} {"=" * 10}')
        