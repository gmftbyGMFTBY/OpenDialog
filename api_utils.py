from header import *
from models import *

def parser_args_api():
    '''chat mode:
    1. mode 0: single-turn chat
    2. mode 1: multi-turn chat
    3. mode 2: knowledge driven multi-turn chat
    '''
    parser = argparse.ArgumentParser(description='flask chatting mode')
    parser.add_argument('--model', type=str, default='bertretrieval')
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--chat_mode', type=int, default=0)
    parser.add_argument('--multi_turn_size', type=int, default=5)
    parser.add_argument('--verbose', dest='verbose', action='store_true')
    parser.add_argument('--no-verbose', dest='verbose', action='store_false')
    return parser.parse_args()

def parse_msg(request):
    data = request.data.decode()
    xml = ET.fromstring(data)
    toUser = xml.find('ToUserName').text
    fromUser = xml.find('FromUserName').text
    msgType = xml.find('MsgType').text
    content = xml.find('Content').text
    return toUser, fromUser, msgType, content

def flask_load_agent(model, gpu, logger):
    '''init the agent'''
    args = {
        'model': model,
        'multi_gpu': gpu
    }
    logger.info(f'[!] begin to init the {args["model"]} agent on GPU {args["multi_gpu"]}')
    if args['model'] == 'bertretrieval':
        agent = BERTRetrievalAgent(args['multi_gpu'], run_mode='test', kb=False)
        agent.load_model(f'ckpt/zh50w/bertretrieval/best.pt')
    elif args['model'] == 'bertmc':
        # [model_type]: bertmc -> mc; bertmcf -> mcf
        agent = BERTMCAgent(args['multi_gpu'], kb=False, model_type='mc')
        agent.load_model(f'ckpt/zh50w/bertmc/best.pt')
    elif args['model'] == 'bertretrieval_multiview':
        agent = BERTMULTIVIEWAgent(args['multi_gpu'], kb=False)
        agent.load_model(f'ckpt/zh50w/bertretrieval_multiview/best.pt')
    elif args['model'] == 'gpt2':
        # available run_mode: test, rerank, rerank_ir
        agent = GPT2Agent(1000, args['multi_gpu'], run_mode='rerank_ir')
        agent.load_model(f'ckpt/train_generative/gpt2/best.pt')
    elif args['model'] == 'lccc':
        agent = LCCCAgent(args['multi_gpu'], run_mode='test')    # run_mode: test/rerank
    elif args['model'] == 'when2talk':
        agent = When2TalkAgent(1000, args['multi_gpu'], run_mode='test')
        agent.load_model(f'ckpt/when2talk/when2talk/best.pt')
    elif args['model'] == 'test':
        agent = TestAgent()
    elif args['model'] == 'multiview':
        agent = MultiViewTestAgent()
    else:
        raise Exception(f'[!] obtain the unknown model name {args["model"]}')
    print(f'[!] init {args["model"]} agent on GPU {args["multi_gpu"]} over ...')
    return agent

def chat(agent, content, args=None, logger=None):
    if args['chat_mode'] == 0:
        return normal_chat_single_turn(agent, content, args=args)
    elif args['chat_mode'] == 1:
        return normal_chat_multi_turn(agent, content, args=args)
    elif args['chat_mode'] == 2:
        return kg_driven_chat_multi_turn(agent, content, args=args)
    else:
        print(f'[!] Unknow chat mode {args["chat_mode"]}')
        return None

def normal_chat_single_turn(agent, content, topic=None, args=None):
    data = {
        'topic': topic,
        'msgs': [{'msg': content}]
    }
    args['content'] = content
    return agent.get_res(data)

def normal_chat_multi_turn(agent, content, topic=None, args=None):
    query = {"$or": [{"fromUser": args["fromUser"]}, {"toUser": args["fromUser"]}]}
    previous_utterances = [i['utterance'] for i in args['table'].find(query)][-args['multi_turn_size']:]
    content_list = [{'msg': i} for i in previous_utterances]
    data = {
        'topic': topic,
        'msgs': content_list,
    }
    args['content'] = ' [SEP] '.join(previous_utterances)
    return agent.get_res(data)

def kg_driven_chat_multi_turn(agent, content, topic=None, args=None):
    query = {"$or": [{"fromUser": args["fromUser"]}, {"toUser": args["fromUser"]}]}
    previous_utterances = [(i['fromUser'], i['utterance']) for i in args['table'].find(query)][-args['multi_turn_size']:]
    content_list = [{'msg': i[1], 'fromUser': i[0]} for i in previous_utterances]
    data = {
        'topic': topic,
        'msgs': content_list,
        'path': args['session'].get('kg_path'),
        'current_node': args['session'].get('node'),
    }
    args['content'] = ' [SEP] '.join(previous_utterances)
    return agent.get_res(data)