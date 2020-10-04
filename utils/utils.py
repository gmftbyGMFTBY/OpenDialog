from header import *

def collect_parameter_4_model(args):
    if args['model'] == 'DualLSTM':
        return (args['multi_gpu'],), {'run_mode': args['mode'], 'lang': args['lang']}
    elif args['model'] == 'bertretrieval':
        return (args['multi_gpu'],), {'run_mode': args['mode'], 'lang': args['lang'], 'local_rank': args['local_rank']}
    elif args['model'] == 'lcccir':
        return (args['multi_gpu'],), {'run_mode': args['mode']}
    elif args['model'] == 'lccc':
        return (args['multi_gpu'],), {'run_mode': args['mode']}
    elif args['model'] == 'bertmc':
        return (args['multi_gpu'],), {'run_mode': args['mode'], 'lang': args['lang'], 'model_type': 'mc'}
    elif args['model'] == 'bertmcf':
        return (args['multi_gpu'],), {'run_mode': args['mode'], 'lang': args['lang'], 'model_type': 'mcf'}
    elif args['model'] == 'bertretrieval_cl':
        return (args['multi_gpu'],), {'run_mode': args['mode'], 'lang': args['lang']}
    elif args['model'] == 'bertretrieval_dis':
        return (args['multi_gpu'],), {'run_mode': args['mode'], 'lang': args['lang']}
    elif args['model'] == 'pone':
        return (args['multi_gpu'],), {'run_mode': args['mode'], 'lang': args['lang']}
    elif args['model'] == 'bertretrieval_multiview':
        return (args['multi_gpu'],), {'run_mode': args['mode'], 'lang': args['lang']}
    elif args['model'] == 'bertnli':
        return (args['multi_gpu'],), {'run_mode': args['mode'], 'lang': args['lang']}
    elif args['model'] == 'bertlogic':
        return (args['multi_gpu'],), {'run_mode': args['mode'], 'lang': args['lang']}
    elif args['model'] == 'seq2seq':
        return (args['multi_gpu'], args['vocab']), {'run_mode': args['mode'], 'lang': args['lang'], 'local_rank': args['local_rank']}
    elif args['model'] == 'gpt2':
        return (args['total_steps'], args['multi_gpu']), {'run_mode': args['mode'], 'lang': args['lang'], 'local_rank': args['local_rank']}
    elif args['model'] == 'gpt2v2':
        return (args['total_steps'], args['multi_gpu']), {'run_mode': args['mode'], 'lang': args['lang'], 'local_rank': args['local_rank']}
    elif args['model'] == 'pfgpt2':
        return (args['total_steps'], args['multi_gpu']), {'run_mode': args['mode'], 'lang': args['lang']}
    elif args['model'] == 'kwgpt2':
        return (args['total_steps'], args['multi_gpu']), {'run_mode': args['mode'], 'lang': args['lang']}
    elif args['model'] == 'gpt2_mmi':
        return (args['total_steps'], args['multi_gpu']), {'run_mode': args['mode'], 'lang': args['lang']}
    elif args['model'] == 'when2talk':
        return (args['total_steps'], args['multi_gpu']), {'run_mode': args['mode'], 'lang': args['lang']}
    elif args['model'] == 'gpt2retrieval':
        return (args['total_steps'], args['multi_gpu']), {'run_mode': args['mode'], 'lang': args['lang']}
    elif args['model'] == 'gpt2lm':
        return (args['total_steps'], args['multi_gpu']), {'run_mode': args['mode'], 'lang': args['lang'], 'lm': True}
    elif args['model'] == 'gpt2gan':
        return (args['multi_gpu'],), {'run_mode': args['mode'], 'lang': args['lang']}
    elif args['model'] == 'gpt2gan_v2':
        return (args['multi_gpu'],), {'run_mode': args['mode'], 'lang': args['lang']}
    elif args['model'] == 'multigpt2':
        return (args['total_steps'],), {'run_mode': args['mode'], 'lang': args['lang']}
    elif args['model'] == 'uni':
        return (args['total_steps'], args['multi_gpu']), {'run_mode': args['mode'], 'lang': args['lang'], 'local_rank': args['local_rank']}
    elif args['model'] == 'bert_na':
        return (args['multi_gpu']), {'run_mode': args['mode'], 'lang': args['lang']}
    elif args['model'] == 'bertirbi':
        return (args['multi_gpu']), {'run_mode': args['mode'], 'local_rank': args['local_rank']}
    elif args['model'] == 'transformer':
        return (args['total_steps'], args['multi_gpu']), {'run_mode': args['mode'], 'lang': args['lang'], 'local_rank': args['local_rank'], 'vocab': args['vocab']}
    else:
        raise Exception(f'[!] unknow model {args["model"]}')

def read_stop_words(path):
    with open(path) as f:
        sw = []
        for line in f.readlines():
            sw.append(line.strip())
        sw = list(set(sw))
    return sw

def read_csv_data(path, lang='zh'):
    with open(path) as f:
        data = []
        reader = csv.reader(f)
        for idx, i in enumerate(reader):
            if idx == 0:
                continue
            i = [j.strip() for j in i]
            data.append(i)
    return data

def read_annotation(path):
    with open(path) as f:
        scores = []
        f_csv = csv.reader(f, delimiter=',')
        for line in f_csv:
            line = list(map(float, line))
            scores.append(line)
    return scores

def read_text_data_noparallel(path):
    with open(path) as f:
        data = f.read().split('\n')
        return data

def read_text_data(path):
    with open(path, encoding='utf-8') as f:
        data = f.read().split('\n\n')
        data = [i.split('\n') for i in data if i.strip()]
        data = [i for i in data if len(i) >= 2]
        return data

def read_text_data_reversed(path):
    with open(path) as f:
        data = f.read().split('\n\n')
        data = [i.split('\n') for i in data if i.strip()]
        # NOTE:
        data = [i for i in data if len(i) >= 2]
        # reversed
        nd = []
        for context, response in data:
            context = ' [SEP] '.join(reversed(context.split(' [SEP] ')))
            nd.append((response, context))
        return nd

def read_text_data_nosep(path):
    with open(path) as f:
        data = f.read().split('\n\n')
        data = [i.split('\n') for i in data if i.strip()]
        nd = []
        for dialog in data:
            context = ' [SEP] '.join(dialog[:-1])
            response = dialog[-1]
            nd.append((context, response))
        return nd

def read_text_data_sep(path):
    '''For LCCC Corpus'''
    with open(path) as f:
        data = f.read().split('\n\n')
        data = [i.split('\n') for i in data if i.strip()]
        nd = []
        for context, response in data:
            context = [i.strip() for i in context.split('[SEP]')]
            nd.append(context + [response])
    return nd

def read_lccc_data(path, debug=False):
    with open(path) as f:
        data = json.load(f)['train']
        if debug:
            data = data[:100000]
    print(f'[!] load LCCC dataset over, find {len(data)} samples')
    return data

def read_json_data(path):
    '''
    read NLI json data
    '''
    with open(path) as f:
        data = []
        for line in tqdm(f.readlines()):
            d = json.loads(line)
            data.append(d)
    return data

def load_best_model(dataset, model_name, model):
    path = f'ckpt/{dataset}/{model_name}/best.pt'
    model.load_state_dict(torch.load(path))
    print(f'[!] already load best checkpoint from {path}')

def bert_as_service_processing4ir(data_path, save_path, batch_size=256):
    print('[!] begin to create the BertClient, make sure the bert-as-service is running')
    client = BertClient()
    data = read_csv_data(data_path, lang='zh')
    contexts, responses = [i[0] for i in data], [i[1] for i in data]
    # processing the contexts
    idx, length = 0, len(contexts)
    cs, rs = [], []
    while idx < length:
        npdata = client.encode(contexts[idx:idx+batch_size])   # [batch, 768]
        cs.append(npdata)
        npdata = client.encode(responses[idx:idx+batch_size])  # [batch, 768]
        rs.append(npdata)
        idx += batch_size
        print(f'[!] processing: {idx}/{length}', end='\r')
    cs, rs = np.concatenate(cs), np.concatenate(rs)
    with open(save_path, 'wb') as f:
        pickle.dump((cs, rs), f)
    print(f'[!] save the processed data into {save_path}')

def when2talk_utils(dialog):
    '''
    one dialog is a list:
    [
        'USER1: xxxx',
        'USER2: xxxx',
        ...
    ]
    return a string:
    "[USER1] xxxx [SEP] [USER1] xxx [STP] [USER2] xxx [SEP] [USER2] xxx [STP]"
    '''
    rest = []
    last_user = None 
    for sentence in dialog:
        current_user = 'USER2' if 'USER2' in sentence else 'USER1'
        if last_user is None:
            pass
        elif last_user == current_user:
            rest.append(' [SEP] ')
        else:
            rest.append(' [STP] ')
        rest.append(sentence)
        last_user = current_user
    rest.append(' [STP] ')
    # rest = rest[:-1]    # ingore the final [SEP]
    return ''.join(rest)

def kwgpt2_utils(sentence, topk=10):
    '''
    obtain the keywords for the sentence and sort them
    '''
    sentence = sentence.replace('[SEP]', '')
    words = jieba.analyse.extract_tags(sentence, topK=topk)
    if len(words) == 0:
        # ignore the cases that have no keywords
        return None
    words = [(word, sentence.index(word)) for word in words]
    words = sorted(words, key=lambda x: x[1])
    words = [i[0] for i in words]
    # conver to string: [w1] [SEP] [w2] [SEP] ...
    return ' [SEP] '.join(words)

def make_vocabs(responses):
    words = set()
    for r in tqdm(response):
        r = list(jieba.cut(r))
        words |= set(r)
    print(f'[!] obtain {len(words)} words from the responses')
    return list(words)

def generate_token_type_ids(s, sep=102):
    rest, type_ = [], 0
    for item in s:
        rest.append(type_)
        if item == sep:
            type_ = 1 if type_ == 0 else 0
    assert len(rest) == len(s)
    return torch.LongTensor(rest)

# ========== wechat api ==========
def reply_text(to_user, from_user, content):
    reply = '''<xml><ToUserName><![CDATA[%s]]></ToUserName><FromUserName><![CDATA[%s]]></FromUserName><CreateTime>%s</CreateTime><MsgType><![CDATA[text]]></MsgType><Content><![CDATA[%s]]></Content><FuncFlag>0</FuncFlag></xml>'''
    reply = reply % (to_user, from_user, str(int(time.time())), content)
    response = make_response(reply)
    response.content_type = 'application/xml'
    return response

def init_mongodb(dbname, table_name):
    client = pymongo.MongoClient('mongodb://localhost:27001/')
    table = client[dbname][table_name]
    return table 

def db_table_counter(table):
    return table.find({}).count()

def load_topic():
    '''
    load topic in the file ./topic.user
    if the file doesn't, default topic is [电影]
    '''
    try:
        with open('cache/topic.user') as f:
            topic = f.read().strip()
        return topic
    except:
        return '电影'

def save_topic(s):
    s = s.replace('#', '')
    if s in ['电影', '数码产品', '体育', '美食', '音乐']:
        with open('cache/topic.user', 'w') as f:
            f.write(s)
        return True, s
    else:
        return False, s
# ========== wechat api ==========

def get_masks(src, trg, PAD=0):
    """generate masks based on inputs and targets
    
    Arguments:
        src {torch.LongTensor} -- input mini-batch in shape (L, B)
        trg {torch.LongTensor} -- target mini-batch in shape (L, B)
    
    Keyword Arguments:
        PAD {int} -- padding value (default: {0})
    
    Returns:
        masks can be directly passed into encode/decode's forward method
    """
    S, S_B = src.shape
    T, T_B = trg.shape
    assert S_B == T_B, "Batch Size of source input and target input inconsistent! %d != %d"%(S_B, T_B)

    trg_mask = nn.Transformer.generate_square_subsequent_mask(T, T)
    src_key_padding_mask = (src == PAD).permute(1, 0)    # [B, S]
    trg_key_padding_mask = (trg == PAD).permute(1, 0)    # [B, T]
    memory_key_padding_mask = (src == PAD).permute(1, 0)    # [B, S]
    return trg_mask, src_key_padding_mask, trg_key_padding_mask, memory_key_padding_mask

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='zh50w', type=str)
    parser.add_argument('--mode', default='irdata')
    parser.add_argument('--batch_size', default=256, type=int)
    
    args = parser.parse_args()
    if args.mode == 'irdata':
        data_path = f'data/{args.dataset}/train.csv'
        save_path = f'data/{args.dataset}/train.pkl'
        bert_as_service_processing4ir(data_path, save_path, args.batch_size)
        data_path = f'data/{args.dataset}/test.csv'
        save_path = f'data/{args.dataset}/test.pkl'
        bert_as_service_processing4ir(data_path, save_path, args.batch_size)
        data_path = f'data/{args.dataset}/dev.csv'
        save_path = f'data/{args.dataset}/dev.pkl'
        bert_as_service_processing4ir(data_path, save_path, args.batch_size)
    else:
        pass

