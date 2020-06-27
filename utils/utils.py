from header import *

def collect_parameter_4_model(args):
    if args['model'] == 'retrieval':
        return (), {}
    elif args['model'] == 'bertretrieval':
        return (args['multi_gpu'],), {'run_mode': args['mode'], 'lang': args['lang']}
    elif args['model'] == 'bertretrieval_multi':
        return (args['multi_gpu'],), {'run_mode': args['mode'], 'lang': args['lang']}
    elif args['model'] == 'bertnli':
        return (args['multi_gpu'],), {'run_mode': args['mode'], 'lang': args['lang']}
    elif args['model'] == 'bertlogic':
        return (args['multi_gpu'],), {'run_mode': args['mode'], 'lang': args['lang']}
    elif args['model'] == 'seq2seq':
        return (args['vocab_size'], args['vocab']), {'run_mode': args['mode'], 'lang': args['lang']}
    elif args['model'] == 'gpt2':
        return (args['total_steps'], args['multi_gpu']), {'run_mode': args['mode'], 'lang': args['lang']}
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
    else:
        raise Exception(f'[!] except model retrieval/seq2seq/gpt2, but got {args["model"]}')

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

def read_text_data_noparallel(path):
    with open(path) as f:
        data = f.read().split('\n')
        return data

def read_text_data(path):
    with open(path) as f:
        data = f.read().split('\n\n')
        data = [i.split('\n') for i in data if i.strip()]
        # NOTE:
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
    with open(path) as f:
        data = f.read().split('\n\n')
        data = [i.split('\n') for i in data if i.strip()]
        nd = []
        for context, response in data:
            context = [i.strip() for i in context.split('[SEP]')]
            nd.append(context + [response])
    return nd

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

def dialog_collate_fn(batch):
    pad = 0
    cid, cid_l, rid, rid_l = [], [], [], []
    for i in batch:
        cid.append(i[0])
        cid_l.append(i[1])
        rid.append(i[2])
        rid_l.append(i[3])
    cid = pad_sequence(cid, batch_first=False, padding_value=pad)
    rid = pad_sequence(rid, batch_first=False, padding_value=pad)
    cid_l = torch.LongTensor(cid_l)
    rid_l = torch.LongTensor(rid_l)
    if torch.cuda.is_available():
        cid = cid.cuda()
        rid = rid.cuda()
        cid_l = cid_l.cuda()
        rid_l = rid_l.cuda()
    return cid, cid_l, rid, rid_l

def nli_collate_fn(batch):
    pad = 0
    sid, label = [], []
    for i in batch:
        sid.append(i['sid'])
        label.append(i['label'])
    sid = pad_sequence(sid, batch_first=True, padding_value=pad)    # [batch, seq]
    label = torch.LongTensor(label)    # [batch]
    if torch.cuda.is_available():
        sid, label = sid.cuda(), label.cuda()
    return sid, label

def gpt2_lm_collate_fn(batch):
    pad = 0
    batch = [i['context_id'] for i in batch]
    # NOTE:
    random.shuffle(batch)
    ids = pad_sequence(batch, batch_first=True, padding_value=pad)
    if torch.cuda.is_available():
        ids = ids.cuda()
    return ids

def gpt2retrieval_train_collate_fn(batch):
    '''
    GPT2Dataset -> gpt2retrieval_train_collate_fn
    '''
    pad = 0
    candidates_k = len(batch[0]['retrieval_embedding'])
    ctx, ir_ctx = [], []
    for i in batch:
        ctx.append(i['context_id'])
        ir_ctx.append(i['retrieval_embedding'])
    ctx = pad_sequence(ctx, batch_first=True, padding_value=pad)    # [batch, seq]
    ir_embed = []    # k*[batch, 300]
    for i in range(candidates_k):
        item_ = torch.tensor([item[i] for item in ir_ctx])    # [batch, 300]
        ir_embed.append(item_)
    ir_embed = torch.stack(ir_embed).mean(dim=0)    # [batch, 300]
    if torch.cuda.is_available():
        ir_embed = ir_embed.cuda()
        ctx = ctx.cuda()
    return ir_embed, ctx

def gpt2retrieval_test_collate_fn(batch):
    '''
    batch must be 1
    '''
    pad = 0
    candidates_k = len(batch[0]['retrieval_embedding'])
    assert len(batch) == 1, f'[!] batch must be 1, but got {len(batch)}'
    ir_embed_, ctx, res = batch[0]['retrieval_embedding'], batch[0]['context_id'], batch[0]['reply_id']
    ir_embed = []
    for i in ir_embed_:
        i = torch.tensor(i)    # [300]
        ir_embed.append(i)
    ir_embed = torch.stack(ir_embed).mean(dim=0)    # [300]
    # ctx/res: [batch, max_len]
    if torch.cuda.is_available():
        ir_embed = ir_embed.cuda()
        ctx, res = ctx.cuda(), res.cuda()
    # 300; seq; seq
    return ir_embed, ctx, res

def gpt2_train_collate_fn(batch):
    pad = 0
    ctx = []
    for i in batch:
        ctx.append(i['context_id'])
    # NOTE: shuffle in the batch
    random.shuffle(ctx)
    ctx = pad_sequence(ctx, batch_first=True, padding_value=pad)
    if torch.cuda.is_available():
        ctx = ctx.cuda()
    return ctx

def gpt2_test_collate_fn(batch):
    '''
    batch must be 1
    '''
    pad = 0
    assert len(batch) == 1, f'[!] batch must be 1, but got {len(batch)}'
    ctx, res = batch[0]['context_id'], batch[0]['reply_id']
    # for i in batch:
    #     ctx.append(i['context_id'])
    #     res.append(i['reply_id'])
    # ctx = pad_sequence(ctx, batch_first=True, padding_value=pad)
    # rid = pad_sequence(res, batch_first=True, padding_value=pad)
    # ctx/res: [batch, max_len]
    if torch.cuda.is_available():
        ctx, res = ctx.cuda(), res.cuda()
    return ctx, res

def gpt2_test_collate_fn_batch(batch):
    pad = 0
    ctx, res = [], []
    for i in batch:
        ctx.append(i['context_id'])
        res.append(i['reply_id'])
    ctx = pad_sequence(ctx, batch_first=True, padding_value=pad)
    res = pad_sequence(res, batch_first=True, padding_value=pad)
    if torch.cuda.is_available():
        ctx = ctx.cuda()
        res = res.cuda()
    return ctx, res

def multigpt2_train_collate_fn(batch):
    '''
    bundle['context_text']
    bundle['context_id']
    bundle['retrieval_list_text']
    bundle['retrieval_list']

    return:
    1. context_id: [batch, seq]
    2. retrieval_list_id: [[batch, seq], [batch, seq], ...]
    '''
    pad = 0
    retrieval_samples = len(batch[0]['retrieval_list'])
    ctx = []
    for i in batch:
        ctx.append(i['context_id'])
    retrieval_list = []
    for idx in range(retrieval_samples):
        r_ = []
        for i in batch:
            r_.append(i['retrieval_list'][idx])
        retrieval_list.append(r_)
    ctx = pad_sequence(ctx, batch_first=True, padding_value=pad)
    r_list = []
    for i in retrieval_list:
        i = pad_sequence(i, batch_first=True, padding_value=pad)
        if torch.cuda.is_available():
            i = i.cuda()
        r_list.append(i)
    if torch.cuda.is_available():
        ctx = ctx.cuda()
    return ctx, r_list

def multigpt2_test_collate_fn(batch):
    '''
    In order to avoid the pad value of the src, batch must be 1
    '''
    pad = 0
    assert len(batch) == 1, f'[!] batch size must be 1, but got {len(batch)}'
    ctx, res_ = batch[0]['context_id'], batch[0]['retrieval_list']
    res = batch[0]['reply_id']
    if torch.cuda.is_available():
        ctx = ctx.cuda()
        res = res.cuda()
        r_ = []
        for i in res_:
            r_.append(i.cuda())
    return ctx, res, r_

def ir_collate_fn(batch):
    cxt, rxt = [], []
    label = []
    samples = len(batch[0][2])
    text_c, text_r = [], []
    for i in batch:
        cxt.extend([i[0]] * (samples+1))
        rxt.extend([i[1]] + i[2])
        label.extend(i[3])
    # cxt: [10*batch_size, 768]
    cxt = torch.tensor(cxt)
    rxt = torch.tensor(rxt)
    assert cxt.shape == rxt.shape, f'ctx: {ctx.shape}; rxt: {rxt.shape}'
    assert len(text_c) == len(text_r), f'ctx: {len(text_c)}; rxt: {len(text_r)}'
    label = torch.tensor(label, dtype=torch.float)
    if torch.cuda.is_available():
        cxt = cxt.cuda()
        rxt = rxt.cuda()
        label = label.cuda()
    return cxt, rxt, label

def bert_ir_train_collate_fn(batch):
    pad = 0
    cxt, label = [], []
    for i in batch:
        cxt.append(i[0])
        label.append(i[1])
    # NOTE:
    random_idx = list(range(len(cxt)))
    random.shuffle(random_idx)
    cxt = pad_sequence(cxt, batch_first=True, padding_value=pad)    # [batch, seq]
    cxt = cxt[random_idx]
    label = label[random_idx]

    label = torch.tensor(label, dtype=torch.long)    # [batch]
    if torch.cuda.is_available():
        cxt, label = cxt.cuda(), label.cuda()
    return cxt, label

def bert_ir_test_collate_fn(batch):
    pad = 0
    cxt, label = [], []
    for i in batch:
        cxt.extend(i[0])
        label.extend(i[1])
    cxt = pad_sequence(cxt, batch_first=True, padding_value=pad)    # [10*batch, seq]
    label = torch.tensor(label, dtype=torch.long)    # [10*batch]
    if torch.cuda.is_available():
        cxt, label = cxt.cuda(), label.cuda()
    return cxt, label

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

# ========== wechat api ==========
def reply_text(to_user, from_user, content):
    reply = '''<xml><ToUserName><![CDATA[%s]]></ToUserName><FromUserName><![CDATA[%s]]></FromUserName><CreateTime>%s</CreateTime><MsgType><![CDATA[text]]></MsgType><Content><![CDATA[%s]]></Content><FuncFlag>0</FuncFlag></xml>'''
    reply = reply % (to_user, from_user, str(int(time.time())), content)
    response = make_response(reply)
    response.content_type = 'application/xml'
    return response

def init_mongodb(dbname, table_name):
    client = pymongo.MongoClient('mongodb://localhost:27017/')
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

