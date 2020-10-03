from .header import *

# ========== utils for GPT2V2RL and GPT2V2 ==========
class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
    
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        
class ActorCritic(nn.Module):
    
    def __init__(self, policy_size, embedding_size, action_std=0.5):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(embedding_size*2, embedding_size),
            nn.Tanh(),
            nn.Linear(embedding_size, policy_size),
            nn.Tanh(),
        )
        self.critic = nn.Sequential(
            nn.Linear(embedding_size*2, embedding_size),
            nn.Tanh(),
            nn.Linear(embedding_size, 1),
        )
        self.action_var = torch.full((action_dim,), action_std*action_std)
        if torch.cuda.is_available():
            self.action_var = self.action_var.cuda()
            
    def forward(self, embedding):
        '''only called by gpt2v2, compatible with gpt2v2 model; only the actor is used and critic is ignored;
        :embedding: [B, E*2]'''
        return self.actor(embedding)    # [B, Policy_size]
    
    def act(self, state, memory):
        action_mean = self.actor(state)
        cov_mat = torch.diag(self.action_var)
        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)
        return action.detach()
    
    def evaluate(self, state, action):   
        action_mean = self.actor(state)
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(device)
        dist = MultivariateNormal(action_mean, cov_mat)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.critic(state)
        return action_logprobs, torch.squeeze(state_value), dist_entropy
# =========== utils for GPT2V2RL and GPT2V2 ==========

class ReplayMemory:

    '''
    Replay Memory for the GPT2RL Model
    It is also a iterator for training the model

    vocab is the BertTokenizer
    '''

    def __init__(self, capacity, vocab, batch_size):
        self.memory = Queue(capacity)
        self.batch_size = batch_size
        self.read_count = 0
        self.vocab = vocab

    def push(self, cid, rid, score):
        '''
        save the positive samples (context, response and corresponding score)
        if full, remove the first one and push it (queue)
        '''
        def filter(string):
            return string.replacce('[PAD]', '')
        c_txt = self.vocab.decode(cid)
        r_txt = self.vocab.decode(rid)
        c_txt, r_txt = filter(c_txt), filter(r_txt)
        data = {
                'cid': cid, 'rid': rid, 'score': score,
                'c_text': c_txt, 'r_text': r_txt}
        if self.memory.full():
            self.memory.get()
        self.memory.put(data)

    def push_many(self, cids, rids, scores):
        '''
        cids/rids: [batch, seq]
        scores: [batch]
        '''
        for c, r, s in zip(cids, rids, scores):
            self.push(c, r, s)

    def obtain(self):
        '''
        get item and put it back
        '''
        data = self.memory.get()
        self.memory.put(data)
        return data

    def __iter__(self):
        return self

    def __next__(self):
        if self.read_count == self.memory.qsize():
            self.read_count = 0
            raise StopIteration
        else:
            read_size = min(self.batch_size, self.memory.qsize()-self.read_count)
            cid, rid = [], []
            for _ in range(read_size):
                data = self.obtain()
                cid.append(torch.cat((data['cid'], data['rid'][1:])))
                self.read_count += 1
            cid = pad_sequence(cid, batch_first=True, padding_value=0)
            if torch.cuda.is_available():
                cid = cid.cuda()
            return cid

    def __len__(self):
        return self.memory.qsize()

class KWParser:

    '''
    Key Word Parser by jieba
    It should be noted that the conversation context may contains multiple utterances, 
    and we focus more on the recent utterances.

    POS Tagger information can be found in: httos://github.com/fxsjy/jieba

    Influence the model very badly, try to remove it
    '''

    def __init__(self):
        # delete the verb and time
        self.allowPOS = ['n', 'nr', 'nz', 'PER', 'LOC', 'ORG', 
                         'ns', 'nt', 'nw', 'vn', 's']
        self.topk, self.one_topk = 10, 5

    def parser(self, msg, topic=None):
        utterances = msg.split('[SEP]')
        kw = set([])
        for utterance in reversed(utterances):
            rest = jieba.analyse.extract_tags(
                        msg, 
                        topK=self.one_topk, 
                        allowPOS=self.allowPOS)
            kw |= set(rest)
            if len(kw) > self.topk:
                break
        return list(kw)

class KBKWParser(KWParser):

    '''
    KB Entity augmentation parser
    '''

    def __init__(self):
        super(KBKWParser, self).__init__()
        with open('data/KG/kg.pkl', 'rb') as f:
            self.kg = pickle.load(f)
        self.map = {'电影': 'movie', '美食': 'food', '数码产品': 'electric', '音乐': 'music', '体育': 'sport'}
        self.topk, self.one_topk = 20, 10
        print(f'[!] load the knowledge graph over')

    def kb_search(self, topic, kw, samples=2):
        '''
        This part can contain lots of strategies
        '''
        if topic not in self.map:
            raise Exception(f'[!] the topic should be in [电影, 美食, 数码产品, 音乐, 体育], but got {topic}')
        data = self.kg[self.map[topic]]
        collector = []
        for item in data:
            for spo in item:
                if kw in spo[0]:
                    # subject contains the keywords
                    collector.extend(spo)
                elif kw in spo[1]:
                    # p contains the keywords
                    collector.append(spo[1])
                elif kw in spo[2]:
                    # object contains the keywords
                    collector.append(spo[2])
        collector = list(set(collector))     # amount of the relative keywords
        # filter stretegy
        # 1. no english; 2. no number; 3. no number and english mixture; 4. 标点符号; 5. length
        rest = []
        for word in collector:
            word = re.sub('[0-9A-Za-z\.]*', '', word.strip())
            if len(word) > 2:
                rest.append(word)
        if samples > len(rest):
            return rest
        else:
            return random.sample(rest, samples)
 
    def parser(self, msg, topic=None):
        utterances = msg.split('[SEP]')
        kw = set([])
        for utterance in reversed(utterances):
            rest = jieba.analyse.extract_tags(
                        msg, 
                        topK=self.one_topk, 
                        allowPOS=self.allowPOS)
            kw |= set(rest)
            for i in rest:
                subrest = self.kb_search(topic, i)
                subrest = jieba.analyse.extract_tags(
                        ' '.join(subrest), 
                        topK=2, 
                        allowPOS=self.allowPOS)
                kw |= set(subrest)
            if len(kw) > self.topk:
                break
        return list(kw)

class ESUtils:

    def __init__(self, index_name, create_index=False):
        self.es = Elasticsearch(http_auth=('XXX', 'XXX'))
        self.index = index_name
        if create_index:
            mapping = {
                'properties': {
                    'context': {
                        'type': 'text',
                        'analyzer': 'ik_max_word',
                        'search_analyzer': 'ik_smart'
                    }
                }
            }
            if self.es.indices.exists(index=self.index):
                print(f'[!] delete the index of the elasticsearch')
                self.es.indices.delete(index=self.index)
            rest = self.es.indices.create(index=self.index)
            print(rest)
            rest = self.es.indices.put_mapping(body=mapping, index=self.index)

    def insert_pairs(self, pairs):
        count = self.es.count(index=self.index)['count']
        actions = []
        for i, qa in enumerate(tqdm(pairs)):
            actions.append({
                '_index': self.index,
                '_id': i + count,
                'context': qa[0],
                'response': qa[1],
            })
        helpers.bulk(self.es, actions) 
        print(f'[!] retrieval database size: {self.es.count(index=self.index)["count"]}')

class ESChat:

    def __init__(self, index_name, kb=True):
        self.es = Elasticsearch(http_auth=('xxx', 'xxx'))
        self.index = index_name
        # if kb:
        #     self.kwparser = KBKWParser()
        # else:
        #     self.kwparser = KWParser()
        self.topic_dict = {
                '电影': '电影 电视剧 明星 动漫',
                '数码产品': '数码产品 数码 相机 手机 电脑 笔记本 iphone ipad',
                '美食': '饮料 美食 饭菜 零食 肉 蛋 奶 烹饪',
                '音乐': '舞曲 歌舞 音乐 流行乐 乐器 DJ 作曲',
                '体育': '体育 运动 健身 减肥 锻炼 养生 竞赛 运动会'
                }

    def search(self, topic, query, samples=10, topk=10):
        '''
        query is the string, which contains the utterances of the conversation context.
        1. topic msg
        2. key word msg
        cantenate with the space operator
        '''
        # 1. topic
        if topic:
            query = f"{self.topic_dict[topic]} [SEP] {query}"
            query = f'{topic}; {query}'
        # 'context': query is Q-Q matching
        # 'response': query is Q-A matching, which seems better
        dsl = {
            'query': {
                'match': {
                    # 'context': query
                    'response': query    # Q-A matching is better
                }
            }
        }
        begin_samples, rest = samples, []
        while len(rest) == 0:
            hits = self.es.search(index=self.index, body=dsl, size=begin_samples)['hits']['hits']
            for h in hits:
                item = {
                    'score': h['_score'], 
                    'context': h['_source']['context'],
                    'response': h['_source']['response']
                }
                if item['response'] in query or 'http' in item['response']:
                    # avoid the repetive responses
                    continue
                else:
                    rest.append(item)
                # rest.append(item)
            begin_samples += 1
        return rest

    def multi_search(self, querys, samples=10):
        # limit the querys length
        querys = [i[-150:] for i in querys]
        search_arr = []
        for query in querys:
            search_arr.append({'index': self.index})
            search_arr.append({'query': {'match': {'response': query}}, 'size': samples})
        request = ''
        for each in search_arr:
            request += f'{json.dumps(each)} \n'
        rest = self.es.msearch(body=request)
        return rest

    def talk(self, topic, msgs):
        rest = self.search(topic, msgs, samples=1)[0]['response']
        # for debug
        # rest = self.search(topic, msgs, samples=10)
        # rest = [i['response'] for i in rest]
        # print(rest)
        # rest = rest[0]
        return rest
    
class CEWithLabelSmoothing(nn.Module):
    
    def __init__(self, vocab_size, label_smoothing=.1, ignore_index=-1, reduction="word_mean"):
        """Cross Entropy Loss with Label Smoothing
        
        Arguments:
            vocab_size {int} -- # of vocabulary in the target language
        
        Keyword Arguments:
            label_smoothing {float} -- label smoothing factor (default: {.1})
            ignore_index {int} -- index need to ignore when calculate the loss (default: {-1})
            reduction {str} -- value in {"word_mean", "sum"}, "word mean": compute word level average loss, "sum":total loss (default: {"word_mean"}) 
        """
        super(CEWithLabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.ignore_index = ignore_index
        self.confidence = 1.0 - label_smoothing
        self.label_smoothing = label_smoothing
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.vocab_size = vocab_size
        self._true_dist = None
        self._reduction = reduction

    def forward(self, logits, target):
        assert logits.size(1) == self.vocab_size, "size mismatch! %d!=%d"%(logits.size(1),self.vocab_size)

        true_dist = logits.clone()
        true_dist.fill_(self.label_smoothing / (self.vocab_size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.ignore_index] = 0.

        mask = (target !=self.ignore_index).float().unsqueeze(1)
        true_dist = true_dist*mask

        loss = self.criterion(self.log_softmax(logits), true_dist)
        
        n_words = torch.sum(mask)
        
        # save some data for debugging
        self._true_dist = true_dist
        self._kl = loss
        self._n_words = n_words

        if self._reduction == "word_mean":
            return loss / n_words
        elif self._reduction == "sum":
            return loss
        else:
            raise ValueError

class IRHead(nn.Module):
    
    def __init__(self, hidden_size, dropout=0.5):
        super(IRHead, self).__init__()
        self.M = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.hidden_layer = nn.Linear(hidden_size*2+1, hidden_size)
        self.opt_layer = nn.Linear(hidden_size, 2)
        self.hidden_drop = nn.Dropout(p=dropout)

    def forward(self, src_embed, tgt_embed):
        '''
        src_embed: [batch, hidden]
        tgt_embed: [batch, hidden]

        return the score: [batch, 2]
        '''
        src_hidden = src_embed.unsqueeze(1)    # [batch, 1, hidden]
        tgt_hidden = tgt_embed.unsqueeze(2)    # [batch, hidden, 1]
        score = torch.bmm(torch.matmul(src_hidden, self.M), tgt_hidden).squeeze(2)  # [batch, 1]
        src_hidden = src_hidden.squeeze(1)
        tgt_hidden = tgt_hidden.squeeze(2)
        inpt = torch.cat([src_hidden, score, tgt_hidden], 1)    # [batch, 2*hidden+1]
        inpt = self.hidden_drop(torch.tanh(self.hidden_layer(inpt)))    # [batch, hidden]
        score = self.opt_layer(inpt)    # [batch, 2]
        return score

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, threshold=-float('Inf'), filter_value=-np.inf):
    assert logits.dim() == 1
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
        
    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value
    return logits

def top_k_top_p_filtering_batch(logits, top_k=0, top_p=0.0, filter_value=-np.inf, 
                                min_token_to_keep=1):
    '''
    :logits: [batch, vocab]
    :return logits: [batch, vocab]
    refer to https://zhuanlan.zhihu.com/p/115076102
    '''
    if top_k > 0:
        top_k = min(max(top_k, min_token_to_keep), logits.size(-1))
        # indices_to_remove: [batch, 1]
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p    # [batch, vocab]
        if min_token_to_keep > 1:
            # make sure the min_token_to_keep token must not to be filtered
            sorted_indices_to_remove[..., :min_token_to_keep] = 0
        # shift the indices to the right to keep also the first token above the threshold
        # avoid the probability of the first token is below the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits

def generate_attention_mask(inpt_ids):
    '''
    generate the corresponding attention mask according to the `input_ids`, which will 
    be fed into the model (BERT or GPT2)
    :inpt_ids: [batch, seq]
    
    return :attn_mask: [batch, seq]; 1 for not masked and 0 for masked tokens
    '''
    attn_mask = torch.zeros_like(inpt_ids)    # [batch, seq]
    not_masked_token_idx = inpt_ids.nonzero().transpose(0, 1).tolist()
    attn_mask[not_masked_token_idx] = 1
    # do not need the .cuda
    return attn_mask

def filter_gpt2rl(x):
    x = [''.join(ii) for ii in x]
    return [ii.replace('[CLS]', '').replace('[PAD]', '').replace('[SEP]', '') for ii in x]

def generate_attention_mask_mc(inpt_ids):
    '''
    inpt_ids: [B, N, S]
    '''
    bsz = inpt_ids.size(0)    # B
    attn_mask = torch.zeros_like(inpt_ids)    # [B, N, S]
    for i in range(bsz):
        not_masked_token_idx = inpt_ids[i].nonzero().transpose(0, 1).tolist()
        attn_mask[i][not_masked_token_idx] = 1
    return attn_mask

class PositionEmbedding(nn.Module):

    '''
    Position embedding for self-attention
    refer: https://pytorch.org/tutorials/beginner/transformer_tutorial.html

    d_model: word embedding size or output size of the self-attention blocks
    max_len: the max length of the input squeezec
    '''

    def __init__(self, d_model, dropout=0.5, max_len=100):
        super(PositionEmbedding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)    # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)    # [1, max_len]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

def to_cuda(x, model=False):
    if torch.cuda.is_available():
        if model:
            x.cuda()
            return None
        else:
            return x.cuda()
    else:
        return x

# ========= BalancedDataParallel ========= #
def scatter(inputs, target_gpus, chunk_sizes, dim=0):
    r"""
    Slices tensors into approximately equal chunks and
    distributes them across given GPUs. Duplicates
    references to objects that are not tensors.
    """

    def scatter_map(obj):
        if isinstance(obj, torch.Tensor):
            try:
                return Scatter.apply(target_gpus, chunk_sizes, dim, obj)
            except Exception:
                print('obj', obj.size())
                print('dim', dim)
                print('chunk_sizes', chunk_sizes)
                quit()
        if isinstance(obj, tuple) and len(obj) > 0:
            return list(zip(*map(scatter_map, obj)))
        if isinstance(obj, list) and len(obj) > 0:
            return list(map(list, zip(*map(scatter_map, obj))))
        if isinstance(obj, dict) and len(obj) > 0:
            return list(map(type(obj), zip(*map(scatter_map, obj.items()))))
        return [obj for targets in target_gpus]

    # After scatter_map is called, a scatter_map cell will exist. This cell
    # has a reference to the actual function scatter_map, which has references
    # to a closure that has a reference to the scatter_map cell (because the
    # fn is recursive). To avoid this reference cycle, we set the function to
    # None, clearing the cell
    try:
        return scatter_map(inputs)
    finally:
        scatter_map = None


def scatter_kwargs(inputs, kwargs, target_gpus, chunk_sizes, dim=0):
    """Scatter with support for kwargs dictionary"""
    inputs = scatter(inputs, target_gpus, chunk_sizes, dim) if inputs else []
    kwargs = scatter(kwargs, target_gpus, chunk_sizes, dim) if kwargs else []
    if len(inputs) < len(kwargs):
        inputs.extend([() for _ in range(len(kwargs) - len(inputs))])
    elif len(kwargs) < len(inputs):
        kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])
    inputs = tuple(inputs)
    kwargs = tuple(kwargs)
    return inputs, kwargs


class BalancedDataParallel(DataParallel):

    def __init__(self, gpu0_bsz, *args, **kwargs):
        self.gpu0_bsz = gpu0_bsz
        super().__init__(*args, **kwargs)

    def forward(self, *inputs, **kwargs):
        if not self.device_ids:
            return self.module(*inputs, **kwargs)
        if self.gpu0_bsz == 0:
            device_ids = self.device_ids[1:]
        else:
            device_ids = self.device_ids
        inputs, kwargs = self.scatter(inputs, kwargs, device_ids)
        if len(self.device_ids) == 1:
            return self.module(*inputs[0], **kwargs[0])
        replicas = self.replicate(self.module, self.device_ids)
        if self.gpu0_bsz == 0:
            replicas = replicas[1:]
        outputs = self.parallel_apply(replicas, device_ids, inputs, kwargs)
        return self.gather(outputs, self.output_device)

    def parallel_apply(self, replicas, device_ids, inputs, kwargs):
        return parallel_apply(replicas, inputs, kwargs, device_ids)

    def scatter(self, inputs, kwargs, device_ids):
        bsz = inputs[0].size(self.dim)
        num_dev = len(self.device_ids)
        gpu0_bsz = self.gpu0_bsz
        bsz_unit = (bsz - gpu0_bsz) // (num_dev - 1)
        if gpu0_bsz < bsz_unit:
            chunk_sizes = [gpu0_bsz] + [bsz_unit] * (num_dev - 1)
            delta = bsz - sum(chunk_sizes)
            for i in range(delta):
                chunk_sizes[i + 1] += 1
            if gpu0_bsz == 0:
                chunk_sizes = chunk_sizes[1:]
        else:
            return super().scatter(inputs, kwargs, device_ids)
        return scatter_kwargs(inputs, kwargs, device_ids, chunk_sizes, dim=self.dim)

# =========== functional utils ========== #
def load_topic_utterances(path):
    with open(path) as f:
        data = {'体育': [], '数码产品': [], '音乐': [], '电影': [], '美食': []}
        trs = {'体育': 'sport', '数码产品': 'electric', '音乐': 'music', '电影': 'movie', '美食': 'food'}
        for line in f.readlines():
            key, key_ = None, None
            for k in data.keys():
                k_ = trs[k]
                if k_ in line:
                    key_ = k_
                    key = k
                    break
            if key:
                line = line.replace(f'__label__{key_}', '').strip()
                line = ''.join(line.split())
                data[key].append(line)
    print(f'[!] load the topic trigger utterances over')
    return data

from torch.optim.lr_scheduler import _LRScheduler
class Noam(_LRScheduler):
    """
    Implements the Noam Learning rate schedule. This corresponds to increasing the learning rate
    linearly for the first ``warmup_steps`` training steps, and decreasing it thereafter proportionally
    to the inverse square root of the step number, scaled by the inverse square root of the
    dimensionality of the model. Time will tell if this is just madness or it's actually important.
    Parameters
    ----------
    warmup_steps: ``int``, required.
        The number of steps to linearly increase the learning rate.
    """
    def __init__(self, optimizer, warmup_steps, d_model):
        self.warmup_steps = warmup_steps
        self.d_model = d_model
        super(Noam, self).__init__(optimizer)
        # self.i_step = 0

    def get_lr(self):
        last_epoch = max(1, self.last_epoch)
        scale = (self.d_model** -0.5) * min([
            last_epoch ** (-0.5), 
            last_epoch * self.warmup_steps ** (-1.5),
            ])

        # return [base_lr  * scale for base_lr in self.base_lrs]
        return [base_lr / base_lr * scale for base_lr in self.base_lrs]

if __name__ == "__main__":
    a = torch.LongTensor([
        [1, 2, 4, 5, 7, 0, 0, 0, 0, 0],
        [1, 2, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 2, 3, 4, 0, 0, 0, 0, 0, 0],
        [1, 3, 4, 5, 6, 7, 8, 9, 1, 0],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 2]
        ])
    rest = generate_attention_mask(a)
    print(rest)
