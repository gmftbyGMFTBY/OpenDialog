from .header import *
from .biencoder import BERTBiEncoder

'''Cross-Attention BertRetrieval'''

class BERTRetrieval(nn.Module):

    def __init__(self, model='bert-base-chinese'):
        super(BERTRetrieval, self).__init__()
        self.model = BertForSequenceClassification.from_pretrained(model, num_labels=2)

    def forward(self, inpt, token_type_ids, attn_mask):
        output = self.model(
            input_ids=inpt,
            attention_mask=attn_mask,
            token_type_ids=token_type_ids,
        )
        logits = output[0]    # [batch, 2]
        return logits
    
class TopicPrediction(nn.Module):
    
    '''bert model as the backbone for semantic embedding;
    follow this work: 2020-COLING Towards Topic-Guided Conversational Recommender System.
    P_{topic}(t)=softmax(e_t^T\cdot \rm{MLP([r^{(1)}; r^{(2)}])}), where r^{(1)} represents the concatenation of the dialog history and the target topic word ([SEP] separating); r^{(2)} represents the concatenation of the historical topic sequence'''
    
    def __init__(self, vocab_size, dropout=0.3, model='bert-base-chinese'):
        super(TopicPrediction, self).__init__()
        self.bert = BertModel.from_pretrained(model)
        self.predictor = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(768, vocab_size),
        )
        
    def forward(self, inpt, attn_mask):
        inpt_embd = self.bert(
            input_ids=inpt,
            attention_mask=attn_mask,
        )[0][:, 0, :]
        rest = self.predictor(inpt_embd)    # [B, V]
        return rest
    
class BERTRetrievalAgent(RetrievalBaseAgent):

    def __init__(self, multi_gpu, run_mode='train', lang='zh', kb=True, local_rank=0):
        super(BERTRetrievalAgent, self).__init__(kb=kb)
        try:
            self.gpu_ids = list(range(len(multi_gpu.split(',')))) 
        except:
            raise Exception(f'[!] multi gpu ids are needed, but got: {multi_gpu}')
        self.args = {
            'lr': 2e-5,
            'grad_clip': 1.0,
            'samples': 10,
            'multi_gpu': self.gpu_ids,
            'talk_samples': 256,
            'max_len': 256,
            'vocab_file': 'bert-base-chinese',
            'pad': 0,
            'model': 'bert-base-chinese',
            'amp_level': 'O2',
            'local_rank': local_rank,
        }
        self.vocab = BertTokenizer.from_pretrained(self.args['vocab_file'])
        self.model = BERTRetrieval(self.args['model'])
        if torch.cuda.is_available():
            self.model.cuda()
        self.optimizer = transformers.AdamW(
            self.model.parameters(), 
            lr=self.args['lr'],
        )
        self.criterion = nn.CrossEntropyLoss()
        if run_mode == 'train':
            self.model, self.optimizer = amp.initialize(
                self.model, 
                self.optimizer, 
                opt_level=self.args['amp_level']
            )
            self.model = nn.parallel.DistributedDataParallel(
                self.model, device_ids=[local_rank], output_device=local_rank,
            )
        self.show_parameters(self.args)

    def train_model(self, train_iter, mode='train', recoder=None, idx_=0):
        self.model.train()
        total_loss, batch_num = 0, 0
        pbar = tqdm(train_iter)
        correct, s = 0, 0
        for idx, batch in enumerate(pbar):
            cid, token_type_ids, attn_mask, label = batch
            self.optimizer.zero_grad()
            output = self.model(cid, token_type_ids, attn_mask)    # [B, 2]
            loss = self.criterion(
                output, 
                label.view(-1),
            )
            
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
            clip_grad_norm_(amp.master_params(self.optimizer), self.args['grad_clip'])
            self.optimizer.step()

            total_loss += loss.item()
            batch_num += 1
            
            now_correct = torch.max(F.softmax(output, dim=-1), dim=-1)[1]    # [batch]
            now_correct = torch.sum(now_correct == label).item()
            correct += now_correct
            s += len(label)
            
            recoder.add_scalar(f'train-epoch-{idx_}/Loss', total_loss/batch_num, idx)
            recoder.add_scalar(f'train-epoch-{idx_}/RunLoss', loss.item(), idx)
            recoder.add_scalar(f'train-epoch-{idx_}/Acc', correct/s, idx)
            recoder.add_scalar(f'train-epoch-{idx_}/RunAcc', now_correct/len(label), idx)

            pbar.set_description(f'[!] train loss: {round(loss.item(), 4)}|{round(total_loss/batch_num, 4)}; acc: {round(now_correct/len(label), 4)}|{round(correct/s, 4)}')
        recoder.add_scalar(f'train-whole/Loss', total_loss/batch_num, idx_)
        recoder.add_scalar(f'train-whole/Acc', correct/s, idx_)
        return round(total_loss / batch_num, 4)

    @torch.no_grad()
    def test_model_(self, test_iter, path):
        self.model.eval()
        total_loss, batch_num = 0, 0
        pbar = tqdm(test_iter)
        rest = []
        for idx, batch in enumerate(pbar):
            cid, token_type_ids, attn_mask, label = batch
            output = self.model(cid, token_type_ids, attn_mask)    # [batch, 2]
            loss = self.criterion(output, label.view(-1))
            total_loss += loss.item()
            batch_num += 1
            output = F.softmax(output, dim=-1)[:, 1]    # [batch]
            preds = [i.tolist() for i in torch.split(output, self.args['samples'])]
            labels = [i.tolist() for i in torch.split(label, self.args['samples'])]
            for label, pred in zip(labels, preds):
                pred = np.argsort(pred, axis=0)[::-1]
                rest.append(([0], pred.tolist()))
        print(f'[!] test loss: {round(total_loss/batch_num, 4)}')
        p_1, r2_1, r10_1, r10_2, r10_5, MAP, MRR = cal_ir_metric(rest)
        print(f'[TEST] P@1: {p_1}; R2@1: {r2_1}; R10@1: {r10_1}; R10@2: {r10_2}; R10@5: {r10_5}; MAP: {MAP}; MRR: {MRR}')
        return round(total_loss/batch_num, 4)
    
    @torch.no_grad()
    def test_model(self, test_iter, path):
        self.model.eval()
        r1, r2, r5, r10, counter, mrr = 0, 0, 0, 0, 0, []
        pbar = tqdm(test_iter)
        for idx, batch in enumerate(pbar):
            cid, token_type_ids, attn_mask, label = batch
            output = self.model(cid, token_type_ids, attn_mask)    # [batch, 2]
            output = F.softmax(output, dim=-1)[:, 1]    # [batch]
            preds = [i.tolist() for i in torch.split(output, self.args['samples'])]
            for pred in preds:
                # pred: [B]
                pred = np.argsort(pred, axis=0)[::-1]
                r1 += (torch.topk(pred, 1, dim=-1)[1] == 0).sum().item()
                r2 += (torch.topk(pred, 2, dim=-1)[1] == 0).sum().item()
                r5 += (torch.topk(pred, 5, dim=-1)[1] == 0).sum().item()
                r10 += (torch.topk(pred, 10, dim=-1)[1] == 0).sum().item()
                pred = pred.numpy()
                y_true = np.zeros(len(pred))
                y_true[0] = 1
                mrr.append(label_ranking_average_precision_score([y_true], [pred]))
                counter += 1
        r1, r2, r5, r10, mrr = round(r1/counter, 4), round(r2/counter, 4), round(r5/counter, 4), round(r10/counter, 4), round(np.mean(mrr), 4)
        print(f'r1@10: {r1}; r2@10: {r2}; r5@10: {r5}; r10@10: {r10}; mrr: {mrr}')
        return round(total_loss/batch_num, 4)

    @torch.no_grad()
    def talk(self, msgs, topic=None):
        self.model.eval()
        utterances_, inpt_ids, token_type_ids, attn_mask = self.process_utterances(
            topic, msgs, max_len=self.args['max_len']
        )
        # prepare the data input
        output = self.model(inpt_ids, token_type_ids, attn_mask)    # [B, 2]
        output = F.softmax(output, dim=-1)[:, 1]    # [B]
        item = torch.argmax(output).item()
        msg = utterances_[item]
        return msg

    def reverse_search(self, ctx, ctx_, res):
        with torch.no_grad():
            utterances_ = self.searcher.search(None, ctx, samples=self.args['talk_samples'])
            utterances_ = [i['context'] for i in utterances_]
            if ctx_ in utterances_:
                mask_index = utterances_.index(ctx_)
            else:
                mask_index = None
            utterances = [f'{i} [SEP] {res}' for i in utterances_]
            ids = [torch.LongTensor(self.vocab.encode(i)[-512:]) for i in utterances]
            ids = pad_sequence(ids, batch_first=True, padding_value=self.args['pad'])
            if torch.cuda.is_available():
                ids = ids.cuda()
            output = self.model(ids)     # [batch, 2]
            if mask_index is not None:
                output[mask_index][1] = -inf
            output = F.softmax(output, dim=-1)[:, 1]
            item = torch.argmax(output)
            rest = utterances_[item]
            return rest

    def rerank(self, topic, msgs, topk=2):
        self.model.eval()
        with torch.no_grad():
            utterances_, ids = self.process_utterances(topic, msgs)
            output = self.model(ids)
            output = F.softmax(output, dim=-1)[:, 1]
            # argsort
            indexs = torch.argsort(output, descending=True)[:topk]
            msgs = [utterances_[index] for index in indexs]
            return msgs
        
class BERTRetrievalLoopAgent:
    
    '''Knowledge Graph Driven Retrieval Dialog System Loop Agent'''
    
    def __init__(self, multi_gpu, run_mode='train', lang='zh', kb=True):
        super(BERTRetrievalLoopAgent, self).__init__()
        self.agent = BERTRetrievalKGAgent(multi_gpu, run_mode='test', lang='zh', kb=True, local_rank=0)
        self.env = BERTRetrievalEnvAgent(multi_gpu, run_mode='test', lang='zh', kb=True, local_rank=0)
        
    def train_model(self, train_iter, mode='train', recoder=None, idx_=0):
        pass
    
    @torch.no_grad()
    def test_model(self):
        pass
        
class BERTRetrievalEnvAgent(BERTRetrievalAgent):
    
    '''Env conversation agent which doesn"t know the kg graph but can return the reward and the next observation;
    the model parameters is different from the BERTRetrievalKGAgent'''
    
    def __init__(self, multi_gpu, run_mode='train', lang='zh', kb=True, local_rank=0, wordnet=None, talk_samples=128):
        super(BERTRetrievalEnvAgent, self).__init__(multi_gpu, run_mode=run_mode, lang=lang, kb=kb, local_rank=local_rank)
        self.args['done_reward'], self.args['smooth_penalty'], self.args['step_penalty'] = 100, 20, 5
        self.wordnet = wordnet
        self.args['talk_sample'] = talk_samples
        self.lac = LAC(mode='lac')
        
    def reset(self):
        self.history = []
        
    def wrap_utterances(self, context, max_len=0):
        '''context is a list of string, which contains the dialog history'''
        context, response =  ' [SEP] '.join(context[:-1]), context[-1]
        # construct inpt_ids, token_type_ids, attn_mask
        inpt_ids = self.vocab.batch_encode_plus([context, response])['input_ids']
        context_inpt_ids, responses_inpt_ids = inpt_ids[0], inpt_ids[1]
        context_token_type_ids = [0] * len(context_inpt_ids)
        responses_token_type_ids = [1] * len(responses_inpt_ids)
        
        # length limitation
        inpt_ids, token_type_ids = context_inpt_ids + responses_inpt_ids[1:], context_token_type_ids + responses_token_type_ids[1:]
        if len(p1) > max_len:
            cut_size = len(p1) - max_len + 1
            inpt_ids = torch.LongTensor([inpt_ids[0]] + inpt_ids[cut_size:])
            token_type_ids = torch.LongTensor([token_type_ids[0]] + token_type_ids[cut_size:])
        else:
            inpt_ids = torch.LongTensor(inpt_ids)
            token_type_ids = torch.LongTensor(token_type_ids)
        attn_mask = torch.ones_like(inpt_ids)
        
        if torch.cuda.is_available():
            inpt_ids, token_type_ids, attn_mask = inpt_ids.cuda(), token_type_ids.cuda(), attn_mask.cuda()
        return inpt_ids, token_type_ids, attn_mask
    
    @torch.no_grad()
    def talk(self, msgs, topic=None):
        self.model.eval()
        utterances_, inpt_ids, token_type_ids, attn_mask = self.process_utterances(
            topic, msgs, max_len=self.args['max_len'],
        )
        # prepare the data input
        output = self.model(inpt_ids, token_type_ids, attn_mask)    # [B, 2]
        output = F.softmax(output, dim=-1)[:, 1]    # [B]
        item = torch.argmax(output).item()
        msg = utterances_[item]
        return msg
        
    @torch.no_grad()
    def get_reward(self, context, done=False, steps=0):
        '''construct the reward'''
        self.model.eval()
        if done:
            return self.args['done_reward'] - steps * self.args['step_penalty']
        else:
            output = self.model(*self.wrap_utterances(context, max_len=self.args['max_len']))    # [2]
            output = F.softmax(output, dim=-1)[0]
            reward = -self.args['smooth_penalty'] * output.item()
            return reward 
        
    def extract_topic_words(self, utterance):
        def filter(word, tag):
            def isChinese():
                for ch in word:
                    if not '\u4e00' <= ch <= '\u9fff':
                        return False
                return True
            def HaveDigital():
                if bool(re.search(r'\d', word)):
                    return False
                else:
                    return True
            def Length():
                if 1 < len(word) < 5:
                    return True
                else:
                    return False
            def HaveAlpha():
                for ch in word:
                    if ch.encode().isalpha():
                        return False
                return True
            def Special():
                for ch in word:
                    if ch in set('一二三四五六七八九十月日周年区东西南北。，|；“”‘’！~·、：=-+#￥%……&*（）【】@？.,?[]{}()!$^`";:'):
                        return False
                return True
            def CheckTag():
                if tag in set(['n', 'nz', 'nw', 'v', 'vn', 'a', 'ad', 'an', 'ORG', 'PER', 'LOC']):
                    return True
                else:
                    return False
            def InWordNet():
                if word in self.wordnet.nodes:
                    return True
                else:
                    return False
            return isChinese() and HaveDigital() and Length() and HaveAlpha() and Special() and CheckTag() and InWordNet()
        words, tags = self.lac.run(utterance)
        topic = []
        for word, tag in zip(words, tags):
            if filter(word, tag):
                topic.append(word)
        return list(set(topic))
        
    @torch.no_grad()
    def get_res(self, data):
        '''return reward and next utterances for the BERTRetrievalEnvAgent'''
        msgs = [i['msg'] for i in data['msgs']]
        # NOTE: in order to make sure the user speak based on the given conversation, use the topic for coarse ranking
        # topic = self.extract_topic_words(msgs[-1])
        topic = None
        msgs = ' [SEP] '.join(msgs)
        res = self.talk(msgs, topic=topic)
        self.history.append(res)
        return res
        
class BERTRetrievalKGGreedyAgent(BERTRetrievalAgent):
    
    '''fix the talk function for BERTRetrievalAgent
    Agent knows the whole knowledge graph path; but the other one doesn"t;
    greedy: ACL 2019 Target-Guided Open-Domain Conversation'''
    
    def __init__(self, multi_gpu, run_mode='train', lang='zh', kb=True, local_rank=0, wordnet=None, talk_samples=128):
        super(BERTRetrievalKGGreedyAgent, self).__init__(multi_gpu, run_mode=run_mode, lang=lang, kb=kb, local_rank=local_rank)
        self.topic_history = []
        self.wordnet = wordnet
        self.args['talk_samples'] = talk_samples
        self.w2v = gensim.models.KeyedVectors.load_word2vec_format(
            'data/chinese_w2v_base.txt', binary=False
        )
        
    def reset(self, target, source):
        self.args['target'], self.args['current_node'] = target, source
        self.topic_history, self.history = [source], []
        print(f'[! Reset the KG target] source: {source}; target: {target}; path: {path}')
        
    def search_candidates(self, msgs, nodes):
        '''Noted that input node maybe multiple topic words
        f(n) = g(n) + h(n)
        1. must have the path to the target    √
        2. compared with current node, more closer to the target    √
        3. retrieval utterance must contain both current and candidate node
        4. as for g(n), 1) word similarity; 2) average bag of utterances coherence
        5. as for f(n), 2) number of the retrieval utterance based on the current node and candidate node and their corresponding average coherence; 3) RL
        '''
        # generate candidates
        candidates = []
        for node in nodes:
            neighbors = []
            base_dis = self.w2v.similarity(node, self.args['target'])
            for n in self.wordnet.neighbors(node):
                if self.w2v.similarity(n, self.args['target']) >= base_dis:
                    continue
                retrieval_rest = self.searcher.must_search(
                    msgs, topic=[node, n], samples=self.args['talk_samples']
                )
                if not retrieval_rest:
                    continue
                try:
                    path = nx.shortest_path(self.wordnet, n, self.args['target'])
                except nx.NetworkXNoPath as e:
                    continue
                neighbors.append((node, n, path))
            candidates.extend(neighbors)
        # score the f(n) and sort
        pass
        
    def move_on_kg(self, current_nodes, size=1):
        '''current nodes are extracted from the human utterance (maybe multiple)'''
        candidates = self.search_candidates(msgs, current_nodes)[:size]
        return candidates
        
    def process_utterances(self, utterances, msgs, max_len=0):
        '''Process the utterances searched by Elasticsearch; input_ids/token_type_ids/attn_mask'''
        # assert lern(topic) > 0, f'[!] topic words must exists'
        # utterances_ = self.searcher.must_search(
        #     msgs, samples=self.args['talk_samples'], topic=topic
        # )
        # utterances_ = [i['utterance'] for i in utterances_]
        # remove the utterances that in the self.history
        # utterances_ = list(set(utterances_) - set(self.history))
        
        # construct inpt_ids, token_type_ids, attn_mask
        inpt_ids = self.vocab.batch_encode_plus([msgs] + utterances_)['input_ids']
        context_inpt_ids, responses_inpt_ids = inpt_ids[0], inpt_ids[1:]
        context_token_type_ids = [0] * len(context_inpt_ids)
        responses_token_type_ids = [[1] * len(i) for i in responses_inpt_ids]
        
        # length limitation
        collection = []
        for r1, r2 in zip(responses_inpt_ids, responses_token_type_ids):
            p1, p2 = context_inpt_ids + r1[1:], context_token_type_ids + r2[1:]
            if len(p1) > max_len:
                cut_size = len(p1) - max_len + 1
                p1, p2 = [p1[0]] + p1[cut_size:], [p2[0]] + p2[cut_size:]
            collection.append((p1, p2))
            
        inpt_ids = [torch.LongTensor(i[0]) for i in collection]
        token_type_ids = [torch.LongTensor(i[1]) for i in collection]
        
        inpt_ids = pad_sequence(inpt_ids, batch_first=True, padding_value=self.args['pad'])
        token_type_ids = pad_sequence(token_type_ids, batch_first=True, padding_value=self.args['pad'])
        attn_mask_index = inpt_ids.nonzero().tolist()
        attn_mask_index_x, attn_mask_index_y = [i[0] for i in attn_mask_index], [i[1] for i in attn_mask_index]
        attn_mask = torch.zeros_like(inpt_ids)
        attn_mask[attn_mask_index_x, attn_mask_index_y] = 1
        
        if torch.cuda.is_available():
            inpt_ids, token_type_ids, attn_mask = inpt_ids.cuda(), token_type_ids.cuda(), attn_mask.cuda()
        return utterances_, inpt_ids, token_type_ids, attn_mask
        
    @torch.no_grad()
    def talk(self, msgs, topics):
        ''':topic: means the current topic node in the knowledge graph path.'''
        self.model.eval()
        # 1) inpt the topic information for the coarse filter in elasticsearch
        utterances, inpt_ids, token_type_ids, attn_mask = self.process_utterances(
            topics, msgs, max_len=self.args['max_len'],
        )
        # 2) neural ranking with the topic information
        output = self.model(inpt_ids, token_type_ids, attn_mask)    # [B, 2]
        output = F.softmax(output, dim=-1)[:, 1]    # [B]
        output = torch.argsort(output, descending=True)
        item = torch.argmax(output).item()
        msg = utterances[item]
        return msg
    
    def obtain_keywords(self, utterance):
        '''select the keyword that most similar to the current_node as the keyword in the human response'''
        keywords = analyse.extract_tags(utterance)
        nodes = [i for i in keywords if i in self.wordnet.nodes]
        assert len(nodes) != 0, f'[!] cannot find the keywords in the human utterances'
        keyword = random.choice(nodes)
        return keyword
    
    def get_res(self, data):
        '''
        data = {
            "msgs": [
                {
                    'fromUser': robot_id,
                    'msg': msg,
                    'timestamp': timestamp
                },
                ...
            ]
        }
        ''' 
        if len(data['msgs']) > 0:
            # 1) move
            self.move_on_kg()
            # 2) obtain the responses based on the next_node
            msgs = [i['msg'] for i in data['msgs']]
            msgs = ' [SEP] '.join(msgs)
            res = self.talk(msgs)
        else:
            res = self.searcher.talk('', topic=[self.args['current_node']])
        self.history.append(res)
        return res
    
class BERTRetrievalClusterGreedyAgent(BERTRetrievalAgent):
    
    '''fix the talk function for BERTRetrievalAgent
    Agent knows the whole knowledge graph path; but the other one doesn"t;
    greedy: ACL 2019 Target-Guided Open-Domain Conversation
    
    1. 簇不能太大，太大容易造成语义不集中，也不能太小，太小并不能快速接近target
    2. 还可以有多个簇，如果一次选出多个候选keyword的话，下一轮你就存在多个簇
    3. 如何选择多个簇可以看作是强化学习的任务，保证选择的多个簇有利于向着target推进并保证会话潜移的平滑性
    4. 如果用RL来选择的话，可以使用transformer将所有的候选信息全部全连接编码，通过对比的方式选择更好的一组keywords
    5. 选出来的一组keyword之间的相关性也要保证
    
    f(x) = g(x) + h(x)
    '''
    
    def __init__(self, multi_gpu, run_mode='train', lang='zh', kb=True, local_rank=0, wordnet=None, talk_samples=128):
        super(BERTRetrievalClusterGreedyAgent, self).__init__(multi_gpu, run_mode=run_mode, lang=lang, kb=kb, local_rank=local_rank)
        self.topic_history = []
        self.wordnet = wordnet
        self.args['talk_samples'] = talk_samples
        self.args['num_candidate'] = 5
        self.args['cluster_width'] = 50
        self.w2v = gensim.models.KeyedVectors.load_word2vec_format(
            'data/chinese_w2v_base.txt', binary=False
        )
        self.cutter = LAC(mode='seg')
        self.g_scorer = BERTBiEncoder(share=True)
        
    def reset(self, target, source):
        self.args['target'], self.args['current_node'] = target, source
        self.topic_history, self.history = [[source]], []
        print(f'[! Reset the KG target] source: {source}; target: {target}')
        
    def obtain_keywords(self, utterance):
        '''select the keyword that most similar to the current_node as the keyword in the human response'''
        return [i for i in self.cutter.run(utterance) if i in self.w2v]
        
    def get_candidates(self, start_node, size=5):
        '''obtain a cluster of nodes which is more closer than the start node'''
        # obtain not only the 1-hop neighboor
        path_lengths = nx.single_source_shortest_path_length(self.wordnet, start_node, cutoff=2)
        candidates = [node for node, length in path_lengths.items() if length <= 2]
        dis = self.w2v.most_similar(start_node, self.args['target'])
        # 1) closer and have the path to the target will be consider as the candidates
        nodes = []
        for node in candidates:
            if self.w2v.similarity(node, self.args['target']) > dis:
                path = nx.dijkstra_path(self.wordnet, node, self.args['target'])
                if path:
                    nodes.append((node, path))
        return nodes
    
    def get_g_function(self, msgs, node):
        # retrieval a bag of responses
        utterances, context_inpt_ids, response_inpt_ids, attn_mask = self.process_utterances_biencoder([node], msgs, max_len=self.args['max_len'])
        # average similarity
        ipdb.set_trace()
        scores = self.g_scorer.predict(context_inpt_ids, response_inpt_ids, attn_mask).tolist()
        return np.mean(scores)
    
    def get_h_function(self, path):
        # search the path
        # calculate the path
        pass
        
    @torch.no_grad()
    def move_on_kg(self):
        '''judge whether meet the end'''
        candidates = self.get_candidates(
            self.args['current_node'],
            size=self.args['num_candidate'],
        )
        g_scores = [self.get_g_function(i[0]) for i in candidates]
        h_scores = [self.get_h_function(i[1]) for i in candidates]
        # resort
        scores = []
        for g, h, candidate in candidates:
            scores.append(((g + h)/2, candidate))
        scores = sorted(scores, key=lambda x:x[0], reverse=True)
        self.args['current_node'] = scores[0][1]
        if self.args['current_node'] == self.args['target']:
            pass
        
    @torch.no_grad()
    def talk(self, msgs):
        ''':topic: means the current topic node in the knowledge graph path.'''
        self.model.eval()
        # 1) inpt the topic information for the coarse filter in elasticsearch
        utterances, inpt_ids, token_type_ids, attn_mask = self.process_utterances(
            self.args['current_node'], msgs, max_len=self.args['max_len'],
        )
        # 2) neural ranking with the topic information
        output = self.model(inpt_ids, token_type_ids, attn_mask)    # [B, 2]
        output = F.softmax(output, dim=-1)[:, 1]    # [B]
        # 3) post ranking with multiple current topic words
        output = torch.argsort(output, descending=True)
        for i in output:
            flag, chosen_word = False, None
            for word in self.args['current_node']:
                if word in utterances[i.item()]:
                    item, flag = i, True
                    chosen_word = word
                    break
            if flag:
                break
        else:
            item = 0
        msg = utterances[item]
        return msg, chosen_word
    
    def get_res(self, data):
        '''
        data = {
            "msgs": [
                {
                    'fromUser': robot_id,
                    'msg': msg,
                    'timestamp': timestamp
                },
                ...
            ]
        }
        ''' 
        if len(data['msgs']) > 0:
            # 1) move
            response = data['msgs'][-1]['msg']
            keywords = self.obtain_keywords(response)
            if self.args['current_node']:
                self.args['current_node'] = list(set(keywords + [self.args['current_node']]))
            else:
                self.args['current_node'] = list(set(keywords))
            self.move_on_kg()
            # 2) obtain the responses based on the next_node
            msgs = [i['msg'] for i in data['msgs']]
            msgs = ' [SEP] '.join(msgs)
            res, chosen_word = self.talk(msgs)
            self.topic_history.append(self.args['current_node'])
            self.args['current_node'] = chosen_word
        else:
            res = self.searcher.talk('', topic=[self.args['current_node']])
        self.history.append(res)
        return res
    
class BERTRetrievalPredictionGreedyAgent(BERTRetrievalAgent):
    
    '''fix the talk function for BERTRetrievalAgent
    Agent knows the whole knowledge graph path; but the other one doesn"t;
    greedy: ACL 2019 Target-Guided Open-Domain Conversation
    
    Compared with the BERTRetrievalKGGreedyAgent, this agent train the classification model on the wordnet for topic selection. 
    '''
    
    def __init__(self, multi_gpu, run_mode='train', lang='zh', kb=True, local_rank=0, talk_samples=128):
        super(BERTRetrievalPredictionGreedyAgent, self).__init__(multi_gpu, run_mode=run_mode, lang=lang, kb=kb, local_rank=local_rank)
        self.topic_history = []
        with open('data/wordnet.pkl', 'rb') as f:
            wordnet = pickle.load(f)
            self.wordnet = list(wordnet.nodes)
        self.args['talk_samples'] = talk_samples
        
        # topic word prediction network
        self.topic_predictor = TopicPrediction(len(self.wordnet))
        
        self.vocab = BertTokenizer.from_pretrained(self.args['vocab_file'])
        if torch.cuda.is_available():
            self.topic_predictor.cuda()
        self.optimizer = transformers.AdamW(
            self.topic_predictor.parameters(), 
            lr=self.args['lr'],
        )
        self.criterion = nn.CrossEntropyLoss()
        if run_mode == 'train':
            self.topic_predictor, self.optimizer = amp.initialize(
                self.topic_predictor, 
                self.optimizer, 
                opt_level=self.args['amp_level']
            )
            self.model = nn.parallel.DistributedDataParallel(
                self.topic_predictor, device_ids=[local_rank], output_device=local_rank,
            )
        
    def train_model(self, train_iter, mode='train', recoder=None, idx_=0):
        '''train the topic predictor'''
        self.model.train()
        total_loss, batch_num = 0, 0
        pbar = tqdm(train_iter)
        correct, s = 0, 0
        for idx, batch in enumerate(pbar):
            cid, attn_mask, label = batch
            self.optimizer.zero_grad()
            output = self.topic_predictor(cid, attn_mask)    # [B, V]
            loss = self.criterion(
                output, 
                label.view(-1),
            )
            
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
            clip_grad_norm_(amp.master_params(self.optimizer), self.args['grad_clip'])
            self.optimizer.step()

            total_loss += loss.item()
            batch_num += 1
            
            now_correct = torch.max(F.softmax(output, dim=-1), dim=-1)[1]
            now_correct = torch.sum(now_correct == label).item()
            correct += now_correct
            s += len(label)
            
            recoder.add_scalar(f'train-epoch-{idx_}/Loss', total_loss/batch_num, idx)
            recoder.add_scalar(f'train-epoch-{idx_}/RunLoss', loss.item(), idx)
            recoder.add_scalar(f'train-epoch-{idx_}/Acc', correct/s, idx)
            recoder.add_scalar(f'train-epoch-{idx_}/RunAcc', now_correct/len(label), idx)

            pbar.set_description(f'[!] train loss: {round(loss.item(), 4)}|{round(total_loss/batch_num, 4)}; acc: {round(now_correct/len(label), 4)}|{round(correct/s, 4)}')
        recoder.add_scalar(f'train-whole/Loss', total_loss/batch_num, idx_)
        recoder.add_scalar(f'train-whole/Acc', correct/s, idx_)
        return round(total_loss / batch_num, 4)
    
    @torch.no_grad()
    def test_model(self, test_iter, path):
        self.model.eval()
        total_loss, batch_num = 0, 0
        pbar = tqdm(test_iter)
        rest = []
        with open(path, 'w') as f:
            for idx, batch in enumerate(pbar):
                cid, attn_mask, label = batch
                output = self.topic_predictor(cid, attn_mask)    # [B, V]
                loss = self.criterion(output, label.view(-1))
                total_loss += loss.item()
                batch_num += 1
                output = torch.max(F.softmax(output, dim=-1), dim=-1)[1]    # [B]
                topic_words = [self.wordnet[i] for i in output.tolist()]
                # write the result into the file
                for cid_, rest_, label_ in zip(cid, topic_words, label):
                    cid_ = self.vocab.decode(cid_).replace('[PAD]', '')
                    f.write(f'CTX: {cid_}\n')
                    f.write(f'REF: {self.wordnet[label_]}\n')
                    f.write(f'TGT: {rest_}\n\n')
        print(f'[!] test loss: {round(total_loss/batch_num, 4)}')
        return round(total_loss/batch_num, 4)
        
    def reset(self, target, source, path):
        self.args['target'], self.args['current_node'], self.args['path'] = target, source, path
        self.topic_history, self.history = [source], []
        print(f'[! Reset the KG target] source: {source}; target: {target}; path: {path}')
        
    def move_on_kg(self):
        '''judge whether meet the end'''
        if self.topic_history[-1] == self.args['target']:
            return
        self.args['current_node'] = self.args['path'][len(self.topic_history)]
        self.topic_history.append(self.args['current_node'])
        
    @torch.no_grad()
    def talk(self, msgs):
        ''':topic: means the current topic node in the knowledge graph path.'''
        self.model.eval()
        # 1) inpt the topic information for the coarse filter in elasticsearch
        utterances, inpt_ids, token_type_ids, attn_mask = self.process_utterances(
            [self.args['current_node']], msgs, max_len=self.args['max_len'],
        )
        # 2) neural ranking with the topic information
        output = self.model(inpt_ids, token_type_ids, attn_mask)    # [B, 2]
        output = F.softmax(output, dim=-1)[:, 1]    # [B]
        # 3) post ranking with current topic word
        output = torch.argsort(output, descending=True)
        for i in output:
            if self.args['current_node'] in utterances[i.item()]:
                item = i
                break
        else:
            item = 0
        # item = torch.argmax(output).item()
        msg = utterances[item]
        return msg
    
    def obtain_keywords(self, utterance):
        '''select the keyword that most similar to the current_node as the keyword in the human response'''
        keywords = analyse.extract_tags(utterance)
        nodes = [i for i in keywords if i in self.wordnet.nodes]
        assert len(nodes) != 0, f'[!] cannot find the keywords in the human utterances'
        keyword = random.choice(nodes)
        return keyword
    
    def get_res(self, data):
        '''
        data = {
            "msgs": [
                {
                    'fromUser': robot_id,
                    'msg': msg,
                    'timestamp': timestamp
                },
                ...
            ]
        }
        ''' 
        if len(data['msgs']) > 0:
            # 1) move
            self.move_on_kg()
            # 2) obtain the responses based on the next_node
            msgs = [i['msg'] for i in data['msgs']]
            msgs = ' [SEP] '.join(msgs)
            res = self.talk(msgs)
        else:
            res = self.searcher.talk('', topic=[self.args['current_node']])
        self.history.append(res)
        return res