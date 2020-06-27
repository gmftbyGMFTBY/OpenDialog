from .header import *
from .test import TestAgent

'''
Pre-training and Fine-tuning the GPT2
In pre-training stage, GPTModel + lm _head
In Fine-tuning stage, GPTModel + Transformer_matrix + lm_head
'''

class PFGPT2(nn.Module):

    def __init__(self, vocab_size, unk_id, sep_id, topk, topp, 
                 config_path='data/config/model_config_dialogue_small.json'):
        super(PFGPT2, self).__init__()
        self.model_config = GPT2Config.from_json_file(config_path)
        self.model = GPT2LMHeadModel(config=self.model_config)
        self.model.resize_token_embeddings(vocab_size)
        self.n_ctx = self.model.config.to_dict().get('n_ctx')
        self.topk, self.topp = topk, topp
        self.unk_id = unk_id
        self.sep_id = sep_id

    def forward(self, inpt_ids):
        # inpt_ids: [batch, seq]
        attn_mask = generate_attention_mask(inpt_ids)
        outputs = self.model(
                    input_ids=inpt_ids, 
                    attention_mask=attn_mask)
        output = outputs[0]    # [batch, seq, vocab]
        return output

    def predict(self, inpt_ids, max_len):
        '''
        batch_size is 1
        inpt_ids: [seq]
        return a list of ids (generated)
        no pad, do not need attention_mask
        '''
        with torch.no_grad():
            generated = []
            for _ in range(max_len):
                outputs = self.model(input_ids=inpt_ids)
                next_token_logits = outputs[0][-1, :]    # [vocab]
                # ignore the [UNK] token
                next_token_logits[self.unk_id] = -np.inf
                filtered_logits = top_k_top_p_filtering(
                        next_token_logits, 
                        top_k=self.topk, 
                        top_p=self.topp)
                next_token = torch.multinomial(
                        F.softmax(filtered_logits, dim=-1),
                        num_samples=1)
                if next_token == self.sep_id:
                    break
                generated.append(next_token.item())
                inpt_ids = torch.cat((inpt_ids, next_token), dim=0)
                # remember to cut off 
                inpt_ids = inpt_ids[-self.n_ctx:]
            return generated

    @torch.no_grad()
    def predict_batch(self, inpt_ids, max_len):
        '''
        inpt_ids: [batch, seq]
        '''
        # change inpt_ids from [seq] to [batch, seq]
        generated = []
        prev, past = inpt_ids, None
        for _ in range(max_len):
            outputs = self.model(input_ids=prev, past=past)    # [batch, seq, vocab]
            output, past = outputs[:2]
            next_token_logits = output[:, -1, :]    # [batch, vocab]
            next_token_logits[:, self.unk_id] = -np.inf
            filtered_logits = top_k_top_p_filtering_batch(
                    next_token_logits, 
                    top_k=self.topk, 
                    top_p=self.topp)
            next_token = torch.multinomial(
                    F.softmax(filtered_logits, dim=-1),
                    num_samples=1)    # [batch, 1]
            generated.append([token.item() for token in next_token.squeeze(1)])
            prev = next_token
        # transpose
        ng, batch_size = [], len(generated[0])
        for i in range(batch_size):
            ng.append([g[i] for g in generated])
        return ng

class PFGPT2Agent(BaseAgent):

    '''
    run_mode:
       1. train: direcctly fine tuning the LM model
       2. train_trs: fine tuning with the semantic transformer matrix
    '''

    def __init__(self, total_steps, multi_gpu, vocab_file='data/vocab/vocab_small', run_mode='train', lang='zh'):
        super(PFGPT2Agent, self).__init__()
        # hyperparameters
        try:
            # self.gpu_ids = [int(i) for i in multi_gpu.split(',')]
            self.gpu_ids = list(range(len(multi_gpu.split(','))))
        except:
            raise Exception(f'[!] multi gpu ids are needed, but got: {multi_gpu}')
        assert run_mode in ['train', 'test', 'train_trs', 'rerank', 'rerank_ir'], f'[!] running mode must be train or test, but got {run_mode}'
        vocab_file = 'data/vocab/vocab_small' if lang == 'zh' else 'data/vocab/vocab_english'
        self.args = {
                'lr': 1e-5,
                'grad_clip': 1.0,
                'pad': 0,
                'tgt_len_size': 50,
                'lr_gamma': 0.5,
                'patience': 5,
                'min_lr': 1e-5,
                'warmup_steps': 2000,
                'total_steps': total_steps,
                'topk': 20,
                'topp': 1.0, 
                'config_path': 'data/config/model_lm_small.json',
                'pretrain_model': 'ckpt/LM/gpt2lm/best.pt',
                'multi_gpu': self.gpu_ids,
                'run_mode': run_mode,
                'vocab_file': vocab_file,
                'lang': lang,
                'topic_transfer': {'音乐': 'music', '体育': 'sport', '数码电子': 'electric', '美食': 'food', '电影': 'movie'}
        }
        # hyperparameters

        self.vocab = BertTokenizer(vocab_file=self.args['vocab_file'])
        self.vocab_size = len(self.vocab)
        self.unk = self.vocab.convert_tokens_to_ids('[UNK]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')

        self.model = PFGPT2(
                self.vocab_size, 
                self.unk, 
                self.sep,
                self.args['topk'], 
                self.args['topp'], 
                config_path=self.args['config_path']
        )

        self.criterion = nn.CrossEntropyLoss(ignore_index=self.args['pad'], reduction='sum')
        self.optimizer = transformers.AdamW(
                self.model.parameters(), 
                lr=self.args['lr'], 
                correct_bias=True)
        # need to obtain the whole iter
        self.warmup_scheduler = transformers.get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.args['warmup_steps'],
                num_training_steps=self.args['total_steps'])
        if torch.cuda.is_available():
            self.model.cuda()
        # train: DataParallel; test: no DataParallel
        if self.args['run_mode'] == 'train':
            self.model = DataParallel(self.model, device_ids=self.gpu_ids)
        
        # run_mode == 'chatbot', use the bertretrieval for reranking
        if run_mode in ['rerank', 'rerank_ir']:
            from multiview import MultiView
            print(f'[!] MultiView reranker model will be initized')
            self.reranker = MultiView(
                    topic=True,
                    logic=False,
                    nli=False,
                    coherence=True,
                    fluency=False,
                    coherence_path='ckpt/train_retrieval/bertretrieval/best.pt',
                    topic_path='ckpt/fasttext/model.bin',
                    )
            print(f'[!] load multiview model over')

        if run_mode == 'rerank_ir':
            self.ir_agent = TestAgent()

        self.show_parameters(self.args)
        
        # load the LM pretrain model
        self.load_model(self.args['pretrain_model'])

    def train_model_transformer(self, train_iter, mode='train'):
        pass

    def train_model(self, train_iter, mode='train'):
        if self.args['run_mode'] == 'train':
            self.train_model_directly(train_iter)
        elif self.args['run_mode'] == 'train_trs':
            self.train_model_transformer(train_iter)
        else:
            raise Exception(f'[!] unknow training mode: {self.args["run_mode"]}')

    def train_model_directly(self, train_iter, mode='train'):
        self.model.train()
        total_loss, total_acc, batch_num = 0, [], 0
        pbar = tqdm(train_iter)
        for idx, batch in enumerate(pbar):
            cid = batch
            
            self.optimizer.zero_grad()

            logits = self.model(cid)    # [batch, seq, vocab]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = cid[..., 1:].contiguous()
            loss = self.criterion(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1))
            _, preds = shift_logits.max(dim=-1)    # [batch, seq]
            # ignore the pad
            not_ignore = shift_labels.ne(self.args['pad'])   # pad is 0 or 1
            num_targets = not_ignore.long().sum().item()    # the number of not pad tokens
            correct = (shift_labels == preds) & not_ignore
            correct = correct.float().sum()
            # loss and token accuracy
            accuracy = correct / num_targets
            total_acc.append(accuracy)
            loss = loss / num_targets
            
            if mode == 'train':
                loss.backward()
                clip_grad_norm_(self.model.parameters(), self.args['grad_clip'])
                self.optimizer.step()
                self.warmup_scheduler.step()
            total_loss += loss.item()
            batch_num += 1

            pbar.set_description(f'[!] lr: {self.args["lr"]}, train loss: {round(loss.item(), 4)}, token acc: {round(accuracy.item(), 4)}')
        return round(total_loss/batch_num, 4)

    def test_model(self, test_iter, path):
        '''
        Generate the test dataset and measure the performance
        '''
        def filter(x):
            return x.replace('[PAD]', '')
        self.model.eval()
        pbar = tqdm(test_iter)
        with open(path, 'w') as f:
            for batch in pbar:
                c, r = batch
                max_size = max(len(r), self.args['tgt_len_size'])
                tgt = self.model.predict(c, max_size)
                text = self.vocab.convert_ids_to_tokens(tgt)
                tgt = ''.join(text)

                ctx = self.vocab.convert_ids_to_tokens(c)
                ctx = filter(''.join(ctx))

                ref = self.vocab.convert_ids_to_tokens(r)
                ref = filter(''.join(ref))

                f.write(f'CTX: {ctx}\n')
                f.write(f'REF: {ref}\n')
                f.write(f'TGT: {tgt}\n\n')
        print(f'[!] translate test dataset over, write into {path}')
        # measure the performance
        (b1, b2, b3, b4), ((r_max_l, r_min_l, r_avg_l), (c_max_l, c_min_l, c_avg_l)), (dist1, dist2, rdist1, rdist2), (average, extrema, greedy) = cal_generative_metric(path, lang=self.args['lang'])
        print(f'[TEST] BLEU: {b1}/{b2}/{b3}/{b4}; Length(max, min, avg): {c_max_l}/{c_min_l}/{c_avg_l}|{r_max_l}/{r_min_l}/{r_avg_l}; Dist: {dist1}/{dist2}|{rdist1}/{rdist2}; Embedding(average/extrema/greedy): {average}/{extrema}/{greedy}')
    
    @torch.no_grad()
    def talk(self, topic, msgs, maxlen=30, batch_size=16):
        '''
        topic, msgs: msgs is a string which split with the [SEP] token
        batch size is 1
        '''
        # tokenizer
        if self.args['run_mode'] == 'test':
            msgs = torch.LongTensor(self.vocab.encode(msgs))
            msgs = to_cuda(msgs)
            tgt = self.model.predict(msgs, maxlen)
            tgt = self.vocab.convert_ids_to_tokens(tgt)
            tgt = ''.join(tgt)
            return tgt
        elif self.args['run_mode'] in ['rerank', 'rerank_ir']:
            # ========== predict_batch ==========
            msgs_ = self.vocab.encode(msgs)
            msgs_ = [deepcopy(msgs_) for _ in range(batch_size)]
            msgs_ = torch.LongTensor(msgs_)    # [batch, seq]
            msgs_ = to_cuda(msgs_)
            tgt = self.model.predict_batch(msgs_, maxlen)
            tgt = [self.vocab.convert_ids_to_tokens(i) for i in tgt]
            # cut from the first [SEP] token
            n_tgt = []
            for i in tgt:
                if '[SEP]' in i:
                    i = i[:i.index('[SEP]')]
                n_tgt.append(''.join(i))
            # multiview scores
            # rerank_ir also use the fast retrieval model
            if self.args['run_mode'] == 'rerank_ir':
                retrieval_rest = self.ir_agent.model.search(topic, msgs, samples=batch_size)
                retrieval_rest = [i['response'] for i in retrieval_rest]
                n_tgt.extend(retrieval_rest)
            contexts = [msgs] * len(n_tgt)
            topic = [self.args['topic_transfer'][topic]]  * len(n_tgt)
            scores = self.reranker(contexts, n_tgt, topic=topic)[0]
            index = np.argmax(scores)
            response = n_tgt[index]
            return response
        else:
            raise Exception(f'[!] error in gpt2 model `talk` function')
