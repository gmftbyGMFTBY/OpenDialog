from .header import *
from .bert_retrieval import *
from .test import *
    
'''
    GPT2Model:
    1. represent the context
    2. represent the retrieval responses

    BertForSequenceClassification:
    1. rerank the searched sentences
    2. rerank the candidates and the generated responses
'''

class GPT2Retrieval(nn.Module):

    def __init__(self, vocab_size, unk_id, sep_id, topk, topp, 
                 config_path='data/config/model_config_dialogue_small.json'):
        super(GPT2Retrieval, self).__init__()
        self.model_config = GPT2Config.from_json_file(config_path)
        self.model = GPT2Model(config=self.model_config)
        self.model.resize_token_embeddings(vocab_size)
        self.n_ctx = self.model.config.to_dict().get('n_ctx')
        self.topk, self.topp = topk, topp
        self.unk_id = unk_id
        self.sep_id = sep_id
        self.lm_head = nn.Linear(768 + 300, vocab_size)

    def forward(self, inpt_ids, candidates):
        '''
        inpt_ids: [batch, seq]
        candidates: [batch, 300], default k:=2
        '''
        # conversation query
        inpt_seq_size = inpt_ids.shape[1]
        attn_mask = generate_attention_mask(inpt_ids)
        outputs = self.model(
                    input_ids=inpt_ids, 
                    attention_mask=attn_mask)
        output = outputs[0]    # [batch, seq, hidden]
        rest = candidates.float()
        # candidates
        '''
        with torch.no_grad():
            rest = []
            for candidate in candidates:
                attn_mask = generate_attention_mask(candidate)    # [batch, seq]
                candidate_output = self.model(
                        input_ids=inpt_ids,
                        attention_mask=attn_mask)[0]    # [batch, seq, hidden]
                # avoid the [PAD] token, use masked mean operation
                candidate_output = candidate_output * attn_mask.unsqueeze(-1).float()
                sum_candidate = torch.sum(candidate_output, dim=1)    # [batch, hidden]
                sum_mask = torch.sum(attn_mask, dim=1)    # [batch]
                candidate_output = sum_candidate / sum_mask.unsqueeze(-1).float()    # [batch, hidden]
                rest.append(candidate_output)
        rest = torch.stack(rest).mean(dim=0)    # [k, batch, hidden] -> [batch, hidden]
        '''
        # rest: [batch, 300] -> [batch, seq, 300]
        rest = rest.view(rest.shape[0], 1, rest.shape[1]).expand(rest.shape[0], inpt_seq_size, rest.shape[1])
        # lm head
        # [batch, seq, hidden] + [batch, seq, 300]
        output = torch.cat([output, rest], dim=2)    # [batch, seq, hidden+300]
        output = self.lm_head(output)    # [batch, seq, vocab]
        return output

    def predict(self, inpt_ids, candidates, max_len):
        '''
        batch_size is 1
        inpt_ids: [seq]
        candidates: k*[seq]

        return a list of ids (generated)
        no pad, do not need attention_mask
        '''
        with torch.no_grad():
            # candidates: rest [seq, hidden]
            '''
            rest = []
            for candidate in candidates:
                attn_mask = generate_attention_mask(candidate)
                candidate_output = self.model(
                        input_ids=inpt_ids,
                        attention_mask=attn_mask)[0]    # [seq, hidden]
                # avoid the [PAD] token, use masked mean operation
                candidate_output = candidate_output * attn_mask.unsqueeze(-1).float()
                sum_candidate = torch.sum(candidate_output, dim=1)    # [batch, hidden]
                sum_mask = torch.sum(attn_mask, dim=1)    # [batch]
                candidate_output = sum_candidate / sum_mask.unsqueeze(-1).float()    # [batch, hidden]
                rest.append(candidate_output)
            rest = torch.stack(rest).mean(dim=0)    # [hidden]
            '''
            rest = candidates.float()
            generated, past = [], None
            for _ in range(max_len):
                outputs = self.model(input_ids=inpt_ids, past=past)
                outputs, past = outputs[:2]
                next_token_logits = outputs[-1, :]    # [hidden]
                next_token_logits = torch.cat((next_token_logits, rest))    # [hidden+300]
                next_token_logits = self.lm_head(next_token_logits)    # [vocab_size]
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
                inpt_ids = next_token
            return generated

class GPT2RetrievalAgent(BaseAgent):

    def __init__(self, total_steps, multi_gpu, vocab_file='data/vocab/vocab_small', run_mode='train', lang='zh'):
        super(GPT2RetrievalAgent, self).__init__()
        # hyperparameters
        try:
            self.gpu_ids = list(range(len(multi_gpu.split(','))))
        except:
            raise Exception(f'[!] multi gpu ids are needed, but got: {multi_gpu}')
        assert run_mode in ['train', 'test'], f'[!] running mode must be train or test, but got {run_mode}'
        vocab_file = 'data/vocab/vocab_small' if lang == 'zh' else 'data/vocab/vocab_english'
        self.args = {
                'lr': 1.5e-4,
                'grad_clip': 1.0,
                'pad': 0,
                'tgt_len_size': 50,
                'patience': 5,
                'min_lr': 1e-5,
                'warmup_steps': 2000,
                'total_steps': total_steps,
                'topk': 200,
                'topp': 1.0,
                'candidates_k': 2,
                'config_path': 'data/config/model_config_dialogue_big.json',
                'multi_gpu': self.gpu_ids,
                'run_mode': run_mode,
                'vocab_file': vocab_file,
                'lang': lang,
                'talk_samples': 256,
        }
        # hyperparameters

        self.vocab = BertTokenizer(vocab_file=self.args['vocab_file'])
        self.vocab_size = len(self.vocab)
        self.unk = self.vocab.convert_tokens_to_ids('[UNK]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')

        self.model = GPT2Retrieval(
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
        to_cuda(self.model, model=True)
        # train: DataParallel; test: no DataParallel
        if self.args['run_mode'] == 'train':
            self.model = DataParallel(self.model, device_ids=self.gpu_ids)
        self.show_parameters(self.args)

    def train_model(self, train_iter, mode='train', recoder=None):
        self.model.train()
        total_loss, batch_num = 0, 0
        pbar = tqdm(train_iter)
        for idx, batch in enumerate(pbar):
            # cid_text for searching the candiddates
            # cid_embed: k*[batch, 300]; cid: [batch, seq]
            cid_embed, cid = batch
            self.optimizer.zero_grad()
            logits = self.model(cid, cid_embed)    # [batch, seq, vocab]
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
            loss = loss / num_targets
            
            if mode == 'train':
                loss.backward()
                clip_grad_norm_(self.model.parameters(), self.args['grad_clip'])
                self.optimizer.step()
                self.warmup_scheduler.step()
            total_loss += loss.item()
            batch_num += 1

            pbar.set_description(f'[!] batch {batch_num}, train loss: {round(loss.item(), 4)}, token acc: {round(accuracy.item(), 4)}')
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
                candidates, c, r = batch
                max_size = max(len(r), self.args['tgt_len_size'])
                tgt = self.model.predict(c, candidates, max_size)
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
    
    def talk(self, topic, msgs, maxlen=30):
        '''
        topic, msgs: msgs is a string which split with the [SEP] token
        batch size is 1
        '''
        # tokenizer
        with torch.no_grad():
            msgs = torch.LongTensor(self.vocab.encode(msgs))
            if torch.cuda.is_available():
                msgs = msgs.cuda()
            tgt = self.model.predict(msgs, candidates, maxlen)
            tgt = self.vocab.convert_ids_to_tokens(tgt)
        tgt = ''.join(tgt)
        if '[sep]' in tgt:
            tgt = tgt[:tgt.index('[sep]')]
        return tgt
