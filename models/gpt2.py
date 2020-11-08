from .header import *
from .test import TestAgent

class GPT2(nn.Module):

    def __init__(self, vocab_size, unk_id, sep_id, cls_id, topk, topp, 
                 repetition_penalty,
                 config_path='data/config/model_config_dialogue_small.json'):
        super(GPT2, self).__init__()
        self.model_config = GPT2Config.from_json_file(config_path)
        self.model = GPT2LMHeadModel(config=self.model_config)
        self.model.resize_token_embeddings(vocab_size)
        self.n_ctx = self.model.config.to_dict().get('n_ctx')
        self.topk, self.topp = topk, topp
        self.unk_id = unk_id
        self.sep_id = sep_id
        self.cls_id = cls_id
        self.repetition_penalty = repetition_penalty

    def forward(self, inpt_ids):
        # inpt_ids: [batch, seq]
        # ipdb.set_trace()
        # attn_mask = generate_attention_mask(inpt_ids)
        outputs = self.model(
            input_ids=inpt_ids, 
            # attention_mask=attn_mask
        )
        output = outputs[0]    # [batch, seq, vocab]
        return output

    @torch.no_grad()
    def predict(self, inpt_ids, max_len):
        '''batch_size is 1; inpt_ids: [seq]; return a list of ids (generated)'''
        generated = [self.cls_id]
        for _ in range(max_len):
            outputs = self.model(
                input_ids=inpt_ids
            )
            next_token_logits = outputs[0][-1, :]    # [V]
            next_token_logits[self.unk_id] = -np.inf
            if generated:
                next_token_logits[list(set(generated))] /= self.repetition_penalty
            filtered_logits = top_k_top_p_filtering(
                    next_token_logits, 
                    top_k=self.topk, 
                    top_p=self.topp)
            next_token = torch.multinomial(
                    F.softmax(filtered_logits, dim=-1),
                    num_samples=1)
            generated.append(next_token.item())
            if next_token == self.sep_id:
                break
            inpt_ids = torch.cat((inpt_ids, next_token), dim=0)
            inpt_ids = inpt_ids[-self.n_ctx:]
        return generated

    @torch.no_grad()
    def predict_batch(self, inpt_ids, attn_mask, position_ids, max_len):
        '''inpt_ids: [batch, seq]; return: samples'''
        # change inpt_ids from [seq] to [batch, seq]
        batch_size = inpt_ids.shape[0]
        generated = [[self.cls_id] * batch_size]
        prev, past = inpt_ids, None
        stop_flag = np.zeros(batch_size)    # [batch]
        for _ in range(max_len):
            outputs = self.model(
                input_ids=prev, 
                attention_mask=attn_mask, 
                past=past,
                position_ids=position_ids,
            )    # [batch, seq, vocab]
            output, past = outputs[:2]
            next_token_logits = output[:, -1, :]    # [batch, vocab]
            next_token_logits[:, self.unk_id] = -np.inf
            # repetition penalty
            for x in range(batch_size):
                y = [item[x] for item in generated]
                next_token_logits[x, y] /= self.repetition_penalty
            filtered_logits = top_k_top_p_filtering_batch(
                next_token_logits, 
                top_k=self.topk, 
                top_p=self.topp,
            )
            next_token = torch.multinomial(
                F.softmax(filtered_logits, dim=-1),
                num_samples=1,
            )    # [batch, 1]
            # set up stop_flag
            for idx, i in enumerate(next_token.squeeze(1)):
                if i == self.sep_id:
                    stop_flag[idx] = 1
            generated.append([token.item() for token in next_token.squeeze(1)])
            prev = next_token
            if sum(stop_flag) == batch_size:
                break
            attn_mask = torch.cat([attn_mask, torch.tensor([1] * batch_size).unsqueeze(1).cuda()], dim=1)
            if past:
                position_ids = (attn_mask.long().cumsum(-1) - 1)
                position_ids.masked_fill_(attn_mask == 0, 0)
                position_ids = position_ids[:, -1].unsqueeze(-1)    # [B, 1]
        # transpose
        ng, batch_size = [], len(generated[0])
        for i in range(batch_size):
            ng.append([g[i] for g in generated])
        return ng

class GPT2Agent(BaseAgent):

    def __init__(self, total_steps, multi_gpu, vocab_file='data/vocab/vocab_small', run_mode='train', lang='zh', lm=False, local_rank=0):
        super(GPT2Agent, self).__init__()
        try:
            self.gpu_ids = list(range(len(multi_gpu.split(','))))
        except:
            raise Exception(f'[!] multi gpu ids are needed, but got: {multi_gpu}')
        assert run_mode in ['train', 'test'], f'[!] running mode must be train or test, but got {run_mode}'
        vocab_file = 'data/vocab/vocab_small' if lang == 'zh' else 'data/vocab/vocab_english'
        self.args = {
            'lr': 1.5e-4,
            'grad_clip': 1.0,
            'tgt_len_size': 30,
            'lr_gamma': 0.5,
            'warmup_steps': 16000,
            'total_steps': total_steps,
            'topk': 2000,
            'topp': 0.97, 
            'config_path': 'data/config/model_config_dialogue_small.json',
            'multi_gpu': self.gpu_ids,
            'run_mode': run_mode,
            'vocab_file': vocab_file,
            'lang': lang,
            'repetition_penalty': 1,
            'amp_level': 'O2',
            'local_rank': local_rank,
        }
        self.vocab = BertTokenizer(vocab_file=self.args['vocab_file'])
        self.vocab_size = len(self.vocab)
        self.unk = self.vocab.convert_tokens_to_ids('[UNK]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')

        self.model = GPT2(
            self.vocab_size, 
            self.unk, 
            self.sep,
            self.cls,
            self.args['topk'], 
            self.args['topp'], 
            self.args['repetition_penalty'],
            config_path=self.args['config_path'],
        )
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad, reduction='sum')
        if torch.cuda.is_available():
            self.model.cuda()
        self.optimizer = transformers.AdamW(
            self.model.parameters(), 
            lr=self.args['lr'],
            correct_bias=True,
        )
        if run_mode == 'train':
            self.model, self.optimizer = amp.initialize(
                self.model, 
                self.optimizer, 
                opt_level=self.args['amp_level'],
            )

        # need to obtain the whole iter
        self.warmup_scheduler = transformers.get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.args['warmup_steps'],
            num_training_steps=self.args['total_steps']
        )
        if self.args['run_mode'] == 'train':
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[local_rank], output_device=local_rank)
        
        if run_mode == 'test':
            from .biencoder import BERTBiEncoderAgent
            self.reranker = BERTBiEncoderAgent(
                multi_gpu, 0, run_mode='test', model='bertirbi', lang=lang,
            )
            self.reranker.load_model('ckpt/zh50w/bertirbi/best.pt')
            print(f'[!] load reranker model over')

        self.show_parameters(self.args)

    def train_model(self, train_iter, mode='train', recoder=None, idx_=0):
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
            not_ignore = shift_labels.ne(self.pad)
            num_targets = not_ignore.long().sum().item()
            correct = (shift_labels == preds) & not_ignore
            correct = correct.float().sum()
            # loss and token accuracy
            accuracy = correct / num_targets
            total_acc.append(accuracy.item())
            loss = loss / num_targets
            
            # loss.backward()
            # clip_grad_norm_(self.model.parameters(), self.args['grad_clip'])
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
            clip_grad_norm_(amp.master_params(self.optimizer), self.args['grad_clip'])
        
            self.optimizer.step()
            self.warmup_scheduler.step()
            total_loss += loss.item()
            batch_num += 1
            
            recoder.add_scalar(f'train-epoch-{idx_}-{self.args["local_rank"]}/Loss', total_loss/batch_num, idx)
            recoder.add_scalar(f'train-epoch-{idx_}-{self.args["local_rank"]}/RunLoss', loss.item(), idx)
            recoder.add_scalar(f'train-epoch-{idx_}-{self.args["local_rank"]}/RunTokenAcc', accuracy, idx)
            recoder.add_scalar(f'train-epoch-{idx_}-{self.args["local_rank"]}/TokenAcc', np.mean(total_acc), idx)
            pbar.set_description(f'[!] local_rank: {self.args["local_rank"]}; train loss: {round(loss.item(), 4)}, token acc: {round(accuracy.item(), 4)}')
        recoder.add_scalar(f'train-whole-{self.args["local_rank"]}/TokenAcc', np.mean(total_acc), idx_)
        recoder.add_scalar(f'train-whole-{self.args["local_rank"]}/Loss', total_loss/batch_num, idx_)
        return round(total_loss/batch_num, 4)
        
    @torch.no_grad()
    def test_model(self, test_iter, path):
        '''Generate the test dataset and measure the performance'''
        def filter_tgt(x):
            if '[SEP]' in x:
                x = x[:x.index('[SEP]')] + '[SEP]'
            return x
        def filter(x):
            return x.replace('[PAD]', '')
        self.model.eval()
        pbar = tqdm(test_iter)
        with open(path, 'w') as f:
            for batch in pbar:
                c, attn_mask, position_ids, r = batch
                max_size = min(len(r), self.args['tgt_len_size'])
                tgt = self.model.predict_batch(
                    c, attn_mask, position_ids, max_size
                )
                
                for tgt_, c_, r_ in zip(tgt, c, r):
                    text = self.vocab.convert_ids_to_tokens(tgt_)
                    text = filter_tgt(''.join(text))

                    ctx = self.vocab.convert_ids_to_tokens(c_)
                    ctx = filter(''.join(ctx))

                    ref = self.vocab.convert_ids_to_tokens(r_)
                    ref = filter(''.join(ref))

                    f.write(f'CTX: {ctx}\n')
                    f.write(f'REF: {ref}\n')
                    f.write(f'TGT: {text}\n\n')
        print(f'[!] translate test dataset over, write into {path}')
        # measure the performance
        (b1, b2, b3, b4), ((r_max_l, r_min_l, r_avg_l), (c_max_l, c_min_l, c_avg_l)), (dist1, dist2, rdist1, rdist2), (average, extrema, greedy) = cal_generative_metric(path, lang=self.args['lang'])
        print(f'[TEST] BLEU: {b1}/{b2}/{b3}/{b4}; Length(max, min, avg): {c_max_l}/{c_min_l}/{c_avg_l}|{r_max_l}/{r_min_l}/{r_avg_l}; Dist: {dist1}/{dist2}|{rdist1}/{rdist2}; Embedding(average/extrema/greedy): {average}/{extrema}/{greedy}')
        
    @torch.no_grad()
    def talk(self, topic, msg, maxlen=50, batch_size=32):
        self.model.eval()
        cid = to_cuda(torch.LongTensor(self.vocab.encode(msg)[-(512-maxlen):]).unsqueeze(0).expand(batch_size, -1))    # [B, S]
        
        # donot need the attention mask
        tgt = self.model.predict_batch(cid, None, None, maxlen)    # [B, S]
        
        # rerank return the best idx
        idx = self.reranker.predict_scores(msg, tgt)
        tgt_text = self.vocab.convert_ids_to_tokens(tgt)[idx]
        tgt_rest = ''.join(tgt_text)
        return tgt
