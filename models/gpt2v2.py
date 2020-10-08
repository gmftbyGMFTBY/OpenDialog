from .header import *

class GPT2V2(nn.Module):

    def __init__(self, vocab_size, unk_id, sep_id, cls_id, topk, topp, 
                 repetition_penalty,
                 config_path='data/config/model_config_dialogue_small.json', 
                 embedding_size=300, policy_size=32):
        super(GPT2V2, self).__init__()
        self.model_config = GPT2Config.from_json_file(config_path)
        self.model = GPT2Model(config=self.model_config)
        
        self.model.resize_token_embeddings(vocab_size)
        self.n_ctx = self.model.config.to_dict().get('n_ctx')
        self.n_embd = self.model.config.to_dict().get('n_embd')
        self.topk, self.topp = topk, topp
        self.unk_id = unk_id
        self.sep_id = sep_id
        self.cls_id = cls_id
        self.repetition_penalty = repetition_penalty
        
        self.agent = ActorCritic(policy_size, embedding_size)
        self.proj = nn.Linear(self.n_embd + policy_size, vocab_size)

    def forward(self, inpt_ids, context_embd, response_embd):
        # inpt_ids: [batch, seq]; [batch, 300]; [batch, 300]
        outputs = self.model(
            input_ids=inpt_ids,
        )
        output = outputs[0]    # [batch, seq, 768]
        policy_embd = self.agent(torch.cat([context_embd, response_embd], dim=-1)).unsqueeze(1).expand(-1, output.shape[1], -1)    # [batch, seq, 32]
        output = torch.cat([output, policy_embd], dim=-1)    # [batch, seq, 768+32]
        output = self.proj(output)    # [batch, seq, vocab]
        return output
    
    @torch.no_grad()
    def predict_batch(self, inpt_ids, attn_mask, position_ids, context_embd, response_embd, max_len):
        '''past parameter and position_ids parameters should be careful
        https://github.com/huggingface/transformers/issues/3021#issuecomment-681792104'''
        batch_size = inpt_ids.shape[0]
        generated = [[self.cls_id] * batch_size]
        prev, past = inpt_ids, None
        stop_flag = np.zeros(batch_size)
        policy_embd = self.agent(
            torch.cat([context_embd, response_embd], dim=-1), 
        )   # [batch, 32]
        for _ in range(max_len):
            outputs = self.model(
                input_ids=prev,
                attention_mask=attn_mask,
                position_ids=position_ids,
                past=past,
            )
            output, past = outputs[:2]
            output = output[:, -1, :]    # [batch, 768]
            output = torch.cat([output, policy_embd], dim=-1)    # [batch, 768+32]
            next_token_logits = self.proj(output)    # [batch, vocab]
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

    @torch.no_grad()
    def predict(self, inpt_ids, context_embd, response_embd, max_len):
        '''inpt_ids: [seq]; return a list of ids (generated)'''
        generated = [self.cls_id]
        policy_embd = self.agent(torch.cat([context_embd, response_embd], dim=-1))    # [32]
        for _ in range(max_len):
            outputs = self.model(
                input_ids=inpt_ids
            )
            output = outputs[0][-1, :]    # [768]
            output = torch.cat([output, policy_embd], dim=-1)    # [768+32]
            next_token_logits = self.proj(output)    # [vocab]
            next_token_logits[self.unk_id] = -np.inf
            if generated:
                next_token_logits[list(set(generated))] /= self.repetition_penalty
            filtered_logits = top_k_top_p_filtering(
                next_token_logits, 
                top_k=self.topk, 
                top_p=self.topp,
            )
            next_token = torch.multinomial(
                F.softmax(filtered_logits, dim=-1),
                num_samples=1,
            )
            generated.append(next_token.item())
            if next_token.item() == self.sep_id:
                break
            inpt_ids = torch.cat((inpt_ids, next_token), dim=0)
            inpt_ids = inpt_ids[-self.n_ctx:]
        return generated

class GPT2V2Agent(BaseAgent):

    def __init__(self, total_steps, multi_gpu, run_mode='train', lang='zh', local_rank=0):
        super(GPT2V2Agent, self).__init__()
        try:
            self.gpu_ids = list(range(len(multi_gpu.split(','))))
        except:
            raise Exception(f'[!] multi gpu ids are needed, but got: {multi_gpu}')
        vocab_file = 'data/vocab/vocab_small' if lang == 'zh' else 'data/vocab/vocab_english'
        self.args = {
            'lr': 1.5e-4,
            'grad_clip': 1.0,
            'tgt_len_size': 30,
            'warmup_steps': 16000,
            'total_steps': total_steps,
            'topk': 2000,
            'topp': 0.98, 
            'config_path': 'data/config/model_config_dialogue_small.json',
            'multi_gpu': self.gpu_ids,
            'run_mode': run_mode,
            'vocab_file': 'data/vocab/vocab_small',
            'lang': lang,
            'repetition_penalty': 1.,
            'amp_level': 'O2',
            'local_rank': local_rank,
            'policy_size': 32,
            'embedding_size': 300,
            'word2vec': 'data/chinese_w2v' if lang == 'zh' else 'data/english_w2v.bin',
        }
        self.vocab = BertTokenizer(vocab_file=self.args['vocab_file'])
        self.vocab_size = len(self.vocab)
        self.unk = self.vocab.convert_tokens_to_ids('[UNK]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.args['pad'] = self.pad

        self.model = GPT2V2(
            self.vocab_size, 
            self.unk, 
            self.sep,
            self.cls,
            self.args['topk'], 
            self.args['topp'], 
            self.args['repetition_penalty'],
            config_path=self.args['config_path'],
            embedding_size=self.args['embedding_size'],
        )
        
        # load the word2vec
        if lang == 'zh':
            self.w2v = load_w2v('data/chinese_w2v')
        else:
            self.w2v = gensim.models.KeyedVectors.load_word2vec_format('data/english_w2v.bin', binary=True)
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.args['pad'], reduction='sum')
        if torch.cuda.is_available():
            self.model.cuda()
        self.optimizer = transformers.AdamW(
            self.model.parameters(), 
            lr=self.args['lr'], 
            correct_bias=True,
        )
        if self.args['run_mode'] == 'train':
            self.model, self.optimizer = amp.initialize(
                self.model, 
                self.optimizer, 
                opt_level=self.args['amp_level'],
            )
        self.warmup_scheduler = transformers.get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.args['warmup_steps'],
            num_training_steps=self.args['total_steps']
        )
        if self.args['run_mode'] == 'train':
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

        self.show_parameters(self.args)

    def train_model(self, train_iter, mode='train', recoder=None, idx_=0):
        self.model.train()
        total_loss, total_acc, batch_num = 0, [], 0
        pbar = tqdm(train_iter)
        for idx, batch in enumerate(pbar):
            cid, labels, context_text, response_text = batch
            # obtain the context and response embeddings
            context_embd = convert_text_embedding(self.w2v, context_text)   # [batch, 300]
            context_embd = torch.tensor(context_embd).cuda()
            r_embd = []
            for j in range(len(response_text[0])):
                response_embd = convert_text_embedding(
                    self.w2v,
                    [i[j] for i in response_text],
                )    # [batch, 300]
                r_embd.append(torch.tensor(response_embd))
            r_embd = torch.stack(r_embd).mean(dim=0).cuda()    # [batch, 300]
            self.optimizer.zero_grad()

            logits = self.model(cid, context_embd, r_embd)    # [batch, seq, vocab]
            shift_logits = logits[..., :-1, :].contiguous()
            loss = self.criterion(
                shift_logits.view(-1, shift_logits.size(-1)),
                labels.view(-1),
            )
            _, preds = shift_logits.max(dim=-1)    # [batch, seq]
            not_ignore = labels.ne(self.args['pad'])
            num_targets = not_ignore.long().sum().item()
            correct = (labels == preds) & not_ignore
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
    def test_model_one_instance(self, test_iter, path):
        '''Generate the test dataset and measure the performance'''
        self.model.eval()
        pbar = tqdm(test_iter)
        with open(path, 'w') as f:
            for batch in pbar:
                cid, rid, context_text, response_text = batch
                context_embd = convert_text_embedding(self.w2v, [context_text])[0]
                context_embd = torch.tensor(context_embd, dtype=torch.float).cuda()
                response_embd = convert_text_embedding(
                    self.w2v,
                    response_text,
                )    # [candidate, 300]
                r_embd = [torch.tensor(i, dtype=torch.float) for i in response_embd]
                r_embd = torch.stack(r_embd).mean(dim=0).cuda()    # [300]
                
                max_size = max(len(rid), self.args['tgt_len_size'])
                tgt = self.model.predict(
                    cid, context_embd, r_embd, max_size,
                )
                text = ''.join(self.vocab.convert_ids_to_tokens(tgt))
                ctx = ''.join(self.vocab.convert_ids_to_tokens(cid))
                ref = ''.join(self.vocab.convert_ids_to_tokens(rid))
                f.write(f'CTX: {ctx}\n')
                f.write(f'REF: {ref}\n')
                f.write(f'TGT: {text}\n\n')
        print(f'[!] translate test dataset over, write into {path}')
        # measure the performance
        (b1, b2, b3, b4), ((r_max_l, r_min_l, r_avg_l), (c_max_l, c_min_l, c_avg_l)), (dist1, dist2, rdist1, rdist2), (average, extrema, greedy) = cal_generative_metric(path, lang=self.args['lang'])
        print(f'[TEST] BLEU: {b1}/{b2}/{b3}/{b4}; Length(max, min, avg): {c_max_l}/{c_min_l}/{c_avg_l}|{r_max_l}/{r_min_l}/{r_avg_l}; Dist: {dist1}/{dist2}|{rdist1}/{rdist2}; Embedding(average/extrema/greedy): {average}/{extrema}/{greedy}')

    @torch.no_grad()
    def test_model(self, test_iter, path):
        '''Generate the test dataset and measure the performance'''
        def filter(x):
            return x.replace('[PAD]', '')
        def filter_generation(x):
            if '[SEP]' in x:
                x = x[:x.index('[SEP]')] + '[SEP]'
            return x
        self.model.eval()
        pbar = tqdm(test_iter)
        with open(path, 'w') as f:
            for batch in pbar:
                cid, rid, attn_mask, position_ids, context_text, response_text = batch
                context_embd = convert_text_embedding(self.w2v, context_text)
                context_embd = torch.tensor(context_embd, dtype=torch.float).cuda()
                r_embd = []
                for idx in range(len(response_text[0])):
                    response_embd = convert_text_embedding(
                        self.w2v,
                        [i[idx] for i in response_text],
                    )    # [batch, 300]
                    r_embd.append(torch.tensor(response_embd, dtype=torch.float))
                r_embd = torch.stack(r_embd).mean(dim=0).cuda()    # [300]
                
                max_size = min(len(rid), self.args['tgt_len_size'])
                tgt = self.model.predict_batch(
                    cid, attn_mask, position_ids, context_embd, r_embd, max_size,
                )
                for tgt_, ctx, ref in zip(tgt, cid, rid):
                    text = self.vocab.convert_ids_to_tokens(tgt_)
                    text = filter_generation(''.join(text))

                    ctx = self.vocab.convert_ids_to_tokens(ctx)
                    ctx = filter(''.join(ctx))

                    ref = self.vocab.convert_ids_to_tokens(ref)
                    ref = filter(''.join(ref))

                    f.write(f'CTX: {ctx}\n')
                    f.write(f'REF: {ref}\n')
                    f.write(f'TGT: {text}\n\n')
        print(f'[!] translate test dataset over, write into {path}')
        # measure the performance
        (b1, b2, b3, b4), ((r_max_l, r_min_l, r_avg_l), (c_max_l, c_min_l, c_avg_l)), (dist1, dist2, rdist1, rdist2), (average, extrema, greedy) = cal_generative_metric(path, lang=self.args['lang'])
        print(f'[TEST] BLEU: {b1}/{b2}/{b3}/{b4}; Length(max, min, avg): {c_max_l}/{c_min_l}/{c_avg_l}|{r_max_l}/{r_min_l}/{r_avg_l}; Dist: {dist1}/{dist2}|{rdist1}/{rdist2}; Embedding(average/extrema/greedy): {average}/{extrema}/{greedy}')
        
    @torch.no_grad()
    def talk(self, topic, msgs, maxlen=50, batch_size=32):
        self.model.eval()
        if self.args['run_mode'] == 'test':
            msgs = torch.LongTensor(self.vocab.encode(msgs)[-(512-maxlen):])
            msgs = to_cuda(msgs)
            tgt = self.model.predict(msgs, maxlen)
            tgt = self.vocab.convert_ids_to_tokens(tgt)
            tgt = ''.join(tgt)
            return tgt
        else:
            raise Exception(f'[!] error in gpt2 model `talk` function')
