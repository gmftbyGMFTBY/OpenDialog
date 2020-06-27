from .header import *

'''
Modified from the GPT2 Model
Existing dialog models only generate one sentence for the user.
But actually, during our daily conversation, we always decide whether to speak based on the conversation context, one sentence cannot hold all the information. So generating multiple sentences will be better and make the chatbot more human-like.
'''

class When2Talk(nn.Module):

    def __init__(self, vocab_size, unk_id, stp_id, topk, topp, 
                 repetition_penalty,
                 config_path='data/config/model_config_dialogue_small.json'):
        super(When2Talk, self).__init__()
        self.model_config = GPT2Config.from_json_file(config_path)
        self.model = GPT2LMHeadModel(config=self.model_config)
        self.model.resize_token_embeddings(vocab_size)
        self.n_ctx = self.model.config.to_dict().get('n_ctx')
        self.topk, self.topp = topk, topp
        self.unk_id = unk_id
        self.stp_id = stp_id
        self.repetition_penalty = repetition_penalty

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
        batch_size is 1; inpt_ids: [seq]
        the user token is [USER2]
        '''
        with torch.no_grad():
            generated = []
            for _ in range(max_len):
                outputs = self.model(input_ids=inpt_ids)[0]
                # outputs = self.model(input_ids=inpt_ids)
                next_token_logits = outputs[-1, :]    # [vocab]
                # penalty on the deplicated tokens
                if generated:
                    next_token_logits[list(set(generated))] /= self.repetition_penalty
                # ignore the [UNK] token
                next_token_logits[self.unk_id] = -np.inf
                filtered_logits = top_k_top_p_filtering(
                        next_token_logits, 
                        top_k=self.topk, 
                        top_p=self.topp)
                next_token = torch.multinomial(
                        F.softmax(filtered_logits, dim=-1),
                        num_samples=1)
                generated.append(next_token.item())
                if next_token.item() == self.stp_id:
                    break
                inpt_ids = torch.cat((inpt_ids, next_token), dim=0)
                inpt_ids = inpt_ids[-self.n_ctx:]
            return generated

class When2TalkAgent(BaseAgent):

    def __init__(self, total_steps, multi_gpu, vocab_file='data/vocab/vocab_small', run_mode='train', lang='zh'):
        super(When2TalkAgent, self).__init__()
        # hyperparameters
        try:
            # self.gpu_ids = [int(i) for i in multi_gpu.split(',')]
            self.gpu_ids = list(range(len(multi_gpu.split(','))))
        except:
            raise Exception(f'[!] multi gpu ids are needed, but got: {multi_gpu}')
        assert run_mode in ['train', 'test'], f'[!] running mode must be train or test, but got {run_mode}'
        vocab_file = 'data/vocab/vocab_small' if lang == 'zh' else 'data/vocab/vocab_english'
        self.args = {
                'lr': 1.5e-4,
                'grad_clip': 1.0,
                'pad': 0,
                'tgt_len_size': 100,
                'lr_gamma': 0.5,
                'patience': 5,
                'min_lr': 1e-5,
                'warmup_steps': 2000,
                'total_steps': total_steps,
                'topk': 5,
                'topp': 0.9,
                'config_path': 'data/config/model_config_dialogue_small.json',
                'multi_gpu': self.gpu_ids,
                'run_mode': run_mode,
                'vocab_file': vocab_file,
                'repetition_penalty': 1.5,
                'lang': lang,
        }
        # hyperparameters

        self.vocab = BertTokenizer(vocab_file=self.args['vocab_file'])
        additional_tokens = {'additional_special_tokens': ['[USER1]', '[USER2]', '[STP]']}
        self.vocab.add_special_tokens(additional_tokens)
        assert self.vocab.convert_tokens_to_ids('[PAD]') == 0, '[PAD] Token must be 0'
        self.vocab_size = len(self.vocab)
        self.unk = self.vocab.convert_tokens_to_ids('[UNK]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')
        self.user1_token_id_ = self.vocab.convert_tokens_to_ids('[USER1]')
        self.user2_token_id_ = self.vocab.convert_tokens_to_ids('[USER2]')
        self.stp_id = self.vocab.convert_tokens_to_ids('[STP]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')

        self.model = When2Talk(
            self.vocab_size, 
            self.unk, 
            self.stp_id,
            self.args['topk'], 
            self.args['topp'], 
            self.args['repetition_penalty'],
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
        to_cuda(self.model.cuda(), model=True)
        # train: DataParallel; test: no DataParallel
        if self.args['run_mode'] == 'train':
            self.model = DataParallel(self.model, device_ids=self.gpu_ids)
        self.show_parameters(self.args)

    def train_model(self, train_iter, mode='train', recoder=None):
        self.model.train()
        total_loss, batch_num = 0, 0
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
        # (b1, b2, b3, b4), ((r_max_l, r_min_l, r_avg_l), (c_max_l, c_min_l, c_avg_l)), (dist1, dist2, rdist1, rdist2), (average, extrema, greedy) = cal_generative_metric(path, lang=self.args['lang'])
        # print(f'[TEST] BLEU: {b1}/{b2}/{b3}/{b4}; Length(max, min, avg): {c_max_l}/{c_min_l}/{c_avg_l}|{r_max_l}/{r_min_l}/{r_avg_l}; Dist: {dist1}/{dist2}|{rdist1}/{rdist2}; Embedding(average/extrema/greedy): {average}/{extrema}/{greedy}')
    
    def talk(self, topic, msgs, maxlen=50):
        '''
        topic, msgs: msgs is a string which split with the [SEP] token
        batch size is 1

        format:
        [USER1] ... [SEP] [USER1] ... [STP] [USER2]
        '''
        # tokenizer
        with torch.no_grad():
            if '[SEP]' not in msgs:
                msgs = f'{msgs} [STP]'
            else:
                msgs = msgs.replace('[SEP]', '[SEP] [USER1]')
                msgs = f'{msgs} [STP]'
            msgs = f'[USER1] {msgs} [USER2]'
            msgs = torch.LongTensor(self.vocab.encode(msgs)[1:-1])
            if torch.cuda.is_available():
                msgs = msgs.cuda()
            tgt = self.model.predict(msgs, maxlen)
            tgt = self.vocab.convert_ids_to_tokens(tgt)
        tgt = ''.join(tgt)
        tgt = tgt.replace('[USER2]', '').replace('[SEP]', '\n\n').replace('[STP]', '')
        return tgt
