from .header import *

class LCCC(nn.Module):
    
    def __init__(self, pretrained_path, repetition_penalty, topk, topp):
        super(LCCC, self).__init__()
        self.model = OpenAIGPTLMHeadModel.from_pretrained(pretrained_path)
        self.vocab = BertTokenizer.from_pretrained(pretrained_path, do_lower_case=True)
        self.unk_id, self.sep_id = self.vocab.convert_tokens_to_ids('[UNK]'), self.vocab.convert_tokens_to_ids('[SEP]')
        self.pad_id = self.vocab.convert_tokens_to_ids('[PAD]')
        self.repetition_penalty = repetition_penalty
        self.topk, self.topp = topk, topp
        self.SPECIAL_TOKENS = ["[CLS]", "[SEP]", "[speaker1]", "[speaker2]"]
        
    def forward(self, inpt_ids):
        # inpt_ids: [B, S]
        outputs = self.model(input_ids=inpt_ids)
        return outputs[0]    # [B, S, V]
    
    def build_input_from_segments(self, history, response, with_eos=True):
        '''borrow from the thu-coai/CDial-GPT'''
        bos, eos, speaker1, speaker2 = self.vocab.convert_tokens_to_ids(self.SPECIAL_TOKENS)
        sequence = [[bos]] + history + [response + ([eos] if with_eos else [])]
        sequence = [sequence[0]] + [[speaker2 if i % 2 else speaker1] + s
                                    for i, s in enumerate(sequence[1:])]
        instance = {}
        instance["input_ids"] = list(chain(*sequence))
        instance["token_type_ids"] = [bos] + [speaker2 if i % 2 else speaker1 for i, s in
                                              enumerate(sequence[1:])
                                              for _ in s]
        return instance
    
    @torch.no_grad()
    def predict(self, inpt_ids, max_len, temperature=0.7, min_length=1):
        '''batch_size is 1
        inpt_ids: without [CLS] and [SEP] token
        '''
        current_output = []
        special_tokens_ids = self.vocab.convert_tokens_to_ids(self.SPECIAL_TOKENS)
        for i in range(max_len):
            instance = self.build_input_from_segments(inpt_ids, current_output, with_eos=False)
            input_ids = to_cuda(torch.LongTensor(instance["input_ids"])).unsqueeze(0)
            token_type_ids = to_cuda(torch.LongTensor(instance["token_type_ids"])).unsqueeze(0)

            logits, *_ = self.model(input_ids, token_type_ids=token_type_ids)
            logits = logits[0, -1, :] / temperature
            logits = top_k_top_p_filtering(logits, top_k=self.topk, top_p=self.topp)
            probs = F.softmax(logits, dim=-1)
            prev = torch.multinomial(probs, 1)
            # ipdb.set_trace()
            if i < min_length and prev.item() in special_tokens_ids:
                while prev.item() in special_tokens_ids:
                    prev = torch.multinomial(probs, num_samples=1)

            if prev.item() in special_tokens_ids:
                break
            current_output.append(prev.item())
        return current_output
    
class LCCCAgent(BaseAgent):
    
    def __init__(self, total_steps, multi_gpu, run_mode):
        super(LCCCAgent, self).__init__()
        # hyperparameters
        try:
            # self.gpu_ids = [int(i) for i in multi_gpu.split(',')]
            self.gpu_ids = list(range(len(multi_gpu.split(','))))
        except:
            raise Exception(f'[!] multi gpu ids are needed, but got: {multi_gpu}')
        self.args = {
            'lr': 5e-5,
            'grad_clip': 1.0,
            'tgt_len_size': 50,
            'warmup_steps': 2000,
            'total_steps': total_steps,
            'topk': 0,
            'topp': 0.9, 
            'temperature': 0.7,
            'pretrained_path': '/home/lt/data/LCCD_GPT',
            'multi_gpu': self.gpu_ids,
            'run_mode': run_mode,
            'repetition_penalty': 1,
        }
        self.model = LCCC(
            self.args['pretrained_path'], 
            self.args['repetition_penalty'],
            self.args['topk'], 
            self.args['topp'],
        )
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.model.pad_id, reduction='sum')
        self.optimizer = transformers.AdamW(
            self.model.parameters(), 
            lr=self.args['lr'], 
            correct_bias=True
        )

        # need to obtain the whole iter
        self.warmup_scheduler = transformers.get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.args['warmup_steps'],
            num_training_steps=self.args['total_steps']
        )
        if torch.cuda.is_available():
            self.model.cuda()
        
        if self.args['run_mode'] == 'train':
            self.model = DataParallel(
                self.model, 
                device_ids=self.gpu_ids
            )
        self.show_parameters(self.args)
        
    def train_model(self, train_iter, mode='train', recoder=None, idx_=0):
        self.model.train()
        total_loss, total_acc, batch_num = 0, [], 0
        pbar = tqdm(train_iter)
        oom_time = 0
        try:
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
                total_acc.append(accuracy.item())
                loss = loss / num_targets
                
                if mode == 'train':
                    loss.backward()
                    clip_grad_norm_(self.model.parameters(), self.args['grad_clip'])
                    self.optimizer.step()
                    self.warmup_scheduler.step()
                total_loss += loss.item()
                batch_num += 1
                
                recoder.add_scalar(f'train-epoch-{idx_}/Loss', total_loss/batch_num, idx)
                recoder.add_scalar(f'train-epoch-{idx_}/RunLoss', loss.item(), idx)
                recoder.add_scalar(f'train-epoch-{idx_}/RunTokenAcc', accuracy, idx)
                recoder.add_scalar(f'train-epoch-{idx_}/TokenAcc', np.mean(total_acc), idx)

                pbar.set_description(f'[!] OOM: {oom_time}, train loss: {round(loss.item(), 4)}, token acc: {round(accuracy.item(), 4)}')
        except RuntimeError as exception:
            if 'out of memory' in str(exception):
                oom_time += 1
                torch.cuda.empty_cache()
            else:
                raise exception
        return round(total_loss/batch_num, 4)
    
    def tokenize_(self, obj):
        return self.model.vocab.convert_tokens_to_ids(self.model.vocab.tokenize(obj))
    
    @torch.no_grad()
    def talk(self, topic, msgs, maxlen=50, batch_size=32):
        self.model.eval()
        msgs = [self.tokenize_(msgs)]
        # msgs = to_cuda(msgs).unsqueeze(0)    # [1, S]
        tgt = self.model.predict(
            msgs, 
            maxlen,
            temperature=self.args['temperature'],
        )
        # ipdb.set_trace()
        tgt = self.model.vocab.convert_ids_to_tokens(tgt)
        tgt = ''.join(tgt)
        return tgt
