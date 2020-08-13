from .header import *

'''
Have the serious, still not be available
'''

class LCCC(nn.Module):
    
    def __init__(self, pretrained_path, repetition_penalty, topk, topp):
        super(LCCC, self).__init__()
        self.model = GPT2LMHeadModel.from_pretrained(pretrained_path)
        self.vocab = BertTokenizer.from_pretrained(pretrained_path)
        self.unk_id, self.sep_id = self.vocab.convert_tokens_to_ids('[UNK]'), self.vocab.convert_tokens_to_ids('[SEP]')
        self.pad_id = self.vocab.convert_tokens_to_ids('[PAD]')
        self.repetition_penalty = repetition_penalty
        self.topk, self.topp = topk, topp
        
    def forward(self, inpt_ids):
        # inpt_ids: [B, S]
        attn_mask = generate_attention_mask(inpt_ids)
        outputs = self.model(input_ids=inpt_ids, attention_mask=attn_mask)
        return outputs[0]    # [B, S, V]
    
    @torch.no_grad()
    def predict(self, inpt_ids, max_len):
        '''
        inpt_ids: [B, S]
        '''
        generated = []
        prev, past = inpt_ids, None
        batch_size = inpt_ids.shape[0]
        stop_flag = np.zeros(batch_size)
        for _ in range(max_len):
            outputs = self.model(input_ids=prev, past=past)
            output, past = outputs[:2]
            next_token_logits = output[:, -1, :]    # [B, V]
            next_token_logits[:, self.unk_id] = -np.inf
            # repetion penalty
            for x in range(batch_size):
                y = [item[x] for item in generated]
                next_token_logits[x, y] /= self.repetition_penalty
            filtered_logits = top_k_top_p_filtering_batch(
                    next_token_logits, 
                    top_k=self.topk, 
                    top_p=self.topp)
            next_token = torch.multinomial(
                    F.softmax(filtered_logits, dim=-1),
                    num_samples=1)    # [batch, 1]
            # set up stop_flag
            for idx, i in enumerate(next_token.squeeze(1)):
                if i == self.sep_id:
                    stop_flag[idx] = 1
            generated.append([token.item() for token in next_token.squeeze(1)])
            prev = next_token
            if sum(stop_flag) == batch_size:
                break
        # transpose
        ng, batch_size = [], len(generated[0])
        for i in range(batch_size):
            ng.append([g[i] for g in generated])
        return ng
    
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
                'topk': 10000,
                'topp': 1.0, 
                'pretrained_path': '/home/lt/data/GPT2_LCCC_base',
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
                correct_bias=True)

        # need to obtain the whole iter
        self.warmup_scheduler = transformers.get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.args['warmup_steps'],
                num_training_steps=self.args['total_steps'])
        if torch.cuda.is_available():
            self.model.cuda()
        
        if self.args['run_mode'] == 'train':
            self.model = DataParallel(
                    self.model, 
                    device_ids=self.gpu_ids)
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
    
    @torch.no_grad()
    def talk(self, topic, msgs, maxlen=50, batch_size=32):
        self.model.eval()
        msgs = torch.LongTensor(self.model.vocab.encode(msgs)[-(512-maxlen):])
        msgs = to_cuda(msgs).unsqueeze(0)    # [1, S]
        tgt = self.model.predict(msgs, maxlen)[0]
        tgt = self.model.vocab.convert_ids_to_tokens(tgt)
        tgt = ''.join(tgt)
        return tgt
