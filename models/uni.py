from .header import *

'''
Pre-train UNI Retrieval and Generative Open-Domain Dialog Model
Partial codes borrowed from thu-coai/CDial-GPT
'''

class UNI(nn.Module):
    
    def __init__(self, config_path, vocab_path, topk, topp):
        super(UNI, self).__init__()
        self.model_config = GPT2Config.from_json_file(config_path)
        self.model = GPT2Model(config=self.model_config)
        self.vocab = BertTokenizer(vocab_file=vocab_path, do_lower_case=True)
        self.lm_head = nn.Linear(self.model_config.n_embd, self.model_config.vocab_size)
        self.topk, self.topp = topk, topp
        self.SPECIAL_TOKENS = ["[CLS]", "[SEP]", "[speaker1]", "[speaker2]"]
        
    def forward(self, inpt_ids, token_type_ids):
        attn_mask = generate_attention_mask(inpt_ids)
        outputs = self.model(
            input_ids=inpt_ids, 
            attention_mask=attn_mask,
            token_type_ids=token_type_ids,
        )
        outputs = outputs[0]    # [B, S, E]
        
        lm_outputs = self.lm_head(outputs)    # [B, S, V]
        return lm_outputs
    
    def build_input_from_segments(self, history, response, with_eos=True):
        '''borrow from the thu-coai/CDial-GPT; for test mode'''
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
    def gen(self, inpt_ids, token_type_ids, max_len=50, temperature=0.7):
        '''batch size is 1'''
        current_output = []
        special_tokens_ids = self.vocab.convert_tokens_to_ids(self.SPECIAL_TOKENS)
        for i in range(max_len):
            instance = self.build_input_from_segments(inpt_ids, generated, with_eos=False)
            input_ids = torch.LongTensor(instance["input_ids"]).unsqueeze(0).to(self.args['device'])
            token_type_ids = to_cuda(torch.LongTensor(instance["token_type_ids"])).unsqueeze(0)
            attn_mask = generate_attention_mask(inpt_ids)
            outputs = self.model(
                input_ids=inpt_ids, 
                attention_mask=attn_mask,
                token_type_ids=token_type_ids,
            )
            outputs = outputs[0]
            logits = self.lm_head(outputs)[0, -1, :] / temperature  # [V]
            logits = top_k_top_p_filtering(logits, top_k=self.topk, top_p=self.topp)
            probs = F.softmax(logits, dim=-1)
            prev = torch.multinomial(probs, 1)
            if i < min_length and prev.item() in special_tokens_ids:
                while prev.item() in special_tokens_ids:
                    prev = torch.multinomial(probs, num_samples=1)

            if prev.item() in special_tokens_ids:
                break
            current_output.append(prev.item())
        return current_output
    
class UNIAgent(BaseAgent):
    
    def __init__(self, total_steps, multi_gpu, run_mode='train', lang='zh', local_rank=-1):
        super(UNIAgent, self).__init__()
        # hyperparameters
        try:
            self.gpu_ids = list(range(len(multi_gpu.split(','))))
        except:
            raise Exception(f'[!] multi gpu ids are needed, but got: {multi_gpu}')
        self.args = {
            'lr': 1.5e-4,
            'grad_clip': 1.0,
            'tgt_len_size': 30,
            'warmup_steps': 2000,
            'total_steps': total_steps,
            'topk': 200,
            'topp': 0.95, 
            'config_path': 'data/config/lccc.json',
            'vocab_path': 'data/vocab/lccc_vocab',
            'multi_gpu': self.gpu_ids,
            'run_mode': run_mode,
            'lang': lang,
            'amp_level': 'O1',
            'device': torch.device("cuda", local_rank),
        }
        # hyperparameter
        self.model = UNI(
            self.args['config_path'],
            self.args['vocab_path'],
            self.args['topk'], 
            self.args['topp'],
        )
        self.model.to(self.args['device'])
        
        self.lm_criterion = nn.CrossEntropyLoss(ignore_index=-1)
        self.optimizer = transformers.AdamW(
            self.model.parameters(), 
            lr=self.args['lr'], 
            correct_bias=True,
        )
        self.model, self.optimizer = amp.initialize(
            self.model, 
            self.optimizer, 
            opt_level=self.args['amp_level']
        )
        # dataparallel
        self.model = DataParallel(self.model, device_ids=[local_rank], output_device=local_rank)
        self.warmup_scheduler = transformers.get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.args['warmup_steps'],
            num_training_steps=self.args['total_steps']
        )
        self.show_parameters(self.args)

    def train_model(self, train_iter, mode='train', recoder=None, idx_=0):
        self.model.train()
        total_loss, total_acc, batch_num = 0, 0, 0
        pbar = tqdm(train_iter)
        for idx, batch in enumerate(pbar):
            inpt_ids, token_type_ids, lm_labels = batch
            inpt_ids = inpt_ids.to(self.args['device'])
            token_type_ids = token_type_ids.to(self.args['device'])
            lm_labels = lm_labels.to(self.args['device'])
            self.optimizer.zero_grad()
            lm_logits = self.model(inpt_ids, token_type_ids)
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = lm_labels[..., 1:].contiguous()
            loss = self.lm_criterion(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            
            _, preds = shift_logits.max(dim=-1)    # [batch, seq]
            # ignore the pad
            not_ignore = shift_labels.ne(-1)   # pad is 0 or 1
            num_targets = not_ignore.long().sum().item()    # the number of not pad tokens
            correct = (shift_labels == preds) & not_ignore
            correct = correct.float().sum()
            accuracy = correct / num_targets
            
            total_loss += loss.item()
            total_acc += accuracy.item()
            batch_num += 1
            
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
            clip_grad_norm_(amp.master_params(self.optimizer), self.args['grad_clip'])
            
            self.optimizer.step()
            self.warmup_scheduler.step()
            recoder.add_scalar(f'train-epoch-{idx_}/Loss', total_loss/batch_num, idx)
            recoder.add_scalar(f'train-epoch-{idx_}/RunLoss', loss.item(), idx)
            recoder.add_scalar(f'train-epoch-{idx_}/RunTokenAcc', accuracy.item(), idx)
            recoder.add_scalar(f'train-epoch-{idx_}/TokenAcc', total_acc/batch_num, idx)
            
            pbar.set_description(f'[!] loss: {round(loss.item(), 4)}|{round(total_loss/batch_num, 4)}, token acc: {round(accuracy.item(), 4)}|{round(total_acc/batch_num, 4)}')
        recoder.add_scalar(f'train-whole/TokenAcc', total_acc/batch_num, idx_)
        recoder.add_scalar(f'train-whole/Loss', total_loss/batch_num, idx_)
        return round(total_loss/batch_num, 4)
    
    @torch.no_grad()
    def test_generative(self, test_iter, path):
        '''test the generative part of the UNI model'''
        pass
    
    @torch.no_grad()
    def test_retrieval(self, test_iter):
        '''test the retrieval part of the UNI model'''
        pass

    @torch.no_grad()
    def test_model(self, test_iter, path):
        '''test the model (retrieval and generative)'''
        self.model.eval()
        generative_iter, retrieval_iter = test_iter
        self.test_generative(generative_iter, path)
        self.test_retrieval(retrieval_iter)
        
    @torch.no_grad()
    def talk(self, topic, msgs, maxlen=50, batch_size=32):
        self.model.eval()
        msgs = torch.LongTensor(self.vocab.encode(msgs)[-(512-maxlen):])
        msgs = to_cuda(msgs)
        tgt = self.model.predict(msgs, maxlen)
        tgt = self.vocab.convert_ids_to_tokens(tgt)
        tgt = ''.join(tgt)
        return tgt
