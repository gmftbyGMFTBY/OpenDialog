from .header import *

class LCCC(nn.Module):
    
    def __init__(self, pretrained_path, topk, topp):
        super(LCCC, self).__init__()
        self.model = OpenAIGPTLMHeadModel.from_pretrained(pretrained_path)
        self.vocab = BertTokenizer.from_pretrained(pretrained_path, do_lower_case=True)
        self.unk_id, self.sep_id = self.vocab.convert_tokens_to_ids('[UNK]'), self.vocab.convert_tokens_to_ids('[SEP]')
        self.pad_id = self.vocab.convert_tokens_to_ids('[PAD]')
        self.topk, self.topp = topk, topp
        self.SPECIAL_TOKENS = ["[CLS]", "[SEP]", "[speaker1]", "[speaker2]"]
        
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
    
    def __init__(self, multi_gpu):
        super(LCCCAgent, self).__init__()
        try:
            self.gpu_ids = list(range(len(multi_gpu.split(','))))
        except:
            raise Exception(f'[!] multi gpu ids are needed, but got: {multi_gpu}')
        self.args = {
            'tgt_len_size': 50,
            'topk': 0,
            'topp': 0.9, 
            'temperature': 0.7,
            'pretrained_path': '/home/lt/data/LCCD_GPT',
            'multi_gpu': self.gpu_ids,
        }
        self.model = LCCC(
            self.args['pretrained_path'],
            self.args['topk'], 
            self.args['topp'],
        )
        
        if torch.cuda.is_available():
            self.model.cuda()
        
        self.show_parameters(self.args)
        
    def tokenize_(self, obj):
        '''borrow from thu-coai/CDial-GPT'''
        return self.model.vocab.convert_tokens_to_ids(self.model.vocab.tokenize(obj))
    
    @torch.no_grad()
    def talk(self, topic, msgs, maxlen=50, batch_size=32):
        self.model.eval()
        msgs = [self.tokenize_(msgs)]
        tgt = self.model.predict(
            msgs, 
            maxlen,
            temperature=self.args['temperature'],
            min_length=1,
        )
        tgt = self.model.vocab.convert_ids_to_tokens(tgt)
        tgt = ''.join(tgt)
        return tgt
    
# ========== LCCC IR ========== #
class LCCCIR(nn.Module):
    
    def __init__(self, pretrained_path, topk, topp):
        super(LCCCIR, self).__init__()
        self.model = OpenAIGPTLMHeadModel.from_pretrained(pretrained_path)
        # binary classification head upon the OpenAIGPTModel (pre-trained)
        self.bs_head = nn.Linear(self.model.config.n_embd, 2)
        self.vocab = BertTokenizer.from_pretrained(pretrained_path, do_lower_case=True)
        self.unk_id, self.sep_id = self.vocab.convert_tokens_to_ids('[UNK]'), self.vocab.convert_tokens_to_ids('[SEP]')
        self.pad_id = self.vocab.convert_tokens_to_ids('[PAD]')
        self.topk, self.topp = topk, topp
        self.SPECIAL_TOKENS = ["[CLS]", "[SEP]", "[speaker1]", "[speaker2]"]
        
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
        
    def forward(self, inpt_ids, token_type_ids):
        '''fine tuning process
        inpt_ids: [B, S]
        token_type_ids: [B, S]
        
        Maybe multi-task:
        (1) Binary classification
        (2) LM
        '''
        transformer_outputs = self.model.transformer(
            inpt_ids,
            attention_mask=None,
            token_type_ids=token_type_ids,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
        )
        hidden_states = transformer_outputs[0]    # [B, S, E]
        bs_state = torch.mean(hidden_states, dim=1)    # [B, E]
        bs_rest = self.bs_head(bs_state)    # [B, 2]
        return bs_rest
    
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

class LCCCIRAgent(RetrievalBaseAgent):
    
    '''
    Fine tunine the LCCC-GPT2-Large for retrieval dialog systems,
    which is hopeful than fine-tuning BERT model.
    '''
    
    def __init__(self, multi_gpu, run_mode='train'):
        super(LCCCIRAgent, self).__init__()
        try:
            self.gpu_ids = list(range(len(multi_gpu.split(','))))
        except:
            raise Exception(f'[!] multi gpu ids are needed, but got: {multi_gpu}')
        self.args = {
            'lr': 5e-5,
            'grad_clip': 1.0,
            'topk': 0,
            'topp': 0.9, 
            'temperature': 0.7,
            'pretrained_path': '/home/lt/data/LCCD_GPT',
            'multi_gpu': self.gpu_ids,
            'run_mode': run_mode,
            'samples': 10,
            'amp_level': 'O2',
        }
        self.model = LCCCIR(
            self.args['pretrained_path'],
            self.args['topk'], 
            self.args['topp'],
        )
        if torch.cuda.is_available():
            self.model.cuda()
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = transformers.AdamW(
            self.model.parameters(), 
            lr=self.args['lr'],
        )
        self.model, self.optimizer = amp.initialize(
            self.model,
            self.optimizer, 
            opt_level=self.args['amp_level']
        )
        if self.args['run_mode'] == 'train':
            self.model = DataParallel(
                self.model, 
                device_ids=self.gpu_ids,
            )
        self.show_parameters(self.args)
        
    def train_model(self, train_iter, mode='train', recoder=None, idx_=0):
        '''similar to bertretrieval model (binary classification)'''
        self.model.train()
        total_loss, total_acc, batch_num = 0, [], 0
        pbar = tqdm(train_iter)
        for idx, batch in enumerate(pbar):
            # [B, S]; [B, S]; [B]
            cid, token_type_ids, label = batch
            self.optimizer.zero_grad()
            output = self.model(cid, token_type_ids)    # [B, 2]
            loss = self.criterion(
                output, 
                label.view(-1),
            )
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
            clip_grad_norm_(amp.master_params(self.optimizer), self.args['grad_clip'])
            # loss.backward()
            # clip_grad_norm_(self.model.parameters(), self.args['grad_clip'])
            self.optimizer.step()
            total_loss += loss.item()
            batch_num += 1
            now_correct = torch.max(F.softmax(output, dim=-1), dim=-1)[1]    # [B]
            now_correct = torch.sum(now_correct == label).item()
            correct += now_correct
            s += len(label)
                
            recoder.add_scalar(f'train-epoch-{idx_}/Loss', total_loss/batch_num, idx)
            recoder.add_scalar(f'train-epoch-{idx_}/RunLoss', loss.item(), idx)
            recoder.add_scalar(f'train-epoch-{idx_}/RunAcc', now_correct/len(label), idx)
            recoder.add_scalar(f'train-epoch-{idx_}/Acc', correct/s, idx)
            pbar.set_description(f'[!] loss(run|all): {round(loss.item(), 4)}|{round(total_loss/batch_num, 4)}; acc(run|all): {round(now_correct/len(label), 4)}|{round(correct/s, 4)}')
        return round(total_loss/batch_num, 4)
    
    @torch.no_grad()
    def test_model(self, test_iter, path):
        self.model.eval()
        total_loss, batch_num = 0, 0
        pbar = tqdm(test_iter)
        rest = []
        for idx, batch in enumerate(pbar):
            cid, token_type_ids, label = batch
            output = self.model(cid, token_type_ids)    # [B, 2]
            loss = self.criterion(output, label.view(-1))
            total_loss += loss.item()
            batch_num += 1
            output = F.softmax(output, dim=-1)[:, 1]    # [batch]
            preds = [i.tolist() for i in torch.split(output, self.args['samples'])]
            for pred in preds:
                pred = np.argsort(pred, axis=0)[::-1]
                rest.append(([0], pred.tolist()))
        print(f'[!] test loss: {round(total_loss/batch_num, 4)}')
        p_1, r2_1, r10_1, r10_2, r10_5, MAP, MRR = cal_ir_metric(rest)
        print(f'[TEST] P@1: {p_1}; R2@1: {r2_1}; R10@1: {r10_1}; R10@2: {r10_2}; R10@5: {r10_5}; MAP: {MAP}; MRR: {MRR}')
        return round(total_loss/batch_num, 4)

    @torch.no_grad()
    def talk(self, topic, msgs):
        self.model.eval()
        ipdb.set_trace()
        utterances_, ids = self.process_utterances(topic, msgs)
        instance = self.model.build_input_from_segments()
        output = self.model(ids)    # [batch, 2]
        output = F.softmax(output, dim=-1)[:, 1]    # [batch]
        item = torch.argmax(output).item()
        msg = utterances_[item]
        return msg