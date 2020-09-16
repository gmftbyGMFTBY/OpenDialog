from .header import *
from .bert_retrieval import BERTRetrieval

class LCCC(nn.Module):
    
    def __init__(self, pretrained_path, topk, topp):
        super(LCCC, self).__init__()
        self.model = OpenAIGPTLMHeadModel.from_pretrained(pretrained_path)
        self.vocab = BertTokenizer.from_pretrained(pretrained_path, do_lower_case=True)
        self.topk, self.topp = topk, topp
        self.SPECIAL_TOKENS = ["[CLS]", "[SEP]", "[speaker1]", "[speaker2]"]
    
    def forward(self, inpt_ids, token_type_ids):
        output = self.model(inpt_ids, token_type_ids=token_type_ids)[0]    # [B, S, V]
        return output
    
    # ========== Following functions are used for test mode    
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
    
    def build_input_from_segments_batch(self, history, response, with_eos=True):
        instances = [self.build_input_from_segments([h], r, with_eos=with_eos) for h, r in zip(history, response)]
        return instances
    
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
            if i < min_length and prev.item() in special_tokens_ids:
                while prev.item() in special_tokens_ids:
                    prev = torch.multinomial(probs, num_samples=1)

            if prev.item() in special_tokens_ids:
                break
            current_output.append(prev.item())
        return current_output
    
    @torch.no_grad()
    def predict_batch(self, inpt_ids, max_len, temperature=0.7):
        '''batch size is not 1; but the length must be the same.
        .predict_batch can speed up the testing and provide the api for generation rerank
        inpt_ids: list of input_ids (length is Batch size)
        
        return: 
        current_output: [B, S] TYPE IS LIST
        '''
        current_output = [[] for _ in range(len(inpt_ids))]    # [B, S']
        stop_flag = [0] * len(inpt_ids)
        special_tokens_ids = self.vocab.convert_tokens_to_ids(self.SPECIAL_TOKENS)
        for i in range(max_len):
            instances = self.build_input_from_segments_batch(
                inpt_ids, current_output, with_eos=False
            )
            input_ids = to_cuda(
                torch.LongTensor([instance["input_ids"] for instance in instances])
            )    # [B, S]
            token_type_ids = to_cuda(
                torch.LongTensor([instance["token_type_ids"] for instance in instances])
            )    # [B, S]
            logits, *_ = self.model(input_ids, token_type_ids=token_type_ids)
            logits = logits[:, -1, :] / temperature    # [B, V]
            logits = top_k_top_p_filtering_batch(logits, top_k=self.topk, top_p=self.topp)
            probs = F.softmax(logits, dim=-1)    # [B, V]
            prev = torch.multinomial(probs, num_samples=1).squeeze(1).tolist()    # [B]
            
            for idx, item in enumerate(prev):
                if item in special_tokens_ids:
                    stop_flag[idx] = 1
                current_output[idx].append(item)
            if sum(stop_flag) == len(stop_flag):
                break
        return current_output
    
class LCCCFTAgent(BaseAgent):
    
    '''
    Supervised Fine-tuning LCCC on the given open-domain dialog corpus
    '''
    
    def __init__(self, multi_gpu, run_mode='test'):
        super(LCCCFTAgent, self).__init__()
        try:
            self.gpu_ids = list(range(len(multi_gpu.split(','))))
        except:
            raise Exception(f'[!] multi gpu ids are needed, but got: {multi_gpu}')
        self.args = {
            'tgt_len_size': 50,
            'grad_clip': 1.0,
            'topk': 0,
            'topp': 0.9,
            'unk': -1,   # For LCCD GPT
            'temperature': 0.7,
            'pretrained_path': '/home/lt/data/LCCD_GPT',
            'multi_gpu': self.gpu_ids,
            'run_mode': run_mode,    # test / rerank
            'samples': 16,
            'lr': 1e-5,
            'amp_level': 'O2',
            'lang': 'zh',
        }
        self.model = LCCC(
            self.args['pretrained_path'],
            self.args['topk'], 
            self.args['topp'],
        )
        if torch.cuda.is_available():
            self.model.cuda()
        # For LCCD GPT, -1 token is [UNK], which is used for padding
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)
        self.optimizer = transformers.AdamW(
            self.model.parameters(),
            lr=self.args['lr'], 
            correct_bias=True
        )
        self.model, self.optimizer = amp.initialize(
            self.model, 
            self.optimizer, 
            opt_level=self.args['amp_level'],
        )
        self.show_parameters(self.args)
        
    def train_model(self, train_iter, mode='train', recoder=None, idx_=0):
        self.model.train()
        total_loss, total_acc, batch_num = 0, [], 0
        pbar = tqdm(train_iter)
        for idx, batch in enumerate(pbar):
            input_ids, token_type_ids, labels = batch
            self.optimizer.zero_grad()
            
            output = self.model(input_ids, token_type_ids)
            output = output[..., :-1, :].contiguous()
            labels = labels[..., 1:].contiguous()
            loss = self.criterion(
                output.view(-1, output.size(-1)),
                labels.view(-1),
            )
            
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
            clip_grad_norm_(amp.master_params(self.optimizer), self.args['grad_clip'])
            # loss.backward()
            # clip_grad_norm_(self.model.parameters(), self.args['grad_clip'])
            self.optimizer.step()
            
            # statis
            _, preds = output.max(dim=-1)    # [B, S]
            not_ignore = labels.ne(self.args['unk'])   # pad is 0 or 1
            num_targets = not_ignore.long().sum().item()    # the number of not pad tokens
            correct = (labels == preds) & not_ignore
            correct = correct.float().sum()
            accuracy = correct / num_targets
            total_acc.append(accuracy.item())
            total_loss += loss.item()
            batch_num += 1
            
            # tfrecoder
            total_acc_ = np.mean(total_acc)
            total_loss_ = total_loss/batch_num
            loss = loss.item()
            accuracy = accuracy.item()
            recoder.add_scalar(f'train-epoch-{idx_}/Loss', total_loss_, idx)
            recoder.add_scalar(f'train-epoch-{idx_}/RunLoss', loss, idx)
            recoder.add_scalar(f'train-epoch-{idx_}/RunTokenAcc', accuracy, idx)
            recoder.add_scalar(f'train-epoch-{idx_}/TokenAcc', total_acc_, idx)
            
            pbar.set_description(f'[!] loss(run|overall): {round(loss, 4)}|{round(total_loss_, 4)}, token acc(run|overall): {round(accuracy, 4)}|{round(total_acc_, 4)}')
        recoder.add_scalar(f'train-Whole/TokenAcc', np.mean(total_acc), idx_)
        recoder.add_scalar(f'train-Whole/Loss', total_loss/batch_num, idx_)
        return round(total_loss_, 4)
    
    @torch.no_grad()
    def test_model(self, test_iter, path):
        '''batch size is 1'''
        self.model.eval()
        pbar = tqdm(test_iter)
        with open(path, 'w') as f:
            for batch in pbar:
                input_ids, ref, _ = batch
                max_size = max(len(ref), self.args['tgt_len_size'])
                tgt = self.model.predict(
                    input_ids, 
                    max_size, 
                    temperature=self.args['temperature'],
                    min_length=1,
                )
                tgt = self.model.vocab.convert_ids_to_tokens(tgt)
                tgt = ''.join(tgt)
                
                ref = self.model.vocab.convert_ids_to_tokens(ref)
                ref = ''.join(ref)
                
                # batch size is 1, so input_ids[0] is needed
                ctx = self.model.vocab.convert_ids_to_tokens(input_ids[0])
                ctx = ''.join(ctx).replace('[PAD]', '')

                f.write(f'CTX: {ctx}\n')
                f.write(f'REF: {ref}\n')
                f.write(f'TGT: {tgt}\n\n')
        print(f'[!] translate test dataset over, write into {path}')
        # measure the performance
        (b1, b2, b3, b4), ((r_max_l, r_min_l, r_avg_l), (c_max_l, c_min_l, c_avg_l)), (dist1, dist2, rdist1, rdist2), (average, extrema, greedy) = cal_generative_metric(path, lang=self.args['lang'])
        print(f'[TEST] BLEU: {b1}/{b2}/{b3}/{b4}; Length(max, min, avg): {c_max_l}/{c_min_l}/{c_avg_l}|{r_max_l}/{r_min_l}/{r_avg_l}; Dist: {dist1}/{dist2}|{rdist1}/{rdist2}; Embedding(average/extrema/greedy): {average}/{extrema}/{greedy}')
        
    def tokenize_(self, obj):
        '''borrow from thu-coai/CDial-GPT'''
        return self.model.vocab.convert_tokens_to_ids(self.model.vocab.tokenize(obj))
    
    @torch.no_grad()
    def talk(self, topic, msgs, maxlen=50, batch_size=32):
        self.model.eval()
        if self.args['run_mode'] == 'test':
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
        else:    # rerank run mode
            msgs_ = [self.tokenize_(msgs)] * self.args['samples']
            tgts = self.model.predict_batch(
                msgs_, maxlen, temperature=self.args['temperature']
            )
            rest = []
            for i in tgts:
                i = self.model.vocab.convert_ids_to_tokens(i)
                i = ''.join(i)
                if '[SEP]' in i:
                    i = i[:i.index('[SEP]')]
                rest.append(i)
            # rerank
            scores = self.reranker([msgs]*len(rest), rest, topic=None)[0]
            index = np.argmax(scores)
            response = rest[index]
            return response
        
# ========== ========== #

    
# ======== Only for Flask server ========== #
class LCCCAgent(BaseAgent):
    
    def __init__(self, multi_gpu, run_mode='test'):
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
            'run_mode': run_mode,    # test / rerank
            'samples': 16,
        }
        self.model = LCCC(
            self.args['pretrained_path'],
            self.args['topk'], 
            self.args['topp'],
        )
        
        # use the BERTRetrieval model as the reranker
        if self.args['run_mode'] == 'rerank':
            from multiview.multiview import MultiView
            print(f'[!] MultiView reranker model will be initized')
            self.reranker = MultiView(
                topic=False,
                length=False,
                nidf_tf=False,
                coherence=True,
                fluency=False,
                repetition_penalty=False,
                mmi=False,
                distinct=False,
                mmi_path='ckpt/train_generative/gpt2_mmi/best.pt',
                coherence_path='ckpt/zh50w/bertretrieval/best.pt',
                topic_path='ckpt/fasttext/model.bin',
                fluency_path='ckpt/LM/gpt2lm/best.pt',
            )
            print(f'[!] load multiview model over')
        
        if torch.cuda.is_available():
            self.model.cuda()
        
        self.show_parameters(self.args)
        
    def tokenize_(self, obj):
        '''borrow from thu-coai/CDial-GPT'''
        return self.model.vocab.convert_tokens_to_ids(self.model.vocab.tokenize(obj))
    
    @torch.no_grad()
    def talk(self, topic, msgs, maxlen=50, batch_size=32):
        self.model.eval()
        if self.args['run_mode'] == 'test':
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
        else:    # rerank run mode
            msgs_ = [self.tokenize_(msgs)] * self.args['samples']
            tgts = self.model.predict_batch(
                msgs_, maxlen, temperature=self.args['temperature']
            )
            rest = []
            for i in tgts:
                i = self.model.vocab.convert_ids_to_tokens(i)
                i = ''.join(i)
                if '[SEP]' in i:
                    i = i[:i.index('[SEP]')]
                rest.append(i)
            # rerank
            scores = self.reranker([msgs]*len(rest), rest, topic=None)[0]
            index = np.argmax(scores)
            response = rest[index]
            return response

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
# ========== Discard ========== #
# ========== LCCC IR ========== #
class LCCCIR(nn.Module):
    
    def __init__(self, pretrained_path, topk, topp):
        super(LCCCIR, self).__init__()
        self.model = OpenAIGPTLMHeadModel.from_pretrained(pretrained_path)
        # binary classification head upon the OpenAIGPTModel (pre-trained)
        self.bs_head = nn.Linear(self.model.config.n_embd, 2)
        self.vocab = BertTokenizer.from_pretrained(pretrained_path, do_lower_case=True)
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
        total_loss, total_acc, batch_num, correct, s = 0, [], 0, 0, 0
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