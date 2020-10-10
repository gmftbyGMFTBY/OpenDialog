from .header import *

'''Cross-Attention BertRetrieval'''

class BERTRetrieval(nn.Module):

    def __init__(self, model='bert-base-chinese'):
        super(BERTRetrieval, self).__init__()
        self.model = BertForSequenceClassification.from_pretrained(model, num_labels=2)

    def forward(self, inpt, token_type_ids, attn_mask):
        output = self.model(
            input_ids=inpt,
            attention_mask=attn_mask,
            token_type_ids=token_type_ids,
        )
        logits = output[0]    # [batch, 2]
        return logits
    
class BERTRetrievalAgent(RetrievalBaseAgent):

    def __init__(self, multi_gpu, run_mode='train', lang='zh', kb=True, local_rank=0):
        super(BERTRetrievalAgent, self).__init__(kb=kb)
        try:
            self.gpu_ids = list(range(len(multi_gpu.split(',')))) 
        except:
            raise Exception(f'[!] multi gpu ids are needed, but got: {multi_gpu}')
        self.args = {
            'lr': 1e-5,
            'grad_clip': 1.0,
            'samples': 10,
            'multi_gpu': self.gpu_ids,
            'talk_samples': 256,
            'max_len': 256,
            'vocab_file': 'bert-base-chinese',
            'pad': 0,
            'model': 'bert-base-chinese',
            'amp_level': 'O2',
            'local_rank': local_rank,
        }
        self.vocab = BertTokenizer.from_pretrained(self.args['vocab_file'])
        self.model = BERTRetrieval(self.args['model'])
        if torch.cuda.is_available():
            self.model.cuda()
        self.optimizer = transformers.AdamW(
            self.model.parameters(), 
            lr=self.args['lr'],
        )
        self.criterion = nn.CrossEntropyLoss()
        if run_mode == 'train':
            self.model, self.optimizer = amp.initialize(
                self.model, 
                self.optimizer, 
                opt_level=self.args['amp_level']
            )
            self.model = nn.parallel.DistributedDataParallel(
                self.model, device_ids=[local_rank], output_device=local_rank,
            )
        self.show_parameters(self.args)

    def train_model(self, train_iter, mode='train', recoder=None, idx_=0):
        self.model.train()
        total_loss, batch_num = 0, 0
        pbar = tqdm(train_iter)
        correct, s = 0, 0
        for idx, batch in enumerate(pbar):
            cid, token_type_ids, attn_mask, label = batch
            self.optimizer.zero_grad()
            output = self.model(cid, token_type_ids, attn_mask)    # [batch]
            loss = self.criterion(
                output, 
                label.view(-1),
            )
            
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
            clip_grad_norm_(amp.master_params(self.optimizer), self.args['grad_clip'])
            self.optimizer.step()

            total_loss += loss.item()
            batch_num += 1
            
            now_correct = torch.max(F.softmax(output, dim=-1), dim=-1)[1]    # [batch]
            now_correct = torch.sum(now_correct == label).item()
            correct += now_correct
            s += len(label)
            
            recoder.add_scalar(f'train-epoch-{idx_}/Loss', total_loss/batch_num, idx)
            recoder.add_scalar(f'train-epoch-{idx_}/RunLoss', loss.item(), idx)
            recoder.add_scalar(f'train-epoch-{idx_}/Acc', correct/s, idx)
            recoder.add_scalar(f'train-epoch-{idx_}/RunAcc', now_correct/len(label), idx)

            pbar.set_description(f'[!] train loss: {round(loss.item(), 4)}|{round(total_loss/batch_num, 4)}; acc: {round(now_correct/len(label), 4)}|{round(correct/s, 4)}')
        recoder.add_scalar(f'train-whole/Loss', total_loss/batch_num, idx_)
        recoder.add_scalar(f'train-whole/Acc', correct/s, idx_)
        return round(total_loss / batch_num, 4)

    @torch.no_grad()
    def test_model(self, test_iter, path):
        self.model.eval()
        total_loss, batch_num = 0, 0
        pbar = tqdm(test_iter)
        rest = []
        for idx, batch in enumerate(pbar):
            cid, token_type_ids, attn_mask, label = batch
            output = self.model(cid, token_type_ids, attn_mask)    # [batch, 2]
            loss = self.criterion(output, label.view(-1))
            total_loss += loss.item()
            batch_num += 1
            output = F.softmax(output, dim=-1)[:, 1]    # [batch]
            preds = [i.tolist() for i in torch.split(output, self.args['samples'])]
            labels = [i.tolist() for i in torch.split(label, self.args['samples'])]
            for label, pred in zip(labels, preds):
                pred = np.argsort(pred, axis=0)[::-1]
                rest.append(([0], pred.tolist()))
        print(f'[!] test loss: {round(total_loss/batch_num, 4)}')
        p_1, r2_1, r10_1, r10_2, r10_5, MAP, MRR = cal_ir_metric(rest)
        print(f'[TEST] P@1: {p_1}; R2@1: {r2_1}; R10@1: {r10_1}; R10@2: {r10_2}; R10@5: {r10_5}; MAP: {MAP}; MRR: {MRR}')
        return round(total_loss/batch_num, 4)

    @torch.no_grad()
    def talk(self, msgs, topic=None):
        self.model.eval()
        utterances_, inpt_ids, token_type_ids, attn_mask = self.process_utterances(
            topic, msgs, max_len=self.args['max_len']
        )
        # prepare the data input
        output = self.model(inpt_ids, token_type_ids, attn_mask)    # [B, 2]
        output = F.softmax(output, dim=-1)[:, 1]    # [B]
        item = torch.argmax(output).item()
        msg = utterances_[item]
        return msg

    def reverse_search(self, ctx, ctx_, res):
        with torch.no_grad():
            utterances_ = self.searcher.search(None, ctx, samples=self.args['talk_samples'])
            utterances_ = [i['context'] for i in utterances_]
            if ctx_ in utterances_:
                mask_index = utterances_.index(ctx_)
            else:
                mask_index = None
            utterances = [f'{i} [SEP] {res}' for i in utterances_]
            ids = [torch.LongTensor(self.vocab.encode(i)[-512:]) for i in utterances]
            ids = pad_sequence(ids, batch_first=True, padding_value=self.args['pad'])
            if torch.cuda.is_available():
                ids = ids.cuda()
            output = self.model(ids)     # [batch, 2]
            if mask_index is not None:
                output[mask_index][1] = -inf
            output = F.softmax(output, dim=-1)[:, 1]
            item = torch.argmax(output)
            rest = utterances_[item]
            return rest

    def rerank(self, topic, msgs, topk=2):
        self.model.eval()
        with torch.no_grad():
            utterances_, ids = self.process_utterances(topic, msgs)
            output = self.model(ids)
            output = F.softmax(output, dim=-1)[:, 1]
            # argsort
            indexs = torch.argsort(output, descending=True)[:topk]
            msgs = [utterances_[index] for index in indexs]
            return msgs
    
class BERTRetrievalDISAgent(RetrievalBaseAgent):
    
    def __init__(self, multi_gpu, run_mode='train', lang='zh', kb=True):
        super(BERTRetrievalDISAgent, self).__init__(kb=kb)
        # hyperparameters
        try:
            self.gpu_ids = list(range(len(multi_gpu.split(',')))) 
        except:
            raise Exception(f'[!] multi gpu ids are needed, but got: {multi_gpu}')
        self.args = {
            'lr': 1e-5,
            'grad_clip': 3.0,
            'samples': 10,
            'multi_gpu': self.gpu_ids,
            'talk_samples': 256,
            'vocab_file': 'bert-base-chinese',
            'pad': 0,
            'model': 'bert-base-chinese',
        }
        # hyperparameters
        self.vocab = BertTokenizer.from_pretrained(self.args['vocab_file'])
        self.model = BERTRetrieval(self.args['model'])
        
        self.agent_ = BERTRetrievalAgent(multi_gpu, kb=False)
        self.agent_.load_model(f'ckpt/zh50w/bertretrieval/best.pt')
        if torch.cuda.is_available():
            self.model.cuda()
            self.agent_.cuda()
        
        self.model = DataParallel(self.model, device_ids=self.gpu_ids)
        # bert model is too big, try to use the DataParallel
        self.optimizer = transformers.AdamW(
                self.model.parameters(), 
                lr=self.args['lr'])
        self.criterion = nn.CrossEntropyLoss()

        self.show_parameters(self.args)
        
    def train_model(self, train_iter, mode='train', recoder=None, idx_=0):
        self.model.train()
        total_loss, batch_num = 0, 0
        correct, s = 0, 0
        length = len(train_iter)
        pbar = tqdm(train_iter)
        for idx, batch in enumerate(pbar):
            cid, label = batch    
            self.optimizer.zero_grad()
            output = self.model(cid)    # [batch, 2]
            loss = self.criterion(
                output, 
                label.view(-1)
            )
            loss.backward()
            clip_grad_norm_(self.model.parameters(), self.args['grad_clip'])
            self.optimizer.step()
            total_loss += loss.item()
            batch_num += 1
            now_correct = torch.max(F.softmax(output, dim=-1), dim=-1)[1]    # [batch]
            now_correct = torch.sum(now_correct == label).item()
            correct += now_correct
            s += len(label)
                
            # recoder
            recoder.add_scalar(f'train-epoch-{idx_}/Loss', total_loss/batch_num, idx)
            recoder.add_scalar(f'train-epoch-{idx_}/RunLoss', loss.item(), idx)
            recoder.add_scalar(f'train-epoch-{idx_}/RunAcc', now_correct/len(label), idx)
            recoder.add_scalar(f'train-eopch-{idx_}/Acc', correct/s, idx)
            recoder.flush()
            pbar.set_description(f'[!] loss: {round(loss.item(), 4)}; acc(run|all): {round(now_correct/len(label), 4)}|{round(correct/s, 4)}')
        print(f'[!] overall acc: {round(correct/s, 4)}')
        return round(total_loss / batch_num, 4)
    
    def construct_tensor(self, cid, rid1s, rid2s, pad=0):
        data = []
        for rid1, rid2 in zip(rid1s, rid2s):
            data.append(torch.LongTensor(cid + rid1[1:] + rid2[2:]))
        data = pad_sequence(data, batch_first=True, padding_value=pad)    # [batch, seq]
        if torch.cuda.is_available():
            data = data.cuda()
        return data

    @torch.no_grad()
    def test_model(self, test_iter, path):
        '''
        tournament algrorithm:
        One batch one instance
        '''
        self.model.eval()
        total_loss, batch_num = 0, 0
        pbar = tqdm(test_iter)
        rest = []
        for idx, batch in enumerate(pbar):
            cid, rids, label = batch
            rids = []
            while len(rids_) != 1:
                output = self.model(cid)
                output = F.softmax(output, dim=-1)[:, 1]    # [batch]
            
            preds = [i.tolist() for i in torch.split(output, self.args['samples'])]
            labels = [i.tolist() for i in torch.split(label, self.args['samples'])]
            for label, pred in zip(labels, preds):
                pred = np.argsort(pred, axis=0)[::-1]
                rest.append(([0], pred.tolist()))
                
        p_1, r2_1, r10_1, r10_2, r10_5, MAP, MRR = cal_ir_metric(rest)
        print(f'[TEST] P@1: {p_1}; R2@1: {r2_1}; R10@1: {r10_1}; R10@2: {r10_2}; R10@5: {r10_5}; MAP: {MAP}; MRR: {MRR}')
        return round(total_loss/batch_num, 4)

    @torch.no_grad()
    def talk(self, topic, msgs):
        # retrieval and process
        utterances_, ids = self.process_utterances(topic, msgs)
        # rerank, ids: [batch, seq]
        output = self.model(ids)    # [batch, 2]
        output = F.softmax(output, dim=-1)[:, 1]    # [batch]
        item = torch.argmax(output).item()
        msg = utterances_[item]
        return msg
    
class BERTRetrievalCLAgent(RetrievalBaseAgent):

    '''
    The only difference between BERTRetrievalAgent is the train_model function
    '''

    def __init__(self, multi_gpu, run_mode='train', lang='zh', kb=True):
        super(BERTRetrievalCLAgent, self).__init__(kb=kb)
        # hyperparameters
        try:
            self.gpu_ids = list(range(len(multi_gpu.split(',')))) 
        except:
            raise Exception(f'[!] multi gpu ids are needed, but got: {multi_gpu}')
        self.args = {
                'lr': 1e-5,
                'grad_clip': 3.0,
                'samples': 10,
                'multi_gpu': self.gpu_ids,
                'talk_samples': 256,
                'vocab_file': 'bert-base-chinese',
                'pad': 0,
                'model': 'bert-base-chinese',
        }
        # hyperparameters
        self.vocab = BertTokenizer.from_pretrained(self.args['vocab_file'])
        self.model = BERTRetrieval(self.args['model'])
        if torch.cuda.is_available():
            self.model.cuda()
        self.model = DataParallel(self.model, device_ids=self.gpu_ids)
        # bert model is too big, try to use the DataParallel
        self.optimizer = transformers.AdamW(
                self.model.parameters(), 
                lr=self.args['lr'])
        self.criterion = nn.CrossEntropyLoss(reduction='none')

        self.show_parameters(self.args)

    def train_model(self, train_iter, mode='train', recoder=None):
        self.model.train()
        total_loss, batch_num = 0, 0
        correct, s = 0, 0
        length = len(train_iter)
        data_size = train_iter.data_size
        with tqdm(total=length) as pbar:
            for idx, batch in enumerate(train_iter):
                p, left, delta_, cid, label = batch
                
                self.optimizer.zero_grad()
                output = self.model(cid)    # [batch, 2]
                losses = self.criterion(
                          output, 
                          label.view(-1))
                train_iter.update_priority(losses.cpu().tolist())
                loss = losses.mean()
                
                loss.backward()
                clip_grad_norm_(self.model.parameters(), self.args['grad_clip'])
                self.optimizer.step()

                total_loss += loss.item()
                batch_num += 1

                now_correct = torch.max(F.softmax(output, dim=-1), dim=-1)[1]    # [batch]
                now_correct = torch.sum(now_correct == label).item()
                correct += now_correct
                s += len(label)
                
                # recoder
                recoder.add_scalar('train/Loss', loss.item(), idx)
                recoder.add_scalar('train/RunAcc', now_correct/len(label), idx)
                recoder.add_scalar('train/WholeAcc', correct/s, idx)
                recoder.add_scalar('train/Progress', p, idx)
                recoder.add_scalar('train/LeftSamples', left, idx)
                recoder.add_scalar('train/AddedSamples', delta_, idx)
                recoder.add_scalar('train/AvailableSamples', int(p*data_size), idx)
                recoder.flush()

                pbar.set_description(f'[!] progress: {p}|1.0; samples(delta|left|available|whole): {delta_}|{left}|{int(p*data_size)}|{data_size}; loss: {round(loss.item(), 4)}; acc: {round(now_correct/len(label), 4)}|{round(correct/s, 4)}')
                pbar.update(1)
        print(f'[!] overall acc: {round(correct/s, 4)}')
        return round(total_loss / batch_num, 4)

    def test_model(self, test_iter, path):
        self.model.eval()
        total_loss, batch_num = 0, 0
        pbar = tqdm(test_iter)
        rest = []
        with torch.no_grad():
            for idx, batch in enumerate(pbar):
                cid, label = batch
                output = self.model(cid)
                loss = self.criterion(output, label.view(-1))
                total_loss += loss.mean().item()
                batch_num += 1

                # output: [batch, 2]
                # only use the positive score as the final score
                output = F.softmax(output, dim=-1)[:, 1]    # [batch]

                preds = [i.tolist() for i in torch.split(output, self.args['samples'])]
                labels = [i.tolist() for i in torch.split(label, self.args['samples'])]
                for label, pred in zip(labels, preds):
                    pred = np.argsort(pred, axis=0)[::-1]
                    rest.append(([0], pred.tolist()))
        print(f'[!] test loss: {round(total_loss/batch_num, 4)}')
        p_1, r2_1, r10_1, r10_2, r10_5, MAP, MRR = cal_ir_metric(rest)
        print(f'[TEST] P@1: {p_1}; R2@1: {r2_1}; R10@1: {r10_1}; R10@2: {r10_2}; R10@5: {r10_5}; MAP: {MAP}; MRR: {MRR}')
        return round(total_loss/batch_num, 4)

    def talk(self, topic, msgs):
        with torch.no_grad():
            # retrieval and process
            utterances_, ids = self.process_utterances(topic, msgs)
            # rerank, ids: [batch, seq]
            output = self.model(ids)    # [batch, 2]
            output = F.softmax(output, dim=-1)[:, 1]    # [batch]
            item = torch.argmax(output).item()
            msg = utterances_[item]
            return msg

    def rerank(self, topic, msgs, topk=2):
        with torch.no_grad():
            utterances_, ids = self.process_utterances(topic, msgs)
            output = self.model(ids)
            output = F.softmax(output, dim=-1)[:, 1]
            # argsort
            indexs = torch.argsort(output, descending=True)[:topk]
            msgs = [utterances_[index] for index in indexs]
            return msgs
