from .header import *

class BERTRetrieval(nn.Module):

    def __init__(self, model='bert-base-chinese'):
        super(BERTRetrieval, self).__init__()
        self.model = BertForSequenceClassification.from_pretrained(
                model,
                num_labels=2)

    def forward(self, inpt):
        '''
        inpt: [batch, seq]
        '''
        attn_mask = generate_attention_mask(inpt)
        output = self.model(
                input_ids=inpt,
                attention_mask=attn_mask)
        logits = output[0]    # [batch, 2]
        return logits 
    
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
        loss_container = []
        with tqdm(total=len(train_iter)) as pbar:
            for idx, batch in train_iter:
                p, cid, label = batch
                self.optimizer.zero_grad()
                output = self.model(cid)    # [batch, 2]
                losses = self.criterion(
                          output, 
                          label.view(-1))
                # 
                train_iter.update_priority(losses.cpu().tolist())
                loss_container.extend(losses.cpu().tolist())
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

                pbar.set_description(f'[!] progress: {p}|1.0; train loss: {round(loss.item(), 4)}; acc: {round(now_correct/len(label), 4)}|{round(correct/s, 4)}')
                pbar.update(len(batch))
        print(f'[!] overall acc: {round(correct/s, 4)}')
        return loss_container

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
                total_loss += loss.item()
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

class BERTRetrievalAgent(RetrievalBaseAgent):

    '''
    Support Multi GPU, for example '1,2'
    '''

    def __init__(self, multi_gpu, run_mode='train', lang='zh', kb=True):
        super(BERTRetrievalAgent, self).__init__(kb=kb)
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
        self.criterion = nn.CrossEntropyLoss()
        self.criterion_ = nn.CrossEntropyLoss(reduction='none')

        self.show_parameters(self.args)

    def train_model(self, train_iter, mode='train', recoder=None):
        self.model.train()
        total_loss, batch_num = 0, 0
        pbar = tqdm(train_iter)
        correct, s = 0, 0
        for idx, batch in enumerate(pbar):
            # label: [batch]
            cid, label = batch
            self.optimizer.zero_grad()
            output = self.model(cid)    # [batch, 2]
            loss = self.criterion(
                    output, 
                    label.view(-1))
            if mode == 'train':
                loss.backward()
                clip_grad_norm_(self.model.parameters(), self.args['grad_clip'])
                self.optimizer.step()

            total_loss += loss.item()
            batch_num += 1
            
            now_correct = torch.max(F.softmax(output, dim=-1), dim=-1)[1]    # [batch]
            now_correct = torch.sum(now_correct == label).item()
            correct += now_correct
            s += len(label)

            pbar.set_description(f'[!] batch: {batch_num}; train loss: {round(loss.item(), 4)}; acc: {round(now_correct/len(label), 4)}|{round(correct/s, 4)}')
        print(f'[!] overall acc: {round(correct/s, 4)}')
        return round(total_loss / batch_num, 4)
    
    @torch.no_grad()
    def predict(self, train_iter):
        '''
        Curriculum learning: 
            predict and collect the loss for each sample in the train_iter
        '''
        self.model.eval()
        loss_container = []
        with tqdm(total=len(train_iter)) as pbar:
            for idx, batch in train_iter:
                cid, label = batch
                output = self.model(cid)    # [batch, 2]
                losses = self.criterion_(
                          output, 
                          label.view(-1))
                loss_container.extend(losses.cpu().tolist())
                pbar.update(len(batch))
                pbar.set_description(f'[!] collect loss for curriculum learning')
        return loss_container

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
                total_loss += loss.item()
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

    def reverse_search(self, ctx, ctx_, res):
        '''
        ctx/res: a list of string
        NOTE: Should remove the F.softmax in this function, set it into the forward
        '''
        with torch.no_grad():
            utterances_ = self.searcher.search(None, ctx, samples=self.args['talk_samples'])
            utterances_ = [i['context'] for i in utterances_]
            # for mask
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
        with torch.no_grad():
            utterances_, ids = self.process_utterances(topic, msgs)
            output = self.model(ids)
            output = F.softmax(output, dim=-1)[:, 1]
            # argsort
            indexs = torch.argsort(output, descending=True)[:topk]
            msgs = [utterances_[index] for index in indexs]
            return msgs
