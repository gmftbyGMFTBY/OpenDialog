from .header import *

'''
BERT for text matching (trained by negative sampling)
Deep mining the relations among the sentences: logic modeling; using the transformers
'''

class BERTRetrieval_Multi(nn.Module):

    def __init__(self, nipt=768, nhead=8, dim_feedforward=512, dropout=0.5, nlayers=6):
        super(BERTRetrieval_Multi, self).__init__()
        self.model = BertModel.from_pretrained(
                'bert-base-chinese')
        # the dataloader already makes sure the turn_size is consistent [NO PAD, NO MASK]
        self.nipt = nipt
        encoder_layer = nn.TransformerEncoderLayer(nipt, nhead, dim_feedforward, dropout)
        self.trs_encoder = nn.TransformerEncoder(encoder_layer, nlayers)
        # classification
        self.classifier = nn.Linear(nipt, 2)

    def forward(self, inpts, sep_index):
        '''
        inpts: [batch, seq]
        sep_index: batch*[turn_size list]
        '''
        # Obtain the Bert embedding of all the utterances (include response)
        seq_size = inpts.shape[-1]
        attn_mask = generate_attention_mask(inpts)
        output = self.model(
                    input_ids=inpts,
                    attention_mask=attn_mask)[0]    # [batch, seq, 768]
        rest = []
        for item, sidx in zip(output, sep_index):
            # item: [seq, 768]; sidx: [turn_size]
            if_pad = True if sum(sidx) < seq_size else False
            if if_pad:
                sidx.append(seq_size - sum(sidx))
                sequences = item.split(sidx, dim=0)[:-1]   # sequence: [sub_seq, 768]
            else:
                sequences = item.split(sidx, dim=0)
            sequences = torch.stack(
                    [torch.mean(seq, dim=0) for seq in sequences])    # sequence: [turn, 768]
            rest.append(sequences)
        rest = torch.stack(rest).permute(1, 0, 2)    # [batch, turn, 768] -> [turn, batch, 768]
        # transformer encoder
        output = self.trs_encoder(F.relu(rest))[-1]    # [turn, batch, 768] -> [batch, 768]
        logits = self.classifier(output)    # [batch, 2]
        return logits 

class BERTRetrievalMultiAgent(RetrievalBaseAgent):

    '''
    Support Multi GPU, for example '1,2'
    '''

    def __init__(self, multi_gpu, run_mode='train', lang='zh', kb=False):
        super(BERTRetrievalMultiAgent, self).__init__(kb=kb)
        # hyperparameters
        try:
            self.gpu_ids = list(range(len(multi_gpu.split(',')))) 
        except:
            raise Exception(f'[!] multi gpu ids are needed, but got: {multi_gpu}')
        self.args = {
                'lr': 3e-5,
                'nipt': 768,
                'nhead': 8,
                'dim_feedforward': 512,
                'dropout': 1.0,
                'nlayers': 6,
                'grad_clip': 3.0,
                'samples': 10,
                'multi_gpu': self.gpu_ids,
                'talk_samples': 512,
                'vocab_file': 'data/vocab/vocab_small',
                'pad': 0,
        }
        # hyperparameters
        self.vocab = BertTokenizer.from_pretrained('bert-base-chinese')
        self.model = BERTRetrieval_Multi(
                nipt=self.args['nipt'],
                nhead=self.args['nhead'],
                dim_feedforward=self.args['dim_feedforward'],
                dropout=self.args['dropout'],
                nlayers=self.args['nlayers'])
        if torch.cuda.is_available():
            self.model.cuda()
        self.model = DataParallel(self.model, device_ids=self.gpu_ids)
        # bert model is too big, try to use the DataParallel
        self.optimizer = transformers.AdamW(
                self.model.parameters(), 
                lr=self.args['lr'])
        self.criterion = nn.CrossEntropyLoss()

        self.show_parameters(self.args)

    def train_model(self, train_iter, mode='train', recoder=None):
        self.model.train()
        total_loss, batch_num = 0, 0
        correct, s = 0, 0
        with tqdm(total=len(train_iter)) as pbar:
            for idx, batch in enumerate(train_iter):
                # label: [batch]
                cid, label, sep_index = batch
                self.optimizer.zero_grad()
                output = self.model(cid, sep_index)    # [batch, 2]
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
                pbar.update(len(label))
        print(f'[!] overall acc: {round(correct/s, 4)}')
        return round(total_loss / batch_num, 4)

    def test_model(self, test_iter, path):
        self.model.eval()
        total_loss, batch_num = 0, 0
        rest = []
        with torch.no_grad():
            with tqdm(total=len(test_iter)) as pbar:
                for idx, batch in enumerate(train_iter):
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
                    pbar.update(len(label))
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
