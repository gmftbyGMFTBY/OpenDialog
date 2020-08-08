from .header import *

class DualLSTM(nn.Module):
    
    '''
    Fuck the LSTM and GRU, stupid motherfucker !!!!!
    '''
    
    def __init__(self, model='bert-base-chinese'):
        super(DualLSTM, self).__init__()
        self.model = BertModel.from_pretrained(model)
        self.head = nn.Linear(768, 2)
        
    def forward(self, inpt):
        with torch.no_grad():
            attn_mask = generate_attention_mask(inpt)
            output = self.model(
                    input_ids=inpt,
                    attention_mask=attn_mask)[0]    # [batch, seq, 768]
            output = torch.mean(output, dim=1)    # [batch, 768]
        output = self.head(output)    # [batch, 2]
        return output 
    
class DualLSTMAgent(RetrievalBaseAgent):
    
    '''
    LSTM Model learn nothing, acc is around 0.5, BERT is better.
    '''
    
    def __init__(self, multi_gpu, run_mode='train', lang='zh', kb=True):
        super(DualLSTMAgent, self).__init__(kb=kb)
        try:
            self.gpu_ids = list(range(len(multi_gpu.split(',')))) 
        except:
            raise Exception(f'[!] multi gpu ids are needed, but got: {multi_gpu}')
        self.args = {
            'lr': 1e-5,
            'grad_clip': 3.0,
            'samples': 10,
            'multi_gpu': self.gpu_ids,
            'model': 'bert-base-chinese',
            'talk_samples': 256,
            'vocab_file': 'bert-base-chinese',
            'pad': 0,
        }
        # hyperparameters
        self.vocab = BertTokenizer.from_pretrained(self.args['vocab_file'])
        self.model = DualLSTM(
            self.args['model']
        )
        if torch.cuda.is_available():
            self.model.cuda()
        self.model = DataParallel(self.model, device_ids=self.gpu_ids)
        # bert model is too big, try to use the DataParallel
        self.optimizer = transformers.AdamW(
            self.model.parameters(), 
            lr=self.args['lr']
        )
        self.criterion = nn.CrossEntropyLoss()
        self.show_parameters(self.args)
        
    def train_model(self, train_iter, mode='train', recoder=None):
        self.model.train()
        total_loss, batch_num = 0, 0
        pbar = tqdm(train_iter)
        correct, s = 0, 0
        for idx, batch in enumerate(pbar):
            cid, label = batch
            self.optimizer.zero_grad()
            output = self.model(cid)    # [B, 2]
            loss = self.criterion(output, label.view(-1))
            loss.backward()
            clip_grad_norm_(
                self.model.parameters(), 
                self.args['grad_clip'],
            )
            self.optimizer.step()

            total_loss += loss.item()
            batch_num += 1
            
            now_correct = torch.max(F.softmax(output, dim=-1), dim=-1)[1]    # [batch]
            now_correct = torch.sum(now_correct == label).item()
            correct += now_correct
            s += len(label)

            pbar.set_description(f'[!] loss: {round(loss.item(), 4)}; acc: {round(now_correct/len(label), 4)}|{round(correct/s, 4)}')
        print(f'[!] overall acc: {round(correct/s, 4)}')
        return round(total_loss / batch_num, 4)
    
    @torch.no_grad()
    def test_model(self, test_iter, path):
        self.model.eval()
        total_loss, batch_num = 0, 0
        pbar = tqdm(test_iter)
        rest = []
        for idx, batch in enumerate(pbar):
            cid, label = batch
            output = self.model(cid)
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
    def talk(self, topic, msgs):
        # retrieval and process
        utterances_, ids = self.process_utterances(topic, msgs)
        # rerank, ids: [batch, seq]
        output = self.model(ids)    # [batch, 2]
        output = F.softmax(output, dim=-1)[:, 1]    # [batch]
        item = torch.argmax(output).item()
        msg = utterances_[item]
        return msg