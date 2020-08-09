from .header import *

'''
BertForMultipleChoice
'''

class BERTMC(nn.Module):
    
    def __init__(self, model='bert-base-chinese'):
        super(BERTMC, self).__init__()
        self.model = BertForMultipleChoice.from_pretrained(model)
        
    def forward(self, inpt):
        # inpt: [B, N, S], default N is 2 (one positive and one negative)
        attn_mask = generate_attention_mask_mc(inpt)
        output = self.model(
            input_ids=inpt,
            attention_mask=attn_mask,
        )
        logits = output[0]    # [B, N]
        return logits
    
class BERTMCFusion(nn.Module):
    
    def __init__(self, model='bert-base-chinese', dropout=0.3, num_layers=1):
        super(BERTMCFusion, self).__init__()
        self.bert = BertModel.from_pretrained(model)
        encoder_layer = nn.TransformerEncoderLayer(
            768, 
            nhead=2, 
            dim_feedforward=768, 
            dropout=dropout,
        )
        # [N, B, 768] -> [N, B, 768]
        self.fusion = nn.TransformerEncoder(
            encoder_layer, 
            num_layers
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(768, 1)
        
    def forward(self, inpt):
        # inpt: [B, N, S], default N is 2 (one positive and one negative)
        num_choices = inpt.shape[1]    # [N]
        inpt = inpt.view(-1, inpt.size(-1))    # [B*N, S]
        attn_mask = generate_attention_mask(inpt)
        output = self.bert(
            input_ids=inpt,
            attention_mask=attn_mask,
        )
        logits = output[0]    # [B*N, S, 768]
        logits = logits.mean(dim=1)    # [B*N, 768]
        logits = torch.stack(logits.split(num_choices)).transpose(0, 1)    # [N, B, 768]
        logits = self.fusion(logits).transpose(0, 1)    # [N, B, 768] -> [B, N, 768]
        logits = self.classifier(self.dropout(logits)).squeeze(-1)    # [B, N, 1] -> [B, N]
        return logits
    
class BERTMCAgent(RetrievalBaseAgent):
    
    def __init__(self, multi_gpu, run_mode='train', lang='zh', kb=True, model_type='mc'):
        super(BERTMCAgent, self).__init__(kb=kb)
        # hyperparameters
        try:
            self.gpu_ids = list(range(len(multi_gpu.split(',')))) 
        except:
            raise Exception(f'[!] multi gpu ids are needed, but got: {multi_gpu}')
        self.args = {
            'lr': 1e-5,
            'grad_clip': 3.0,
            'test_N': 10,
            'multi_gpu': self.gpu_ids,
            'talk_samples': 256,
            'vocab_file': 'bert-base-chinese',
            'pad': 0,
            'dropout': 0.5,
            'num_layer': 1,
            'model': 'bert-base-chinese',
            'model_type': model_type,
        }
        # hyperparameters
        self.vocab = BertTokenizer.from_pretrained(self.args['vocab_file'])
        if self.args['model_type'] == 'mc':
            self.model = BERTMC(self.args['model'])
        elif self.args['model_type'] == 'mcf':
            self.model = BERTMCFusion(
                self.args['model'], 
                dropout=self.args['dropout'], 
                num_layers=self.args['num_layer']
            )
        else:
            raise Exception(f'[!] unknow model type {self.args["model_type"]}')
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

    def train_model(self, train_iter, mode='train', recoder=None, idx=0):
        self.model.train()
        total_loss, batch_num = 0, 0
        pbar = tqdm(train_iter)
        correct, s = 0, 0
        for idx, batch in enumerate(pbar):
            cid, label = batch
            self.optimizer.zero_grad()
            output = self.model(cid)
            loss = self.criterion(
                output, 
                label.view(-1),
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
            
            recoder.add_scalar(f'train-epoch-{idx}/Loss', loss.item(), idx)
            recoder.add_scalar(f'train-epoch-{idx}/Acc', correct/s, idx)
            recoder.add_scalar(f'train-epoch-{idx}/RunAcc', now_correct/len(label), idx)

            pbar.set_description(f'[!] loss: {round(loss.item(), 4)}; acc: {round(now_correct/len(label), 4)}|{round(correct/s, 4)}')
        return round(total_loss / batch_num, 4)
    
    @torch.no_grad()
    def test_model(self, test_iter, path):
        self.model.eval()
        pbar = tqdm(test_iter)
        rest = []
        for idx, batch in enumerate(pbar):
            cid, label = batch    # [B, N(10), S], [B]
            output = self.model(cid)    # [B, N]
            output = F.softmax(output, dim=-1)    # [B, N]
            preds = [i.tolist() for i in output]    # [B]
            for pred in preds:
                pred = np.argsort(pred)[::-1]
                rest.append(([0], pred.tolist()))
        p_1, r2_1, r10_1, r10_2, r10_5, MAP, MRR = cal_ir_metric(rest)
        print(f'[TEST] P@1: {p_1}; R2@1: {r2_1}; R10@1: {r10_1}; R10@2: {r10_2}; R10@5: {r10_5}; MAP: {MAP}; MRR: {MRR}')

    @torch.no_grad()
    def talk(self, topic, msgs):
        '''
        Batch size is the Num choice {B=N}
        '''
        # retrieval and process
        utterances_, ids = self.process_utterances(topic, msgs)
        # ids: [N, S]
        ids = ids.unsqueeze(0)    # [1, N, S]
        output = self.model(ids).squeeze(0)    # [1, N] -> [N]
        output = F.softmax(output, dim=-1)    # [N]
        item = torch.argmax(output).item()
        msg = utterances_[item]
        return msg

        