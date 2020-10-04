from .header import *

'''Bert bi-encoder for binary classification'''

class BertEmbedding(nn.Module):
    
    def __init__(self, fine_tune=True, pool=False):
        super(BertEmbedding, self).__init__()
        self.model = BertModel.from_pretrained('bert-base-chinese')
        self.fine_tune = fine_tune
        self.pool = pool
        self.proj = nn.Sequential(
            nn.Linear(768*2, 768),
            nn.Tanh(),
            nn.Linear(768, 2)
        )

    def obtain_embedding(self, ids):
        '''convert ids to embedding tensor; Return: [B, 768]'''
        if self.fine_tune:
            embd = self.model(ids)[0]    # [B, S, 768]
        else:
            with torch.no_grad():
                embd = self.model(ids)[0]
        if self.pool:
            rest = torch.mean(embd, dim=1)    # [B, 768]
        else:
            rest = embd[:, 0, :]
        return rest    # [B, 768]
    
    def forward(self, cid, rid):
        cembd = self.obtain_embedding(cid)
        rembd = self.obtain_embedding(rid)
        embd = torch.cat([cembd, rembd], dim=1)    # [B, 768*2]
        rest = self.proj(embd)    # [B, 2]
        return rest
    
class BERTBiEncoderAgent(RetrievalBaseAgent):
    
    def __init__(self, multi_gpu, run_mode='train', local_rank=0):
        super(BERTBiEncoderAgent, self).__init__(kb=False)
        try:
            self.gpu_ids = list(range(len(multi_gpu.split(',')))) 
        except:
            raise Exception(f'[!] multi gpu ids are needed, but got: {multi_gpu}')
        self.args = {
            'lr': 1e-4,
            'grad_clip': 1.0,
            'samples': 10,
            'multi_gpu': self.gpu_ids,
            'talk_samples': 256,
            'vocab_file': 'bert-base-chinese',
            'pad': 0,
            'model': 'bert-base-chinese',
            'amp_level': 'O2',
            'local_rank': local_rank,
            'fine-tune': True,
            'pool': False,
        }
        self.vocab = BertTokenizer.from_pretrained(self.args['vocab_file'])
        self.model = BertEmbedding(
            fine_tune=self.args['fine-tune'], pool=self.args['pool'])
        if torch.cuda.is_available():
            self.model.cuda()
        self.optimizer = transformers.AdamW(
            self.model.parameters(), 
            lr=self.args['lr'],
        )
        if run_mode == 'train':
            self.model, self.optimizer = amp.initialize(
                self.model, 
                self.optimizer, 
                opt_level=self.args['amp_level']
            )
            self.model = nn.parallel.DistributedDataParallel(
                self.model, device_ids=[local_rank], output_device=local_rank,
                find_unused_parameters=True,
            )
        self.criterion = nn.CrossEntropyLoss()
        self.show_parameters(self.args)
        
    def train_model(self, train_iter, mode='train', recoder=None, idx_=0):
        self.model.train()
        total_loss, batch_num = 0, 0
        pbar = tqdm(train_iter)
        correct, s = 0, 0
        for idx, batch in enumerate(pbar):
            cid, rid, label = batch
            self.optimizer.zero_grad()
            output = self.model(cid, rid)    # [B, 2]
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
            
            now_correct = torch.max(F.softmax(output, dim=-1), dim=-1)[1]    # [B]
            now_correct = torch.sum(now_correct == label).item()
            correct += now_correct
            s += len(label)
            
            recoder.add_scalar(f'train-epoch-L{self.args["local_rank"]}-{idx_}/Loss', total_loss/batch_num, idx)
            recoder.add_scalar(f'train-epoch-L{self.args["local_rank"]}-{idx_}/RunLoss', loss.item(), idx)
            recoder.add_scalar(f'train-epoch-L{self.args["local_rank"]}-{idx_}/Acc', correct/s, idx)
            recoder.add_scalar(f'train-epoch-L{self.args["local_rank"]}-{idx_}/RunAcc', now_correct/len(label), idx)

            pbar.set_description(f'[!] loss: {round(loss.item(), 4)}|{round(total_loss/batch_num, 4)}; acc: {round(now_correct/len(label), 4)}|{round(correct/s, 4)}')
        recoder.add_scalar(f'train-whole-L{self.args["local_rank"]}/Loss', total_loss/batch_num, idx_)
        recoder.add_scalar(f'train-whole-L{self.args["local_rank"]}/Acc', correct/s, idx_)
        return round(total_loss / batch_num, 4)
    
    @torch.no_grad()
    def test_model(self, test_iter, mode='test', recoder=None, idx_=0):
        self.model.eval()
        rest, total_loss, batch_num = [], 0, 0
        pbar = tqdm(test_iter)
        for idx, batch in enumerate(pbar):
            cid, rid, label = batch
            output = self.model(cid, rid)    # [B, 2]
            loss = self.criterion(output, label.view(-1))
            total_loss += loss.item()
            batch_num += 1

            # output: [batch, 2]
            # only use the positive score as the final score
            output = F.softmax(output, dim=-1)[:, 1]    # [B]

            preds = [i.tolist() for i in torch.split(output, self.args['samples'])]
            labels = [i.tolist() for i in torch.split(label, self.args['samples'])]
            for label, pred in zip(labels, preds):
                pred = np.argsort(pred, axis=0)[::-1]
                rest.append(([0], pred.tolist()))
        print(f'[!] test loss: {round(total_loss/batch_num, 4)}')
        p_1, r2_1, r10_1, r10_2, r10_5, MAP, MRR = cal_ir_metric(rest)
        print(f'[TEST] P@1: {p_1}; R2@1: {r2_1}; R10@1: {r10_1}; R10@2: {r10_2}; R10@5: {r10_5}; MAP: {MAP}; MRR: {MRR}')
        return round(total_loss/batch_num, 4)