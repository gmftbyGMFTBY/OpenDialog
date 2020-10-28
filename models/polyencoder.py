from .header import *

'''
PolyEncoder: https://arxiv.org/pdf/1905.01969v2.pdf
1. Bi-encoder
2. Cross-encoder (refer to bertretrieval)
3. Poly-Encoder
'''

class BertEmbedding(nn.Module):
    
    '''squeeze strategy: 1. first; 2. first-m; 3. average'''
    
    def __init__(self, m=0):
        super(BertEmbedding, self).__init__()
        self.model = BertModel.from_pretrained('bert-base-chinese')
        self.m = m

    def forward(self, ids, attn_mask, strategy='first'):
        '''convert ids to embedding tensor; Return: [B, 768]'''
        embd = self.model(ids, attention_mask=attn_mask)[0]    # [B, S, 768]
        if strategy == 'first':
            rest = embd[:, 0, :]
        elif strategy == 'first-m':
            rest = embd[:, :self.m, :]    # [B, M, 768]
        elif strategy == 'average':
            rest = embd.mean(dim=1)    # [B, 768]
        else:
            raise Exception(f'[!] Unknow squeeze strategy {self.squeeze_strategy}')
        return rest
    
class PolyEncoder(nn.Module):
    
    def __init__(self, m=16):
        super(PolyEncoder, self).__init__()
        self.ctx_encoder = BertEmbedding(m=m)
        self.can_encoder = BertEmbedding()
        
    def _encode(self, cid, rid, cid_mask, rid_mask):
        cid_rep = self.ctx_encoder(cid, cid_mask, strategy='first-m')
        rid_rep = self.can_encoder(rid, rid_mask, strategy='first')
        # cid_rep: [B, M, E]; rid_rep: [B, E]
        return cid_rep, rid_rep
        
    @torch.no_grad()
    def predict(self, cid, rid, rid_mask):
        batch_size = rid.shape[0]
        cid_rep, rid_rep = self._encode(cid.unsqueeze(0), rid, None, rid_mask)
        cid_rep = cid_rep.squeeze(0)    # [M, E]
        # cid_rep/rid_rep: [M, E], [B, E]
        
        # POLY ENCODER ATTENTION
        # [M, E] X [E, S] -> [M, S] -> [S, M]
        weights = F.softmax(
            torch.matmul(cid_rep, rid_rep.t()).transpose(0, 1),
            dim=-1,
        )
        # [S, M] X [M, E] -> [S, E]
        cid_rep = torch.matmul(weights, cid_rep)
        dot_product = (cid_rep * rid_rep).sum(-1)    # [S]
        return dot_product
        
    def forward(self, cid, rid, cid_mask, rid_mask):
        batch_size = cid.shape[0]
        assert batch_size > 1, f'[!] batch size must bigger than 1, cause other elements in the batch will be seen as the negative samples'
        cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask)
        # cid_rep/rid_rep: [B, M, E]; [B, E]
        
        # POLY ENCODER ATTENTION
        # [B, M, E] X [E, S] -> [B, M, S] -> [B, S, M]
        weights = F.softmax(
            torch.matmul(cid_rep, rid_rep.t()).permute(0, 2, 1),    # [B, M, S] -> [B, S, M]
            dim=-1,
        )
        cid_rep = torch.bmm(weights, cid_rep)    # [B, S, M] X [B, M, E] -> [B, S, E]
        # [B, S, E] x [B, S, E] -> [B, S]
        dot_product = (cid_rep * rid_rep.unsqueeze(0).expand(batch_size, -1, -1)).sum(-1)
        # use half for supporting the apex
        mask = to_cuda(torch.eye(batch_size)).half()    # [B, B]
        # calculate accuracy
        acc_num = (F.softmax(dot_product, dim=-1).max(dim=-1)[1] == torch.LongTensor(torch.arange(batch_size)).cuda()).sum().item()
        acc = acc_num / batch_size
        # calculate the loss
        loss = F.log_softmax(dot_product, dim=-1) * mask
        loss = (-loss.sum(dim=1)).mean()
        return loss, acc
    
class RUBERTBiEncoder(nn.Module):
    
    '''Re-used Bert bi-encoder model'''
    
    def __init__(self, max_turn_length=10):
        super(RUBERTBiEncoder, self).__init__()
        self.ctx_encoder = BertEmbedding()
        self.can_encoder = BertEmbedding()
        self.turn_weight = F.softmax(torch.log2(torch.arange(1, max_turn_length+1, dtype=torch.float)), dim=-1)
        self.turn_weight = to_cuda(self.turn_weight).half()    # apex
        self.max_chunk = 64
        
    def _encode(self, ids, ids_mask, ctx=True):
        if ctx:
            # ids: [B*T, S]
            # context may raise the OOM error
            id_rep = []
            for idx in range(0, len(ids), self.max_chunk):
                subids = ids[idx:idx+self.max_chunk]
                submask = ids_mask[idx:idx+self.max_chunk]
                sub_id_rep = self.ctx_encoder(subids, submask)
                id_rep.append(sub_id_rep)
            id_rep = torch.cat(id_rep)
        else:
            id_rep = self.can_encoder(ids, ids_mask)
        return id_rep
    
    def squeeze(self, cid):
        '''squeeze the context embeddings; cid: [B, T, E]'''
        rest = []
        for sample in cid:
            weight = self.turn_weight[:len(sample)].unsqueeze(1).expand(-1, 768)
            # [T, E] * [T, E]
            rest.append(torch.sum(weight * sample, dim=0))    # [E]
        rest = torch.stack(rest)    # [B, E]
        return rest
    
    @torch.no_grad()
    def talk_predict(self, cid, rid, rid_mask):
        '''self.history_embd is used'''
        cid_rep = self._encode(cid, ctx=True)    # [E]
        self.history_embd.append(cid_rep)
        cid_rep = torch.stack(self.history_embd)    # [T, E]
        cid_rep = self.squeeze([cid_rep]).squeeze(0)    # [E]
        rid_rep = self._encode(rid, rid_mask, ctx=False)   # [B, E]
        dot_product = torch.matmul(cid_rep, rid_rep.t())  # [B]
        return dot_product
    
    @torch.no_grad()
    def predict(self, cid, rid, cid_mask, rid_mask):
        '''return the dot product of this turn and the rid_rep for the agent'''
        batch_size = rid.shape[0]
        cid_rep = self._encode(cid, cid_mask, ctx=True)   # [T, E]
        cid_rep = self.squeeze([cid_rep]).squeeze(0)    # [E]
        rid_rep = self._encode(rid, rid_mask, ctx=False)   # [B, E]
        # cid_rep/rid_rep: [E], [B, E]
        dot_product = torch.matmul(cid_rep, rid_rep.t())  # [B]
        return dot_product
        
    def forward(self, cid, turn_length, rid, cid_mask, rid_mask):
        '''cid: [B*T, S]; rid: [B, S]; cid_mask: [B*T, S]; rid_mask: [B, S]'''
        batch_size = rid.shape[0]
        cid_rep = self._encode(cid, cid_mask, ctx=True)   # [B*T, E]
        cid_rep = self.squeeze(torch.split(cid_rep, turn_length))    # [B, E]
        rid_rep = self._encode(rid, rid_mask, ctx=False)   # [B, E]
        
        # cid_rep/rid_rep: [B, 768]
        dot_product = torch.matmul(cid_rep, rid_rep.t())  # [B, B]
        # use half for supporting the apex
        mask = to_cuda(torch.eye(batch_size)).half()    # [B, B]
        # calculate accuracy
        acc_num = (F.softmax(dot_product, dim=-1).max(dim=-1)[1] == torch.LongTensor(torch.arange(batch_size)).cuda()).sum().item()
        acc = acc_num / batch_size
        # calculate the loss
        loss = F.log_softmax(dot_product, dim=-1) * mask
        loss = (-loss.sum(dim=1)).mean()
        return loss, acc
    
class BERTBiEncoder(nn.Module):
    
    '''During training, the other elements in the batch are seen as the negative samples, which will lead to the fast training speed. More details can be found in paper: https://arxiv.org/pdf/1905.01969v2.pdf
    reference: https://github.com/chijames/Poly-Encoder/blob/master/encoder.py
    '''
    
    def __init__(self):
        super(BERTBiEncoder, self).__init__()
        self.ctx_encoder = BertEmbedding()
        self.can_encoder = BertEmbedding()
        
    def _encode(self, cid, rid, cid_mask, rid_mask):
        cid_rep = self.ctx_encoder(cid, cid_mask)
        rid_rep = self.can_encoder(rid, rid_mask)
        return cid_rep, rid_rep
    
    @torch.no_grad()
    def predict(self, cid, rid, rid_mask):
        batch_size = rid.shape[0]
        cid_rep, rid_rep = self._encode(cid.unsqueeze(0), rid, None, rid_mask)
        cid_rep = cid_rep.squeeze(0)    # [768]
        # cid_rep/rid_rep: [768], [B, 768]
        dot_product = torch.matmul(cid_rep, rid_rep.t())  # [B]
        return dot_product
        
    def forward(self, cid, rid, cid_mask, rid_mask):
        batch_size = cid.shape[0]
        assert batch_size > 1, f'[!] batch size must bigger than 1, cause other elements in the batch will be seen as the negative samples'
        cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask)
        # cid_rep/rid_rep: [B, 768]
        dot_product = torch.matmul(cid_rep, rid_rep.t())  # [B, B]
        # use half for supporting the apex
        mask = to_cuda(torch.eye(batch_size)).half()    # [B, B]
        # calculate accuracy
        acc_num = (F.softmax(dot_product, dim=-1).max(dim=-1)[1] == torch.LongTensor(torch.arange(batch_size)).cuda()).sum().item()
        acc = acc_num / batch_size
        # calculate the loss
        loss = F.log_softmax(dot_product, dim=-1) * mask
        loss = (-loss.sum(dim=1)).mean()
        return loss, acc
    
class BERTBiCompEncoder(nn.Module):
    
    '''During training, the other elements in the batch are seen as the negative samples, which will lead to the fast training speed. More details can be found in paper: https://arxiv.org/pdf/1905.01969v2.pdf
    reference: https://github.com/chijames/Poly-Encoder/blob/master/encoder.py
    
    Set the different learning ratio
    '''
    
    def __init__(self, nhead, dim_feedforward, num_encoder_layers, dropout=0.1):
        super(BERTBiCompEncoder, self).__init__()
        self.ctx_encoder = BertEmbedding()
        self.can_encoder = BertEmbedding()
        
        encoder_layer = nn.TransformerEncoderLayer(
            768, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
        )
        encoder_norm = nn.LayerNorm(768)
        self.trs_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_encoder_layers, 
            encoder_norm,
        )
        
        self.proj1 = nn.Linear(768*2, 768)
        self.gate = nn.Linear(768*3, 768)
        self.dropout = nn.Dropout(p=dropout)
        self.layernorm = nn.LayerNorm(768)
        
    def _encode(self, cid, rid, cid_mask, rid_mask):
        cid_rep = self.ctx_encoder(cid, cid_mask)
        rid_rep = self.can_encoder(rid, rid_mask)
        return cid_rep, rid_rep
    
    @torch.no_grad()
    def predict(self, cid, rid, rid_mask):
        # cid_rep: [1, E]; rid_rep: [S, E]
        batch_size = rid.shape[0]
        cid_rep, rid_rep = self._encode(cid.unsqueeze(0), rid, None, rid_mask)
        cid_rep = cid_rep.squeeze(0)    # [E]
        cross_rep = torch.cat(
            [
                cid_rep.unsqueeze(0).expand(batch_size, -1), 
                rid_rep,
            ], 
            dim=1,
        )    # [S, 2*E]
        
        cross_rep = self.dropout(
            torch.tanh(
                self.trs_encoder(
                    torch.tanh(
                        self.proj1(cross_rep).unsqueeze(1)
                    )
                )
            ).squeeze(1)
        )    # [S, E]
        
        gate = torch.sigmoid(
            self.gate(
                torch.cat(
                    [
                        rid_rep,    # [S, E]
                        cid_rep.unsqueeze(0).expand(batch_size, -1),    # [S, E]
                        cross_rep,    # [S, E]
                    ],
                    dim=-1,
                )
            )
        )    # [S, E]
        # cross_rep: [S, E]
        cross_rep = self.layernorm(gate * rid_rep + (1 - gate) * cross_rep)
        # cid: [E]; cross_rep: [S, E]
        dot_product = torch.matmul(cid_rep, cross_rep.t())    # [S]
        return dot_product
        
    def forward(self, cid, rid, cid_mask, rid_mask):
        batch_size = cid.shape[0]
        assert batch_size > 1, f'[!] batch size must bigger than 1, cause other elements in the batch will be seen as the negative samples'
        cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask)    # [B, E]
        
        # cross attention for all the candidates
        cross_rep = []
        for cid_rep_ in cid_rep:
            cid_rep_ = cid_rep_.unsqueeze(0).expand(batch_size, -1)    # [S, E]
            cross_rep.append(
                torch.cat([cid_rep_, rid_rep], dim=-1)
            )    # [S, E*2]
        cross_rep = torch.stack(cross_rep).permute(1, 0, 2)    # [B, S, 2*E] -> [S, B, E*2]
        cross_rep = self.dropout(
            torch.tanh(
                self.trs_encoder(
                    torch.tanh(self.proj1(cross_rep))
                )
            ).permute(1, 0, 2)
        )    # [B, S, E]
        
        gate = torch.sigmoid(
            self.gate(
                torch.cat(
                    [
                        rid_rep.unsqueeze(0).expand(batch_size, -1, -1), 
                        cid_rep.unsqueeze(1).expand(-1, batch_size, -1),
                        cross_rep,
                    ], 
                    dim=-1
                )
            )
        )    # [B, S, E]
        cross_rep = self.layernorm(gate * rid_rep.unsqueeze(0).expand(batch_size, -1, -1) + (1 - gate) * cross_rep)    # [B, S, E]
        
        # reconstruct rid_rep
        cid_rep = cid_rep.unsqueeze(1)    # [B, 1, E]
        dot_product = torch.bmm(cid_rep, cross_rep.permute(0, 2, 1)).squeeze(1)    # [B, S]
        # use half for supporting the apex
        mask = to_cuda(torch.eye(batch_size)).half()    # [B, B]
        # calculate accuracy
        acc_num = (F.softmax(dot_product, dim=-1).max(dim=-1)[1] == torch.LongTensor(torch.arange(batch_size)).cuda()).sum().item()
        acc = acc_num / batch_size
        # calculate the loss
        loss = F.log_softmax(dot_product, dim=-1) * mask
        loss = (-loss.sum(dim=1)).mean()
        return loss, acc
    
class BERTBiEncoderAgent(RetrievalBaseAgent):
    
    '''model parameter can be:
    1. compare: bi-encoder with comparsion module
    2. no-compare: pure bi-encoder
    3. polyencoder: polyencoder'''
    
    def __init__(self, multi_gpu, total_step, run_mode='train', local_rank=0, kb=True, model='no-compare'):
        super(BERTBiEncoderAgent, self).__init__(kb=kb)
        try:
            self.gpu_ids = list(range(len(multi_gpu.split(',')))) 
        except:
            raise Exception(f'[!] multi gpu ids are needed, but got: {multi_gpu}')
        self.args = {
            'lr': 5e-5,
            'lr_': 5e-4,
            'grad_clip': 1.0,
            'multi_gpu': self.gpu_ids,
            'talk_samples': 256,
            'vocab_file': 'bert-base-chinese',
            'pad': 0,
            'samples': 10,
            'model': 'bert-base-chinese',
            'amp_level': 'O2',
            'local_rank': local_rank,
            'warmup_steps': 8000,
            'total_step': total_step,
            'dmodel': model,
            'num_encoder_layers': 2,
            'dim_feedforward': 512,
            'nhead': 6,
            'dropout': 0.1,
            'max_len': 256,
            'poly_m': 16,
        }
        self.vocab = BertTokenizer.from_pretrained(self.args['vocab_file'])
        if model == 'no-compare':
            self.model = BERTBiEncoder()
        elif model == 'polyencoder':
            self.model = PolyEncoder( 
                m=self.args['poly_m'],
            )
        else:
            self.model = BERTBiCompEncoder(
                self.args['nhead'], 
                self.args['dim_feedforward'], 
                self.args['num_encoder_layers'], 
                dropout=self.args['dropout'],
            )
        if torch.cuda.is_available():
            self.model.cuda()
        if run_mode == 'train':
            if model in ['polyencoder', 'no-compare']:
                self.optimizer = transformers.AdamW(
                    self.model.parameters(), 
                    lr=self.args['lr'],
                )
            else:
                self.optimizer = transformers.AdamW(
                    [
                        {
                            'params': self.model.ctx_encoder.parameters(),
                        },
                        {
                            'params': self.model.can_encoder.parameters(),
                        },
                        {
                            'params': self.model.trs_encoder.parameters(), 
                            'lr': self.args['lr_'],
                        },
                        {
                            'params': self.model.proj1.parameters(), 
                            'lr': self.args['lr_'],
                        },
                        {
                            'params': self.model.gate.parameters(), 
                            'lr': self.args['lr_'],
                        }
                    ], 
                    lr=self.args['lr'],
                )
                print(f'[!] set the different learning ratios for comparsion module')
            self.model, self.optimizer = amp.initialize(
                self.model, 
                self.optimizer,
                opt_level=self.args['amp_level'],
            )
            self.scheduler = transformers.get_linear_schedule_with_warmup(
                self.optimizer, 
                num_warmup_steps=self.args['warmup_steps'], 
                num_training_steps=total_step,
            )
            self.model = nn.parallel.DistributedDataParallel(
                self.model, device_ids=[local_rank], output_device=local_rank,
                find_unused_parameters=True,
            )
        self.show_parameters(self.args)
        
    def train_model(self, train_iter, mode='train', recoder=None, idx_=0):
        self.model.train()
        total_loss, total_acc, batch_num = 0, 0, 0
        pbar = tqdm(train_iter)
        correct, s = 0, 0
        for idx, batch in enumerate(pbar):
            self.optimizer.zero_grad()
            cid, rid, cid_mask, rid_mask = batch
            loss, acc = self.model(cid, rid, cid_mask, rid_mask)
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
            clip_grad_norm_(amp.master_params(self.optimizer), self.args['grad_clip'])
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            total_acc += acc
            batch_num += 1
            
            recoder.add_scalar(f'train-epoch-{idx_}/Loss', total_loss/batch_num, idx)
            recoder.add_scalar(f'train-epoch-{idx_}/RunLoss', loss.item(), idx)
            recoder.add_scalar(f'train-epoch-{idx_}/Acc', total_acc/batch_num, idx)
            recoder.add_scalar(f'train-epoch-{idx_}/RunAcc', acc, idx)

            pbar.set_description(f'[!] loss: {round(loss.item(), 4)}|{round(total_loss/batch_num, 4)}; acc: {round(acc, 4)}|{round(total_acc/batch_num, 4)}')
        recoder.add_scalar(f'train-whole/Loss', total_loss/batch_num, idx_)
        recoder.add_scalar(f'train-whole/Acc', total_acc/batch_num, idx_)
        return round(total_loss / batch_num, 4)
        
    @torch.no_grad()
    def test_model(self, test_iter, mode='test', recoder=None, idx_=0):
        '''there is only one context in the batch, and response are the candidates that are need to reranked; batch size is the self.args['samples']; the groundtruth is the first one. For douban300w and E-Commerce datasets'''
        self.model.eval()
        r1, r2, r5, r10, counter, mrr = 0, 0, 0, 0, 0, []
        pbar = tqdm(test_iter)
        for idx, batch in tqdm(list(enumerate(pbar))):                
            cid, rids, rids_mask = batch
            batch_size = len(rids)
            if batch_size != self.args['samples']:
                continue
            dot_product = self.model.predict(cid, rids, rids_mask).cpu()    # [B]
            r1 += (torch.topk(dot_product, 1, dim=-1)[1] == 0).sum().item()
            r2 += (torch.topk(dot_product, 2, dim=-1)[1] == 0).sum().item()
            r5 += (torch.topk(dot_product, 5, dim=-1)[1] == 0).sum().item()
            r10 += (torch.topk(dot_product, 10, dim=-1)[1] == 0).sum().item()
            preds = torch.argsort(dot_product, dim=-1).tolist()    # [B, B]
            # mrr
            dot_product = dot_product.numpy()
            y_true = np.zeros(len(dot_product))
            y_true[0] = 1
            mrr.append(label_ranking_average_precision_score([y_true], [dot_product]))
            counter += 1
            
        r1, r2, r5, r10, mrr = round(r1/counter, 4), round(r2/counter, 4), round(r5/counter, 4), round(r10/counter, 4), round(np.mean(mrr), 4)
        print(f'r1@10: {r1}; r2@10: {r2}; r5@10: {r5}; r10@10: {r10}; mrr: {mrr}')
    
    @torch.no_grad()
    def talk(self, msgs, topic=None):
        self.model.eval()
        utterances, inpt_ids, res_ids, attn_mask = self.process_utterances_biencoder(topic, msgs, max_len=self.args['max_len'])
        output = self.model.predict(inpt_ids, res_ids, attn_mask)    # [B]
        item = torch.argmax(output).item()
        msg = utterances[item]
        return msg
    
class RUBERTBiEncoderAgent(RetrievalBaseAgent):
    
    def __init__(self, multi_gpu, total_step, run_mode='train', local_rank=0, kb=True):
        super(RUBERTBiEncoderAgent, self).__init__(kb=kb)
        try:
            self.gpu_ids = list(range(len(multi_gpu.split(',')))) 
        except:
            raise Exception(f'[!] multi gpu ids are needed, but got: {multi_gpu}')
        self.args = {
            'lr': 5e-5,
            'grad_clip': 1.0,
            'multi_gpu': self.gpu_ids,
            'talk_samples': 256,
            'vocab_file': 'bert-base-chinese',
            'pad': 0,
            'samples': 10,
            'model': 'bert-base-chinese',
            'amp_level': 'O2',
            'local_rank': local_rank,
            'warmup_steps': 8000,
            'total_step': total_step,
            'max_len': 256,
            'max_turn_size': 10
        }
        self.history_text, self.history_embd = [], []
        self.vocab = BertTokenizer.from_pretrained(self.args['vocab_file'])
        self.model = RUBERTBiEncoder(
            max_turn_length=self.args['max_turn_size'],
        )
        if torch.cuda.is_available():
            self.model.cuda()
        if run_mode == 'train':
            self.optimizer = transformers.AdamW(
                self.model.parameters(), 
                lr=self.args['lr'],
            )
            self.model, self.optimizer = amp.initialize(
                self.model, 
                self.optimizer,
                opt_level=self.args['amp_level'],
            )
            self.scheduler = transformers.get_linear_schedule_with_warmup(
                self.optimizer, 
                num_warmup_steps=self.args['warmup_steps'], 
                num_training_steps=total_step,
            )
            self.model = nn.parallel.DistributedDataParallel(
                self.model, device_ids=[local_rank], output_device=local_rank,
                find_unused_parameters=True,
            )
        self.show_parameters(self.args)
        
    def reset(self):
        '''clear the memory'''
        del self.history_text
        del self.history_embd
        self.history_text, self.history_embd = [], []
        print(f'[!] reset the dialog history over ...')
        
    def train_model(self, train_iter, mode='train', recoder=None, idx_=0):
        self.model.train()
        total_loss, total_acc, batch_num = 0, 0, 0
        pbar = tqdm(train_iter)
        correct, s = 0, 0
        for idx, batch in enumerate(pbar):
            self.optimizer.zero_grad()
            cid, turn_length, rid, cid_mask, rid_mask = batch
            loss, acc = self.model(cid, turn_length, rid, cid_mask, rid_mask)
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
            clip_grad_norm_(amp.master_params(self.optimizer), self.args['grad_clip'])
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            total_acc += acc
            batch_num += 1
            
            recoder.add_scalar(f'train-epoch-{idx_}/Loss', total_loss/batch_num, idx)
            recoder.add_scalar(f'train-epoch-{idx_}/RunLoss', loss.item(), idx)
            recoder.add_scalar(f'train-epoch-{idx_}/Acc', total_acc/batch_num, idx)
            recoder.add_scalar(f'train-epoch-{idx_}/RunAcc', acc, idx)

            pbar.set_description(f'[!] loss: {round(loss.item(), 4)}|{round(total_loss/batch_num, 4)}; acc: {round(acc, 4)}|{round(total_acc/batch_num, 4)}')
        recoder.add_scalar(f'train-whole/Loss', total_loss/batch_num, idx_)
        recoder.add_scalar(f'train-whole/Acc', total_acc/batch_num, idx_)
        return round(total_loss / batch_num, 4)
        
    @torch.no_grad()
    def test_model(self, test_iter, mode='test', recoder=None, idx_=0):
        '''there is only one context in the batch, and response are the candidates that are need to reranked; batch size is the self.args['samples']; the groundtruth is the first one. For douban300w and E-Commerce datasets'''
        self.model.eval()
        r1, r2, r5, r10, counter, mrr = 0, 0, 0, 0, 0, []
        pbar = tqdm(test_iter)
        for idx, batch in tqdm(list(enumerate(pbar))):                
            cid, rids, cid_mask, rids_mask = batch
            batch_size = len(rids)
            if batch_size != self.args['samples']:
                continue
            dot_product = self.model.predict(cid, rids, cid_mask, rids_mask).cpu()    # [B]
            r1 += (torch.topk(dot_product, 1, dim=-1)[1] == 0).sum().item()
            r2 += (torch.topk(dot_product, 2, dim=-1)[1] == 0).sum().item()
            r5 += (torch.topk(dot_product, 5, dim=-1)[1] == 0).sum().item()
            r10 += (torch.topk(dot_product, 10, dim=-1)[1] == 0).sum().item()
            preds = torch.argsort(dot_product, dim=-1).tolist()    # [B, B]
            # mrr
            dot_product = dot_product.numpy()
            y_true = np.zeros(len(dot_product))
            y_true[0] = 1
            mrr.append(label_ranking_average_precision_score([y_true], [dot_product]))
            counter += 1
            
        r1, r2, r5, r10, mrr = round(r1/counter, 4), round(r2/counter, 4), round(r5/counter, 4), round(r10/counter, 4), round(np.mean(mrr), 4)
        print(f'r1@10: {r1}; r2@10: {r2}; r5@10: {r5}; r10@10: {r10}; mrr: {mrr}')
    
    @torch.no_grad()
    def talk(self, msgs, topic=None):
        '''msg only contain one utterance'''
        self.model.eval()
        utterances, inpt_ids, res_ids, attn_mask = self.process_utterances_biencoder(topic, msgs, max_len=self.args['max_len'])
        
        output = self.model.talk_predict(inpt_ids, res_ids, attn_mask)    # [B]
        item = torch.argmax(output).item()
        msg = utterances[item]
        
        # 需要注意的是，用在context的msg也是可以缓存的
        self.history_text.append(msg)
        return msg
    
    def process_utterances_biencoder(self, topic, msgs, max_len=0):
        history = self.history_text
        def _length_limit(ids):
            if len(ids) > max_len:
                ids = [ids[0]] + ids[-(max_len-1):]
            return ids
        utterances = self.searcher.search(msgs, samples=self.args['talk_samples'], topic=topic)
        utterances = [i['utterance'] for i in utterances]
        utterances = list(set(utterances) - set(self.history))
        inpt_ids = self.vocab.batch_encode_plus([msgs] + utterances)['input_ids']
        context_inpt_ids, response_inpt_ids = inpt_ids[0], inpt_ids[1:]
        context_inpt_ids = torch.LongTensor(_length_limit(context_inpt_ids))
        response_inpt_ids = [torch.LongTensor(_length_limit(i)) for i in response_inpt_ids]
        response_inpt_ids = pad_sequence(response_inpt_ids, batch_first=True, padding_value=self.args['pad'])
        attn_mask_index = response_inpt_ids.nonzero().tolist()
        attn_mask_index_x, attn_mask_index_y = [i[0] for i in attn_mask_index], [i[1] for i in attn_mask_index]
        attn_mask = torch.zeros_like(response_inpt_ids)
        attn_mask[attn_mask_index_x, attn_mask_index_y] = 1
        
        if torch.cuda.is_available():
            context_inpt_ids, response_inpt_ids, attn_mask = context_inpt_ids.cuda(), response_inpt_ids.cuda(), attn_mask.cuda()
        return utterances, context_inpt_ids, response_inpt_ids, attn_mask