from .header import *

'''
PolyEncoder: https://arxiv.org/pdf/1905.01969v2.pdf
1. Bi-encoder
2. Cross-encoder (refer to bertretrieval)
3. Poly-Encoder
'''

class BertEmbedding(nn.Module):
    
    '''squeeze strategy: 1. first; 2. first-m; 3. average'''
    
    def __init__(self, squeeze_strategy='first', m=0):
        super(BertEmbedding, self).__init__()
        self.model = BertModel.from_pretrained('bert-base-chinese')
        self.squeeze_strategy = squeeze_strategy
        self.m = m

    def forward(self, ids, attn_mask, strategy='first'):
        '''convert ids to embedding tensor; Return: [B, 768]'''
        embd = self.model(ids, attention_mask=attn_mask)[0]    # [B, S, 768]
        if self.squeeze_strategy == 'first':
            rest = embd[:, 0, :]
        elif self.squeeze_strategy == 'first-m':
            rest = embd[:, :self.m, :]    # [B, M, 768]
        elif self.squeeze_strategy == 'average':
            rest = embd.mean(dim=1)    # [B, 768]
        else:
            raise Exception(f'[!] Unknow squeeze strategy {self.squeeze_strategy}')
        return rest
    
class PolyEncoder(nn.Module):
    
    def __init__(self, share=True, m=16):
        super(PolyEncoder, self).__init__()
        if share:
            self.encoder = BertEmbedding(squeeze_strategy='first-m', m=m)
        else:
            self.ctx_encoder = BertEmbedding(squeeze_strategy='first-m', m=m)
            self.can_encoder = BertEmbedding(squeeze_strategy='first-m', m=m)
        self.share = share
        
    def _encode(self, cid, rid, cid_mask, rid_mask):
        if self.share:
            cid_rep = self.encoder(cid, cid_mask, strategy='first-m')
            rid_rep = self.encoder(rid, rid_mask, strategy='first')
        else:
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
        # weight    [B, M]
        weights = F.softmax(
            torch.matmul(rid_rep, cid_rep.t())
            dim=-1,
        )
        # context embedding
        cid_rep = torch.matmul(weights, cid_rep)    # [B, M] * [M, E] -> [B, E]
        
        dot_product = (cid_rep, rid_rep).sum(-1)  # [B, E] * [B, E] -> [B, E] -> [B]
        return F.softmax(dot_product, dim=-1)
        
    def forward(self, cid, rid, cid_mask, rid_mask):
        batch_size = cid.shape[0]
        assert batch_size > 1, f'[!] batch size must bigger than 1, cause other elements in the batch will be seen as the negative samples'
        cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask)
        # cid_rep/rid_rep: [B, M, E]; [B, E]
        
        # POLY ENCODER ATTENTION
        # weight    [B, M]
        weights = F.softmax(
            torch.bmm(
                rid_rep.unsqueeze(1),    # [B, 1, E]
                cid_rep.permute(0, 2, 1),    # [B, E, M]
            ).squeeze(1),    # [B, 1, M] -> [B, M]
            dim=-1,
        )
        # context embedding
        cid_rep = torch.bmm(
            weights.unsqueeze(1),    # [B, 1, M],
            cid_rep,    # [B, M, E]
        ).squeeze(1)    # [B, 1, E] -> [B, E] 
        
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
    
    def __init__(self, share=False):
        super(BERTBiEncoder, self).__init__()
        if share:
            self.encoder = BertEmbedding(squeeze_strategy='first')
        else:
            self.ctx_encoder = BertEmbedding(squeeze_strategy='first')
            self.can_encoder = BertEmbedding(squeeze_strategy='first')
        self.share = share
        
    def _encode(self, cid, rid, cid_mask, rid_mask):
        if self.share:
            cid_rep = self.encoder(cid, cid_mask)
            rid_rep = self.encoder(rid, rid_mask)
        else:
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
        return F.softmax(dot_product, dim=-1)
        
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
    '''
    
    def __init__(self, nhead, dim_feedforward, num_encoder_layers, dropout=0.2, share=False):
        super(BERTBiCompEncoder, self).__init__()
        if share:
            self.encoder = BertEmbedding(squeeze_strategy='first')
        else:
            self.ctx_encoder = BertEmbedding(squeeze_strategy='first')
            self.can_encoder = BertEmbedding(squeeze_strategy='first')
        self.share = share
        
        encoder_layer = nn.TransformerEncoderLayer(
            768, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout
        )
        encoder_norm = nn.LayerNorm(768)
        self.trs_encoder = nn.TransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm
        )
        
        self.proj1 = nn.Linear(768*2, 768)
        self.proj2 = nn.Linear(768*2, 768)
        
    def _encode(self, cid, rid, cid_mask, rid_mask):
        if self.share:
            cid_rep = self.encoder(cid, cid_mask)
            rid_rep = self.encoder(rid, rid_mask)
        else:
            cid_rep = self.ctx_encoder(cid, cid_mask)
            rid_rep = self.can_encoder(rid, rid_mask)
        return cid_rep, rid_rep
    
    @torch.no_grad()
    def predict(self, cid, rid, rid_mask):
        # cid_rep: [1, E]; rid_rep: [B, E]
        batch_size = rid.shape[0]
        cid_rep, rid_rep = self._encode(cid.unsqueeze(0), rid, None, rid_mask)
        cid_rep = cid_rep.squeeze(0)    # [E]
        cross_rep = torch.cat([cid_rep.unsqueeze(0).expand(batch_size, -1), rid_rep], dim=1)    # [S, E*2]
        cross_rep = self.proj1(
            self.trs_encoder(
                cross_rep.unsqueeze(1),
            ).squeeze(1)
        )    # [S, E*2] -> [S, E] 
        cross_rep = self.proj2(
            torch.cat(
                [
                    rid_rep,    # [S, E]
                    cross_rep,    # [S, E]
                ],
                dim=-1,
            )
        )    # [B, E]
        # cid: [E]; rid: [B, E]
        dot_product = torch.matmul(cid_rep, cross_rep.t())    # [B]
        return F.softmax(dot_product, dim=-1)    # [B]
        
    def forward(self, cid, rid, cid_mask, rid_mask):
        batch_size = cid.shape[0]
        assert batch_size > 1, f'[!] batch size must bigger than 1, cause other elements in the batch will be seen as the negative samples'
        cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask)    # [B, 768]
        
        # cross attention for all the candidates: [B, 768] -> [B, S, 768]
        cross_rep = []
        for cid_rep_ in cid_rep:
            cid_rep_ = cid_rep_.unsqueeze(0).expand(batch_size, -1)    # [S, 768]
            cross_rep.append(torch.cat([cid_rep_, rid_rep], dim=1))    # [S, 768*2]
        cross_rep = torch.stack(cross_rep).permute(1, 0, 2)    # [B, S, 768*2] -> [S, B, E]
        cross_rep = self.trs_encoder(
            self.proj1(cross_rep)
        ).permute(1, 0, 2)    # [S, B, E] -> [B, S, E]
        cross_rep = self.proj2(
            torch.cat(
                [
                    rid_rep.unsqueeze(0).expand(batch_size, -1, -1),    # [B, S, E]
                    cross_rep,    # [B, S, E]
                ],
                dim=-1,
            )
        )    # [B, S, E]
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
            'grad_clip': 1.0,
            'multi_gpu': self.gpu_ids,
            'talk_samples': 256,
            'vocab_file': 'bert-base-chinese',
            'pad': 0,
            'samples': 10,
            'model': 'bert-base-chinese',
            'amp_level': 'O2',
            'local_rank': local_rank,
            'pool': False,
            'share_bert': True,
            'warmup_steps': 8000,
            'total_step': total_step,
            'model': model,
            'num_encoder_layers': 1,
            'dim_feedforward': 512,
            'nhead': 8,
            'dropout': 0.1,
            'max_len': 256,
            'm': 16,
        }
        self.vocab = BertTokenizer.from_pretrained(self.args['vocab_file'])
        if model == 'no-compare':
            self.model = BERTBiEncoder(self.args['share_bert'])
        elif model == 'polyencoder':
            self.model = PolyEncoder(
                share=self.args['share_bert'], m=self.args['m']
            )
        else:
            self.model = BERTBiCompEncoder(
                self.args['nhead'], 
                self.args['dim_feedforward'], 
                self.args['num_encoder_layers'], 
                dropout=self.args['dropout'], 
                share=self.args['share_bert'],
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
                opt_level=self.args['amp_level']
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
            cid, rid, cid_mask, rid_mask = batch
            
            self.optimizer.zero_grad()
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
    def _test_model(self, test_iter, mode='test', recoder=None, idx_=0):
        '''there is only one context in the batch, and response are the candidates that are need to reranked; batch size is the self.args['samples']; the groundtruth is the first one.'''
        self.model.eval()
        r1, r2, r5, r10, counter, mrr = 0, 0, 0, 0, 0, []
        pbar = tqdm(test_iter)
        for idx, batch in tqdm(list(enumerate(pbar))):
            cid, rid, cid_mask, rid_mask = batch
            batch_size = len(rid)
            if batch_size != self.args['samples']:
                continue
            dot_product = self.model(cid, rid, cid_mask, rid_mask).cpu()    # [B, B]
            helper = torch.arange(10).unsqueeze(1)
            r1 += (torch.topk(dot_product, 1, dim=1)[1] == helper).sum().item()
            r2 += (torch.topk(dot_product, 2, dim=1)[1] == helper).sum().item()
            r5 += (torch.topk(dot_product, 5, dim=1)[1] == helper).sum().item()
            r10 += (torch.topk(dot_product, 10, dim=1)[1] == helper).sum().item()
            preds = torch.argsort(dot_product, dim=1).tolist()    # [B, B]
            counter += batch_size
            # mrr
            for idx, logit in enumerate(dot_product.numpy()):
                y_true = np.zeros(len(logit))
                y_true[idx] = 1
                mrr.append(label_ranking_average_precision_score([y_true], [logit]))
        r1, r2, r5, r10, mrr = round(r1/counter, 4), round(r2/counter, 4), round(r5/counter, 4), round(r10/counter, 4), round(np.mean(mrr), 4)
        print(f'r1@100: {r1}; r2@100: {r2}; r5@100: {r5}; r10@100: {r10}; mrr: {mrr}')
        
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
        print(f'r1@100: {r1}; r2@100: {r2}; r5@100: {r5}; r10@100: {r10}; mrr: {mrr}')
    
    @torch.no_grad()
    def talk(self, msgs, topic=None):
        self.model.eval()
        utterances, inpt_ids, res_ids, attn_mask = self.process_utterances_biencoder(topic, msgs, max_len=self.args['max_len'])
        output = self.model.predict(inpt_ids, res_ids, attn_mask)    # [B]
        item = torch.argmax(output).item()
        msg = utterances[item]
        return msg