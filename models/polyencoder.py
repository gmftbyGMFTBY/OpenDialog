from .header import *

'''PolyEncoder: https://arxiv.org/pdf/1905.01969v2.pdf
1. Bi-encoder
2. Cross-encoder
3. Poly-Encoder

A little bit different from the original implementation in the paper,
we don't use the segment embedding for cross-encoder (easier for data processing, forgive me :)).
'''

class BertEmbedding(nn.Module):
    
    '''squeeze strategy:
    1. first:
    2. first-m:
    3. average
    '''
    
    def __init__(self, squeeze_strategy='first', m=0):
        super(BertEmbedding, self).__init__()
        self.model = BertModel.from_pretrained('bert-base-chinese')
        self.squeeze_strategy = squeeze_strategy
        self.m = m

    def forward(self, ids, attn_mask):
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
    
class BERTBIEncoder(nn.Module):
    
    '''During training, the other elements in the batch are seen as the negative samples, which will lead to the fast training speed. More details can be found in paper: https://arxiv.org/pdf/1905.01969v2.pdf'''
    
    def __init__(self, share=False):
        super(BERTBIEncoder, self).__init__()
        if share:
            self.encoder = BertEmbedding(squeeze_strategy='first')
        else:
            self.ctx_encoder = BertEmbedding(squeeze_strategy='first')
            self.can_encoder = BertEmbedding(squeeze_strategy='first')
        self.share = share
        
    def forward(self, cid, rid, cid_mask, rid_mask, loss=True):
        batch_size = cid.shape[0]
        assert batch_size > 1, f'[!] batch size must bigger than 1, cause other elements in the batch will be seen as the negative samples'
        if self.share:
            cid_rep = self.encoder(cid, cid_mask)
            rid_rep = self.encoder(rid, rid_mask)
        else:
            cid_rep = self.ctx_encoder(cid, cid_mask)
            rid_rep = self.can_encoder(rid, rid_mask)
        # cid_rep/rid_rep: [B, 768]
        dot_product = torch.matmul(cid_rep, rid_rep.t())  # [B, B]
        # use half for supporting the apex
        mask = to_cuda(torch.eye(batch_size)).half()    # [B, B]
        if loss:
            loss = F.log_softmax(dot_product, dim=-1) * mask
            loss = (-loss.sum(dim=1)).mean()
            return loss
        else:
            return dot_product
    
class BERTBiEncoderAgent(RetrievalBaseAgent):
    
    def __init__(self, multi_gpu, run_mode='train', local_rank=0, kb=True):
        super(BERTBiEncoderAgent, self).__init__(kb=kb)
        try:
            self.gpu_ids = list(range(len(multi_gpu.split(',')))) 
        except:
            raise Exception(f'[!] multi gpu ids are needed, but got: {multi_gpu}')
        self.args = {
            'lr': 5e-5,
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
            'share_bert': True,
        }
        self.vocab = BertTokenizer.from_pretrained(self.args['vocab_file'])
        self.model = BERTBIEncoder(self.args['share_bert'])
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
        self.show_parameters(self.args)
        
    def train_model(self, train_iter, mode='train', recoder=None, idx_=0):
        self.model.train()
        total_loss, batch_num = 0, 0
        pbar = tqdm(train_iter)
        correct, s = 0, 0
        for idx, batch in enumerate(pbar):
            cid, rid, cid_mask, rid_mask = batch
            self.optimizer.zero_grad()
            loss = self.model(cid, rid, cid_mask, rid_mask)    # [B, 2]
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
            clip_grad_norm_(amp.master_params(self.optimizer), self.args['grad_clip'])
            self.optimizer.step()

            total_loss += loss.item()
            batch_num += 1
            
            recoder.add_scalar(f'train-epoch-L{self.args["local_rank"]}-{idx_}/Loss', total_loss/batch_num, idx)
            recoder.add_scalar(f'train-epoch-L{self.args["local_rank"]}-{idx_}/RunLoss', loss.item(), idx)

            pbar.set_description(f'[!] loss: {round(loss.item(), 4)}|{round(total_loss/batch_num, 4)}')
        recoder.add_scalar(f'train-whole-L{self.args["local_rank"]}/Loss', total_loss/batch_num, idx_)
        return round(total_loss / batch_num, 4)
    
    @torch.no_grad()
    def test_model(self, test_iter, mode='test', recoder=None, idx_=0):
        '''there is only one context in the batch, and response are the candidates that are need to reranked; batch size is the self.args['samples']; the groundtruth is the first one.'''
        self.model.eval()
        rest, total_loss, batch_num = [], 0, 0
        pbar = tqdm(test_iter)
        for idx, batch in enumerate(pbar):
            cid, rid, cid_mask, rid_mask = batch
            batch_size = len(rid)
            assert batch_size == self.args['samples'], f'samples attribute must be the same as the batch size'
            dot_product = self.model(cid, rid, cid_mask, rid_mask, loss=False)
            dot_product = F.softmax(dot_product.squeeze(0), dim=-1)    # [B]
            
            preds = dot_product.tolist()    # [10]
            preds = np.argsort(pred, axis=0)[::-1]
            rest.append(([0], preds.tolist()))
        p_1, r2_1, r10_1, r10_2, r10_5, MAP, MRR = cal_ir_metric(rest)
        print(f'[TEST] P@1: {p_1}; R2@1: {r2_1}; R10@1: {r10_1}; R10@2: {r10_2}; R10@5: {r10_5}; MAP: {MAP}; MRR: {MRR}')
        return round(total_loss/batch_num, 4)