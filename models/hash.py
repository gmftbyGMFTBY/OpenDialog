from .header import *
from .biencoder import BERTBiEncoderAgent

'''Deep Hashing for high-efficient ANN search'''

class HashModel(nn.Module):
    
    def __init__(self, hidden_size, hash_code_size, dropout=0.3):
        super(HashModel, self).__init__()
        self.hash_code_size = hash_code_size
        self.loss_weight = loss_weight
        
        self.ctx_hash_encoder = nn.Sequential(
            nn.Linear(768, hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, hash_code_size),
        )
        
        self.ctx_hash_decoder = nn.Sequential(
            nn.Linear(hash_code_size, hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, 768)
        )
        
        self.can_hash_encoder = nn.Sequential(
            nn.Linear(768, hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, hash_code_size),
        )
        
        self.can_hash_decoder = nn.Sequential(
            nn.Linear(hash_code_size, hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, 768)
        )
    
    @torch.no_grad()
    def get_query_hash_code(self, cid_rep):
        '''cid_rep: [B, 768]'''
        hash_code = torch.sign(self.ctx_hash_encoder(cid_rep)).tolist()    # [B, H]
        return hash_code
    
    @torch.no_grad()
    def get_res_hash_code(self, rid_rep):
        '''rid_rep: [B, S]'''
        hash_code = torch.sign(self.can_hash_encoder(rid_rep)).tolist()    # [B, H]
        return hash_code
        
    def forward(self, cid_rep, rid_rep):
        batch_size = cid.shape[0]
        assert batch_size > 1, f'[!] batch size must bigger than 1, cause other elements in the batch will be seen as the negative samples'
        
        # Hash function
        ctx_hash_code = self.ctx_hash_encoder(cid_rep)    # [B, Hash]
        can_hash_code = self.can_hash_encoder(rid_rep)    # [B, Hash]
        cid_rep_recons = self.ctx_hash_decoder(ctx_hash_code)    # [B, 768]
        rid_rep_recons = self.can_hash_decoder(can_hash_code)    # [B, 768]
        
        # ===== calculate preserved loss ===== #
        preserved_loss = torch.norm(cid_rep_recons - cid_rep, p=2, dim=1).sum() + torch.norm(rid_rep_recons - rid_rep, p=2, dim=1).sum()
        
        # ===== calculate quantization loss ===== #
        ctx_hash_code_h, can_hash_code_h = torch.sign(ctx_hash_code), torch.sign(can_hash_code)
        quantization_loss = torch.norm(ctx_hash_code - ctx_hash_code_h, p=2, dim=1).sum() + torch.norm(can_hash_code - can_hash_code_h, p=2, dim=1).sum()
        
        # ===== calculate hash loss (hamming distance) ===== #
        matrix = torch.matmul(ctx_hash_code, can_hash_code.t())    # [B, B]
        hash_loss = (matrix - self.hash_code_size * mask).mean()
        
        loss = preserved_loss + quantization_loss + hash_loss
        return loss, (preserved_loss, quantization_loss, hash_loss)

class HashModelAgent(BaseAgent):
    
    def __init__(self, multi_gpu, run_mode='train', local_rank=0, kb=True):
        super(HashModelAgent, self).__init__(kb=kb)
        try:
            self.gpu_ids = list(range(len(multi_gpu.split(',')))) 
        except:
            raise Exception(f'[!] multi gpu ids are needed, but got: {multi_gpu}')
        self.args = {
            'lr': 5e-4,
            'grad_clip': 1.0,
            'multi_gpu': self.gpu_ids,
            'local_rank': local_rank,
            'dropout': 0.1,
            'hidden_size': 512,
            'hash_code_size': 128,
            'path': 'ckpt/ecommerce/bertirbi/best.pt',
        }
        self.bert_encoder = BERTBiEncoderAgent(multi_gpu, total_step, run_mode='test', local_rank=local_rank, kb=True, model='bertirbi')
        # load the pre-trained parameters of the bertbiencoder model
        self.bert_encoder.load_model(self.args['path'])
        
        self.model = HashModel(self.args['hidden_size'], self.args['hash_code_size'])
        if torch.cuda.is_available():
            self.model.cuda()
        if run_mode == 'train':
            self.optimizer = optim.Adam(
                self.model.parameters(), 
                lr=self.args['lr'],
            )
            self.model = nn.parallel.DistributedDataParallel(
                self.model, device_ids=[local_rank], output_device=local_rank,
                find_unused_parameters=True,
            )
        self.show_parameters(self.args)
        
    def train_model(self, train_iter, mode='train', recoder=None, idx_=0):
        self.model.train()
        total_loss, total_p_loss, total_q_loss, total_h_loss, batch_num = 0, 0, 0, 0, 0
        pbar = tqdm(train_iter)
        correct, s = 0, 0
        for idx, batch in enumerate(pbar):
            self.optimizer.zero_grad()
            cid, rid, cid_mask, rid_mask = batch
            cid_rep, rid_rep = self.bert_encoder.get_embedding(cid, cid_mask, context=True), self.bert_encoder.get_embedding(rid, rid_mask, context=False)
            
            loss, (preserved_loss, quantization_loss, hash_loss) = self.model(cid_rep, rid_rep)
            loss.backward()
            clip_grad_norm_(self.model.parameters(), self.args['grad_clip'])
            self.optimizer.step()

            total_loss += loss.item()
            total_p_loss += preserved_loss.item()
            total_q_loss += quantization_loss.item()
            total_h_loss += hash_loss.item()
            batch_num += 1
            
            recoder.add_scalar(f'train-epoch-{idx_}/PreservedLoss', total_p_loss/batch_num, idx)
            recoder.add_scalar(f'train-epoch-{idx_}/RunPreservedLoss', preserved_loss.item(), idx)
            recoder.add_scalar(f'train-epoch-{idx_}/QuantizationLoss', total_q_loss/batch_num, idx)
            recoder.add_scalar(f'train-epoch-{idx_}/RunQuantizationLoss', quantization_loss.item(), idx)
            recoder.add_scalar(f'train-epoch-{idx_}/HashLoss', total_h_loss/batch_num, idx)
            recoder.add_scalar(f'train-epoch-{idx_}/RunHashLoss', hash_loss.item(), idx)
            recoder.add_scalar(f'train-epoch-{idx_}/Loss', total_loss/batch_num, idx)
            recoder.add_scalar(f'train-epoch-{idx_}/RunLoss', loss.item(), idx)

            pbar.set_description(f'[!] loss(preserved|quantization|hash|total): {round(total_p_loss/batch_num, 4)}|{round(total_q_loss/batch_num, 4)}|{round(total_h_loss/batch_num, 4)}|{round(total_loss/batch_num, 4)}')
        recoder.add_scalar(f'train-whole/Loss', total_loss/batch_num, idx_)
        recoder.add_scalar(f'train-whole/PreservedLoss', total_p_loss/batch_num, idx_)
        recoder.add_scalar(f'train-whole/QuantizationLoss', total_q_loss/batch_num, idx_)
        recoder.add_scalar(f'train-whole/HashLoss', total_h_loss/batch_num, idx_)
        return round(total_loss / batch_num, 4)
        
    @torch.no_grad()
    def predict_hash_code(self, test_iter, recoder=None, idx_=0):
        '''predict and generate the hash code for each sample'''
        self.model.eval()
        pbar = tqdm(test_iter)
        counter, collections = 0, []
        for idx, batch in enumerate(pbar):
            ids, ids_mask = batch
            # ids_rep: [B, 768]
            ids_rep = self.bert_encoder.get_embedding(ids, ids_mask, context=False)
            ids_code = self.model.get_res_hash_code(ids_rep)
            collections.exend(ids_code)
        print(f'[!] obtain {len(collections)} hash codes')
        return collections