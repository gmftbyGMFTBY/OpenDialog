from .header import *
from .biencoder import BertEmbedding

'''Deep Hashing for high-efficient ANN search'''

class HashBERTBiEncoderModel(nn.Module):
    
    '''Joint learn the hsahing code and the contextual embedding:
    1. hashing module is the regularization for the bi-encoder module,
    2. bi-encoding module can bring better performance for the hashing code.'''
    
    def __init__(self, hidden_size, hash_code_size, dropout=0.3, lang='zh'):
        super(HashBERTBiEncoderModel, self).__init__()
        self.ctx_encoder = BertEmbedding(lang=lang)
        self.can_encoder = BertEmbedding(lang=lang)
        
        self.hash_code_size = hash_code_size
        
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
        self.mseloss = nn.MSELoss()
        
    def _encode(self, cid, rid, cid_mask, rid_mask):
        cid_rep = self.ctx_encoder(cid, cid_mask)
        rid_rep = self.can_encoder(rid, rid_mask)
        return cid_rep, rid_rep
    
    @torch.no_grad()
    def predict(self, cid, rid, rid_mask):
        batch_size = rid.shape[0]
        cid_rep, rid_rep = self._encode(cid.unsqueeze(0), rid, None, rid_mask)
        cid_rep = cid.squeeze(0)
        # [768]; [B, 768] -> [H]; [B, H]
        ctx_hash_code = torch.sign(self.ctx_hash_encoder(cid_rep))    # [Hash]
        can_hash_code = torch.sign(self.can_hash_encoder(rid_rep))    # [B, Hash]
        matrix = torch.matmul(cid_hash_code, can_hash_code.t())    # [B]
        distance = 0.5 * (self.hash_code_size - matrix)    # hamming distance: ||b_i, b_j||_{H} = 0.5 * (K - b_i^Tb_j); distance: [B]
        return distance
        
    def forward(self, cid, rid, cid_mask, rid_mask):
        '''do we need the dot production loss? In my opinion, the hash loss is the replaction of the dot production loss. But need the experiment results to show it.'''
        batch_size = cid.shape[0]
        assert batch_size > 1, f'[!] batch size must bigger than 1, cause other elements in the batch will be seen as the negative samples'
        cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask)
        
        # Hash function
        ctx_hash_code = self.ctx_hash_encoder(cid_rep)    # [B, Hash]
        can_hash_code = self.can_hash_encoder(rid_rep)    # [B, Hash]
        cid_rep_recons = self.ctx_hash_decoder(ctx_hash_code)    # [B, 768]
        rid_rep_recons = self.can_hash_decoder(can_hash_code)    # [B, 768]
        
        # ===== calculate preserved loss ===== #
        preserved_loss = torch.norm(cid_rep_recons - cid_rep, p=2, dim=1).mean() + torch.norm(rid_rep_recons - rid_rep, p=2, dim=1).mean()
        
        # ===== calculate quantization loss ===== #
        ctx_hash_code_h, can_hash_code_h = torch.sign(ctx_hash_code), torch.sign(can_hash_code)
        quantization_loss = torch.norm(ctx_hash_code - ctx_hash_code_h, p=2, dim=1).mean() + torch.norm(can_hash_code - can_hash_code_h, p=2, dim=1).mean()
        
        # ===== calculate hash loss (hamming distance) ===== #
        matrix = torch.matmul(ctx_hash_code, can_hash_code.t())    # [B, B]
        mask = to_cuda(torch.eye(batch_size)).half()
        size_matrix = torch.ones_like(mask) * self.hash_code_size
        zero_matrix = torch.zeros_like(mask)
        mask = torch.where(mask == 0, size_mask, zero_mask)
        hamming_distance = 0.5 * (self.hash_code_size - matrix)    # hamming distance: ||b_i, b_j||_{H} = 0.5 * (K - b_i^Tb_j); [B, B]
        # use MSELoss, regulazation
        hash_loss = self.mseloss(mask, hamming_diatance)
        acc_num = (torch.softmax(hamming_distance, dim=-1).min(dim=-1)[1] == torch.LongTensor(torch.arange(batch_size)).cuda()).sum().item()
        acc = acc_num / batch_size
        
        loss = preserved_loss + quantization_loss + hash_loss
        return loss, acc, (preserved_loss, quantization_loss, hash_loss)

class HashModelAgent(RetrievalBaseAgent):
    
    def __init__(self, multi_gpu, total_step, run_mode='train', local_rank=0, kb=True, lang='zh'):
        super(HashModelAgent, self).__init__(kb=kb)
        try:
            self.gpu_ids = list(range(len(multi_gpu.split(',')))) 
        except:
            raise Exception(f'[!] multi gpu ids are needed, but got: {multi_gpu}')
        self.args = {
            'lr': 5e-5,
            'lr_': 5e-4,
            'grad_clip': 1.0,
            'multi_gpu': self.gpu_ids,
            'local_rank': local_rank,
            'dropout': 0.1,
            'hidden_size': 512,
            'hash_code_size': 128,
            'lang': lang,
            'total_steps': total_step,
            'warmup_steps': int(0.1 * total_step),
            'samples': 10,
            'amp_level': 'O2',
        }
        
        self.model = HashBERTBiEncoderModel(
            self.args['hidden_size'], 
            self.args['hash_code_size'],
            dropout=self.args['dropout'],
            lang=self.args['lang'],
        )
        if torch.cuda.is_available():
            self.model.cuda()
        if run_mode == 'train':
            self.optimizer = transformers.AdamW(
                [
                    {
                        'params': self.model.ctx_encoder.parameters(),
                    },
                    {
                        'params': self.model.can_encoder.parameters(),
                    },
                    {
                        'params': self.model.ctx_hash_encoder.parameters(), 
                        'lr': self.args['lr_'],
                    },
                    {
                        'params': self.model.ctx_hash_decoder.parameters(), 
                        'lr': self.args['lr_'],
                    },
                    {
                        'params': self.model.can_hash_encoder.parameters(), 
                        'lr': self.args['lr_'],
                    },
                    {
                        'params': self.model.can_hash_decoder.parameters(), 
                        'lr': self.args['lr_'],
                    },
                ], 
                lr=self.args['lr'],
            )
            print(f'[!] set the different learning ratios for hashing module')
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
        total_acc, total_loss, total_p_loss, total_q_loss, total_h_loss, batch_num = 0, 0, 0, 0, 0, 0
        pbar = tqdm(train_iter)
        correct, s = 0, 0
        for idx, batch in enumerate(pbar):
            self.optimizer.zero_grad()
            cid, rid, cid_mask, rid_mask = batch
            loss, acc, (preserved_loss, quantization_loss, hash_loss) = self.model(
                cid, rid, cid_mask, rid_mask
            )
            loss.backward()
            clip_grad_norm_(self.model.parameters(), self.args['grad_clip'])
            self.optimizer.step()

            total_loss += loss.item()
            total_acc += acc
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
            recoder.add_scalar(f'train-epoch-{idx_}/RunAcc', acc, idx)
            recoder.add_scalar(f'train-epoch-{idx_}/Acc', total_acc/batch_num, idx)

            pbar.set_description(f'[!] loss(p|q|hash|t): {round(total_p_loss/batch_num, 4)}|{round(total_q_loss/batch_num, 4)}|{round(total_h_loss/batch_num, 4)}|{round(total_loss/batch_num, 4)}; acc: {round(acc, 4)}|{round(total_acc/batch_num, 3)}')
        
        recoder.add_scalar(f'train-whole/Loss', total_loss/batch_num, idx_)
        recoder.add_scalar(f'train-whole/PreservedLoss', total_p_loss/batch_num, idx_)
        recoder.add_scalar(f'train-whole/QuantizationLoss', total_q_loss/batch_num, idx_)
        recoder.add_scalar(f'train-whole/HashLoss', total_h_loss/batch_num, idx_)
        recoder.add_scalar(f'train-whole/Acc', total_acc/batch_num, idx_)
        return round(total_loss / batch_num, 4)
    
    @torch.no_grad()
    def test_model(self, test_iter, mode='test', recoder=None, idx_=0):
        '''replace the dot production with the hamming distance calculated by the hash encoder'''
        self.model.eval()
        r1, r2, r5, r10, counter, mrr = 0, 0, 0, 0, 0, []
        pbar = tqdm(test_iter)
        for idx, batch in tqdm(list(enumerate(pbar))):                
            cid, rids, rids_mask = batch
            batch_size = len(rids)
            if batch_size != self.args['samples']:
                continue
            hamming_diatance = self.model.predict(cid, rids, rids_mask).cpu()    # [B]
            r1 += (torch.topk(hamming_diatance, 1, dim=-1)[1] == 0).sum().item()
            r2 += (torch.topk(hamming_diatance, 2, dim=-1)[1] == 0).sum().item()
            r5 += (torch.topk(hamming_diatance, 5, dim=-1)[1] == 0).sum().item()
            r10 += (torch.topk(hamming_diatance, 10, dim=-1)[1] == 0).sum().item()
            preds = torch.argsort(hamming_diatance, dim=-1).tolist()    # [B, B]
            # mrr
            hamming_diatance = hamming_diatance.numpy()
            y_true = np.zeros(len(hamming_diatance))
            y_true[0] = 1
            mrr.append(label_ranking_average_precision_score([y_true], [hamming_diatance]))
            counter += 1
            
        r1, r2, r5, r10, mrr = round(r1/counter, 4), round(r2/counter, 4), round(r5/counter, 4), round(r10/counter, 4), round(np.mean(mrr), 4)
        print(f'r1@10: {r1}; r2@10: {r2}; r5@10: {r5}; r10@10: {r10}; mrr: {mrr}')
        
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