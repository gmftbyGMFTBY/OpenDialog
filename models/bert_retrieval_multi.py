from .header import *

class BERTMULTIVIEW(nn.Module):
    
    '''
    Multi-view for automatic evaluation, retrieval-based dialog system and generation rerank:
    1. Fluency
    2. Coherence
    3. Diversity
    4. Naturalness
    5. Relatedness
    '''

    def __init__(self):
        super(BERTMULTIVIEW, self).__init__()
        self.model = BertModel.from_pretrained('bert-base-chinese')
        
        self.fluency_m = nn.Linear(768, 256)
        self.fluency_head = nn.Linear(256, 2)
        self.coherence_m = nn.Linear(768, 256)
        self.coherence_head = nn.Linear(256, 2)
        self.diversity_m = nn.Linear(768, 256)
        self.diversity_head = nn.Linear(256, 2)
        self.naturalness_m = nn.Linear(768, 256)
        self.naturalness_head = nn.Linear(256, 2)
        self.relatedness_m = nn.Linear(768, 256)
        self.relatedness_head = nn.Linear(256, 2)
        
        self.head = nn.Linear(256, 2)

    def forward(self, inpt, aspect='coherence'):
        if aspect != 'overall':
            with torch.no_grad():
                attn_mask = generate_attention_mask(inpt)
                output = self.model(input_ids=inpt, attention_mask=attn_mask)[0]
                output = torch.mean(output, dim=1)    # [batch, 768]
        else:
            attn_mask = generate_attention_mask(inpt)
            output = self.model(input_ids=inpt, attention_mask=attn_mask)[0]
            output = torch.mean(output, dim=1)    # [batch, 768]

        if aspect == 'coherence':
            coherence_m = torch.tanh(self.coherence_m(output))
            coherence_rest = self.coherence_head(coherence_m)    # [batch, 2]
            return coherence_rest
        elif aspect == 'fluency':
            fluency_m = torch.tanh(self.fluency_m(output))
            fluency_rest = self.fluency_head(fluency_m)    # [batch, 2]
            return fluency_rest
        elif aspect == 'diversity':
            diversity_m = torch.tanh(self.diversity_m(output))
            diversity_rest = self.diversity_head(diversity_m)    # [batch, 2]
            return diversity_rest
        elif aspect == 'naturalness':
            naturalness_m = torch.tanh(self.naturalness_m(output))
            naturalness_rest = self.naturalness_head(naturalness_m)    # [batch, 2]
            return naturalness_rest
        elif aspect == 'relatedness':
            relatedness_m = torch.tanh(self.relatedness_m(output))
            relatedness_rest = self.relatedness_head(relatedness_m)    # [batch, 2]
            return relatedness_rest
        elif aspect == 'overall':
            fluency_m = torch.tanh(self.fluency_m(output))
            coherence_m = torch.tanh(self.coherence_m(output))
            diversity_m = torch.tanh(self.diversity_m(output))
            naturalness_m = torch.tanh(self.naturalness_m(output))
            relatedness_m = torch.tanh(self.relatedness_m(output))
            output = torch.stack(
                [fluency_m, coherence_m, diversity_m, naturalness_m, relatedness_m]
            ).mean(dim=0)    # 5*[batch, 256] -> [5, batch, 256] -> [batch, 256]
            output = self.head(output)    # [batch, 2]
            return output
        else:
            raise Exception(f'[!] target aspect {aspect} is unknown')


class BERTMULTIVIEWAgent(RetrievalBaseAgent):
    
    '''
    Only train with 1 epoch: 1 epoch contains 5 warmup training and 1 fine tuning steps for each aspect.
    '''
    
    def __init__(self, multi_gpu, run_mode='train', lang='zh', kb=False):
        super(BERTMULTIVIEWAgent, self).__init__(kb=kb)
        try:
            self.gpu_ids = list(range(len(multi_gpu.split(','))))
        except:
            raise Exception(f'[!] multi gpu ids are needed, but got: {multi_gpu}')
        self.args = {
                'lr': 1e-5,
                'pad': 0,
                'vocab_file': 'data/vocab/small',
                'talk_samples': 256,
                'multi_gpu': self.gpu_ids,
                'grad_clip': 3.0,
                'warmup': 5,
                'fine_tuning_step': 1,
        }
        self.vocab = BertTokenizer.from_pretrained('bert-base-chinese')
        self.model = BERTMULTIVIEW()
        if torch.cuda.is_available():
            self.model.cuda()
        self.model = DataParallel(self.model, device_ids=self.gpu_ids)
        self.optimizer = transformers.AdamW(
                self.model.parameters(),
                lr=self.args['lr'])
        self.criterion = nn.CrossEntropyLoss()

        self.show_parameters(self.args)
        
    def train_model(self, train_iters, mode='train', recoder=None):
        '''
        Stage 1: warmup 
        Stage 2: fine tuning 5 aspect heads
        '''
        self.model.train()
        
        # stage 1: warm up
        pbar = tqdm(range(self.args['warmup']))
        for i in pbar:
            loss, acc = self.train_model_aspect(train_iters[-1], aspect='overall')
            pbar.set_description(f'[!] warmup epoch {i+1}|{self.args["warmup"]} finish ...')
            
        # stage 2: fine tuning aspect heads
        pbar = tqdm(range(self.args['fine_tuning_step']))
        for i in pbar:
            order = ['coherence', 'fluency', 'diversity', 'naturalness', 'relatedness']
            for aspect, iter_ in tqdm(zip(order, train_iters[:-1])):
                # continue
                print(f'[!] begin train the `{aspect}` negative aspect')
                loss, acc = self.train_model_aspect(iter_, aspect=aspect)
            pbar.set_description(f'[!] fine tuning stage epoch {i+1}|{self.args["fine_tuning_step"]}')
        # return useless value for compatiblilty
        return 1.0

    def train_model_aspect(self, train_iter, aspect='coherence'):
        batch_num = 0
        correct, s, total_loss = 0, 0, 0
        pbar = tqdm(train_iter)
        for batch in pbar:
            cid, label = batch
            self.optimizer.zero_grad()
            output = self.model(cid, aspect=aspect)    # [batch, 2]
            loss = self.criterion(
                    output, 
                    label.view(-1))
            loss.backward()
            clip_grad_norm_(self.model.parameters(), self.args['grad_clip'])
            self.optimizer.step()
                        
            total_loss += loss.item()
            now_correct = torch.max(F.softmax(output, dim=-1), dim=-1)[1]
            now_correct = torch.sum(now_correct == label).item()
            correct += now_correct
            s += len(label)
            batch_num += 1
            
            pbar.set_description(f'[!] aspect: {aspect}; loss: {round(loss.item(), 4)}; acc(run|overall): {round(now_correct/len(label), 4)}|{round(correct/s, 4)}')
        print(f'[!] overall acc: {round(correct/s, 4)}')
        return round(total_loss / batch_num, 4), round(correct/s, 4)
    
    def talk(self, topic, msgs):
        with torch.no_grad():
            # retrieval and process
            utterances_, ids = self.process_utterances(topic, msgs)
            # rerank, ids: [batch, seq]
            outputs = self.model(ids, aspect='overall')    # 4*[batch, 2]
            outputs = [F.softmax(output, dim=-1)[:, 1] for output in outputs]    # 4*[batch]
            # combine these scores
            output = torch.stack(outputs).mean(dim=0)    # [4, batch] -> [batch]
            item = torch.argmax(output).item()
            msg = utterances_[item]
            return msg