from .header import *

'''
Bert for Non-Autoregressive Generation
'''

class BERTNA(nn.Module):
    
    def __init__(self, config_path, max_size=10):
        super(BERTNA, self).__init__()
        self.model = BertModel.from_pretrained(config_path)
        hidden_size = self.model.config.get_config_dict('bert-base-chinese')[0]['hidden_size']
        vocab_size = self.model.config.get_config_dict('bert-base-chinese')[0]['vocab_size']
        self.head = nn.Linear(
            hidden_size, vocab_size,
        )
        self.max_size = max_size
        
    def forward(self, inpt_ids):
        '''inpt_ids: [B, S]; [MASK] tokens represent the predicted tokens'''
        # [B, S, E]
        outputs = self.model(input_ids=inpt_ids)[0]
        outputs = self.head(outputs)    # [B, S, V]
        return outputs
    
    @torch.no_grad()
    def predict(self, inpt_ids):
        outputs = self.model(input_ids=inpt_ids)[0]
        outputs = F.softmax(self.head(outputs), dim=-1)    # [B, S, V]
        outputs = outputs[:, 1:self.max_size+1, :]    # [B, max_size, V]
        batch_size, vocab_size = outputs.size(0), outputs.size(-1)
        outputs = outputs.reshape(-1, vocab_size)    # [B*max_size, V]
        outputs = torch.multinomial(outputs, 1).squeeze(-1)    # [B*max_size]
        outputs = outputs.view(batch_size, self.max_size)    # [B, max_size]
        return outputs
    
class BERTNAAgent(BaseAgent):
    
    def __init__(self, multi_gpu, run_mode='train', lang='zh'):
        super(BERTNAAgent, self).__init__()
        try:
            self.gpu_ids = list(range(len(multi_gpu.split(','))))
        except:
            raise Exception(f'[!] multi gpu ids are needed, but got: {multi_gpu}')
        self.args = {
            'lr': 1e-4,
            'grad_clip': 1.0,
            'max_size': 10,
            'pad': 0,
            'config_path': 'bert-base-chinese',
            'multi_gpu': self.gpu_ids,
            'run_mode': run_mode,
            'lang': lang,
            'amp_level': 'O2',
        }
        self.model = BERTNA(
            self.args['config_path'],
            max_size=self.args['max_size'],
        )
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=self.args['pad']
        )
        if torch.cuda.is_available():
            self.model.cuda()
        self.optimizer = transformers.AdamW(
            self.model.parameters(), 
            lr=self.args['lr'], 
            correct_bias=True
        )
        self.model, self.optimizer = amp.initialize(
            self.model, 
            self.optimizer, 
            opt_level=self.args['amp_level']
        )
        self.vocab = BertTokenizer.from_pretrained(self.args['config_path'])
        if self.args['run_mode'] == 'train':
            self.model = DataParallel(
                self.model, 
                device_ids=self.gpu_ids
            )
        self.show_parameters(self.args)
        
    def train_model(self, train_iter, mode='train', recoder=None, idx_=0):
        self.model.train()
        total_loss, total_acc, batch_num = 0, [], 0
        pbar = tqdm(train_iter)
        for idx, batch in enumerate(pbar):
            cid, label = batch
            cid, label = cid.cuda(), label.cuda()
            self.optimizer.zero_grad()
            logits = self.model(cid)    # [B, S, V]
            
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = label[..., 1:].contiguous()
            loss = self.criterion(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            _, preds = shift_logits.max(dim=-1)    # [B, S]
            not_ignore = shift_labels.ne(self.args['pad'])
            num_targets = not_ignore.long().sum().item()
            correct = (shift_labels == preds) & not_ignore
            correct = correct.float().sum()
            accuracy = correct / num_targets
            total_acc.append(accuracy.item())
        
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
            clip_grad_norm_(amp.master_params(self.optimizer), self.args['grad_clip'])
            
            self.optimizer.step()
            total_loss += loss.item()
            batch_num += 1
            
            recoder.add_scalar(f'train-epoch-{idx_}/Loss', total_loss/batch_num, idx)
            recoder.add_scalar(f'train-epoch-{idx_}/RunLoss', loss.item(), idx)
            recoder.add_scalar(f'train-epoch-{idx_}/RunTokenAcc', accuracy, idx)
            recoder.add_scalar(f'train-epoch-{idx_}/TokenAcc', np.mean(total_acc), idx)

            pbar.set_description(f'[!] loss: {round(loss.item(), 4)}|{round(total_loss/batch_num, 4)}, token acc: {round(accuracy.item(), 4)}|{round(np.mean(total_acc), 4)}')
        recoder.add_scalar(f'train-whole/TokenAcc', np.mean(total_acc), idx_)
        recoder.add_scalar(f'train-whole/Loss', total_loss/batch_num, idx_)
        return round(total_loss/batch_num, 4)
    
    @torch.no_grad()
    def test_model(self, test_iter, path):
        '''batch version'''
        def filter(x):
            if '[PAD]' in x:
                x = x[:x.index('[PAD]')]
            return x
        self.model.eval()
        pbar = tqdm(test_iter)
        with open(path, 'w') as f:
            for batch in pbar:
                cid, label = batch    # [B, S]; [B, S]
                cid, label = cid.cuda(), label.cuda()
                tgt = self.model.predict(cid)    # [B, max_size]
                for tgt_, ctx_, label_ in zip(tgt, cid, label):
                    tgt_ = self.vocab.convert_ids_to_tokens(tgt_)
                    tgt_ = filter(''.join(tgt_))

                    ctx_ = self.vocab.convert_ids_to_tokens(ctx_)
                    ctx_ = filter(''.join(ctx_))

                    label_ = self.vocab.convert_ids_to_tokens(label_)
                    label_ = filter(''.join(label_))

                    f.write(f'CTX: {ctx_}\n')
                    f.write(f'REF: {label_}\n')
                    f.write(f'TGT: {tgt_}\n\n')
        print(f'[!] translate test dataset over, write into {path}')
