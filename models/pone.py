from .header import *

class PONE(nn.Module):

    def __init__(self, lang='zh'):
        super(PONE, self).__init__()
        model = 'bert-base-chinese' if lang == 'zh' else 'bert-base-uncased'
        self.model = BertForSequenceClassification.from_pretrained(
                model_name, num_labels=2)

    @torch.no_grad()
    def forward_(self, inpt):
        '''
        inpt: [batch, seq]
        '''
        attn_mask = generate_attention_mask(inpt)
        output = self.model(input_ids=inpt, attention_mask=attn_mask)
        logits = output[0]    # [batch, 2]
        return logits

    def forward(self, inpt):
        logits = self.forward_(inpt)
        return logits

    def predict(self, inpt):
        '''
        return the positive scores as the final results
        '''
        logits = self.forward_(inpt)    # [batch, 2]
        return logits[:, 1]    # [batch]
    
class PONEAgent(BaseAgent):
    
    '''
    The bert-based automatic evaluation for open-domain conversation chatbot
    mode:
        1. origin: Bert-RUBER
        2. weighted: weighted enhanced Bert-RUBER
        3. positive: positive enhanced Bert-RUBER
        4. pone: combining the weighted and positive mode
    The Bert parameters are not fine-tuned
    '''

    def __init__(self, multi_gpu, run_mode='train', lang='zh'):
        super(PONEAgent, self).__init__()
        try:
            self.gpu_ids = list(range(len(multi_gpu.split(',')))) 
        except:
            raise Exception(f'[!] multi gpu ids are needed, but got: {multi_gpu}')
        self.args = {
                'lr': 1e-5,
                'grad_clip': 3.0,
                'multi_gpu': self.gpu_ids,
                'pad': 0,
                'lang': lang,
        }
        self.model = PONE(lang=lang)
        if torch.cuda.is_available():
            self.model.cuda()
        self.model = DataParallel(self.model, device_ids=self.gpu_ids)
        # bert model is too big, try to use the DataParallel
        self.optimizer = transformers.AdamW(
                self.model.parameters(),
                lr=self.args['lr'])
        self.criterion = nn.CrossEntropyLoss()
        self.show_parameters(self.args)
    
    def train_model_origin(self, train_iter):
        self.model.train()
        total_loss, batch_num = 0, 0
        pbar = tqdm(train_iter)
        correct, s = 0, 0
        for idx, batch in enumerate(pbar):
            # label: [batch]
            cid, label = batch
            self.optimizer.zero_grad()
            output = self.model(cid)    # [batch, 2]
            loss = self.criterion(
                    output, 
                    label.view(-1))
            if mode == 'train':
                loss.backward()
                clip_grad_norm_(self.model.parameters(), self.args['grad_clip'])
                self.optimizer.step()

            total_loss += loss.item()
            batch_num += 1
            
            now_correct = torch.max(F.softmax(output, dim=-1), dim=-1)[1]    # [batch]
            now_correct = torch.sum(now_correct == label).item()
            correct += now_correct
            s += len(label)

            pbar.set_description(f'[!] loss(run/total): {round(loss.item(), 4)}|{round(loss.item(), 4)}; acc: {round(now_correct/len(label), 4)}|{round(correct/s, 4)}')
        print(f'[!] overall loss: {total_loss}; overall acc: {round(correct/s, 4)}')
        return round(total_loss / batch_num, 4)

    def train_model(self, train_iter, mode='origin'):
        if mode == 'origin':
            self.train_model_origin(train_iter)
        elif mode == 'weighted':
            self.train_model_weighted(train_iter)
        elif mode == 'positive':
            self.train_model_positive(train_iter)
        elif mode == 'pone':
            self.train_model_pone(train_iter)
        else:
            raise Exception(f'[!] unknown training mode: {mode}')
