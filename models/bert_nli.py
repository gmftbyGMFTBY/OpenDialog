from .header import *

'''
BERT NLI First, then transfering to Retrieval mode 
'''

class BERTNLI(nn.Module):

    def __init__(self):
        super(BERTNLI, self).__init__()
        self.model = BertForSequenceClassification.from_pretrained(
                'bert-base-chinese',
                num_labels=3)

    def forward(self, inpt):
        '''
        inpt: [batch, seq]
        '''
        attn_mask = generate_attention_mask(inpt)
        output = self.model(input_ids=inpt, attention_mask=attn_mask)
        logits = output[0]    # [batch, 3]
        return logits 

class BERTNLIAgent(RetrievalBaseAgent):

    '''
    Support Multi GPU, for example '1,2'
    '''

    def __init__(self, multi_gpu):
        super(BERTNLIAgent, self).__init__()
        # hyperparameters
        try:
            self.gpu_ids = [int(i) for i in multi_gpu.split(',')]
        except:
            raise Exception(f'[!] multi gpu ids are needed, but got: {multi_gpu}')
        self.args = {
                'lr': 3e-5,
                'grad_clip': 3.0,
                'samples': 10,
                'multi_gpu': self.gpu_ids,
                'vocab_file': 'data/vocab/vocab_small',
                'pad': 0,
        }
        # hyperparameters
        self.vocab = BertTokenizer(vocab_file=self.args['vocab_file'])
        self.model = BERTNLI()
        if torch.cuda.is_available():
            self.model.cuda()
        self.model = DataParallel(self.model, device_ids=[0,1,2,3])
        # bert model is too big, try to use the DataParallel
        self.optimizer = transformers.AdamW(
                self.model.parameters(), 
                lr=self.args['lr'])
        self.criterion = nn.CrossEntropyLoss()

        self.show_parameters(self.args)

    def train_model(self, train_iter, mode='train', recoder=None):
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

            pbar.set_description(f'[!] batch: {batch_num}; train loss: {round(loss.item(), 4)}; acc: {round(now_correct/len(label), 4)}|{round(correct/s, 4)}')
        print(f'[!] overall acc: {round(correct/s, 4)}')
        return round(total_loss / batch_num, 4)

    def test_model(self, test_iter, path):
        self.model.eval()
        total_loss, batch_num = 0, 0
        correct, s = 0, 0
        pbar = tqdm(test_iter)
        rest = []
        with torch.no_grad():
            for idx, batch in enumerate(pbar):
                cid, label = batch
                output = self.model(cid)
                loss = self.criterion(output, label.view(-1))
                total_loss += loss.item()
                batch_num += 1

                # output: [batch, 3]
                # only use the positive score as the final score
                output = F.softmax(output, dim=-1)    # [batch, 3]
                # only need the accuarcy to show the performance
                now_correct = torch.max(output, dim=-1)[1]    # [batch]
                now_correct = torch.sum(now_correct == label).item()
                correct += now_correct
                s += len(label)
        print(f'[!] test loss: {round(total_loss/batch_num, 4)}')
        print(f'[TEST] Accuracy: {round(correct/s, 4)}')
        return round(total_loss/batch_num, 4)
