from .header import *

'''
Traditional text-match retrieval chatbot: Dual-LSTM (2015-acl Ubuntu corpus)
input:
    src: [seq, batch]
    tgt: [seq, batch]
    src_l / tgt_l: [batch]
output:
    score: [batch]

use function is the interface of the Agent
'''

class RetrievalModel(nn.Module):

    def __init__(self, hidden_size, dropout=0.5):
        super(RetrievalModel, self).__init__()
        self.model = IRHead(hidden_size, dropout=dropout)

    def forward(self, src, tgt):
        # src/tgt: [batch, hidden]
        score = self.model(src, tgt)    # [batch]
        return score

class RetrievalAgent:

    def __init__(self):
        # hyperparameters
        self.hidden_size = 768    # for bert
        self.dropout = 0.5 
        self.lr = 1e-3
        self.grad_clip = 3.0
        self.samples = 10    # 1 positive and 9 negative, you can change it
        # hyperparameters

        self.model = RetrievalModel(self.hidden_size, dropout=self.dropout)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.BCELoss()

        if torch.cuda.is_available():
            self.model.cuda()
        # show model
        print('========== Model ===========')
        print(self.model)
        print('========== Model ===========')
        self.show_parameters()

    def show_parameters(self):
        print(f'========== Model Parameters ==========')
        print(f'hidden size: {self.hidden_size}')
        print(f'dropout ratio: {self.dropout}')
        print(f'learning ratio: {self.lr}')
        print(f'grad_clip: {self.grad_clip}')
        print(f'samples: {self.samples}/1+{self.samples-1}-')
        print(f'========== Model Parameters ==========')

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        print(f'[!] save model into {path}')

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        print(f'[!] load model from {path}')

    def train_model(self, train_iter, mode='train'):
        self.model.train()
        total_loss, batch_num = 0, 0
        pbar = tqdm(train_iter)
        correct, s = 0, 0
        for idx, batch in enumerate(pbar):
            cid, rid, label = batch
            self.optimizer.zero_grad()
            output = self.model(cid, rid)
            loss = self.criterion(output, label)

            if mode == 'train':
                loss.backward()
                clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

            total_loss += loss.item()
            batch_num += 1

            cls = (output > 0.5).type_as(label)
            now_correct = torch.sum(cls == label).item()
            correct += now_correct
            s += len(label)

            pbar.set_description(f'[!] batch: {batch_num}; train loss: {round(loss.item(), 4)}; accuracy: {round(now_correct/len(label), 4)}/{round(correct/s, 4)}')
        print(f'[!] overall accuracy: {round(correct/s, 4)}')
        return round(total_loss/batch_num, 4)

    def test_model(self, test_iter):
        self.model.eval() 
        total_loss, batch_num = 0, 0
        pbar = tqdm(test_iter)
        rest = []
        with torch.no_grad():
            for idx, batch in enumerate(pbar):
                cid, rid, label = batch
                output = self.model(cid, rid)
                loss = self.criterion(output, label)
                total_loss += loss.item()
                batch_num += 1

                preds = [i.tolist() for i in torch.split(output, self.samples)]
                labels = [i.tolist() for i in torch.split(label, self.samples)]
                for label, pred in zip(labels, preds):
                    pred = np.argsort(pred, axis=0)[::-1]
                    rest.append(([0], pred.tolist()))
        print(f'[!] test loss: {round(total_loss/batch_num, 4)}')
        p_1, r2_1, r10_1, r10_2, r10_5, MAP, MRR = cal_ir_metric(rest)
        print(f'[TEST] P@1: {p_1}; R2@1: {r2_1}; R10@1: {r10_1}; R10@2: {r10_2}; R10@5: {r10_5}; MAP: {MAP}; MRR: {MRR}')
        return round(total_loss/batch_num, 4)

    def use(self, ctx, rest):
        '''
        ctx: a string of the context
        rest: a list of the string, which holds the responses
        need the BertClient
        '''
        self.model.eval()
        print('[!] make sure the bert-as-service is running')
        self.client = BertClient()
        emb = self.client.encode([ctx] + rest)     # emb: len(rest) + 1
        ctx_emb, res_emb = emb[0], emb[1:]
        ctx_emb = np.tile(ctx_emb, (len(res_emb), 1))
        ctx_emb = torch.tensor(ctx_emb)
        res_emb = torch.tensor(res_emb)
        if torch.cuda.is_available():
            ctx_emb = ctx_emb.cuda()
            res_emb = res_emb.cuda()
        with torch.no_grad():
            score = self.model(ctx_emb, res_emb)    # len(rest)
        return score.cpu().numpy()    # len(rest)
        
