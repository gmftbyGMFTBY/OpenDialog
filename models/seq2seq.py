from .header import *
from .encoder import *
from .decoder import *

'''
RNN-based Seq2Seq with attention
'''

class Seq2Seq(nn.Module):

    def __init__(self, vocab_size, embed_size, hidden_size, 
                 dropout=0.5, bidirectional=True, n_layers=1):
        super(Seq2Seq, self).__init__()
        self.encoder = GRUEncoder(
                embed_size,
                hidden_size,
                n_layers=n_layers,
                dropout=dropout,
                bidirectional=bidirectional)
        self.decoder = GRUDecoder(
                vocab_size,
                embed_size,
                hidden_size,
                n_layers=n_layers,
                dropout=dropout)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.vocab_size = vocab_size
        self.n_layer = n_layers

    def forward(self, src, tgt, src_l):
        '''
        src: [seq, batch]
        tgt: [seq, batch]
        src_l: [batch]
        '''
        batch_size, max_len = src.shape[1], tgt.shape[0]
        final_opt = torch.zeros(max_len, batch_size, self.vocab_size)
        if torch.cuda.is_available():
            final_opt = final_opt.cuda()

        # context: [seq, batch, hidden]; hiddne: [batch, hidden]
        src = self.embedding(src)    # [seq, batch, embed]
        context, hidden = self.encoder(src, src_l)
        hidden = hidden.repeat(self.n_layer, 1, 1)    # [layer, batch, hidden]
        inpt = tgt[0, :]    # <res> token

        for t in range(1, max_len):
            inpt = self.embedding(inpt)    # [1, batch, embed]
            output, hidden = self.decoder(inpt, hidden, context)
            final_opt[t] = output
            inpt = tgt[t]
        # [max_len, batch, opt_size]
        return final_opt

    def predict(self, src, src_l, max_len):
        '''
        special tokens: <pad>, <unk>, <res>, <eos>, <ctx>, <eou>
        '''
        with torch.no_grad():
            batch_size = src.shape[1]
            final_opt = torch.zeros(max_len, batch_size)
            if torch.cuda.is_available():
                final_opt = final_opt.cuda()
            src = self.embedding(src)    # [seq, batch, embed]
            context, hidden = self.encoder(src, src_l)
            hidden = hidden.repeat(self.n_layer, 1, 1)
            # <res> token
            inpt = torch.zeros(batch_size, dtype=torch.long).fill_(2)
            if torch.cuda.is_available():
                inpt = inpt.cuda()

            for t in range(1, max_len):
                inpt = self.embedding(inpt)    # [batch, embed]
                inpt, hidden = self.decoder(inpt, hidden, context)
                inpt = inpt.topk(1)[1].squeeze()
                final_opt[t] = inpt

            return final_opt    # [max_len, batch]

class Seq2SeqAgent(BaseAgent):

    def __init__(self, vocab_size, vocab):
        super(Seq2SeqAgent, self).__init__()
        try:
            # self.gpu_ids = [int(i) for i in multi_gpu.split(',')]
            self.gpu_ids = list(range(len(multi_gpu.split(','))))
        except:
            raise Exception(f'[!] multi gpu ids are needed, but got: {multi_gpu}')
        assert run_mode in ['train', 'test', 'rerank', 'rerank_ir'], f'[!] running mode must be train or test, but got {run_mode}'
        # hyperparameters
        self.args= {
            'hidden_size': 512,
            'vocab_file': 'bert-base-chinese',
            'embed_size': 300,
            'dropout': 0.5,
            'bidirectional': True,
            'n_layers': 2,
            'grad_clip': 3.0,
            'lr': 1e-4,
            'pad': 0,
            'tgt_len_size': 50,
        }
        self.vocab = BertTokenizer.from_pretrained(self.args['vocab_file'])
        self.args['inpt_size'] = len(self.vocab)
        self.model = Seq2Seq(
                self.args['vocab_size'],
                self.args['embed_size'],
                self.args['hidden_size'],
                dropout=self.args['dropout'],
                bidirectional=self.args['bidirectional'],
                n_layers=self.args['n_layers']
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args['lr'])
        self.criteiron = nn.NLLLoss(ignore_index=self.args['pad'])
        self.scheduler = lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=self.lr_gamma,
            patience=self.patience,
            verbose=True,
            cooldown=0,
            min_lr=self.min_lr)

        if torch.cuda.is_available():
            self.model.cuda()
        self.model = DataParallel(self.model, device_ids=self.gpu_ids)
        self.show_parameters(self.args)

    def train_model(self, train_iter, mode='train', recoder=None):
        self.model.train()
        total_loss, batch_num = 0, 0
        pbar = tqdm(train_iter)
        for idx, batch in enumerate(pbar):
            cid, cid_l, rid, rid_l = batch
            self.optimizer.zero_grad()
            output = self.model(cid, rid, cid_l)
            loss = self.criteiron(
                output[1:].view(-1, self.vocab_size),
                rid[1:].contiguous().view(-1)
            )
            loss.backward()
            clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            total_loss += loss.item()
            batch_num += 1

            pbar.set_description(f'[!] train loss: {round(loss.item(), 4)}; token acc: ')
        return round(total_loss/batch_num, 4)

    def test_model(self, test_iter, path):
        '''
        translate the test dataset, measure the performance
        '''
        def filter(x):
            # x: a list of string; return the final string
            x = list(map(int, x.tolist()))[1:]
            x = self.vocab.idx2toks(x)
            end_ = x.index('<eos>')
            x = x[1:end_]
            return ' '.join(x)

        self.model.eval()
        pbar = tqdm(test_iter)
        with open(path, 'w') as f:
            for idx, batch in enumerate(pbar):
                cid, cid_l, rid, rid_l = batch
                batch_size = cid.shape[1]
                max_size = max(len(rid), self.tgt_len_size)
                output = self.model.predict(cid, cid_l, max_size)
                
                # idx to tokens
                for i in range(batch_size):
                    ctx, ref, tgt = cid[:, i], rid[:, i], output[:, i]
                    ctx_s, ref_s, tgt_s = filter(ctx), filter(ref), filter(tgt)

                    f.write(f'CTX: {ctx_s}\n')
                    f.write(f'REF: {ref_s}\n')
                    f.write(f'TGT: {tgt_s}\n\n')
        print(f'[!] translate test dataset over')
        # measure the performance
        (b1, b2, b3, b4), (dist1, dist2, rdist1, rdist2), (average, extrema, greedy) = cal_generative_metric(path)
        print(f'[TEST] BLEU: {b1}/{b2}/{b3}/{b4}; Dist: {dist1}/{dist2}|{rdist1}/{rdist2}; Embedding(average/extrema/greedy): {average}/{extrema}/{greedy}')
