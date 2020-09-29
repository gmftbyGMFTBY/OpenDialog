from .header import *

'''
RNN-based Seq2Seq with attention:
Encoder component
inpt: 
    src: [seq, batch, embedding]
    src_l: [batch]
opt:
    context: [seq, batch, hidden]
    hidden: [batch, hidden]
'''

class Attention(nn.Module):

    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.randn(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)

    def forward(self, hidden, context):
        '''
        hidden: [batch, hidden_size]
        context: [seq, batch, hidden_size]

        return the context vector for decoding: [batch, hidden]
        '''
        timestep = context.shape[0]
        h = hidden.repeat(timestep, 1, 1).transpose(0, 1)    # [batch, seq, hidden_size]
        context = context.transpose(0, 1)    # [batch, seq, hidden_size]
        attn_energies = self.score(h, context)    # [batch, seq]
        score = F.softmax(attn_energies, dim=1).unsqueeze(1)    # [batch, 1, seq]
        context = torch.bmm(score, context).squeeze(1)    # [batch, hidden]
        return context

    def score(self, hidden, context):
        '''
        hidden: [batch, seq, hidden]
        context: [batch, seq, hidden]
        '''
        energy = torch.tanh(self.attn(torch.cat([hidden, context], 2)))    # [batch, seq, hidden]
        energy = energy.transpose(1, 2)    # [batch, hidden, seq]
        v = self.v.repeat(context.shape[0], 1).unsqueeze(1)    # [batch, 1, hidden]
        energy = torch.bmm(v, energy)   # [batch, 1, seq]
        return energy.squeeze(1)    # [batch, seq]

class GRUEncoder(nn.Module):

    def __init__(self, embed_size, hidden_size, 
                 n_layers=1, dropout=0.5, bidirectional=True):
        super(GRUEncoder, self).__init__()
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.rnn = nn.GRU(
            embed_size, 
            hidden_size, 
            num_layers=n_layers, 
            dropout=(0 if n_layers == 1 else dropout),
            bidirectional=bidirectional
        )
        self.times = n_layers * 2 if bidirectional else n_layers
        self.hidden_project = nn.Linear(self.times * hidden_size, hidden_size)
        self.init_weight()

    def init_weight(self):
        init.xavier_normal_(self.rnn.weight_hh_l0)
        init.xavier_normal_(self.rnn.weight_ih_l0)
        self.rnn.bias_ih_l0.data.fill_(0.0)
        self.rnn.bias_hh_l0.data.fill_(0.0)

    def forward(self, src, src_l):
        # src: [S, B, E]
        embed = nn.utils.rnn.pack_padded_sequence(src, src_l, enforce_sorted=False)
        output, hidden = self.rnn(embed)
        output, _ = nn.utils.rnn.pad_packed_sequence(output)

        # output: [seq, batch, hidden * bidirectional * nlayer]
        # hidden: [n_layer * bidirectional, batch, hidden]
        if self.bidirectional:
            output = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]
        hidden = hidden.permute(1, 2, 0)    # [batch, hidden, n_layer * bidirectional]
        hidden = hidden.reshape(hidden.shape[0], -1)    # [batch, hidden*...]
        hidden = torch.tanh(self.hidden_project(hidden))     # [batch, hidden]
        return output, hidden

class GRUDecoder(nn.Module):

    def __init__(self, output_size, embed_size, hidden_size,
                 n_layers=2, dropout=0.5):
        super(GRUDecoder, self).__init__()
        self.attention = Attention(hidden_size)
        self.rnn = nn.GRU(
            hidden_size + embed_size,
            hidden_size,
            num_layers=n_layers,
            dropout=(0 if n_layers == 1 else dropout)
        )
        self.opt_layer = nn.Linear(hidden_size*2, output_size)
        self.init_weight()

    def init_weight(self):
        init.xavier_normal_(self.rnn.weight_hh_l0)
        init.xavier_normal_(self.rnn.weight_ih_l0)
        self.rnn.bias_ih_l0.data.fill_(0.0)
        self.rnn.bias_hh_l0.data.fill_(0.0)

    def forward(self, src, last_hidden, context):
        embed = src.unsqueeze(0)    # [1, batch, embed]
        key = last_hidden.sum(axis=0)    # [batch, hidden]
        context_v = self.attention(key, context)    # [batch, hidden]
        context_v = context_v.unsqueeze(0)    # [1, batch, hidden]
        inpt = torch.cat([embed, context_v], 2)    # [1, batch, hidden+embed]
        
        output, hidden = self.rnn(inpt, last_hidden)
        output = output.squeeze(0)

        output = torch.cat([output, context_v.squeeze(0)], 1)     # [batch, hidden*2]
        output = self.opt_layer(output)    # [batch, vocab]
        return output, hidden

class Seq2Seq(nn.Module):

    def __init__(self, vocab_size, embed_size, hidden_size, 
                 dropout=0.5, bidirectional=True, n_layers=1, 
                 cls=0, sep=0, unk=0):
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
        self.cls = cls
        self.sep = sep
        self.unk = unk

    def forward(self, src, tgt, src_l):
        '''src/tgt [seq, batch]; src_l: [batch]'''
        batch_size, max_len = src.shape[1], tgt.shape[0]
        final_opt = torch.zeros(max_len-1, batch_size, self.vocab_size).cuda()

        src = self.embedding(src)    # [seq, batch, embed]
        # context: [seq, batch, hidden]; hiddne: [batch, hidden]
        context, hidden = self.encoder(src, src_l)
        hidden = hidden.repeat(self.n_layer, 1, 1)    # [layer, batch, hidden]
        inpt = tgt[0]    # [B], full of [CLS] tokens

        for t in range(1, max_len):
            inpt = self.embedding(inpt)    # [batch, embed]
            output, hidden = self.decoder(inpt, hidden, context)
            final_opt[t-1] = output
            inpt = tgt[t]
        # [max_len, batch, opt_size]
        return final_opt

    @torch.no_grad()
    def predict(self, src, src_l, max_len):
        batch_size = src.shape[1]
        final_opt = torch.zeros(max_len, batch_size, dtype=torch.long).cuda()
        src = self.embedding(src)    # [seq, batch, embed]
        context, hidden = self.encoder(src, src_l)
        hidden = hidden.repeat(self.n_layer, 1, 1)
        inpt = torch.zeros(batch_size, dtype=torch.long).fill_(self.cls).cuda()
        final_opt[0] = inpt
        
        # early break
        stop_flag = [0] * batch_size

        for t in range(1, max_len):
            inpt = self.embedding(inpt)    # [batch, embed]
            inpt, hidden = self.decoder(inpt, hidden, context)
            # ignore the [UNK] token
            inpt[:, self.unk] = -np.inf
            next_token = torch.multinomial(
                F.softmax(inpt, dim=-1),
                num_samples=1,
            ).squeeze(1)   # [B, 1] -> [B]
            final_opt[t] = next_token
            inpt = next_token
            
            for idx, item in enumerate(next_token):
                if stop_flag[idx] == 0 and item == self.sep:
                    stop_flag[idx] = 1
            if sum(stop_flag) == batch_size:
                break
            
        return final_opt    # [max_len, batch]

class Seq2SeqAgent(BaseAgent):

    def __init__(self, multi_gpu, vocab, run_mode='train', lang='zh', local_rank=0):
        super(Seq2SeqAgent, self).__init__()
        try:
            # self.gpu_ids = [int(i) for i in multi_gpu.split(',')]
            self.gpu_ids = list(range(len(multi_gpu.split(','))))
        except:
            raise Exception(f'[!] multi gpu ids are needed, but got: {multi_gpu}')
        self.args= {
            'hidden_size': 512,
            'vocab_file': 'bert-base-chinese',
            'embed_size': 512,
            'dropout': 0.5,
            'bidirectional': True,
            'n_layers': 2,
            'grad_clip': 3.0,
            'lr': 5e-4,
            'tgt_len_size': 50,
            'run_mode':run_mode,
            'lang': 'zh',
            'local_rank': local_rank,
        }
        
        self.vocab = vocab
        self.args['vocab_size'] = len(self.vocab)
        self.args['pad'] = self.vocab.vocab.stoi['[PAD]']
        self.args['sep'] = self.vocab.vocab.stoi['[SEP]']
        self.args['cls'] = self.vocab.vocab.stoi['[CLS]']
        self.args['unk'] = self.vocab.vocab.stoi['[UNK]']
        
        self.model = Seq2Seq(
            self.args['vocab_size'],
            self.args['embed_size'],
            self.args['hidden_size'],
            dropout=self.args['dropout'],
            bidirectional=self.args['bidirectional'],
            n_layers=self.args['n_layers'],
            cls=self.args['cls'],
            sep=self.args['sep'],
            unk=self.args['unk'],
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args['lr'])
        self.criteiron = nn.CrossEntropyLoss(ignore_index=self.args['pad'])

        if torch.cuda.is_available():
            self.model.cuda()
        if self.args['run_mode'] == 'train':
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.args['local_rank']],
                output_device=self.args['local_rank'],
            )
        self.show_parameters(self.args)

    def train_model(self, train_iter, mode='train', recoder=None, idx_=0):
        self.model.train()
        total_loss, batch_num = 0, 0
        pbar = tqdm(train_iter)
        for idx, batch in enumerate(pbar):
            cid, cid_l, rid, rid_l = batch
            self.optimizer.zero_grad()
            output = self.model(cid, rid, cid_l)
            loss = self.criteiron(
                output.view(-1, self.args['vocab_size']),
                rid[1:].contiguous().view(-1),
            )
            loss.backward()
            clip_grad_norm_(self.model.parameters(), self.args['grad_clip'])
            self.optimizer.step()
            total_loss += loss.item()
            batch_num += 1
            
            recoder.add_scalar(f'train-epoch-{idx_}/Loss', total_loss/batch_num, idx)
            recoder.add_scalar(f'train-epoch-{idx_}/RunLoss', loss.item(), idx)
            recoder.add_scalar(f'train-epoch-{idx_}/PPL', math.exp(total_loss/batch_num), idx)
            recoder.add_scalar(f'train-epoch-{idx_}/RunPPL', math.exp(loss.item()), idx)
            
            pbar.set_description(f'[!] loss: {round(loss.item(), 4)}|{round(total_loss/batch_num, 4)}; ppl: {round(math.exp(loss.item()), 4)}|{round(math.exp(total_loss/batch_num), 4)}')
        recoder.add_scalar(f'train-whole/Loss', total_loss/batch_num, idx_)
        return round(total_loss/batch_num, 4)

    @torch.no_grad()
    def test_model(self, test_iter, path):
        def filter_(x):
            x_ = ''.join(self.vocab.decode(x))
            x_ = x_.replace('[PAD]', '')
            return x_
            
        def filter(x):
            try:
                x_ = ''.join(self.vocab.decode(x))
                if '[SEP]' in x_:
                    x_ = x_[:x_.index('[SEP]')] + '[SEP]'
                return x_   
            except:
                ipdb.set_trace()
        self.model.eval()
        pbar = tqdm(test_iter)
        with open(path, 'w') as f:
            for batch in pbar:
                cid, cid_l, rid, rid_l = batch
                batch_size = cid.shape[1]
                max_size = min(len(rid), self.args['tgt_len_size'])
                output = self.model.predict(cid, cid_l, max_size)
                
                for i in range(batch_size):
                    ctx, ref, tgt = cid[:, i], rid[:, i], output[:, i]
                    ctx_s, ref_s, tgt_s = filter_(ctx), filter_(ref), filter(tgt)

                    f.write(f'CTX: {ctx_s}\n')
                    f.write(f'REF: {ref_s}\n')
                    f.write(f'TGT: {tgt_s}\n\n')
        print(f'[!] translate test dataset over')
        # measure the performance
        (b1, b2, b3, b4), ((r_max_l, r_min_l, r_avg_l), (c_max_l, c_min_l, c_avg_l)), (dist1, dist2, rdist1, rdist2), (average, extrema, greedy) = cal_generative_metric(path, lang=self.args['lang'])
        print(f'[TEST] BLEU: {b1}/{b2}/{b3}/{b4}; Length(max, min, avg): {c_max_l}/{c_min_l}/{c_avg_l}|{r_max_l}/{r_min_l}/{r_avg_l}; Dist: {dist1}/{dist2}|{rdist1}/{rdist2}; Embedding(average/extrema/greedy): {average}/{extrema}/{greedy}')
