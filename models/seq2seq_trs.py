from .header import *

'''
Transformer seq2seq model
'''

class Transformer(nn.Module):
    def __init__(self, n_vocab, d_model=512, n_head=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1, max_len=512, share_word_embedding=False, pad=0):
        """
        Transformer Arguments:
            n_vocab {int} -- # of vocabulary
        
        Keyword Arguments:
            d_model {int} -- dimension of hidden state (default: {512})
            n_head {int} -- # of heads used in multi-head attention (default: {8})
            num_encoder_layers {int} -- # of transformer encoder layers (default: {6})
            num_decoder_layers {int} -- # of transformer decoder blocks (default: {6})
            dim_feedforward {int} -- dimension of hidden layer of position wise feed forward layer(default: {2048})
            dropout {float} -- dropout rate (default: {0.1})
            max_len {int} -- max input length (default: {512})
            share_word_embedding {bool} -- share word embedding between encoder and decoder
        """
        super(Transformer, self).__init__()
        self.n_vocab = n_vocab
        self.enc_word_embed = nn.Embedding(n_vocab, d_model, padding_idx=pad)
        # self.pos_embed  = nn.Embedding(max_len+1, d_model, padding_idx=pad)
        self.pos_embed = PositionEmbedding(d_model, dropout=dropout, max_len=512)
        if share_word_embedding:
            self.dec_word_embed = self.enc_word_embed
        else:
            self.dec_word_embed = nn.Embedding(n_vocab, d_model, padding_idx=pad)

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead=n_head, dim_feedforward=dim_feedforward, dropout=dropout)
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead=n_head, dim_feedforward=dim_feedforward, dropout=dropout)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)
        
        self.proj = nn.Linear(d_model, self.n_vocab)

    def forward(self, src, trg, 
                src_mask=None, trg_mask=None,
                memory_mask=None, src_key_padding_mask=None,
                trg_key_padding_mask=None, memory_key_padding_mask=None):
        """forward computation for Transformer
        
        Arguments:
            src {torch.LongTensor} -- input mini-batch in shape (L_S, B)
            trg {torch.LongTensor} -- target mini-batch in shape (L_T, B)
        
        Keyword Arguments:
            src_turn {torch.LongTensor} -- turn ids in range (1, T) (default: {None})
        
        Returns:
            torch.Tensor -- logits in shape (L_T, V)
        """
        # src_embed = self.enc_word_embed(src) + self.pos_embed(src)
        # trg_embed = self.dec_word_embed(trg) + self.pos_embed(trg)
        src_embed = self.pos_embed(self.enc_word_embed(src))
        trg_embed = self.pos_embed(self.dec_word_embed(trg))

        memory = self.encoder(
            src_embed,
            mask=src_mask, 
            src_key_padding_mask=src_key_padding_mask,
        )

        output = self.decoder(
            trg_embed, memory,
            tgt_mask=trg_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=trg_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )

        logits = self.proj(output)
        return logits
    
    @torch.no_grad()
    def predict(self, src, src_mask=None, trg_mask=None,
                memory_mask=None, src_key_padding_mask=None,
                trg_key_padding_mask=None, memory_key_padding_mask=None, 
                max_size=0, cls=0, sep=0, topk=0, topp=0.0):
        '''return the trg_generated; shape of the returned tensor is [S, B]'''
        batch_size = src.size(1)
        stop_flag = [False] * batch_size
        src_embed = self.pos_embed(self.enc_word_embed(src))
        memory = self.encoder(
            src_embed,
            mask=src_mask,
            src_key_padding_mask=src_key_padding_mask,
        )
        # construct the trg tensor
        trg = torch.LongTensor([cls] * batch_size).unsqueeze(0)    # [1, B]
        if torch.cuda.is_available():
            trg = trg.cuda()
        for idx in range(1, max_size + 1):
            trg_embed = self.pos_embed(self.dec_word_embed(trg))
            trg_mask = nn.Transformer.generate_square_subsequent_mask(idx, idx)
            if torch.cuda.is_available():
                trg_mask = trg_mask.cuda()
            output = self.decoder(
                trg_embed, memory,
                tgt_mask=trg_mask,
                memory_mask=None,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=memory_key_padding_mask,
            )    # [1, B, E]
            logits = self.proj(output[-1, :, :])    # [B, V]
            logits = top_k_top_p_filtering_batch(logits, top_k=topk, top_p=topp)    # [B, V]
            next_token = torch.multinomial(
                F.softmax(logits, dim=-1),
                num_samples=1,
            ).transpose(0, 1)    # [B, 1] -> [1, B]
            trg = torch.cat([trg, next_token], dim=0)    # [S+1, B]
            
            # stop flag update
            for idx, token_i in enumerate(next_token.squeeze(0)):
                if token_i == sep:
                    stop_flag[idx] = True
            if sum(stop_flag) == len(stop_flag):
                break
        return trg
    
class TransformerAgent(BaseAgent):
    
    '''learning rate setting (1.0 or 2.0): 
    https://forum.opennmt.net/t/why-different-adam-learning-rate-in-transformer-config-file-and-recommend-setting/2606
    '''
    
    def __init__(self, total_steps, multi_gpu, run_mode='train', lang='zh', local_rank=0, vocab=None):
        super(TransformerAgent, self).__init__()
        try:
            self.gpu_ids = list(range(len(multi_gpu.split(','))))
        except:
            raise Exception(f'[!] multi gpu ids are needed, but got: {multi_gpu}')
        if lang == 'zh':
            if vocab is None:
                self.vocab = BertTokenizer.from_pretrained('bert-base-chinese')
            else:
                self.vocab = vocab
        else:
            self.vocab = BertTokenizer.from_pretrained('bert-base-uncased')
        if run_mode == 'train':
            assert total_steps >= 100000, f'[!] Add the epoch up; make sure the total training steps are larger than 100000'
        self.args = {
            'lr': 2.0,
            'grad_clip': 0.5,
            'tgt_len_size': 50,
            'topk': 500,
            'topp': 0.97,
            'multi_gpu': self.gpu_ids,
            'run_mode': run_mode,
            'lang': lang,
            'label_smooth': 0.1,
            'amp_level': 'O2',
            # transformer parameters
            'd_model': 512,
            'n_head': 8,
            'n_enc_layers': 6,
            'n_dec_layers': 6,
            'd_ff': 1024,
            'share_embed': True,
            'dropout': 0.1,
            'warmup_steps': 8000,
            'total_steps': total_steps,
            # transformer paramters
            'local_rank': local_rank,
            'init': 'glorot',
        }
        if vocab:
            self.vocab_size = self.vocab.size
        else:
            self.vocab_size = len(self.vocab)
        if vocab:
            self.unk = self.vocab.vocab.stoi['[UNK]']
            self.sep = self.vocab.vocab.stoi['[SEP]']
            self.cls = self.vocab.vocab.stoi['[CLS]']
            self.pad = self.vocab.vocab.stoi['[PAD]']
        else:
            self.unk = self.vocab.convert_tokens_to_ids('[UNK]')
            self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
            self.cls = self.vocab.convert_tokens_to_ids('[CLS]')
            self.pad = self.vocab.convert_tokens_to_ids('[PAD]')

        self.model = Transformer(
            len(self.vocab),
            d_model=self.args['d_model'],
            n_head=self.args['n_head'],
            num_encoder_layers=self.args['n_enc_layers'],
            num_decoder_layers=self.args['n_dec_layers'],
            dim_feedforward=self.args['d_ff'],
            share_word_embedding=self.args['share_embed'],
            dropout=self.args['dropout'],
        )
        # Glorot init
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        if torch.cuda.is_available():
            self.model.cuda()
        
        self.criterion = CEWithLabelSmoothing(self.vocab_size, label_smoothing=self.args['label_smooth'], ignore_index=self.pad)
        # self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args['lr'])
        self.model, self.optimizer = amp.initialize(
            self.model, 
            self.optimizer, 
            opt_level=self.args['amp_level'],
        )
        self.warmup_scheduler = Noam(
            self.optimizer, self.args['warmup_steps'], self.args['d_model'],
        )
        if self.args['run_mode'] == 'train':
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.args['local_rank']],
                output_device=self.args['local_rank'],
            )
        self.show_parameters(self.args)
        
    def train_model(self, train_iter, mode='train', recoder=None, idx_=0):
        self.model.train()
        total_loss, total_acc, batch_num = 0, [], 0
        pbar = tqdm(train_iter)
        for idx, batch in enumerate(pbar):
            src, trg, trg_mask, src_key_padding_mask, trg_key_padding_mask, memory_key_padding_mask = batch
            self.optimizer.zero_grad()
            logits = self.model(
                src, trg,
                trg_mask=trg_mask,
                src_key_padding_mask=src_key_padding_mask,
                trg_key_padding_mask=trg_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )    # [S, B, V]
            shift_logits = logits[:-1, ..., :].contiguous()
            shift_labels = trg[1:, ...].contiguous()
            loss = self.criterion(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            
            _, preds = shift_logits.max(dim=-1)    # [S, B]
            not_ignore = shift_labels.ne(self.pad)
            num_targets = not_ignore.long().sum().item()    # the number of not pad tokens
            correct = (shift_labels == preds) & not_ignore
            correct = correct.float().sum()
            accuracy = correct / num_targets
            total_acc.append(accuracy.item())
            
            # loss.backward()
            # clip_grad_norm_(self.model.parameters(), self.args['grad_clip'])
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
            clip_grad_norm_(amp.master_params(self.optimizer), self.args['grad_clip'])
        
            self.optimizer.step()
            self.warmup_scheduler.step()
            total_loss += loss.item()
            batch_num += 1
            
            recoder.add_scalar(f'train-epoch-{idx_}/Loss-{self.args["local_rank"]}', total_loss/batch_num, idx)
            recoder.add_scalar(f'train-epoch-{idx_}/RunLoss-{self.args["local_rank"]}', loss.item(), idx)
            recoder.add_scalar(f'train-epoch-{idx_}/RunTokenAcc-{self.args["local_rank"]}', accuracy, idx)
            recoder.add_scalar(f'train-epoch-{idx_}/TokenAcc-{self.args["local_rank"]}', np.mean(total_acc), idx)
            pbar.set_description(f'[!] lr: {round(self.optimizer.param_groups[0]["lr"], 6)}, loss: {round(loss.item(), 4)}|{round(total_loss/batch_num, 4)}, token acc: {round(accuracy.item(), 4)}|{round(np.mean(total_acc), 4)}')
        recoder.add_scalar(f'train-whole/TokenAcc-{self.args["local_rank"]}', np.mean(total_acc), idx_)
        recoder.add_scalar(f'train-whole/Loss-{self.args["local_rank"]}', total_loss/batch_num, idx_)
        return round(total_loss/batch_num, 4)
    
    @torch.no_grad()
    def test_model(self, test_iter, path):
        spliter = '' if self.args['lang'] == 'zh' else ' '
        def filter(x):
            return x.replace('[PAD]', '')
        def filter_generation(x):
            x = filter(x)
            if '[SEP]' in x:
                x = x[:x.index('[SEP]')] + '[SEP]'
            return x
        self.model.eval()
        pbar = tqdm(test_iter)
        with open(path, 'w') as f:
            for batch in pbar:
                src, trg, trg_mask, src_key_padding_mask, trg_key_padding_mask, memory_key_padding_mask = batch
                max_size = min(trg.size(0), self.args['tgt_len_size'])
                trg_generated = self.model.predict(
                    src,
                    src_key_padding_mask=src_key_padding_mask,
                    trg_key_padding_mask=trg_key_padding_mask,
                    memory_key_padding_mask=memory_key_padding_mask,
                    max_size=max_size, cls=self.cls, sep=self.sep,
                    topk=self.args['topk'], topp=self.args['topp'],
                )    # [S, B]
                # write the results
                for src_, trg_, trg_generated_ in zip(
                    src.transpose(0, 1), 
                    trg.transpose(0, 1), 
                    trg_generated.transpose(0, 1), 
                ):
                    trg_generated_ = self.vocab.convert_ids_to_tokens(trg_generated_)
                    trg_generated_ = filter_generation(spliter.join(trg_generated_))

                    src_ = self.vocab.convert_ids_to_tokens(src_)
                    src_ = filter(spliter.join(src_))

                    trg_ = self.vocab.convert_ids_to_tokens(trg_)
                    trg_ = filter(spliter.join(trg_))

                    f.write(f'CTX: {src_}\n')
                    f.write(f'REF: {trg_}\n')
                    f.write(f'TGT: {trg_generated_}\n\n')
        print(f'[!] translate test dataset over, write into {path}')
        # measure the performance
        (b1, b2, b3, b4), ((r_max_l, r_min_l, r_avg_l), (c_max_l, c_min_l, c_avg_l)), (dist1, dist2, rdist1, rdist2), (average, extrema, greedy) = cal_generative_metric(path, lang=self.args['lang'])
        print(f'[TEST] BLEU: {b1}/{b2}/{b3}/{b4}; Length(max, min, avg): {c_max_l}/{c_min_l}/{c_avg_l}|{r_max_l}/{r_min_l}/{r_avg_l}; Dist: {dist1}/{dist2}|{rdist1}/{rdist2}; Embedding(average/extrema/greedy): {average}/{extrema}/{greedy}')
