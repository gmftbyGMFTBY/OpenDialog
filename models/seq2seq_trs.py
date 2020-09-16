from .header import *

'''
Transformer seq2seq model
'''

class Transformer(nn.Module):
    def __init__(self, n_vocab, d_model=512, n_head=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1, max_len=512, share_word_embedding=True, pad=0):
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
        super().__init__()
        self.n_vocab = n_vocab
        self.enc_word_embed = nn.Embedding(n_vocab, d_model, padding_idx=pad)
        self.pos_embed  = nn.Embedding(max_len+1, d_model, padding_idx=pad)
        if share_word_embedding:
            self.dec_word_embed = self.enc_word_embed
        else:
            self.dec_word_embed = nn.Embedding(n_dec_vocab, d_model, padding_idx=pad)

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead=n_head, dim_feedforward=dim_feedforward, dropout=dropout)
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead=n_head, dim_feedforward=dim_feedforward, dropout=dropout)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)
        
        self.proj = nn.Linear(d_model, self.n_vocab)

    def forward(self, src, trg, src_pos, trg_pos, 
                src_mask=None, trg_mask=None,
                memory_mask=None, src_key_padding_mask=None,
                trg_key_padding_mask=None, memory_key_padding_mask=None):
        """forward computation for Transformer
        
        Arguments:
            src {torch.LongTensor} -- input mini-batch in shape (L_S, B)
            trg {torch.LongTensor} -- target mini-batch in shape (L_T, B)
            src_pos {torch.LongTensor} -- input position ids in range (1, L), padded position filled with 0
            trg_pos {torch.LongTensor} -- target position ids in range (1, L), padded position filled with 0
        
        Keyword Arguments:
            src_turn {torch.LongTensor} -- turn ids in range (1, T) (default: {None})
        
        Returns:
            torch.Tensor -- logits in shape (L_T, V)
        """
        src_embed = self.enc_word_embed(src) + self.pos_embed(src_pos)
        trg_embed = self.dec_word_embed(trg) + self.pos_embed(trg_pos)

        memory = self.encoder(src_embed,
                              mask=src_mask,
                              src_key_padding_mask=src_key_padding_mask)

        output = self.decoder(trg_embed, memory,
                              tgt_mask=trg_mask,
                              memory_mask=memory_mask,
                              tgt_key_padding_mask=trg_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)

        logits = self.proj(output)
        return logits
    
    @torch.no_grad()
    def predict(self, src, trg, src_pos, trg_pos,
                src_mask=None, trg_mask=None,
                memory_mask=None, src_key_padding_mask=None,
                trg_key_padding_mask=None, memory_key_padding_mask=None, max_size=0):
        '''
        src: [S, B];
        trg: [1, B];
        '''
        src_embed = self.enc_word_embed(src) + self.pos_embed(src_pos)
        trg_embed = self.dec_word_embed(trg) + self.pos_embed(trg_pos)

        memory = self.encoder(src_embed,
                              mask=src_mask,
                              src_key_padding_mask=src_key_padding_mask)
        for _ in range(max_size):
            output = self.decoder(trg_embed, memory,
                                  tgt_mask=trg_mask,
                                  memory_mask=memory_mask,
                                  tgt_key_padding_mask=trg_key_padding_mask,
                                  memory_key_padding_mask=memory_key_padding_mask)

            logits = self.proj(output)
    
class TransformerAgent(BaseAgent):
    
    def __init__(self, total_steps, multi_gpu, run_mode='train', lang='zh'):
        super(TransformerAgent, self).__init__()
        try:
            self.gpu_ids = list(range(len(multi_gpu.split(','))))
        except:
            raise Exception(f'[!] multi gpu ids are needed, but got: {multi_gpu}')
        self.vocab = BertTokenizer.from_pretrained('bert-base-chinese')
        self.args = {
            'lr': 1e-4,
            'grad_clip': 1.0,
            'tgt_len_size': 50,
            'topk': 200,
            'topp': 0.95,
            'multi_gpu': self.gpu_ids,
            'run_mode': run_mode,
            'lang': lang,
            'amp_level': 'O2',
            # transformer parameters
            'd_model': 768,
            'n_head': 8,
            'n_enc_layers': 6,
            'n_dec_layers': 6,
            'd_ff': 1024,
            'share_embed': True,
            'dropout': 0.2,
            'warmup_steps': 4000,
            'total_steps': total_steps,
        }
        self.vocab_size = len(self.vocab)
        self.unk = self.vocab.convert_tokens_to_ids('[UNK]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')

        self.model = Transformer(
            len(self.vocab),
            self.args['d_model'],
            self.args['n_head'],
            self.args['n_enc_layers'],
            self.args['n_dec_layers'],
            self.args['d_ff'],
            share_word_embedding=self.args['share_embed'],
            dropout=self.args['dropout'],
        )
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad)
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

        # need to obtain the whole iter
        self.warmup_scheduler = transformers.get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.args['warmup_steps'],
            num_training_steps=self.args['total_steps'],
        )
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
            src, trg, src_pos, trg_pos, trg_mask, src_key_padding_mask, trg_key_padding_mask, memory_key_padding_mask = batch
            
            self.optimizer.zero_grad()
            logits = self.model(
                src, trg, src_pos, trg_pos,
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
            
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
            clip_grad_norm_(amp.master_params(self.optimizer), self.args['grad_clip'])
        
            self.optimizer.step()
            self.warmup_scheduler.step()
            total_loss += loss.item()
            batch_num += 1
            
            recoder.add_scalar(f'train-epoch-{idx_}/Loss', total_loss/batch_num, idx)
            recoder.add_scalar(f'train-epoch-{idx_}/RunLoss', loss.item(), idx)
            recoder.add_scalar(f'train-epoch-{idx_}/RunTokenAcc', accuracy, idx)
            recoder.add_scalar(f'train-epoch-{idx_}/TokenAcc', np.mean(total_acc), idx)
            pbar.set_description(f'[!] train loss: {round(loss.item(), 4)}, token acc: {round(accuracy.item(), 4)}')
        recoder.add_scalar(f'train-whole/TokenAcc', np.mean(total_acc), idx_)
        recoder.add_scalar(f'train-whole/Loss', total_loss/batch_num, idx_)
        return round(total_loss/batch_num, 4)
    
    @torch.no_grad()
    def test_model(self, test_iter, path):
        def filter(x):
            return x.replace('[PAD]', '')
        self.model.eval()
        pbar = tqdm(test_iter)
        with open(path, 'w') as f:
            for batch in pbar:
                src, trg, src_pos, trg_pos, trg_mask, src_key_padding_mask, trg_key_padding_mask, memory_key_padding_mask = batch
                max_size = max(trg.size(0), self.args['tgt_len_size'])
                tgt = self.model.predict(c, max_size)
                text = self.vocab.convert_ids_to_tokens(tgt)
                tgt = ''.join(text)

                ctx = self.vocab.convert_ids_to_tokens(c)
                ctx = filter(''.join(ctx))

                ref = self.vocab.convert_ids_to_tokens(r)
                ref = filter(''.join(ref))

                f.write(f'CTX: {ctx}\n')
                f.write(f'REF: {ref}\n')
                f.write(f'TGT: {tgt}\n\n')
        print(f'[!] translate test dataset over, write into {path}')
        # measure the performance
        (b1, b2, b3, b4), ((r_max_l, r_min_l, r_avg_l), (c_max_l, c_min_l, c_avg_l)), (dist1, dist2, rdist1, rdist2), (average, extrema, greedy) = cal_generative_metric(path, lang=self.args['lang'])
        print(f'[TEST] BLEU: {b1}/{b2}/{b3}/{b4}; Length(max, min, avg): {c_max_l}/{c_min_l}/{c_avg_l}|{r_max_l}/{r_min_l}/{r_avg_l}; Dist: {dist1}/{dist2}|{rdist1}/{rdist2}; Embedding(average/extrema/greedy): {average}/{extrema}/{greedy}')
