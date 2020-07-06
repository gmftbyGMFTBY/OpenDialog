from .header import *

class PONE(nn.Module):

    def __init__(self, lang='zh'):
        super(PONE, self).__init__()
        model_name = 'bert-base-chinese' if lang == 'zh' else 'bert-base-uncased'
        self.model = BertForSequenceClassification.from_pretrained(
                model_name, num_labels=1)    # regression

    def forward_(self, inpt):
        '''
        inpt: [batch, seq]
        '''
        attn_mask = generate_attention_mask(inpt)
        output = self.model(input_ids=inpt, attention_mask=attn_mask)
        logits = output[0]    # [batch, 1]
        return logits

    def forward(self, inpt):
        logits = self.forward_(inpt)
        logits = torch.sigmoid(logits.squeeze(1))    # [batch]
        return logits

class PONE_cls(nn.Module):

    def __init__(self, lang='zh'):
        super(PONE_cls, self).__init__()
        model_name = 'bert-base-chinese' if lang == 'zh' else 'bert-base-uncased'
        self.model = BertModel.from_pretrained(model_name)
        self.head = nn.Linear(768, 1)

    @torch.no_grad()
    def forward_(self, inpt):
        '''
        inpt: [batch, seq]
        '''
        attn_mask = generate_attention_mask(inpt)
        output = self.model(input_ids=inpt, attention_mask=attn_mask)[0]
        output = torch.mean(output, dim=1)    # [batch, 768]
        return output

    def forward(self, inpt):
        logits = self.forward_(inpt)
        logits = self.head(logits).squeeze(1)    # [batch]
        logits = torch.sigmoid(logits) 
        return logits

class PONE_bi(nn.Module):

    def __init__(self, lang='zh', dropout=0.5):
        super(PONE_bi, self).__init__()
        model_name = 'bert-base-chinese' if lang == 'zh' else 'bert-base-uncased'
        self.model = BertModel.from_pretrained(model_name)
        self.head = IRHead(768, dropout=dropout)

    @torch.no_grad()
    def forward_(self, inpt):
        attn_mask = generate_attention_mask(inpt)
        output = self.model(input_ids=inpt, attention_mask=attn_mask)[0]    # [batch, seq, 768]
        # reduce_mean for time axis
        output = torch.mean(output, dim=1)    # [batch, 768]
        return output

    def forward(self, src, tgt):
        src = self.forward_(src)    # [batch, 768]
        tgt = self.forward_(tgt)    # [batch, 768]
        score = self.head(src, tgt)    # [batch]
        return score

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
                'lang': lang,
        }
        self.model = PONE_cls(lang=lang)
        if torch.cuda.is_available():
            self.model.cuda()
        if run_mode == 'train':
            self.model = DataParallel(self.model, device_ids=self.gpu_ids)
        self.optimizer = transformers.AdamW(
                self.model.parameters(),
                lr=self.args['lr'])
        # self.criterion = nn.CrossEntropyLoss()
        self.criterion = nn.BCELoss()
        self.show_parameters(self.args)
    
    def train_model(self, train_iter, mode='train'):
        self.model.train()
        total_loss, batch_num = 0, 0
        pbar = tqdm(train_iter)
        correct, s = 0, 0
        for idx, batch in enumerate(pbar):
            # label: [batch]
            cid, label = batch
            label = label.float()
            self.optimizer.zero_grad()
            output = self.model(cid)    # [batch]
            loss = self.criterion(output, label)
            if mode == 'train':
                loss.backward()
                clip_grad_norm_(self.model.parameters(), self.args['grad_clip'])
                self.optimizer.step()

            total_loss += loss.item()
            batch_num += 1

            output = output >= 0.5
            now_correct = torch.sum(output.float() == label).item()
            # now_correct = torch.max(F.softmax(output, dim=-1), dim=-1)[1]    # [batch]
            # now_correct = torch.sum(now_correct == label).item()
            correct += now_correct
            s += len(label)

            pbar.set_description(f'[!] loss(run/total): {round(loss.item(), 4)}|{round(loss.item(), 4)}; acc: {round(now_correct/len(label), 4)}|{round(correct/s, 4)}')
        print(f'[!] overall loss: {total_loss}; overall acc: {round(correct/s, 4)}')
        return round(total_loss / batch_num, 4)

    def show_correlation(self, score1, score2, mode):
        p, pp = pearsonr(score1,score2)
        s, ss = spearmanr(score1, score2)
        p, pp, s, ss = round(p, 5), round(pp, 5), round(s, 5), round(ss, 5)
        print(f'[!] {mode} pearson(p): {p}({pp}); spearman(p): {s}({ss})')

    def test_model(self, test_iter, mode='test'):
        '''
        calculate the human correlation
        '''
        self.model.eval()
        pbar = tqdm(test_iter)
        fluency, coherence, engagement, overall, predscores = {}, {}, {}, {}, []
        for idx, batch in enumerate(pbar):
            cid, scores = batch
            output = self.model(cid)    # [batch]
            predscores.extend(output.tolist())
            for annotations in scores: 
                for aidx, annotation in enumerate(annotations):
                    try:
                        fluency[aidx].append(annotation[0])
                        coherence[aidx].append(annotation[1])
                        engagement[aidx].append(annotation[2])
                        overall[aidx].append(annotation[3])
                    except:
                        fluency[aidx] = [annotation[0]]
                        coherence[aidx] = [annotation[1]]
                        engagement[aidx] = [annotation[2]]
                        overall[aidx] = [annotation[3]]
        # human annotations
        fluency1, fluency2, fluency3 = fluency[0], fluency[1], fluency[2]
        coherence1, coherence2, coherence3 = coherence[0], coherence[1], coherence[2]
        engagement1, engagement2, engagement3 = engagement[0], engagement[1], engagement[2]
        overall1, overall2, overall3 = overall[0], overall[1], overall[2]

        self.show_correlation(fluency1, fluency2, 'Human 1-2 Fluency')
        self.show_correlation(coherence1, coherence2, 'Human 1-2 Coherence')
        self.show_correlation(engagement1, engagement2, 'Human 1-2 Engagement')
        self.show_correlation(overall1, overall2, 'Human 1-2 Overall')
        print('=' * 30)
        self.show_correlation(fluency1, fluency3, 'Human 1-3 Fluency')
        self.show_correlation(coherence1, coherence3, 'Human 1-3 Coherence')
        self.show_correlation(engagement1, engagement3, 'Human 1-3 Engagement')
        self.show_correlation(overall1, overall3, 'Human 1-3 Overall')
        print('=' * 30)
        self.show_correlation(fluency2, fluency3, 'Human 2-3 Fluency')
        self.show_correlation(coherence2, coherence3, 'Human 2-3 Coherence')
        self.show_correlation(engagement2, engagement3, 'Human 2-3 Engagement')
        self.show_correlation(overall2, overall3, 'Human 2-3 Overall')
        print('=' * 30)

        # calculate the correlation (spearman and perason)
        for i in range(3):
            fluency_ = fluency[i]
            coherence_ = coherence[i]
            engagement_ = engagement[i]
            overall_ = overall[i]

            self.show_correlation(fluency_, predscores, 'Fluency')
            self.show_correlation(coherence_, predscores, 'Coherence')
            self.show_correlation(engagement_, predscores, 'Engagement')
            self.show_correlation(overall_, predscores, 'Overall')
            print('=' * 30)
