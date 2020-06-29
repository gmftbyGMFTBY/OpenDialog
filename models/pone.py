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
        self.model = PONE(lang=lang)
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
        # calculate the correlation (spearman and perason)
        for i in range(3):
            fluency_ = fluency[i]
            coherence_ = coherence[i]
            engagement_ = engagement[i]
            overall_ = overall[i]

            p, pp = pearsonr(fluency_, predscores)
            s, ss = spearmanr(fluency_, predscores)
            p, pp, s, ss = round(p, 5), round(pp, 5), round(s, 5), round(ss, 5)
            print(f'[!] Fluency pearson(p): {p}({pp}); spearman(p): {s}({ss})')
            p, pp = pearsonr(coherence_, predscores)
            s, ss = spearmanr(coherence_, predscores)
            p, pp, s, ss = round(p, 5), round(pp, 5), round(s, 5), round(ss, 5)
            print(f'[!] Coherence pearson(p): {p}({pp}); spearman(p): {s}({ss})')
            p, pp = pearsonr(engagement_, predscores)
            s, ss = spearmanr(engagement_, predscores)
            p, pp, s, ss = round(p, 5), round(pp, 5), round(s, 5), round(ss, 5)
            print(f'[!] Engagement pearson(p): {p}({pp}); spearman(p): {s}({ss})')
            p, pp = pearsonr(overall_, predscores)
            s, ss = spearmanr(overall_, predscores)
            p, pp, s, ss = round(p, 5), round(pp, 5), round(s, 5), round(ss, 5)
            print(f'[!] Overall pearson(p): {p}({pp}); spearman(p): {s}({ss})')
            print(f'========== ========== ==========')
