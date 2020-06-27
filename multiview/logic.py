from .header import *

'''
NLI model 计算生成文本和上下文的文本蕴含关系
'''

class LOGIC(RetrievalBaseAgent):

    def __init__(self):
        super(LOGIC, self).__init__(searcher=False)
        self.vocab = BertTokenizer(vocab_file='data/vocab/vocab_small')
        self.model = BERTRetrieval()
        self.pad = 0
        if torch.cuda.is_available():
            self.model.cuda()

    @torch.no_grad()
    def scores(self, msgs, resps):
        '''
        msgs: {context} [SEP] {response}, a batch version[batch can be 1]
        ids: [batch, seq]/[1, seq](batch is 1)
        '''
        msgs = [f'{m} [SEP] {r}' for m, r in zip(msgs, resps)]
        ids = [torch.LongTensor(self.vocab.encode(i)[-300:]) for i in msgs]
        ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
        if torch.cuda.is_available():
            ids = ids.cuda()
        output = self.model(ids)    # [batch, 3]
        output = F.softmax(output, dim=-1)[:, 1]    # [batch]
        output = output.cpu().tolist()    # [list]
        return output

    @torch.no_grad()
    def scores_(self, cid, rid):
        cid = torch.cat((cid, rid), dim=1)
        output = self.model(cid)
        output = F.softmax(output, dim=-1)[:, 1]
        output = output.cpu().tolist()
        return output

if __name__ == "__main__":
    model = LOGIC()
    # in root path, test it with command: python -m multiview.nli
    model.load_model('ckpt/NLI/bertnli/best.pt')
