from .header import *

'''
COHERENCE(BERT-RUBER) 指标本质上就是一个检索模型
TODO:
    1. 直接加载BERT检索模型作为BERT-RUBER
    2. 使用NLI迁移后训练的BERT检索模型
    3. 人工相似度评价，之后如果结果理想，可以直接使用BERT-RUBER做后期迭代的依据
'''

class COHERENCE(RetrievalBaseAgent):

    '''
    do not train it, just inference for scoring
    '''

    def __init__(self):
        super(COHERENCE, self).__init__(searcher=False)
        self.vocab = BertTokenizer(vocab_file='data/vocab/vocab_small')
        self.model = BERTRetrieval()
        self.pad = 0
        if torch.cuda.is_available():
            self.model.cuda()

    def reload_model(self, state_dict):
        self.model.load_state_dict(state_dict)
        print(f'[!] reload the coherence model parameters')

    @torch.no_grad()
    def scores(self, msgs, resps):
        '''
        msgs: {context}[SEP]{response}, a batch of the pair of context and response
        '''
        msgs = [f'{m} [SEP] {r}' for m, r in zip(msgs, resps)]
        ids = [torch.LongTensor(self.vocab.encode(i)[-300:]) for i in msgs]
        ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
        if torch.cuda.is_available():
            ids = ids.cuda()    # [batch, seq]
        output = self.model(ids)
        output = F.softmax(output, dim=-1)[:, 1]    # [batch] gather the positive scores
        output = output.cpu().tolist()
        return output    # [batch]

    @torch.no_grad()
    def scores_(self, cid, rid):
        ipdb.set_trace()
        cid = torch.cat((cid, rid), dim=1)    # [batch, seq]
        output = self.model(cid)
        output = F.softmax(output, dim=-1)[:, 1]
        output = output.cpu().tolist()
        return output

if __name__ == "__main__":
    model = COHERENCE()
    # run this in the root path: python -m multiview.coherence
    model.load_model('ckpt/zh50w/bertretrieval/best.pt')
