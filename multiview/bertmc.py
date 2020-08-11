from .header import *

class BERTMCF(RetrievalBaseAgent):

    def __init__(self):
        super(BERTMCF, self).__init__(searcher=False)
        self.vocab = BertTokenizer(vocab_file='data/vocab/vocab_small')
        self.model = BERTMCFusion()
        self.pad = 0
        if torch.cuda.is_available():
            self.model.cuda()

    @torch.no_grad()
    def scores(self, msgs, groundtruths, resps):
        '''
        msgs: {context}[SEP]{response}, a batch of the pair of context and response
        '''
        msgs = [[f'{m} [SEP] {g}', f'{m} [SEP] {r}'] for m, g, r in zip(msgs, groundtruths, resps)]    # [B]
        ids = []
        for i, j in msgs:
            ids.append(torch.LongTensor(self.vocab.encode(i)[-512:]))
            ids.append(torch.LongTensor(self.vocab.encode(j)[-512:]))
        ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
        ids = torch.stack(ids.split(2))    # [B, N, S]
        if torch.cuda.is_available():
            ids = ids.cuda()    # [B, N, S]
        output = self.model(ids)    # [B, N]
        output = F.softmax(output, dim=-1)[:, 1]    # [batch] gather the positive scores
        output = output.cpu().tolist()
        return output    # [B]

if __name__ == "__main__":
    model = BERTMCF()
    # run this in the root path: python -m multiview.coherence
    model.load_model('ckpt/zh50w/bertmcf/best.pt')
