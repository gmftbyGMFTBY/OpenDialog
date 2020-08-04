from .header import *

class BERT_MULTIVIEW(RetrievalBaseAgent):
    
    def __init__(self):
        super(BERT_MULTIVIEW, self).__init__(searcher=False)
        self.vocab = BertTokenizer.from_pretrained('bert-base-chinese')
        self.model = BERTMULTIVIEW()
        self.pad = 0
        if torch.cuda.is_available():
            self.model.cuda()

    @torch.no_grad()
    def scores(self, msgs, resps, details=False):
        '''
        msgs: {context}[SEP]{response}, a batch of the pair of context and response
        default aggregation strategy is average, min or max are also needed
        '''
        msgs = [f'{m} [SEP] {r}' for m, r in zip(msgs, resps)]
        ids = [torch.LongTensor(self.vocab.encode(i)[-512:]) for i in msgs]
        ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
        if torch.cuda.is_available():
            ids = ids.cuda()    # [batch, seq]
        outputs = self.model(ids, aspect='null')
        outputs = [F.softmax(output, dim=-1)[:, 1] for output in outputs]    # 5*[batch]
        if details:
            outputs = [output.cpu().tolist() for output in outputs]
            return outputs
        else:
            # combine these scores, average
            output = torch.stack(outputs).min(dim=0)[0]    # [5, batch] -> [batch]
            output = output.cpu().tolist()
            return output    # [batch]

if __name__ == "__main__":
    model = BERT_MULTIVIEW()
    # run this in the root path: python -m multiview.multiview
    model.load_model('ckpt/zh50w/bertretrieval_multiview/best.pt')
