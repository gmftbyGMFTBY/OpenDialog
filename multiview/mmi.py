from .header import *

class MMI(BaseAgent):

    '''
    DialoGPT MMI Model for rerank the generated responses;
    In this work, we use it as a part of the multiview evaluation module

    The code in `GPT2-chitchat` use the loss to represent the MMI scores,
    but it is not good.
    For example, the generated responses is much longer than the original candidates,
    but actually it will obtain the very big loss (small score).

    So, the paper of DialoGPT mentioned that the MMI scores are obtained by 
    P(Source|Hypothesis), so in this code, we use the language model probability to do it.
    '''

    def __init__(self):
        super(MMI, self).__init__()
        self.model_path = 'ckpt/train_generative/gpt2_mmi/best.pt'
        self.vocab = BertTokenizer(vocab_file='data/vocab/vocab_small')
        self.unk = self.vocab.convert_tokens_to_ids('[UNK]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.model = GPT2(
                len(self.vocab), self.unk, self.sep, len(self.vocab), 1.0, 1.0,
                config_path='data/config/model_config_dialogue.json')
        if torch.cuda.is_available():
            self.model.cuda()

    def scores(self, sources, targets):
        s_ = []
        for s, t in zip(sources, targets):
            probs = self.score(s, t)
            probs = np.mean(probs)
            s_.append(probs)
        return s_

    def score(self, source, target):
        c_ids = self.vocab.encode(source)[1:]
        r_ids = self.vocab.encode(target)
        # length
        c_ids_l = len(c_ids)
        ids = r_ids + c_ids
        if len(ids) >= 300:
            return [0.2]
        ids = torch.LongTensor(ids)
        if torch.cuda.is_available():
            ids = ids.cuda()
        output = self.model.model(input_ids=ids)[0]    # [seq, vocab]
        output = F.softmax(output, dim=-1)    # [seq, vocab]
        # obtain the index
        index_x = list(range(len(ids)))[-(c_ids_l+1):-1]
        index_y = c_ids
        assert len(index_x) == len(index_y), f'[!] x and y must have the same length'
        # probs
        probs = output[index_x, index_y].tolist()
        return probs

if __name__ == "__main__":
    pass
