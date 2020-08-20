from .header import *

'''
Safety Model 本质上使用语言模型和逆语言模型判断生成的句子是不是保守性回复
一个保守性回复具有的性质是:
    1. 很容易从上文推断到下文
    2. 但是很难从下文推断到上文
使用预训练的 GPT2 模型计算语言模型分数
'''

class SAFETY_FLUENCY(BaseAgent):

    '''
    Use the pre-trained Language Model to calculate the fluency scores.
    Use the pre-trained dialog model to calculate the safety scores.
    
    LCCC-GPT Model
    '''

    def __init__(self, path):
        super(SAFETY_FLUENCY, self).__init__()
        self.vocab = BertTokenizer.from_pretrained(path)
        self.model = LCCC(
            path,
            0,    # topk
            0.9,    # topp
        )
        self.SPECIAL_TOKENS = ["[CLS]", "[SEP]", "[speaker1]", "[speaker2]"]
        if torch.cuda.is_available():
            self.model.cuda()
            
    def tokenize_(self, obj):
        '''borrow from thu-coai/CDial-GPT'''
        return self.model.vocab.convert_tokens_to_ids(self.model.vocab.tokenize(obj))
    
    def build_input_from_segments(self, history, response, with_eos=True):
        '''borrow from the thu-coai/CDial-GPT'''
        bos, eos, speaker1, speaker2 = self.vocab.convert_tokens_to_ids(self.SPECIAL_TOKENS)
        sequence = [[bos]] + history + [response + ([eos] if with_eos else [])]
        sequence = [sequence[0]] + [[speaker2 if i % 2 else speaker1] + s
                                    for i, s in enumerate(sequence[1:])]
        instance = {}
        instance["input_ids"] = list(chain(*sequence))
        instance["token_type_ids"] = [bos] + [speaker2 if i % 2 else speaker1 for i, s in
                                              enumerate(sequence[1:])
                                              for _ in s]
        return instance

    def scores(self, msgs, resps):
        fluency_scores, safety_scores = [], []
        for m, r in zip(msgs, resps):
            probs = self.score(m, r)
            # probs, mmi_probs = np.mean(probs), np.mean(mmi_probs)
            probs = np.mean(probs)
            # mmi_probs = np.mean(self.weight[:len(mmi_probs)] * mmi_probs)
            fluency_scores.append(probs)
            # safety_scores.append(mmi_probs)
        return fluency_scores

    @torch.no_grad()
    def score(self, msg, res):
        self.model.eval()
        msgs = [self.tokenize_(msg), self.tokenize_(res)]
        
        c_ids = self.vocab.encode(msg)
        r_ids = self.vocab.encode(res)[1:]    # ignore the [CLS]
        l_r_ids = len(r_ids)
        c_ids = c_ids + r_ids
        if len(c_ids) >= 300:
            return [0.2]
        c_ids = torch.LongTensor(c_ids)
        if torch.cuda.is_available():
            c_ids = c_ids.cuda()
        # gpt2 model
        output = self.model.model(input_ids=c_ids)[0]    # [seq, vocab]
        output = F.softmax(output, dim=-1)
        # index
        index_x = list(range(len(c_ids)))[-(l_r_ids+1):-1]
        index_y = r_ids
        probs = output[index_x, index_y].tolist()
        return probs

if __name__ == "__main__":
    model = SAFETY_FLUENCY()
    model.load_model('ckpt/train_generative/gpt2/best.pt')
