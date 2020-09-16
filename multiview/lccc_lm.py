from .header import *

'''
LCCC Chinese Open-domain Dialog Pre-train GPT2 Model for Language Model Evaluation
'''

def top_filtering(logits, top_k=0, top_p=0.0, threshold=-float('Inf'), filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
            threshold: a minimal threshold to keep logits
    """
    assert logits.dim() == 1  # Only work for batch size 1 for now - could update but it would obfuscate a bit the code
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        # Remove all tokens with a probability less than the last token in the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Back to unsorted indices and set them to -infinity
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value

    return logits

class LCCCLM(BaseAgent):
    
    def __init__(self, path, topk, topp, t=0.7):
        super(LCCCLM, self).__init__()
        self.model = OpenAIGPTLMHeadModel.from_pretrained(path)
        self.vocab = BertTokenizer.from_pretrained(path, do_lower_case=True)
        self.topk, self.topp = topk, topp
        self.SPECIAL_TOKENS = ["[CLS]", "[SEP]", "[speaker1]", "[speaker2]"]
        self.temperature = t
        if torch.cuda.is_available():
            self.model.cuda()
    
    def tokenize_(self, obj):
        '''borrow from thu-coai/CDial-GPT'''
        return self.vocab.convert_tokens_to_ids(self.vocab.tokenize(obj))
    
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
    
    def scores(self, msgs, resps, temperature=0.7):
        s = [self.score(m, r, temperature=temperature) for m, r in list(zip(msgs, resps))]
        return s

    @torch.no_grad()
    def score(self, msg, res, temperature=0.7, alpha=0.2):
        self.model.eval()
        inpt_ids = [self.tokenize_(msg)]
        opt_res = self.tokenize_(res)
        special_tokens_ids = self.vocab.convert_tokens_to_ids(self.SPECIAL_TOKENS)
        probs, current_output = [], []
        for res_idx in opt_res:
            instance = self.build_input_from_segments(inpt_ids, current_output, with_eos=False)
            input_ids = torch.LongTensor(instance["input_ids"]).unsqueeze(0).cuda()
            token_type_ids = torch.LongTensor(instance["token_type_ids"]).unsqueeze(0).cuda()
            logits, *_ = self.model(input_ids, token_type_ids=token_type_ids)
            logits = logits[0, -1, :] / temperature
            probs.append(F.softmax(logits, dim=-1)[res_idx].item())
            current_output.append(res_idx)
            if len(instance['input_ids']) >= 512:
                # the easiest way is to break
                break
        # beam search score
        rest = sum(np.log(probs))
        length_norm = (5 + len(opt_res))**alpha/(5 + 1)**alpha
        rest /= length_norm
        return rest

if __name__ == "__main__":
    model = LCCCLM()
    model.load_model('/data/lantian/data/LCCD_GPT')

    
    