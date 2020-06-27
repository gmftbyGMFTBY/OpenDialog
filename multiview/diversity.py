from .header import *
from .utils import *

class Distinct:

    '''
    Micro-Distinct: instance level
    Macro-Distinct: corpus level (obtained dialog history)
    '''

    def __init__(self):
        pass

    def filter(self, msg):
        msg = msg.replace('[SEP]', '')
        msg = list(jieba.cut(msg))
        return msg

    def make_corpus(self, batch):
        data = []
        for i in batch:
            data.extend(i)
        return data

    def _micro(self, r):
        return cal_distinct(r)

    def _macro(self, h):
        h = self.make_corpus(h)
        return cal_distinct(h)

    def scores(self, responses, history):
        '''
        :response: a batch of response string
        :history: a batch of history string
        '''
        r = [self.filter(i) for i in responses]
        micro_s = [self._micro(r_) for r_ in r]

        if history:
            h_ = [self.filter(i) for i in history]
            h = []
            for r_ in r:
                h.append(h_ + r_)
            macro_s = [self._macro(h_) for h_ in h]
            s = [(mi + ma) / 2 for mi, ma in zip(micro_s, macro_s)]
        else:
            s = micro_s
        return s

class RepetitionPenalty:

    '''
    ACL 2020 Conversational Graph Grounded Policy Learning for Open-Domain Conversation Generation
    When the generated responses have 60% (threshold that can be set) terms which are the same as the terms in context or the generated responses.
    '''

    def __init__(self, inner_count=3, context_count=3):
        self.ic = inner_count
        self.cc = context_count

    def _repetition_context(self, contexts, responses):
        s = []
        for c, r in zip(contexts, responses):
            c_terms = list(jieba.cut(c))
            r_terms = list(jieba.cut(r))
            if len(r_terms) == 1:
                s.append(0)
                continue
            counter = 0
            for r_item in r_terms:
                if r_item in c_terms:
                    counter += 1
            try:
                ratio = counter / len(r_terms)
            except:
                ratio = 1
            s.append(1-ratio)
        return s

    def _repetition_inner(self, responses):
        '''
        avoid the cases like: '我喜欢吃面条，喜欢吃面条，吃面条'
        y = 1 - x (x is the ratio of the repetition tokens, bigger x lower score)
        '''
        s = []
        for response in responses:
            terms = list(jieba.cut(response))
            terms = Counter(terms)
            values = list(terms.values())
            if len(values) == 1:
                s.append(0)
                continue
            counter = 0
            for v in values:
                if v >= self.ic:
                    counter += v
            ratio = counter / sum(values) 
            s.append(1-ratio)
        return s

    def scores(self, contexts, responses): 
        s1 = self._repetition_context(contexts, responses)
        s2 = self._repetition_inner(responses)
        # mean strategy is not good as the min
        # s = [(s1_+s2_)/2 for s1_, s2_ in zip(s1, s2)]
        s = [min(s1_, s2_) for s1_, s2_ in zip(s1, s2)]
        return s

class Length():

    '''
    Penalty for the short messages for the given conversations
    score = 1 - \frac{1}{len(response)}
    '''

    def __init__(self):
        self.weight_scores = {
                0: 0, 1: 0, 
                2: 0.1, 3: 0.1, 
                4: 0.2, 5: 0.2, 
                6: 0.4, 7: 0.4,
                8: 0.5, 9: 0.5,
                10: 0.6, 11: 0.6,
                12: 0.7, 13: 0.7,
                14: 0.8, 15: 0.8,
                16: 0.9, 17: 0.9}
        self.filter_tokens = [' ', ',', '。', '!', '?', '！', '？', ';', '.', '，', '#', '$', '*', '(', ')', '（', '）', '[', ']', '-', '+', '=', '\t', '\n', '{', '}']

    def _filter_l(self, s):
        s_ = []
        for i in list(s):
            if i not in self.filter_tokens:
                s_.append(i)
        return len(s)

    def _scores(self, l):
        if l > max(self.weight_scores.keys()):
            return 1.0
        else:
            return self.weight_scores[l]

    def scores(self, responses):
        response_length = [self._filter_l(i) for i in responses]
        scores = [self._scores(i) for i in response_length]
        return scores

class NIDF_TF():

    '''
    Inverse Document Frequency for measuring the diversity: range from 0 to 1, 1 means very diverse and 0 means very non-diverse

    Refer to the paper (AAAI 2020):
    Learning from Easy to Complex: Adaptive Multi-curricula Learning for Neural Dialogue Generation
    '''

    def __init__(self):
        self.args = {
                'corpus_path': 'data/zh50w/train_.txt',
                'rest_path': 'ckpt/NIDF_TF/data.pkl',
                'stopwords_path': 'data/stopwords.txt',
                'stopwords': True,
                'factor_tf': 0.5,
                'factor_idf': 0.5,
                }
        self.cutter = thulac.thulac(seg_only=True)
        if os.path.exists(self.args['rest_path']):
            self._load()
        else:
            # generate the IDF matrix
            if self.args['stopwords']:
                self.stopwords = load_stopwords(self.args['stopwords_path'])
            else:
                self.stopwords = None
            self._train()

    def _train(self):
        # read the file and tokenized
        data = load_corpus(self.args['corpus_path'])
        self.whole_doc = len(data)
        self.words, self.idf_count, self.tf_count = obtain_word_idf(data)
        self.idf_count = np.log(self.whole_doc / self.idf_count)
        self.idf_max, self.idf_min = max(self.idf_count), min(self.idf_count)
        print(f'[!] save the words({len(self.words)}) and TF-IDF')
        with open(self.args['rest_path'], 'wb') as f:
            pickle.dump((self.words, self.whole_doc, self.idf_count, self.tf_count), f)

    def _load(self):
        with open(self.args['rest_path'], 'rb') as f:
            self.words, self.whole_doc, self.idf_count, self.tf_count = pickle.load(f)
        self.idf_max, self.idf_min = max(self.idf_count), min(self.idf_count)
        print(f'[!] load IDF data and words from {self.args["rest_path"]}')

    def scores(self, responses, topk=3):
        responses_ = []
        for i in responses:
            i = [j[0] for j in self.cutter.cut(i)]
            i = [j for j in i if j in self.words]
            # i = list(set(i))    # filter the duplicated terms
            responses_.append(i)
        scores = []
        for response in responses_:
            # tf
            p_tf = []
            for w in response:
                index = self.words.index(w)
                ntf = self.tf_count[index]
                p_tf.append(ntf)
            if len(p_tf) == 0:
                p_tf = 0
            else:
                p_tf = np.mean(p_tf)
            # idf
            response = list(set(response))    # duplicated words influence the NIDF performance
            p_idf = []
            for w in response:
                index = self.words.index(w)
                nidf = (self.idf_count[index] - self.idf_min) / (self.idf_max - self.idf_min)
                p_idf.append(nidf)
            if len(p_idf) == 0:
                # scores.append(np.mean(self.idf_count))
                p_idf = 0
            else:
                # average is not appriproate, the long responses will obtain the low scores
                # should average the topk idf(s)
                p_idf = np.mean(sorted(p_idf, reverse=True)[:topk])
            scores.append(self.args['factor_tf'] * p_tf + self.args['factor_idf'] * p_idf)
        return scores

if __name__ == "__main__":
    model = NIDF_TF()
    responses = [
            '哈哈哈',
            '真的吗',
            '我不知道',
            '我不是很清楚',
            '好的',
            '真的是这样么，我不是很清楚',
            ]
    scores = model.scores(responses)
    print(scores)
