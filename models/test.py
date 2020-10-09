from .header import *

'''
ElasticSearch Baseline
'''

class TestAgent(RetrievalBaseAgent):

    def __init__(self, kb=True):
        super(TestAgent, self).__init__()
        self.model = ESChat('retrieval_database', kb=kb)

    def talk(self, msgs, topic=None):
        return self.model.talk(msgs, topic=topic) 

    def obtain_qa_pair(self, msgs, samples=2):
        rest = self.model.search(msgs, samples=samples)
        pairs = []
        for item in rest:
            c, r = item['context'], item['response']
            pairs.append(f'{c} [SEP] {r}')
        return pairs

    def MultiSearch(self, msgs, samples=2):
        '''
        msgs: k query for searching
        DO NOT RETURN THE RESPONSES THAT THE CONTEXT IS THE SAME AS THE QUERY
        '''
        rest = self.model.multi_search(msgs, samples=samples)
        rest = rest['responses']
        data = []
        for item, q in zip(rest, msgs):
            try:
                item = item['hits']['hits']
            except:
                ipdb.set_trace()
            p = []
            for i in item:
                i = i['_source']
                # p.append(f'{i["context"]} [SEP] {i["response"]}')
                p.append(i["response"])
            data.append(p)
        return data

class MultiViewTestAgent(RetrievalBaseAgent):

    def __init__(self, kb=True):
        super(MultiViewTestAgent, self).__init__(kb=kb)
        self.args = {'talk_samples': 128, 'topic_threshold': 0.5}
        from multiview import MultiView
        print(f'[!] MultiView reranker model will be initized')
        self.reranker = MultiView(
                topic=True,
                length=True,
                nidf_tf=True,
                coherence=True,
                fluency=True,
                repetition_penalty=True,
                mmi=True,
                distinct=True,
                mmi_path='ckpt/train_generative/gpt2_mmi/best.pt',
                coherence_path='ckpt/train_retrieval/bertretrieval/best.pt',
                topic_path='ckpt/fasttext/model.bin',
                fluency_path='ckpt/LM/gpt2lm/best.pt',
                )
        print(f'[!] load multiview model over')

    @torch.no_grad()
    def talk(self, topic, msgs):
        '''
        If the topic of the context is not consistant with the given topic.
        Try to use the trgigger sentence as the context.
        '''
        # detect the topic of the msgs, if the msgs's topic is noised
        # use the trigger sentences

        utterances_ = self.searcher.search(topic, msgs, samples=self.args['talk_samples'])
        utterances_ = [i['response'] for i in utterances_]
        utterances_ = list(set(utterances_) - set(self.history))

        msgs_ = len(utterances_) * [msgs]
        topic = len(utterances_) * [topic]
        scores = self.reranker(msgs_, utterances_, topic=topic, history=self.history)
        scores = scores[0]

        index = np.argmax(scores)
        response = utterances_[index]
        return response

if __name__ == "__main__":
    agent = TestAgent()
    print(agent.talk(None, '你今天可真勇猛啊 [SEP] 哈哈哈，哥哥来保护你 [SEP] 妹妹今天可娇弱了 [SEP] 你算个屁'))
    # data = agent.MultiSearch(['公主岭'])
