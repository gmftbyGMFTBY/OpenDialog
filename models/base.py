from .header import *

'''
Base Agent
'''

class BaseAgent:

    def __init__(self):
        self.history = []
        # trigger utterances
        self.trigger_utterances = load_topic_utterances('data/topic/train.txt')

    def show_parameters(self, args):
        print('========== Model ==========')
        print(self.model)
        print('========== Model ==========')
        print(f'========== Model Parameters ==========')
        for key, value in args.items():
            print(f'{key}: {value}')
        print(f'========== Model Parameters ==========')

    def save_model(self, path):
        '''
        Only save the model (without the module. for DatatParallel)
        '''
        try:
            state_dict = self.model.module.state_dict()
        except:
            state_dict = self.model.state_dict()
        torch.save(state_dict, path)
        print(f'[!] save model into {path}')

    def load_model(self, path):
        '''
        add the `module.` before the state_dict keys if the error are raised,
        which means that the DataParallel are used to load the model
        '''
        state_dict = torch.load(path)
        try:
            self.model.load_state_dict(state_dict)
        except:
            current_module = True if 'module' in [i[0] for i in self.model.state_dict().items()][0] else False
            saved_module = True if 'module' in [i[0] for i in state_dict.items()][0] else False
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if saved_module and not current_module:
                    # save dataparallel and load no dataparallel
                    name = k[7:]
                    new_state_dict[name] = v
                elif not saved_module and current_module:
                    # save no dataparallel and load have dataparallel
                    name = f"module.{k}"
                    new_state_dict[name] = v
                else:
                    # this case will not raise the exception
                    pass
            self.model.load_state_dict(new_state_dict)
        print(f'[!] load model from {path}')

    def train_model(self, train_iter, mode='train'):
        raise NotImplementedError

    def test_model(self, test_iter, path):
        raise NotImplementedError

    def talk(self, topic, msgs):
        '''
        topic: topic of the conversation
        msgs: a string of the conversation context
        '''
        raise NotImplementedError

    def get_res(self, data):
        '''
        SMP-MCC 2020
        data = {
            "group_id": group_id,
            "topic": topic,
            "robot_id": robot_id
            "msgs": [
                {
                    'from_id': robot_id,
                    'msg': msg,
                    'timestamp': timestamp
                },
                ...
            ]
        }
        '''
        topic = data['topic']
        msgs = [i['msg'] for i in data['msgs']]
        msgs = '[SEP]'.join(msgs)
        # feed the model and obtain the result
        res = self.talk(topic, msgs)
        self.history.append(res)
        return res

class RetrievalBaseAgent:

    def __init__(self, searcher=True, kb=True):
        if searcher:
            self.searcher = ESChat('retrieval_database', kb=kb)
        self.history = []    # save the history during the SMP-MCC test

    def show_parameters(self, args):
        print('========== Model ==========')
        print(self.model)
        print('========== Model ==========')
        print(f'========== Model Parameters ==========')
        for key, value in args.items():
            print(f'{key}: {value}')
        print(f'========== Model Parameters ==========')

    def save_model(self, path):
        try:
            state_dict = self.model.module.state_dict()
        except:
            state_dict = self.model.state_dict()
        torch.save(state_dict, path)
        print(f'[!] save model into {path}')
    
    def load_model(self, path):
        '''
        add the `module.` before the state_dict keys if the error are raised,
        which means that the DataParallel(self.model) are used to load the model
        '''
        state_dict = torch.load(path)
        try:
            self.model.load_state_dict(state_dict)
        except:
            current_module = True if 'module' in [i[0] for i in self.model.state_dict().items()][0] else False
            saved_module = True if 'module' in [i[0] for i in state_dict.items()][0] else False
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if saved_module and not current_module:
                    name = k[7:]
                    new_state_dict[name] = v
                elif not saved_module and current_module:
                    name = f"module.{k}"
                    new_state_dict[name] = v
            self.model.load_state_dict(new_state_dict)
        print(f'[!] load model from {path}')

    def train_model(self, train_iter, mode='train'):
        raise NotImplementedError

    def test_model(self, test_iter, path):
        raise NotImplementedError

    def process_utterances(self, topic, msgs):
        '''
        Process the utterances searched by Elasticsearch
        '''
        utterances_ = self.searcher.search(topic, msgs, samples=self.args['talk_samples'])
        utterances_ = [i['response'] for i in utterances_]
        # remove the utterances that in the self.history
        utterances_ = list(set(utterances_) - set(self.history))
        utterances = [f'{msgs} [SEP] {i}' for i in utterances_]
        # 512 length limitations for BERT Module
        ids = [torch.LongTensor(self.vocab.encode(i)[-512:]) for i in utterances_]
        ids = pad_sequence(ids, batch_first=True, padding_value=self.args['pad'])
        if torch.cuda.is_available():
            ids = ids.cuda()
        return utterances_, ids

    def talk(self, topic, msgs):
        '''
        topic: topic of the conversation
        msgs: a string of the conversation context
        '''
        raise NotImplementedError

    def get_res(self, data):
        '''
        SMP-MCC 2020
        data = {
            "group_id": group_id,
            "topic": topic,
            "robot_id": robot_id
            "msgs": [
                {
                    'from_id': robot_id,
                    'msg': msg,
                    'timestamp': timestamp
                },
                ...
            ]
        }
        '''
        topic = data['topic']
        msgs = [i['msg'] for i in data['msgs']]
        msgs = '[SEP]'.join(msgs)
        # feed the model and obtain the result
        res = self.talk(topic, msgs)
        self.history.append(res)
        return res
