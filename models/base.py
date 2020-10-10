from .header import *

'''
Base Agent
'''

class BaseAgent:

    def __init__(self):
        self.history = []
        # trigger utterances
        # self.trigger_utterances = load_topic_utterances('data/topic/train.txt')

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
        msgs = [i['msg'] for i in data['msgs']]
        msgs = '[SEP]'.join(msgs)
        # feed the model and obtain the result
        res = self.talk(msgs)
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

    def process_utterances(self, topic, msgs, max_len=0):
        '''Process the utterances searched by Elasticsearch; input_ids/token_type_ids/attn_mask'''
        utterances_ = self.searcher.search(msgs, samples=self.args['talk_samples'], topic=topic)
        utterances_ = [i['response'] for i in utterances_]
        # remove the utterances that in the self.history
        utterances_ = list(set(utterances_) - set(self.history))
        
        # construct inpt_ids, token_type_ids, attn_mask
        inpt_ids = self.vocab.batch_encode_plus([msgs] + utterances_)['input_ids']
        context_inpt_ids, responses_inpt_ids = inpt_ids[0], inpt_ids[1:]
        context_token_type_ids = [0] * len(context_inpt_ids)
        responses_token_type_ids = [[1] * len(i) for i in responses_inpt_ids]
        
        # length limitation
        collection = []
        for r1, r2 in zip(responses_inpt_ids, responses_token_type_ids):
            p1, p2 = context_inpt_ids + r1[1:], context_token_type_ids + r2[1:]
            if len(p1) > max_len:
                cut_size = len(p1) - max_len + 1
                p1 = torch.LongTensor([p1[0]] + p1[cut_size:])
                p2 = torch.LongTensor([p2[0]] + p2[cut_size:])
            collection.append((p1, p2))
            
        inpt_ids = [torch.LongTensor(i[0]) for i in collection]
        token_type_ids = [torch.LongTensor(i[1]) for i in collection]
        
        inpt_ids = pad_sequence(inpt_ids, batch_first=True, padding_value=self.args['pad'])
        token_type_ids = pad_sequence(token_type_ids, batch_first=True, padding_value=self.args['pad'])
        attn_mask_index = inpt_ids.nonzero().tolist()
        attn_mask_index_x, attn_mask_index_y = [i[0] for i in attn_mask_index], [i[1] for i in attn_mask_index]
        attn_mask = torch.zeros_like(inpt_ids)
        attn_mask[attn_mask_index_x, attn_mask_index_y] = 1
        
        if torch.cuda.is_available():
            inpt_ids, token_type_ids, attn_mask = inpt_ids.cuda(), token_type_ids.cuda(), attn_mask.cuda()
        return utterances_, inpt_ids, token_type_ids, attn_mask

    def talk(self, msgs, topic=None):
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
        msgs = [i['msg'] for i in data['msgs']]
        msgs = '[SEP]'.join(msgs)
        topic = data['topic'] if 'topic' in data else None
        res = self.talk(msgs, topic=topic)
        self.history.append(res)
        return res
