from header import *
from utils import *
from data import *

'''
The Dataset Object can handle the single-turn and multi-turn (<eou> seperator) dialog format.
'''

# ========== For LCCC ========== #
SPECIAL_TOKENS = ["[CLS]", "[SEP]", "[speaker1]", "[speaker2]"]
# ========== For LCCC ========== #

class ChineseTokenizer:
    
    '''
    Only for Chinese RNN based model, parameters:
    :corpus: a list of pair (context string, response string)
    '''

    def __init__(self, corpus, n_vocab=50000, min_freq=1):
        self.allowPOS = ['n', 'nr', 'nz', 'PER', 'LOC', 'ORG', 'ns', 'nt', 'nw', 'vn', 's']
        self.topk = 10
        special_tokens = ['[PAD]', '[UNK]', '[CLS]', '[SEP]']
        self.vocab = vocab.Vocab(
            self._build_vocab(corpus),
            max_size=n_vocab,
            min_freq=min_freq, 
            specials=special_tokens,
        )
        assert self.vocab.stoi['[PAD]'] == 0, f'[PAD] id should be 0, but got {self.vocab.stoi["[PAD]"]}'
        print(f'[!] init the vocabulary over, vocab size: {len(self.vocab)}')

    def __len__(self):
        return len(self.vocab)

    @property
    def size(self):
        return len(self.vocab)

    def decode(self, idx_seq, spliter=''):
        '''chinese spliter: ''; english spliter: ' '
        '''
        words = self.idx2toks(idx_seq)
        return spliter.join(words)

    def encode(self, tok_seq, len_size_limit):
        '''Careful about the special tokens'''
        sentences = re.split('(\[SEP\])', tok_seq)
        sep_token = self.vocab.stoi['[SEP]']
        cls_token = self.vocab.stoi['[CLS]']
        idxs = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence == '[SEP]':
                continue
            sentence = list(jieba.cut(sentence))
            sentence = list(map(lambda i: self.vocab.stoi[i] if i in self.vocab.stoi else self.vocab.stoi['[UNK]'], sentence))
            idxs.extend(sentence)
            idxs.append(sep_token)
        idxs = idxs[-(len_size_limit-2):]
        idxs = [cls_token] + idxs
        return idxs

    def idx2toks(self, idx_seq):
        return list(map(lambda i: self.vocab.itos[i], idx_seq))

    def _build_vocab(self, corpus):
        vocab_counter = Counter()
        for context, response in tqdm(corpus):
            c_words = list(jieba.cut(context))
            r_words = list(jieba.cut(response))
            vocab_counter.update(c_words + r_words)
        print(f'[!] whole vocab size: {len(vocab_counter)}')
        return vocab_counter

    def _build_keywords(self, corpus):
        keywords = Counter()
        for dialog in tqdm(corpus):
            for utterance in dialog:
                words = jieba.analyse.extract_tags(
                    utterance, 
                    topK=self.topk, 
                    allowPOS=self.allowPOS
                )
                keywords.update(words)
        print(f'[!] collect {len(keywords)} keywords')
        return keywords

class When2talkDataset(Dataset):

    def __init__(self, path, mode='train', min_length=15, lang='zh', src_len_size=512, tgt_len_size=128):
        if lang == 'zh':
            vocab_file = 'data/vocab/vocab_small'
        else:
            vocab_file = 'data/vocab/vocab_english'
        self.mode = mode
        # tokenizer with addtional tokens
        self.vocab = BertTokenizer(vocab_file=vocab_file)
        additional_tokens = {'additional_special_tokens': ['[USER1]', '[USER2]', '[STP]']}
        self.vocab.add_special_tokens(additional_tokens)
        self.src_len_size, self.tgt_len_size = src_len_size, tgt_len_size
        #
        self.pp_path = f'{os.path.splitext(path)[0]}.pkl'
        if os.path.exists(self.pp_path):
            with open(self.pp_path, 'rb') as f:
                self.data = pickle.load(f)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None
        # 
        data = read_text_data(path)
        self.data = []
        if self.mode in ['train', 'dev']:
            contexts = [when2talk_utils(i) for i in data]
            for sample in tqdm(contexts):
                bundle = dict()
                bundle['context_text'] = sample
                ids = self.vocab.encode(sample)[1:-1]
                if len(ids) < min_length:
                    continue
                ids = ids[-self.src_len_size:]
                bundle['context_id'] = torch.LongTensor(ids)
                self.data.append(bundle)
            self.data = sorted(self.data, key=lambda x: len(x['context_id']))
        else:
            contexts = [when2talk_utils(i) for i in data]
            for sample in tqdm(contexts):
                bundle = dict()
                bundle['context_text'] = sample
                ids = self.vocab.encode(sample)[1:-1]
                user2_token = self.vocab.convert_tokens_to_ids('[USER2]')
                context, response = ids[:ids.index(user2_token) + 1], ids[ids.index(user2_token) + 1:]
                if len(context) < min_length:
                    continue
                context = context[-self.src_len_size:]
                bundle['context_id'] = torch.LongTensor(context)
                bundle['reply_id'] = torch.LongTensor(response)
                self.data.append(bundle)
        print(f'[!] read and process raw data from {path} over')
        # save the data
        with open(self.pp_path, 'wb') as f:
            pickle.dump(self.data, f)
        print(f'[!] save dataset into {self.pp_path}')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]
    
# ========== LCCC-GPT2 ========== #
# For LCCC IR Model, fine tuning the LCCC GPT2 model for retrieval dialog systems
class UNIDataset(Dataset):

    def __init__(self, vocab, path, samples=1, max_history=5, batch_first=True, uni=False):
        self.tokenizer = BertTokenizer.from_pretrained(vocab)
        self.max_history = max_history
        self.pad = self.tokenizer.pad_token_id
        self.batch_first = batch_first
        self.pp_path = f'{os.path.splitext(path)[0]}_uni.pt'
        
        # load the dataset
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
        else:
            with open(path, 'r', encoding='utf-8') as f:
                dataset = read_lccc_data(path, debug=True)
                # ipdb.set_trace()
                responses = [i[-1] for i in dataset]
                dataset = [self.tokenize_(item) for item in tqdm(dataset)]
                self.data = []
                # construct the negative samples and positive samples
                for dialog in tqdm(dataset):
                    bundle = {'context': dialog}
                    # context, response = dialog[:-1], dialog[-1]
                    # negatives = generate_negative_samples(
                    #     response, 
                    #     responses, 
                    #     samples=samples
                    # )
                    # tokenize negative samples
                    # negatives = self.tokenize_(negatives)
                    # for i, r in enumerate([response] + negatives):
                    #     bundle = {
                    #         'context': context + [r],
                    #         'label': 1 if i == 0 else 0,
                    #     }
                    # bundle = {'context': context + [response]}
                    self.data.append(bundle)
                print(f'[!] collect {len(self.data)} samples for training')
            torch.save(self.data, self.pp_path)
            print(f'[!] process the dataset and write it into {self.pp_path}')
            
    def tokenize_(self, obj):
        if isinstance(obj, str):
            return self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(obj))
        return list(self.tokenize_(o) for o in obj)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        history = self.data[index]['context'][-2 * self.max_history:-1]
        resposne = self.data[index]['context'][-1]
        # label = self.data[index]['label']
        return self.process(history, resposne)

    def process(self, history, resposne, with_eos=True):
        bos, eos, speaker1, speaker2 = self.tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
        sequence = [[bos]] + history + [resposne + ([eos] if with_eos else [])]
        sequence = [sequence[0]] + [[speaker2 if i % 2 else speaker1] + s
                                    for i, s in enumerate(sequence[1:])]
        instance = {}
        instance["input_ids"] = list(chain(*sequence))[-512:]
        instance["token_type_ids"] = [bos] + [speaker2 if i % 2 else speaker1 for i, s in
                                              enumerate(sequence[1:])
                                              for _ in s]
        instance["token_type_ids"] = instance["token_type_ids"][-512:]
        # if label == 0:
        #     # negative samples donot do the language model training
        #     instance["lm_labels"] = [-1] * len(instance["input_ids"])
        # else:
        instance["lm_labels"] = ([-1] * sum(len(s) for s in sequence[:-1])) + [-1] + sequence[-1][1:]
        instance["lm_labels"] = instance["lm_labels"][-512:]
        # instance["label"] = label
        return instance;

    def collate(self, batch):
        input_ids = pad_sequence(
            [torch.tensor(instance["input_ids"], dtype=torch.long) for instance in batch],
            batch_first=self.batch_first, padding_value=self.pad
        )
        token_type_ids = pad_sequence(
            [torch.tensor(instance["token_type_ids"], dtype=torch.long) for instance in batch],
            batch_first=self.batch_first, padding_value=self.pad
        )
        # labels = torch.LongTensor([instance['label'] for instance in batch])
        lm_labels = pad_sequence(
            [torch.tensor(instance["lm_labels"], dtype=torch.long) for instance in batch],
            batch_first=self.batch_first, padding_value=-1
        )
        # if torch.cuda.is_available():
        #     input_ids, token_type_ids, lm_labels = input_ids.cuda(), token_type_ids.cuda(), lm_labels.cuda()
        # [B, S]; [B, S]; [B, S]
        return input_ids, token_type_ids, lm_labels
# ========== LCCC-GPT2 ========== #

class GPT2Dataset(Dataset):
    
    '''
    Training GPT2 model doesn't need the target sentence, just training the Language Model
    GPT2 model can leverage the ability for all the pure text information, which is better than Seq2Seq architecture
    '''

    def __init__(self, path, mode='train', lang='zh', min_length=20, src_len_size=512, tgt_len_size=128):
        if lang == 'zh':
            vocab_file = 'data/vocab/vocab_small'
        else:
            vocab_file = 'data/vocab/vocab_english'
        self.mode = mode
        self.pad = '[PAD]'
        self.vocab = BertTokenizer(vocab_file=vocab_file)
        self.pad_id = self.vocab.convert_tokens_to_ids(self.pad)
        self.src_len_size, self.tgt_len_size = src_len_size, tgt_len_size
        self.pp_path = f'{os.path.splitext(path)[0]}.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None 

        data = read_text_data(path)
        self.data = []
        if self.mode in ['train', 'dev']:
            contexts = [' [SEP] '.join(sample) for sample in data]
            for sample in tqdm(contexts):
                bundle = dict()
                bundle['context_text'] = sample
                ids = self.vocab.encode(sample)
                if len(ids) < min_length:
                    continue
                bundle['context_id'] = ids[-self.src_len_size:]
                self.data.append(bundle)
            self.data = sorted(self.data, key=lambda x: len(x['context_id']))
        else:
            contexts, responses = [], []
            for sample in data:
                contexts.append(' [SEP] '.join(sample[:-1]))
                responses.append(sample[-1])
            for c, r in tqdm(list(zip(contexts, responses))):
                bundle = dict()
                bundle['context_text'] = c
                bundle['reply_text'] = r
                ids = self.vocab.encode(c)
                bundle['context_id'] = ids[-self.src_len_size:]
                ids = self.vocab.encode(r)
                bundle['reply_id'] = ids[:self.tgt_len_size]
                self.data.append(bundle)
        print(f'[!] read and process raw data from {path} over')

    def save_pickle(self):
        torch.save(self.data, self.pp_path)
        print(f'[!] save dataset into {self.pp_path}')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]
    
    def collate(self, batch):
        if self.mode in ['train', 'dev']:
            ctx = [torch.LongTensor(i['context_id']) for i in batch]
            random.shuffle(ctx)
            ctx = pad_sequence(ctx, batch_first=True, padding_value=self.pad_id)
            if torch.cuda.is_available():
                ctx = ctx.cuda()
            return ctx
        else:
            # assert len(batch) == 1, f'[!] batch must be 1, but got {len(batch)}'
            # ctx, res = torch.LongTensor(batch[0]['context_id']), torch.LongTensor(batch[0]['reply_id'])
            # if torch.cuda.is_available():
            #     ctx, res = ctx.cuda(), res.cuda()
            # return ctx, res
            
            # NOTE: BATCH VERSION, PAD IN THE LEFT;
            max_len = max([len(i['context_id']) for i in batch])
            ctx = torch.LongTensor([[self.pad_id] * (max_len - len(i['context_id'])) + i['context_id'] for i in batch])
            position_ids = torch.LongTensor([[0] * (max_len - len(i['context_id'])) + list(range(len(i['context_id']))) for i in batch])
            attn_mask_index = ctx.nonzero().tolist()
            attn_mask_index_x, attn_mask_index_y = [i[0] for i in attn_mask_index], [i[1] for i in attn_mask_index]
            attn_mask = ctx.clone()
            attn_mask[attn_mask_index_x, attn_mask_index_y] = 1
            res = [i['reply_id'] for i in batch]
            if torch.cuda.is_available():
                ctx = ctx.cuda()
                attn_mask = attn_mask.cuda()
                position_ids = position_ids.cuda()
            return ctx, attn_mask, position_ids, res

class MultiGPT2Dataset(Dataset):

    '''
    GPT2 Dataset with the multiple retrieval samples
    '''

    def __init__(self, path, mode='train', vocab_file='data/vocab/vocab_small',
                 src_len_size=512, tgt_len_size=128, retrieval_size=2):
        self.mode = mode
        self.pad = '[PAD]'
        self.vocab = BertTokenizer(vocab_file=vocab_file)
        self.pad_id = self.vocab.convert_tokens_to_ids(self.pad)
        self.src_len_size, self.tgt_len_size = src_len_size, tgt_len_size

        self.data = []
        data = read_csv_data(path)

        if self.mode in ['train', 'dev']:
            contexts, retrieval_list = [], []
            for sample in data:
                c = sample[0].split('<eou>')
                c = [i.strip() for i in c]
                c.append(sample[1])
                contexts.append(c)
                # retrieval list
                retrieval_list.append(sample[2:2+retrieval_size])
            for context, retrieval in tqdm(list(zip(contexts, retrieval_list))):
                bundle = dict()
                bundle['context_text'] = ''.join(context)
                ids = [self.vocab.cls_token_id]
                for utterance in context:
                    ids.extend([self.vocab.convert_tokens_to_ids(word) for word in utterance])
                    ids.append(self.vocab.sep_token_id)
                # length size of the context
                ids = ids[-self.src_len_size:]
                bundle['context_id'] = torch.LongTensor(ids)
                # retrieval list
                bundle['retrieval_list_text'] = retrieval
                retrieval_list_ = []
                for i in retrieval:
                    ids = [self.vocab.cls_token_id]
                    ids.extend([self.vocab.convert_tokens_to_ids(word) for word in i])
                    ids.append(self.vocab.sep_token_id)
                    ids = ids[:self.src_len_size]
                    retrieval_list_.append(torch.LongTensor(ids))
                bundle['retrieval_list'] = retrieval_list_
                self.data.append(bundle)
        else:
            contexts, responses, retrieval_list = [], [], []
            for sample in data:
                c = sample[0].split('<eou>')
                c = [i.strip() for i in c]
                contexts.append(c)
                responses.append(sample[1])
                # retrieval list
                retrieval_list.append(sample[2:2+retrieval_size])
            for c, r, r_ in tqdm(list(zip(contexts, responses, retrieval_list))):
                bundle = dict()
                bundle['context_text'] = ''.join(c)
                bundle['reply_text'] = r
                bundle['retrievl_list_text'] = r_
                # ids
                ids = [self.vocab.cls_token_id]
                for utterance in c:
                    ids.extend([self.vocab.convert_tokens_to_ids(word) for word in utterance])
                    ids.append(self.vocab.sep_token_id)
                ids = ids[-self.src_len_size:]
                bundle['context_id'] = torch.LongTensor(ids)
                ids = [self.vocab.cls_token_id]
                ids.extend([self.vocab.convert_tokens_to_ids(word) for word in r])
                ids.append(self.vocab.sep_token_id)
                ids = ids[:self.tgt_len_size]
                bundle['reply_id'] = torch.LongTensor(ids)
                # retrieval ids
                retrieval_list_ = []
                for i in r_:
                    ids = [self.vocab.cls_token_id]
                    ids.extend([self.vocab.convert_tokens_to_ids(word) for word in i])
                    ids.append(self.vocab.sep_token_id)
                    ids = ids[:self.src_len_size]
                    retrieval_list_.append(torch.LongTensor(ids))
                bundle['retrieval_list'] = retrieval_list_
                self.data.append(bundle)
        print(f'[!] read and process raw dara from {path} over')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

class DialogDataset(Dataset):

    '''
    Construct the dataset, use once for one epoch
    Tokenizer is the function, default jieba.cut; You can also set is the list for no tokenization
    '''

    def __init__(self, path, mode='train', vocab=None, tokenizer=jieba.cut,
                 n_vocab=50000, src_len_size=100, tgt_len_size=20):
        self.mode = mode
        self.tokenizer = tokenizer
        self.src_len_size = src_len_size
        self.tgt_len_size = tgt_len_size
        # load data
        data = read_csv_data(path)
        responses, contexts = [], []
        for sample in tqdm(data):
            responses.append(list(self.tokenizer(sample[1])))
            rc, c = [], sample[0].split('<eou>')
            for utterance in c:
                rc.extend(list(self.tokenizer(utterance.strip())))
                rc.append('<eou>')
            rc = rc[:-1]
            contexts.append(rc)
        print(f'[!] read raw data from {path} over')
        # process the dataset
        if mode == 'train':
            self.vocab = vocabulary(
                    (contexts, responses), 
                    n_vocab=n_vocab)
        else:
            assert vocab, 'vocab not the NoneType for test/dev mode'
            self.vocab = vocab
        self.data = []
        # init the data
        for c, r in zip(contexts, responses):
            bundle = dict()
            bundle['context_text'] = ' '.join(c)
            bundle['reply_text'] = ' '.join(r)
            bundle['context_id'] = torch.LongTensor(self.vocab.toks2idx(c, self.src_len_size))
            bundle['reply_id'] = torch.LongTensor(self.vocab.toks2idx(r, self.tgt_len_size))
            bundle['context_l'] = bundle['context_id'].shape[0]
            bundle['reply_l'] = bundle['reply_id'].shape[0]
            self.data.append(bundle)
        print(f'[!] {mode} dataset init over, size: {len(self.data)}')
        print(f'[!] example:')
        example = random.choice(self.data)
        print(f'CTX: {example["context_text"]}')
        print(f'REF: {example["reply_text"]}')

    def __getitem__(self, i):
        bundle = self.data[i]
        cid, cid_l, rid, rid_l = bundle['context_id'], \
                bundle['context_l'], bundle['reply_id'], bundle['reply_l']
        return cid, cid_l, rid, rid_l

    def __len__(self):
        return len(self.data)

class BERTNLIDataset(Dataset):

    '''
    BERT NLI Datset for Chinese
    '''

    def __init__(self, path, max_len=300, vocab_file='data/vocab/vocab_small'):
        data = read_json_data(path)
        self.vocab = BertTokenizer(vocab_file=vocab_file)
        self.max_len = max_len
        self.pp_path = f'{os.path.splitext(path)[0]}.pkl'
        if os.path.exists(self.pp_path):
            with open(self.pp_path, 'rb') as f:
                self.data = pickle.load(f)
            print(f'[!] load preprocessed file from {self.pp_path}')
            # Dataset object must return None
            return None 
        self.data = []
        d_ = []
        label_map = {'contradiction': 0, 'neutral': 1, 'entailment': 2}
        for i in data:
            s1, s2, label = i['sentence1'], i['sentence2'], i['gold_label']
            d_.append((s1, s2, label))
        for item in tqdm(d_):
            bundle = {}
            s1, s2, label = item
            s = f'{s1} [SEP] {s2}'
            sid = self.vocab.encode(s)
            bundle['sid'] = torch.LongTensor(sid)
            bundle['label'] = label_map[label] 
            self.data.append(bundle)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        return bundle

    def save_pickle(self):
        with open(self.pp_path, 'wb') as f:
            pickle.dump(self.data, f)
        print(f'[!] save dataset into {self.pp_path}')

class BERTLOGICDataset(Dataset):

    '''
    BERT LOGIC Dataset: similar with the BERTIRDataset
    The negative samples are chosen by the IR systems, which have the high semantic coherence but low logic coherence.

    The whole `train_retrieval` corpus is huge, only use 500000 samples
    '''

    def __init__(self, path, mode='train', max_len=300, samples=1, vocab_file='data/vocab/vocab_small'):
        self.mode = mode
        self.max_len = max_len
        # data = read_csv_data(path)
        data = read_text_data(path)
        data = random.sample(data, 500000)
        # context and response are all the negative samples 
        contexts = [i[0] for i in data]
        self.vocab = BertTokenizer(vocab_file=vocab_file)
        self.pp_path = f'{os.path.splitext(path)[0]}_logic.pkl'
        if os.path.exists(self.pp_path):
            with open(self.pp_path, 'rb') as f:
                self.data = pickle.load(f)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None
        self.data = []
        self.max_len = max_len 
        self.es = Elasticsearch() 
        # collect the data samples
        d_ = []
        with tqdm(total=len(data)) as pbar:
            idx, batch_size = 0, 1000
            while idx < len(data):
                contexts = [i[0] for i in data[idx:idx+batch_size]]
                responses = [i[1] for i in data[idx:idx+batch_size]]
                negatives = generate_logic_negative_samples(
                        contexts, self.es, "retrieval_chatbot", 
                        samples=samples)
                for each in zip(contexts, responses, negatives):
                    d_.append((each[0], [each[1]] + each[2]))
                idx += batch_size
                pbar.update(batch_size)
        if mode in ['train', 'dev']:
            # concatenate the context and the response
            for item in tqdm(d_):
                context, response = item
                context_id = self.vocab.encode(context)
                for idx, r in enumerate(response):
                    bundle = dict()
                    rid = self.vocab.encode(r)
                    bundle['context_id'] = context_id + rid[1:]
                    bundle['label'] = 1 if idx == 0 else 0
                    self.data.append(bundle)
        else:
            for item in tqdm(d_):
                context, response = item
                context_id = self.vocab.encode(context)
                res_ids = [self.vocab.encode(i) for i in response]
                bundle = dict()
                bundle['context_id'] = context_id
                bundle['reply_id'] = res_ids
                bundle['label'] = [1] + [0] * samples
                self.data.append(bundle)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        if self.mode in ['train', 'dev']:
            ids = torch.LongTensor(bundle['context_id'][-self.max_len:])
        else:
            ids = []
            for i in range(len(bundle['reply_id'])):
                p = bundle['context_id'] + bundle['reply_id'][i][1:]
                ids.append(torch.LongTensor(p[-self.max_len:]))
        return ids, bundle['label']
    
    def save_pickle(self):
        with open(self.pp_path, 'wb') as f:
            pickle.dump(self.data, f)
        print(f'[!] save dataset into {self.pp_path}')

class BERTIRMultiDataset(Dataset):

    '''
    training samples (positive:negative): 1:1
    test samples (positive:negative) 1:9

    turn_size controls the turn_size of the multi-turn conversations
    '''

    def __init__(self, path, mode='train', max_len=300, samples=9, turn_size=3, vocab_file='data/vocab/vocab_small'):
        self.mode = mode
        data = read_text_data(path)
        responses = [i[-1] for i in data]
        self.vocab = BertTokenizer.from_pretrained('bert-base-chinese')
        self.pp_path = f'{os.path.splitext(path)[0]}_multiir.pkl'
        if os.path.exists(self.pp_path):
            with open(self.pp_path, 'rb') as f:
                self.data = pickle.load(f)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None
        self.data = []
        sep_id = self.vocab.convert_tokens_to_ids('[SEP]')
        self.max_len = max_len
        # collect the samples
        d_ = []
        for context, response in data:
            negative = generate_negative_samples(response, responses, samples=1)
            d_.append((context, [response] + negative))
        if mode in ['train', 'dev']:
            for contexts, responses in tqdm(d_):
                # recode the [SEP] index after tokenize
                if contexts.count('[SEP]') < turn_size:
                    continue
                contexts_id = self.vocab.encode(contexts)
                for idx, r in enumerate(responses):
                    bundle = dict()
                    rid = self.vocab.encode(r)[1:]    # without [CLS]
                    ids = contexts_id + rid
                    if len(ids) > 512:
                        continue
                    bundle['ids'] = ids
                    bundle['label'] = 1 if idx == 0 else 0
                    bundle['turn_length'] = bundle['ids'].count(sep_id)
                    sep_index = (np.array(ids) == sep_id).astype(np.int).nonzero()[0]
                    sep_chunk_size, last_sep = [], 0 
                    for sep_idx in sep_index:
                        sep_chunk_size.append(sep_idx - last_sep + 1)
                        last_sep = sep_idx + 1
                    bundle['sep_index'] = sep_chunk_size
                    self.data.append(bundle)
        else:
            for item in tqdm(d_):
                contexts, responses = item
                contexts_id = [self.vocab.encode(context)[-self.max_len:] for context in contexts]
                res_ids = [self.vocab.encode(i)[-self.max_len] for i in responses]
                bundle = dict()
                bundle['ids'] = contexts_id
                bundle['replys_id'] = res_ids
                bundle['label'] = [1] + [0] * samples
                bundle['turn_length'] = len(bundle['ids']) + 1
                self.data.append(bundle)
        self.data = sorted(self.data, key=lambda i:i['turn_length'])
        print(f'[!] read the processed raw data from {path} over')

    def __len__(self):
        return len(self.data)

    def save_pickle(self):
        with open(self.pp_path, 'wb') as f:
            pickle.dump(self.data, f)
        print(f'[!] save the dataset into {self.pp_path}')

    def __getitem__(self, i):
        return self.data[i]

class BERTIRMultiDataLoader:

    def __init__(self, data, shuffle=True, batch_size=16):
        self.data = data
        self.data_size = len(data)
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.lengths = [i['turn_length'] for i in self.data.data]
        self.index, self.pad = 0, 0

    def __iter__(self):
        return self

    def __len__(self):
        return self.data_size

    def __next__(self):
        if self.index >= self.data_size:
            self.index = 0
            raise StopIteration
        else:
            idx, start1 = self.index, self.lengths[self.index]
            for l in self.lengths[self.index:self.index+self.batch_size]:
                if l != start1:
                    break
                idx += 1
            batch = self.data[self.index:idx]    # batch*[turn, seq]
            if self.shuffle:
                random.shuffle(batch)
            self.index = idx
            if self.data.mode in ['train', 'dev']:
                # construct the tensor
                # batch: batch*[turn, seq] -> turn*[batch, seq] with the [PAD]
                ids = [torch.LongTensor(i['ids']) for i in batch]
                ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)   # [batch, seq]
                sep_index = [i['sep_index'] for i in batch]
                labels = torch.LongTensor([i['label'] for i in batch])
                # rest: turn_size*[batch, seq]; labels: [batch]
                if torch.cuda.is_available():
                    ids = ids.cuda()
                    labels = labels.cuda()
                return ids, labels, sep_index
            else:
                rest, turn_size = [], len(batch[0])
                contexts, responses, labels = [item['ids'] for item in batch], [item['replys_ids'] for item in batch], [item['label'] for item in batch]
                sentences, labels = [], []
                for i in range(len(batch)):
                    for r in range(len(responses)):
                        item = contexts[i] + [responses[i][j]]
                        sentences.append(item)
                    labels.extend(batch[i]['label'])
                # sentences: batch*samples; labels: batch*samples
                rest = []
                for i in range(turn_size):
                    n_batch = [torch.LongTensor(item[i]) for item in sentences]
                    n_batch = pad_sequence(n_batch, batch_first=True, padding_value=self.pad)
                    if torch.cuda.is_available():
                        n_batch = n_batch.cuda()
                    rest.append(n_batch)
                if torch.cuda.is_available():
                    labels = torch.LongTensor(labels).cuda()
                return rest, labels
            
class BERTMCDataset(Dataset):
    
    def __init__(self, path, mode='train', src_min_length=20, tgt_min_length=15, 
                 max_len=300, samples=1, vocab_file='data/vocab/vocab_small', 
                 model_type='mc', harder=False):
        self.mode = mode
        self.max_len = max_len 
        data = read_text_data(path)
        responses = [i[1] for i in data]
        self.vocab = BertTokenizer.from_pretrained('bert-base-chinese')
        
        if mode == 'test':
            # load the test dataset generated by the BERTIRDataset directly
            self.pp_path = f'{os.path.splitext(path)[0]}_hard.pkl'
        else:
            self.pp_path = f'{os.path.splitext(path)[0]}_{model_type}.pkl'
        if os.path.exists(self.pp_path):
            with open(self.pp_path, 'rb') as f:
                self.data = pickle.load(f)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None
        self.data, d_ = [], []
        
        for i in tqdm(data):
            context, response = i[0], i[1]
            negative = generate_negative_samples(response, responses, samples=samples)
            d_.append((context, [response] + negative))
        
        if mode in ['train', 'dev']:
            for context, responses in tqdm(d_):
                response, negative = responses
                context_id = self.vocab.encode(context)
                response_id = self.vocab.encode(response)
                negative_id = self.vocab.encode(negative)
                choice1 = context_id + response_id[1:]
                choice2 = context_id + negative_id[1:]
                bundle = dict()
                if random.random() < 0.5:
                    bundle['ids'] = [choice1[-self.max_len:], choice2[-self.max_len:]]
                    bundle['label'] = 0
                else:
                    bundle['ids'] = [choice2[-self.max_len:], choice1[-self.max_len:]]
                    bundle['label'] = 1
                self.data.append(bundle)
        else:
            for context, response in tqdm(d_):
                context_id = self.vocab.encode(context)
                # delete the [CLS] token of the response sequence for combining
                ids = [self.vocab.encode(r) for r in response]
                bundle = dict()
                bundle['context_id'] = context_id
                bundle['reply_id'] = ids
                bundle['label'] = [1] + [0] * samples
                self.data.append(bundle)
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        if self.mode in ['train', 'dev']:
            return bundle
        else:
            ids = []
            for i in range(len(bundle['reply_id'])):
                p = bundle['context_id'] + bundle['reply_id'][i][1:]
                ids.append(torch.LongTensor(p[-self.max_len:]))
            return ids
    
    def save_pickle(self):
        with open(self.pp_path, 'wb') as f:
            pickle.dump(self.data, f)
        print(f'[!] save dataset into {self.pp_path}')
        
class BERTIRBIDataset(Dataset):
    
    '''test mode batch size must be 1'''
    
    def __init__(self, path, mode='train', max_len=300, samples=9):
        self.mode = mode
        self.max_len = max_len
        data = read_text_data(path)
        self.vocab = BertTokenizer.from_pretrained('bert-base-chinese')
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.pp_path = f'{os.path.splitext(path)[0]}_irbi.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None
        self.data = []
        if mode in ['train', 'dev']:
            d_ = [(context, response) for context, response in data]
            for context, response in tqdm(d_):
                item = self.vocab.batch_encode_plus([context, response])
                cid, cid_mask = item['input_ids'][0], item['attention_mask'][0]
                rid, rid_mask = item['input_ids'][1], item['attention_mask'][1]
                cid, cid_mask = self._length_limit(cid, cid_mask)
                rid, rid_mask = self._length_limit(rid, rid_mask)
                self.data.append({
                    'cid': cid,
                    'rid': rid,
                    'cid_mask': cid_mask,
                    'rid_mask': rid_mask,
                })
        else:
            d_ = []
            responses = [i[1] for i in data]
            for i in tqdm(data):
                context, response = i[0], i[1]
                negative = generate_negative_samples(
                    response, responses, samples=samples
                )
                d_.append((context, [response] + negative))
            for context, response in tqdm(d_):
                item = self.vocab.batch_encode_plus([context] + response)
                cid, cid_mask = item['input_ids'][0], item['attention_mask'][0]
                cid, cid_mask = self._length_limit(cid, cid_mask)
                rids, rids_mask = [], []
                for sample_ids, sample_mask in zip(item['input_ids'][1:], item['attention_mask'][1:]):
                    sample_ids, sample_mask = self._length_limit(sample_ids, sample_mask)
                    rids.append(sample_ids)
                    rids_mask.append(sample_mask)
                self.data.append({
                    'cid': cid,
                    'cid_mask': cid_mask,
                    'rids': rids,
                    'rids_mask': rids_mask,
                })
                
    def _length_limit(self, ids, ids_mask):
        if len(ids) > self.max_len:
            ids = [ids[0]] + ids[-(self.max_len-1):]
            ids_mask = ids_mask[-self.max_len:]    # all 1
        return ids, ids_mask
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        if self.mode in ['train', 'dev']:
            cid = torch.LongTensor(bundle['cid'])
            cid_mask = torch.LongTensor(bundle['cid_mask'])
            rid = torch.LongTensor(bundle['rid'])
            rid_mask = torch.LongTensor(bundle['rid_mask'])
        else:
            cid = torch.LongTensor(bundle['cid'])
            cid_mask = torch.LongTensor(bundle['cid_mask'])
            rid = [torch.LongTensor(i) for i in bundle['rids']]
            rid_mask = [torch.LongTensor(i) for i in bundle['rids_mask']]
        return cid, rid, cid_mask, rid_mask 
    
    def save_pickle(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save dataset into {self.pp_path}')
        
    def collate(self, batch):
        cid, cid_mask, rid, rid_mask = [], [], [], []
        if self.mode == 'train':
            for i in batch:
                cid.append(i[0])
                rid.append(i[1])
                cid_mask.append(i[2])
                rid_mask.append(i[3])
            cid = pad_sequence(cid, batch_first=True, padding_value=self.pad)
            cid_mask = pad_sequence(cid_mask, batch_first=True, padding_value=0)
            rid = pad_sequence(rid, batch_first=True, padding_value=self.pad)
            rid_mask = pad_sequence(rid_mask, batch_first=True, padding_value=0)
        else:
            assert len(batch) == 1, f'[!] test mode batch size must be 1'
            batch = batch[0]
            cid, rid, cid_mask, rid_mask = batch[0], batch[1], batch[2], batch[3]
            cid = cid.unsqueeze(0)    # [1, S]
            cid_mask = cid_mask.unsqueeze(0)    # [1, S]
            rid = pad_sequence(rid, batch_first=True, padding_value=self.pad)
            rid_mask = pad_sequence(rid, batch_first=True, padding_value=0)
        if torch.cuda.is_available():
            cid, rid, cid_mask, rid_mask = cid.cuda(), rid.cuda(), cid_mask.cuda(), rid_mask.cuda()
        return cid, rid, cid_mask, rid_mask

class BERTIRDataset(Dataset):

    '''
    BERT IR Dataset Chinese
    1. train and dev mode only need the binary classification
    2. test mode needs to consider about the measurement
    '''

    def __init__(self, path, mode='train', max_len=300, samples=9, negative_aspect='coherence'):
        self.mode = mode
        self.max_len = max_len 
        data = read_text_data(path)
        # data = read_text_data_nosep(path)
        responses = [i[1] for i in data]
        self.vocab = BertTokenizer.from_pretrained('bert-base-chinese')
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.pp_path = f'{os.path.splitext(path)[0]}_{negative_aspect}.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None
        self.data = []
        
        d_ = []
        for item in tqdm(data):
            context, response = item[:-1], item[-1]
            context = ' [SEP] '.join(context)
            negative = generate_negative_samples(
                response, responses, samples=samples
            )
            d_.append((context, [response] + negative))
        
        if mode in ['train', 'dev']:
            for context, response in tqdm(d_):
                item = self.vocab.batch_encode_plus([context] + response, return_token_type_ids=True)
                context_id, response_ids = item['input_ids'][0], item['input_ids'][1:]
                for idx, rid in enumerate(response_ids):
                    ids = context_id + rid[1:]
                    tids = [0] * len(context_id) + [1] * (len(rid) - 1)
                    self.data.append({
                        'ids': ids,
                        'tids': tids,
                        'label': 1 if idx == 0 else 0,
                    })
        else:
            for context, responses in tqdm(d_):
                item = self.vocab.batch_encode_plus([context] + response, return_token_type_ids=True)
                context_id, response_ids = item['input_ids'][0], item['input_ids'][1:]
                self.data.append({
                    'cids': context_id,
                    'rids': response_ids,
                    'label': [1] + [0] * samples,
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        if self.mode in ['train', 'dev']:
            if len(bundle['ids']) > self.max_len:
                cut_size = len(bundle['ids']) - self.max_len + 1    # ignore [CLS]
                context_ids = torch.LongTensor([bundle['ids'][0]] + bundle['ids'][cut_size:])
                token_type_ids = torch.LongTensor([bundle['tids'][0]] + bundle['tids'][cut_size:])
            else:
                context_ids = torch.LongTensor(bundle['ids'])
                token_type_ids = torch.LongTensor(bundle['tids'])
            return context_ids, token_type_ids, bundle['label'] 
        else:
            ids, token_ids = [], []
            for i in range(len(bundle['rids'])):
                p = bundle['cids'] + bundle['rids'][i][1:]
                tids = [0] * len(bundle['cids']) + [1] * (len(bundle['rids'][i]) - 1)
                if len(p) > self.max_len:
                    cut_size = len(p) - self.max_len + 1    # ignore [CLS]
                    context_ids = torch.LongTensor([p[0]] + p[cut_size:])
                    token_type_ids = torch.LongTensor([tids[0]] + tids[cut_size:])
                else:
                    context_ids = torch.LongTensor(p)
                    token_type_ids = torch.LongTensor(tids)
                ids.append(context_ids)
                token_ids.append(token_type_ids)
            return ids, token_ids, bundle['label']
    
    def collate(self, batch):
        if self.mode == 'train':
            ids, token_type_ids, label = [], [], []
            for i in batch:
                ids.append(torch.LongTensor(i[0]))
                token_type_ids.append(torch.LongTensor(i[1]))
                label.append(i[2])
            ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)    # [batch, seq]
            token_type_ids = pad_sequence(token_type_ids, batch_first=True, padding_value=self.pad)
            attn_mask_index = ids.nonzero().tolist()
            attn_mask_index_x, attn_mask_index_y = [i[0] for i in attn_mask_index], [i[1] for i in attn_mask_index]
            attn_mask = torch.zeros_like(ids)
            attn_mask[attn_mask_index_x, attn_mask_index_y] = 1
            label = torch.LongTensor(label)
        else:
            ids, token_type_ids, label = [], [], []
            for i in batch:
                ids.extend(torch.LongTensor(i[0]))
                token_type_ids.extend(torch.LongTensor(i[1]))
                label.extend(i[2])
            ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)    # [batch, seq]
            token_type_ids = pad_sequence(token_type_ids, batch_first=True, padding_value=self.pad)
            attn_mask_index = ids.nonzero().tolist()
            attn_mask_index_x, attn_mask_index_y = [i[0] for i in attn_mask_index], [i[1] for i in attn_mask_index]
            attn_mask = torch.zeros_like(ids)
            attn_mask[attn_mask_index_x, attn_mask_index_y] = 1
            label = torch.LongTensor(label)
        if torch.cuda.is_available():
            ids, token_type_ids, attn_mask, label = ids.cuda(), token_type_ids.cuda(), attn_mask.cuda(), label.cuda()
        return ids, token_type_ids, attn_mask, label
            
    def save_pickle(self):
        torch.save(self.data, self.pp_path)
        print(f'[!] save dataset into {self.pp_path}')
        
class BERTIRDISDataset(Dataset):

    '''
    BERT IR Curriculum Learning Dataset
    The negative samples maybe use the different negative sampling strategies to collect the different difficult samples.
    '''

    def __init__(self, path, mode='train', src_min_length=20, tgt_min_length=15, 
                 max_len=300, samples=1, vocab_file='data/vocab/vocab_small'):
        agent = BERTRetrievalAgent(args['multi_gpu'], kb=False)
        agent.load_model(f'ckpt/zh50w/bertretrieval/best.pt')
        print(f'[!] load the bert retrieval model over')
        self.mode = mode
        self.max_len = max_len 
        # data = read_csv_data(path)
        data = read_text_data(path)
        # context and response are all the negative samples 
        responses = [i[1] for i in data]
        # self.vocab = BertTokenizer(vocab_file=vocab_file)
        self.vocab = BertTokenizer.from_pretrained('bert-base-chinese')
        
        self.pp_path = f'{os.path.splitext(path)[0]}_dis.pkl'
        if os.path.exists(self.pp_path):
            with open(self.pp_path, 'rb') as f:
                self.data = pickle.load(f)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None
        self.data = []

        # collect the data samples
        d_ = []
        for i in tqdm(data):
            context, response = i[0], i[1]
            negative = generate_negative_samples(response, responses, samples=samples)
            d_.append((context, [response] + negative))
        
        if mode in ['train', 'dev']:
            # concatenate the context and the response
            for context, responses in tqdm(d_):
                response, negative = responses
                context_id = self.vocab.encode(context)
                response_id = self.vocab.encode(response)
                negative_id = self.vocab.encode(negative)
                bundle = dict()
                if random.random() < 0.5:
                    ids = context_id + response_id[1:] + negative_id[1:]
                    bundle['label'] = 1
                else:
                    ids = context_id + negative_id[1:] + response_id[1:]
                    bundle['label'] = 0
                bundle['context_id'] = ids[-self.max_len:]
                self.data.append(bundle)
        else:
            for item in tqdm(d_):
                context, response = item
                response, negative = response[0], response[1:]
                context_id = self.vocab.encode(context)
                response_id = self.vocab.encode(response)
                negative_ids = [self.vocab.encode(i) for i in negative]
                bundle = dict()
                bundle['context_id'] = context_id
                bundle['reply_id'] = [response_id] + res_ids
                bundle['label'] = [1] + [0] * samples
                self.data.append(bundle)
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        return bundle
    
    def save_pickle(self):
        with open(self.pp_path, 'wb') as f:
            pickle.dump(self.data, f)
        print(f'[!] save dataset into {self.pp_path}')
        
class BERTIRCLDataset(Dataset):

    '''
    BERT IR Curriculum Learning Dataset
    The negative samples maybe use the different negative sampling strategies to collect the different difficult samples.
    '''

    def __init__(self, path, mode='train', src_min_length=20, tgt_min_length=15, 
                 max_len=300, samples=9, vocab_file='data/vocab/vocab_small'):
        self.mode = mode
        self.max_len = max_len 
        # data = read_csv_data(path)
        data = read_text_data(path)
        # context and response are all the negative samples 
        responses = [i[1] for i in data]
        # self.vocab = BertTokenizer(vocab_file=vocab_file)
        self.vocab = BertTokenizer.from_pretrained('bert-base-chinese')
        self.pp_path = f'{os.path.splitext(path)[0]}_cl.pkl'
        if os.path.exists(self.pp_path):
            with open(self.pp_path, 'rb') as f:
                self.data = pickle.load(f)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None
        self.data = []

        # collect the data samples
        d_ = []
        for i in tqdm(data):
            context, response = i[0], i[1]
            negative = generate_negative_samples(response, responses, samples=samples)
            d_.append((context, [response] + negative))
        
        if mode in ['train', 'dev']:
            # concatenate the context and the response
            for context, response in tqdm(d_):
                context_id = self.vocab.encode(context)
                for idx, r in enumerate(response):
                    bundle = dict()
                    rid = self.vocab.encode(r)
                    p = context_id + rid[1:]
                    bundle['context_id'] = p[-self.max_len:]
                    bundle['label'] = 1 if idx == 0 else 0
                    self.data.append(bundle)
            # speed up the predict procedure
            self.data = sorted(self.data, key=lambda i:len(i['context_id']))
        else:
            for item in tqdm(d_):
                context, response = item
                context_id = self.vocab.encode(context)
                res_ids = [self.vocab.encode(i) for i in response]
                bundle = dict()
                bundle['context_id'] = context_id
                bundle['reply_id'] = res_ids
                bundle['label'] = [1] + [0] * samples
                self.data.append(bundle)
                
    def reset_order(self, losses):
        '''
        separate the right and the wrong labels
        '''
        rtrue = [idx for idx, i in enumerate(self.data) if i['label'] == 1]
        rfalse = [idx for idx, i in enumerate(self.data) if i['label'] == 0]
        true_losses = [losses[i] for i in rtrue]
        false_losses = [losses[i] for i in rfalse]
        true_index = np.argsort(true_losses)
        true_index = [rtrue[i] for i in true_index]
        false_index = np.argsort(false_losses)
        false_index = [rfalse[i] for i in false_index]
        index = []
        for t, f in zip(true_index, false_index):
            index.extend([t, f])
        self.data = [self.data[i] for i in index]
        print(f'[!] reset the training order over according to the losses')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        if self.mode in ['train', 'dev']:
            return bundle
        else:
            ids = []
            for i in range(len(bundle['reply_id'])):
                p = bundle['context_id'] + bundle['reply_id'][i][1:]
                ids.append(torch.LongTensor(p[-self.max_len:]))
            return ids, bundle['label']
    
    def save_pickle(self):
        with open(self.pp_path, 'wb') as f:
            pickle.dump(self.data, f)
        print(f'[!] save dataset into {self.pp_path}')
        
class BERTIRCLDataLoader:
    
    def __init__(self, data, T, batch_size=16, window_ratio=0.05,
                 forLoss=False):
        self.data = data    # dataset object
        self.data_size = len(data)
        self.batch_size = batch_size
        self.Lbatch_size = batch_size
        print(f'[!] bsz(cl|predict): {self.batch_size}|{self.Lbatch_size}')
        self.T = T
        self.c0 = 0.01    # begin with training 1% samples
        self.t = 0
        self.forLoss = forLoss
        self.index = 0
        self.pad = 0
        self.pool_size = 1024
        self.last_p = 0
        
        self.priority = [10000.0] * self.data_size
        self.last_index = None
        
    def reset_order(self, losses):
        self.data.reset_order(losses)
        
    def update_priority(self, losses):
        assert len(self.last_index) == len(losses), f'[!] error during updating the losses'
        for idx, l in zip(self.last_index, losses):
            self.priority[idx] = l
        
    def progress(self):
        '''
        use self.t to obtain the available samples ratio
        '''
        s = np.sqrt(
            self.t * (1 - self.c0**2) / self.T + self.c0**2
        )
        return s
    
    def normal(self, x):
        s = np.array(x)
        s = s / np.sum(s)
        return s
    
    def __iter__(self):
        return self
    
    def __len__(self):
        if self.forLoss:
            return self.data_size
        else:
            return self.T
    
    def __next__(self):
        if self.forLoss:
            # extract the data samples one by one in order
            if self.index >= self.data_size:
                self.index = 0
                raise StopIteration
            else:
                batch = self.data[self.index:self.index+self.Lbatch_size]
                ids = [torch.LongTensor(i['context_id']) for i in batch]
                ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
                labels = torch.LongTensor([i['label'] for i in batch])
                if torch.cuda.is_available():
                    ids = ids.cuda()
                    labels = labels.cuda()
                self.index += len(batch)
                return ids, labels
        else:
            p_ = self.progress()
            p = int(p_ * self.data_size)
            delta_ = p - self.last_p
            self.last_p = p
            if p >= self.data_size:
                self.t = 0
                self.last_p = 0
                raise StopIteration
            else:
                # define the sample range
                data = self.data[:p]
                
                # ========== uniform sample ==========
                self.last_index = random.sample(range(len(data)), self.batch_size)
                # ========== priority sample (very slow) ==========
                # make sure the samples that are not trained will have the high priority to be trained, and focus on the hard sample in the `easy` curriculum learning duration.
                # self.last_index = np.random.choice(
                #     len(data), self.batch_size, p=self.normal(p), replace=False,
                # )
                # ========== priority sample with pool size (not so slow but also priority) ==========
                # pool_index = random.sample(range(len(data)), self.pool_size)
                # probability = [self.priority[i] for i in pool_index]
                # self.last_index = np.random.choice(
                #     pool_index, 
                #     self.batch_size, 
                #     p=self.normal(probability), 
                #     replace=False,
                # )
                
                batch = [data[i] for i in self.last_index]
                # construct the tensor
                ids = [torch.LongTensor(i['context_id']) for i in batch]
                ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
                labels = torch.LongTensor([i['label'] for i in batch])
                if torch.cuda.is_available():
                    ids = ids.cuda()
                    labels = labels.cuda()
                self.t += 1
                return round(p_, 4), self.priority[:p].count(10000.0), delta_, ids, labels
        
class DecoupleGPT2RLDataset(Dataset):

    '''
    GPT2RL Dataset cannot contain the [PAD] token in the batch.
    So we rewrite the Dataset (sorted with the length) and DataLoader to make sure it.

    The batch size is variable (different length has different the number of the istances),
    which won't affect the model.

    train/dev/test mode: return the contexts and the responses
    length of the utterances less than `self.length_threshold` will be droped (useless)

    RL Dataset must make sure that the reply_id + context_id is small than gpt2.n_ctx (300)
    Default the src_len_size is 280 and tgt_len_size is 20 (faster)

    Construct the keywords vocabulary
    '''

    def __init__(self, path, vocab_file='data/vocab/vocab_small', kw_length=10, src_len_size=200, tgt_len_size=50, length_threshold=5):
        assert src_len_size + tgt_len_size <= 300, f'[!] the sum of src and tgt length must small than 300, but got {src_len_size} + {tgt_len_size}'
        self.vocab = BertTokenizer(vocab_file=vocab_file)
        self.src_len_size, self.tgt_len_size = src_len_size, tgt_len_size
        self.length_threshold = length_threshold
        self.pp_path = f'{os.path.splitext(path)[0]}.pkl'
        if os.path.exists(self.pp_path):
            with open(self.pp_path, 'rb') as f:
                self.data = pickle.load(f)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None

        # PREPROCESS
        data = read_text_data(path)
        # keywords
        self.keywords = vocabulary(data, n_vocab=1000)
        self.data, contexts, responses = [], [], []
        for idx, sample in enumerate(data):
            # extremely conversation cases are useless for RL
            if len(sample[0]) < 10 or len(sample[1]) < 5:
                continue
            if len(sample[0]) > src_len_size or len(sample[1]) > tgt_len_size:
                continue
            contexts.append(sample[0])
            responses.append(sample[1])
        for c, r in tqdm(list(zip(contexts, responses))):
            bundle = dict()
            ids = self.vocab.encode(c)
            ids = ids[-self.src_len_size:]
            if len(ids) <= self.length_threshold:
                continue
            bundle['context_id'] = torch.LongTensor(ids)
            bundle['context_length'] = len(ids)
            # keywords
            keywords = jieba.analyse.extract_tags(
                    r, kw_length, allowPOS=self.keywords.allowPOS)
            bundle['keywords_text'] = keywords
            bundle['keywords_id'] = torch.LongTensor(
                    self.keywords.toks2idx(keywords, kw_length))
            ids = self.vocab.encode(r)
            ids = ids[:self.tgt_len_size]
            if len(ids) <= self.length_threshold:
                continue
            bundle['reply_id'] = torch.LongTensor(ids)
            self.data.append(bundle)
        # sort by the context_length from small to big
        self.data = sorted(self.data, key=lambda i:i['context_length'])
        print(f'[!] read and process raw data from {path} over')

    def save_pickle(self):
        with open(self.pp_path, 'wb') as f:
            pickle.dump(self.data, f)
        print(f'[!] save dataset into {self.pp_path}')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

class DecoupleGPT2RLDataLoader:

    '''
    Custom DataLoader for non-pad batch
    Implement __next__ and __iter__ method

    :shuffle: shuffle in the batch
    :batch_size: default max size of the batch (maybe smaller)
    '''

    def __init__(self, data, shuffle=True, batch_size=16):
        self.data = data
        self.data_size = len(data)
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.lengths = [i['context_length'] for i in self.data.data]
        self.index, self.pad = 0, 0

    def __iter__(self):
        return self

    def __next__(self):
        '''
        return batch as a iterator
        '''
        if self.index >= self.data_size:
            self.index = 0    # reset
            raise StopIteration
        else:
            # decide the dynamic batch size
            idx, startl = self.index, self.lengths[self.index]
            for l in self.lengths[self.index:self.index+self.batch_size]:
                if l != startl:
                    break
                idx += 1
            batch = self.data[self.index:idx]
            if self.shuffle:
                random.shuffle(batch)
            self.index = idx
            # construct the pytorch tensor and return
            contexts = [i['context_id'] for i in batch]
            responses = [i['reply_id'] for i in batch]
            keywords = [i['keywords_id'] for i in batch]
            reply_length = [len(i['reply_id']) for i in batch]
            ctx = torch.stack(contexts)    # [batch, seq]
            keywords = pad_sequence(
                    keywords, batch_first=True, 
                    padding_value=self.data.vocabulary.vocab.stoi['<pad>'])
            res = pad_sequence(responses, batch_first=True, padding_value=self.pad)
            if torch.cuda.is_available():
                ctx, res, keywords = ctx.cuda(), res.cuda(), keywords.cuda()
            return ctx, keywords, res, reply_length

    def __len__(self):
        # for tqdm
        return self.data_size

class GPT2RLDataset(Dataset):

    '''
    GPT2RL Dataset cannot contain the [PAD] token in the batch.
    So we rewrite the Dataset (sorted with the length) and DataLoader to make sure it.

    The batch size is variable (different length has different the number of the istances),
    which won't affect the model.

    train/dev/test mode: return the contexts and the responses
    length of the utterances less than `self.length_threshold` will be droped (useless)

    RL Dataset must make sure that the reply_id + context_id is small than gpt2.n_ctx (300)
    Default the src_len_size is 280 and tgt_len_size is 20 (faster)

    Construct the keywords vocabulary
    '''

    def __init__(self, path, vocab_file='data/vocab/vocab_small', src_len_size=200, tgt_len_size=50, length_threshold=5):
        assert src_len_size + tgt_len_size <= 512, f'[!] the sum of src and tgt length must small than 300, but got {src_len_size} + {tgt_len_size}'
        self.vocab = BertTokenizer(vocab_file=vocab_file)
        self.src_len_size, self.tgt_len_size = src_len_size, tgt_len_size
        self.length_threshold = length_threshold
        self.pp_path = f'{os.path.splitext(path)[0]}_rl.pkl'
        if os.path.exists(self.pp_path):
            with open(self.pp_path, 'rb') as f:
                self.data = pickle.load(f)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None

        # PREPROCESS
        data = read_text_data(path)
        self.data, contexts, responses = [], [], []
        for idx, sample in enumerate(data):
            # extremely conversation cases are useless for RL
            if len(sample[0]) < 10 or len(sample[1]) < 5:
                continue
            if len(sample[0]) > src_len_size or len(sample[1]) > tgt_len_size:
                continue
            contexts.append(sample[0])
            responses.append(sample[1])
        for c, r in tqdm(list(zip(contexts, responses))):
            bundle = dict()
            ids = self.vocab.encode(c)
            ids = ids[-self.src_len_size:]
            if len(ids) <= self.length_threshold:
                continue
            bundle['context_id'] = torch.LongTensor(ids)
            bundle['context_length'] = len(ids)
            ids = self.vocab.encode(r)
            ids = ids[:self.tgt_len_size]
            if len(ids) <= self.length_threshold:
                continue
            bundle['reply_id'] = torch.LongTensor(ids)
            self.data.append(bundle)
        # sort by the context_length from small to big
        self.data = sorted(self.data, key=lambda i:i['context_length'])
        print(f'[!] read and process raw data from {path} over')

    def save_pickle(self):
        with open(self.pp_path, 'wb') as f:
            pickle.dump(self.data, f)
        print(f'[!] save dataset into {self.pp_path}')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

class GPT2RLDataLoader:

    '''
    Custom DataLoader for non-pad batch
    Implement __next__ and __iter__ method

    :shuffle: shuffle in the batch
    :batch_size: default max size of the batch (maybe smaller)
    '''

    def __init__(self, data, shuffle=True, batch_size=16):
        self.data = data
        self.data_size = len(data)
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.lengths = [i['context_length'] for i in self.data.data]
        self.index, self.pad = 0, 0

    def __iter__(self):
        return self

    def __next__(self):
        '''
        return batch as a iterator
        '''
        if self.index >= self.data_size:
            self.index = 0    # reset
            raise StopIteration
        else:
            # decide the dynamic batch size
            idx, startl = self.index, self.lengths[self.index]
            for l in self.lengths[self.index:self.index+self.batch_size]:
                if l != startl:
                    break
                idx += 1
            batch = self.data[self.index:idx]
            if self.shuffle:
                random.shuffle(batch)
            self.index = idx
            # construct the pytorch tensor and return
            contexts = [i['context_id'] for i in batch]
            responses = [i['reply_id'] for i in batch]
            reply_length = [len(i['reply_id']) for i in batch]
            ctx = torch.stack(contexts)    # [batch, seq]
            res = pad_sequence(responses, batch_first=True, padding_value=self.pad)
            if torch.cuda.is_available():
                ctx, res = ctx.cuda(), res.cuda()
            return ctx, res, reply_length

    def __len__(self):
        # for tqdm
        return self.data_size

class GPT2LMDataset(Dataset):

    '''
    GPT2 Chinese LM 
    corpus from: https://github.com/CLUEbenckmark/CLUE; wiki2019zh

    The preprocessed procedure already makes sure that the max length is 300
    '''

    def __init__(self, path, min_length=15, src_length_size=300, vocab_file='data/vocab/vocab_small'):
        super(GPT2LMDataset, self).__init__()
        self.pad = '[PAD]'
        self.vocab = BertTokenizer(vocab_file=vocab_file)
        self.pad_id = self.vocab.convert_tokens_to_ids(self.pad)
        self.pp_path = f'{os.path.splitext(path)[0]}.pkl'
        if os.path.exists(self.pp_path):
            with open(self.pp_path, 'rb') as f:
                self.data = pickle.load(f)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None
        data = read_text_data_noparallel(path)
        self.data = []
        for sample in tqdm(data):
            bundle = dict()
            ids = self.vocab.encode(sample)
            if len(ids) < min_length:
                continue
            ids = ids[:src_length_size]
            bundle['context_id'] = torch.LongTensor(ids)
            self.data.append(bundle)
        # NOTE:
        self.data = sorted(self.data, key=lambda x: len(x['context_id']))
        print(f'[!] read and process raw data from {path} over')

    def save_pickle(self):
        with open(self.pp_path, 'wb') as f:
            pickle.dump(self.data, f)
        print(f'[!] save dataset into {self.pp_path}')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]
    
class GPT2V2RLDataset(Dataset):
    
    '''batch inference issue has been solved, we don"t need to customize the dataloader'''
    
    def __init__(self, path, mode='train', min_length=20, lang='zh', src_len_size=512, tgt_len_size=128, candidate=5):
        vocab_file = 'data/vocab/vocab_small' if lang == 'zh' else 'data/vocab/vocab_english'
        self.mode = mode
        
        self.vocab = BertTokenizer(vocab_file=vocab_file)
        self.pad_id = self.vocab.convert_tokens_to_ids('[PAD]')
        self.cls_id = self.vocab.convert_tokens_to_ids('[CLS]')
        self.sep_id = self.vocab.convert_tokens_to_ids('[SEP]')
        
        self.src_len_size, self.tgt_len_size, self.candidate = src_len_size, tgt_len_size, candidate
        
        # init the elasticsearch retrieval
        self.retrieval = TestAgent(kb=False)
        print(f'[!] elasticsearch retrieval module init over')
        
        self.pp_path = f'{os.path.splitext(path)[0]}_v2rl.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None
        
        data = read_text_data(path)
        self.data = []
        contexts = [i[0] for i in data]
        responses = [i[1] for i in data]
        index, query_batch_size, dataset_size = 0, 256, len(data)
        for begin_index in tqdm(list(range(0, dataset_size, query_batch_size))):
            inner_contexts = contexts[begin_index:begin_index+query_batch_size]
            inner_responses = responses[begin_index:begin_index+query_batch_size]
            # [query_batch_size, candidates] type is list
            inner_candidates = self.retrieval.MultiSearch(inner_contexts, samples=self.candidate)
            for ctx, res, can in list(zip(inner_contexts, inner_responses, inner_candidates)):
                if len(can) < self.candidate:
                    continue
                bundle = dict()
                res_tokens = self.vocab.encode(res)
                item = self.vocab.batch_encode_plus([ctx])
                cid, cid_mask = item['input_ids'][0][-self.src_len_size:], item['attention_mask'][0][-self.src_len_size:]
                if len(cid) < min_length:
                    continue
                bundle['cids'] = cid
                bundle['cids_mask'] = cid_mask
                bundle['rids'] = res_tokens[:self.tgt_len_size]
                bundle['context_text'] = ctx.replace('[SEP]', '')
                bundle['response_text'] = can
                self.data.append(bundle)
        print(f'[!] read and process raw data from {path} over')
        torch.save(self.data, self.pp_path)
        print(f'[!] save dataset into {self.pp_path}')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]
    
    def collate(self, batch):
        max_len = max([len(i['cids']) for i in batch])
        position_ids = torch.LongTensor([[0] * (max_len - len(i['cids'])) + list(range(len(i['cids']))) for i in batch])
        cids = torch.LongTensor([[self.pad_id] * (max_len - len(i['cids'])) + i['cids'] for i in batch])
        rids = pad_sequence(
            [torch.LongTensor(i['rids']) for i in batch], 
            batch_first=True, padding_value=self.pad_id,
        )
        attn_mask_index = cids.nonzero().tolist()
        attn_mask_index_x, attn_mask_index_y = [i[0] for i in attn_mask_index], [i[1] for i in attn_mask_index]
        attn_mask = cids.clone()
        attn_mask[attn_mask_index_x, attn_mask_index_y] = 1
        if torch.cuda.is_available():
            cids, rids = cids.cuda(), rids.cuda()
            attn_mask = attn_mask.cuda()
            position_ids = position_ids.cuda()
        ctx_text = [i['context_text'] for i in batch]
        candidate_text = [i['response_text'] for i in batch]
        return cids, rids, attn_mask, position_ids, ctx_text, candidate_text

class GPT2V2Dataset(Dataset):
    
    '''retrieval some candidate for generation; the context tokens are masked; retrieval responses and the conversation context will be used as the semantic controllor
    Support parallel prediction by padding in front of the sequences inner a batch'''

    def __init__(self, path, mode='train', min_length=20, lang='zh', src_len_size=512, tgt_len_size=128, candidate=5):
        vocab_file = 'data/vocab/vocab_small' if lang == 'zh' else 'data/vocab/vocab_english'
        self.mode = mode
        
        self.vocab = BertTokenizer(vocab_file=vocab_file)
        self.pad_id = self.vocab.convert_tokens_to_ids('[PAD]')
        self.cls_id = self.vocab.convert_tokens_to_ids('[CLS]')
        self.sep_id = self.vocab.convert_tokens_to_ids('[SEP]')
        
        self.src_len_size, self.tgt_len_size, self.candidate = src_len_size, tgt_len_size, candidate
        
        # init the elasticsearch retrieval
        self.retrieval = TestAgent(kb=False)
        print(f'[!] elasticsearch retrieval module init over')
        
        self.pp_path = f'{os.path.splitext(path)[0]}_v2.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None
        
        data = read_text_data(path)
        self.data = []
        if self.mode in ['train', 'dev']:
            contexts = [i[0] for i in data]
            responses = [i[1] for i in data]
            index, query_batch_size, dataset_size = 0, 256, len(data)
            for begin_index in tqdm(list(range(0, dataset_size, query_batch_size))):
                inner_contexts = contexts[begin_index:begin_index+query_batch_size]
                inner_responses = responses[begin_index:begin_index+query_batch_size]
                # [query_batch_size, candidates] type is list
                inner_candidates = self.retrieval.MultiSearch(inner_contexts, samples=self.candidate)
                for ctx, res, can in list(zip(inner_contexts, inner_responses, inner_candidates)):
                    if len(can) < self.candidate:
                        continue
                    # NOTE: faster for training?
                    # can = [list(jieba.cut(i_can)) for i_can in can]
                    bundle = dict()
                    ctx_tokens = self.vocab.convert_tokens_to_ids(self.vocab.tokenize(ctx))
                    res_tokens = self.vocab.convert_tokens_to_ids(self.vocab.tokenize(res))
                    sequence = [self.cls_id] + ctx_tokens + [self.sep_id] + res_tokens + [self.sep_id]
                    if len(sequence) < min_length:
                        continue
                    bundle['context_text'] = ctx.replace('[SEP]', '')
                    bundle['response_text'] = can
                    
                    bundle['ids'] = sequence
                    bundle['labels'] = [self.pad_id] * (len(ctx_tokens) + 1) + res_tokens + [self.sep_id]
                    
                    # NOTE:
                    if len(bundle['ids']) > self.src_len_size:
                        cut_size = len(bundle['ids']) - self.src_len_size
                        bundle['ids'] = bundle['ids'][cut_size:]
                        bundle['labels'] = bundle['labels'][cut_size:]
                    self.data.append(bundle)
            self.data = sorted(self.data, key=lambda x: len(x['ids']))
        else:
            contexts = [i[0] for i in data]
            responses = [i[1] for i in data]
            index, query_batch_size, dataset_size = 0, 128, len(data)
            for begin_index in tqdm(list(range(0, dataset_size, query_batch_size))):
                inner_contexts = contexts[begin_index:begin_index+query_batch_size]
                inner_responses = responses[begin_index:begin_index+query_batch_size]
                # [query_batch_size, candidates] type is list
                inner_candidates = self.retrieval.MultiSearch(inner_contexts, samples=self.candidate)
                for ctx, res, can in list(zip(inner_contexts, inner_responses, inner_candidates)):
                    if len(can) < self.candidate:
                        continue
                    bundle = dict()
                    ctx_tokens = self.vocab.encode(ctx)
                    res_tokens = self.vocab.encode(res)
                    if len(ctx_tokens) < min_length:
                        continue
                    bundle['ids'] = ctx_tokens
                    # NOTE:
                    if len(bundle['ids']) > self.src_len_size:
                        cut_size = len(bundle['ids']) - self.src_len_size
                        bundle['ids'] = ctx_tokens[cut_size:]
                    bundle['rids'] = res_tokens
                    bundle['context_text'] = ctx.replace('[SEP]', '')
                    bundle['response_text'] = can
                    self.data.append(bundle)
        print(f'[!] read and process raw data from {path} over')
        torch.save(self.data, self.pp_path)
        print(f'[!] save dataset into {self.pp_path}')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]
    
    def collate(self, batch):
        if self.mode in ['train', 'dev']:
            ids = [torch.LongTensor(i['ids']) for i in batch]
            labels = [torch.LongTensor(i['labels']) for i in batch]
            ctx_text = [i['context_text'] for i in batch]
            candidate_text = [i['response_text'] for i in batch]
            
            # shuffle
            random_idx = list(range(len(batch)))
            random.shuffle(random_idx)
            ids = [ids[i] for i in random_idx]
            labels = [labels[i] for i in random_idx]
            ctx_text = [ctx_text[i] for i in random_idx]
            candidate_text = [candidate_text[i] for i in random_idx]
            
            ids = pad_sequence(ids, batch_first=True, padding_value=self.pad_id)
            labels = pad_sequence(labels, batch_first=True, padding_value=self.pad_id)
            if torch.cuda.is_available():
                ids, labels = ids.cuda(), labels.cuda()
            return ids, labels, ctx_text, candidate_text
        else:
            # ========== BATCH SIZE BIGGER THAN 1 ========== #
            max_len = max([len(i['ids']) for i in batch])
            position_ids = torch.LongTensor([[0] * (max_len - len(i['ids'])) + list(range(len(i['ids']))) for i in batch])
            ids = torch.LongTensor([[self.pad_id] * (max_len - len(i['ids'])) + i['ids'] for i in batch])
            attn_mask_index = ids.nonzero().tolist()
            attn_mask_index_x, attn_mask_index_y = [i[0] for i in attn_mask_index], [i[1] for i in attn_mask_index]
            attn_mask = ids.clone()
            attn_mask[attn_mask_index_x, attn_mask_index_y] = 1
            res = [i['rids'] for i in batch]
            if torch.cuda.is_available():
                ids = ids.cuda()
                attn_mask = attn_mask.cuda()
                position_ids = position_ids.cuda()
            ctx_text = [i['context_text'] for i in batch]
            candidate_text = [i['response_text'] for i in batch]
            rids = [torch.LongTensor(i['rids']) for i in batch]
            return ids, rids, attn_mask, position_ids, ctx_text, candidate_text
        
            # ========== BATCH SIZE IS 1 VERSION ========== #
            # assert len(batch) == 1, f'[!] batch must be 1, but got {len(batch)}'
            # ids = torch.LongTensor(batch[0]['ids'])
            # rids = torch.LongTensor(batch[0]['rids'])
            # ctx_text = batch[0]['context_text']
            # candidate_text = batch[0]['response_text']
            # if torch.cuda.is_available():
            #     ids = ids.cuda()
            # return ids, rids, ctx_text, candidate_text

# ========== PONE ========== #
class PONEDataset(Dataset):

    def __init__(self, path, mode='train', lang='zh', src_len_size=512, samples=10, bert=False, human_annotations=None, train_mode='origin'):
        vocab_file = 'bert-base-chinese' if lang == 'zh' else 'bert-base-uncased'
        self.mode = mode
        self.max_len = src_len_size
        self.vocab = BertTokenizer.from_pretrained(vocab_file)
        if mode == 'train':
            self.pp_path = f'{os.path.splitext(path)[0]}_pone.pkl'
            if os.path.exists(self.pp_path):
                with open(self.pp_path, 'rb') as f:
                    self.data = pickle.load(f)
                print(f'[!] load preprocessed file from {self.pp_path}')
                return None

        # process dataset
        if mode in ['train']:
            self.data = []
            # data = read_text_data_nosep(path)
            data = read_text_data(path)
            responses = [i[1] for i in data]
            d_ = []
            # negatives = generate_negative_samples_bm25(
            #         responses, samples=samples, lang=lang, bert=bert)
            # for i, n in zip(data, negatives):
            #     context, response = i[0], i[1]
            #     d_.append((context, [response] + n))
            for i in data:
                context, response = i[0], i[1]
                n = generate_negative_samples(response, responses, samples=samples)
                d_.append((context, [response] + n))
            # concatenate the context and the response
            for context, response in tqdm(d_):
                context_id = self.vocab.encode(context)
                for idx, r in enumerate(response):
                    bundle = dict()
                    rid = self.vocab.encode(r)
                    bundle['context_id'] = context_id + rid[1:]
                    bundle['label'] = 1 if idx == 0 else 0
                    self.data.append(bundle)
            # NOTE:
            self.data = sorted(self.data, key=lambda x: len(x['context_id']))
        else:
            # test stage only predict
            assert type(path) == list, f'[!] test stage must input multiple files'
            assert len(human_annotations) == 3, f'[!] 3 annotators'
            context_p, groundtruth_p, pred_p = path
            annotator1, annotator2, annotator3 = human_annotations
            self.data = []

            context_data, groundtruth_data, pred_data = read_text_data_noparallel(context_p), read_text_data_noparallel(groundtruth_p), read_text_data_noparallel(pred_p)
            annotator1, annotator2, annotator3 = read_annotation(annotator1), read_annotation(annotator2), read_annotation(annotator3)

            for context, response, pred, a1, a2, a3 in tqdm(list(zip(context_data, groundtruth_data, pred_data, annotator1, annotator2, annotator3))):
                bundle = dict()
                context_id = self.vocab.encode(context)
                rid = self.vocab.encode(response)
                bundle['pred'] = context_id + rid[1:]
                bundle['human_scores'] = (a1, a2, a3)
                self.data.append(bundle)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        if self.mode in ['train']:
            ids = torch.LongTensor(bundle['context_id'][-self.max_len:])
            return ids, bundle['label'] 
        else:
            ids = torch.LongTensor(bundle['pred'][-self.max_len:])
            annotations = bundle['human_scores']
            return ids, annotations 
    
    def save_pickle(self):
        with open(self.pp_path, 'wb') as f:
            pickle.dump(self.data, f)
        print(f'[!] save dataset into {self.pp_path}')
        
class TransformerDataset(Dataset):
    
    '''
    Seq2Seq-attn or Transformer DataLoader
    '''
    
    def __init__(self, path, mode='train', lang='zh', max_length=256, zh_tokenizer=False, n_vocab=80000):
        self.mode = mode
        if lang == 'zh':
            if not zh_tokenizer:
                self.vocab = BertTokenizer.from_pretrained('bert-base-chinese')
            else:
                print(f'[!] use chinese tokenizer ...')
        else:
            self.vocab = BertTokenizer.from_pretrained('bert-base-uncased')
        
        self.pp_path = f'{os.path.splitext(path)[0]}_trs.pt'
        if os.path.exists(self.pp_path):
            if not zh_tokenizer:
                self.data = torch.load(self.pp_path)
                self.pad_id = self.vocab.convert_tokens_to_ids('[PAD]')
            else:
                self.vocab, self.data = torch.load(self.pp_path)
                self.pad_id = self.vocab.vocab.stoi['[PAD]']
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None
        
        with open(path, 'r', encoding='utf-8') as f:
            dataset = read_text_data(path)
            if zh_tokenizer:
                self.vocab = ChineseTokenizer(dataset, n_vocab=n_vocab)
                self.vocab_size = self.vocab.size
                print(f'[!] create the chinese vocab over ... vocab size: {self.vocab_size}')
                self.pad_id = self.vocab.vocab.stoi['[PAD]']
            else:
                self.pad_id = self.vocab.convert_tokens_to_ids('[PAD]')
            responses = [i[-1] for i in dataset]
            contexts = [' [SEP] '.join(i[:-1]) for i in dataset]
            self.data = []
            for context, response in tqdm(list(zip(contexts, responses))):
                if zh_tokenizer:
                    rid = self.vocab.encode(response, max_length)
                    cid = self.vocab.encode(context, max_length)
                else:
                    rid = self.vocab.encode(response)[-max_length:]
                    cid = self.vocab.encode(context)[-max_length:]
                bundle = {'cid': cid, 'rid': rid}
                self.data.append(bundle)
            self.data = sorted(self.data, key=lambda x: len(x['cid']))
            print(f'[!] collect {len(self.data)} samples for training')
            
            if mode == 'train':
                if zh_tokenizer:
                    torch.save([self.vocab, self.data], self.pp_path)
                else:
                    torch.save(self.data, self.pp_path)
            print(f'[!] process the dataset and write it into {self.pp_path}')
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        return self.data[i]
    
    def collate(self, batch):
        src = pad_sequence(
            [torch.tensor(instance['cid'], dtype=torch.long) for instance in batch],
            batch_first=False, padding_value=self.pad_id,
        )
        trg = pad_sequence(
            [torch.tensor(instance['rid'], dtype=torch.long) for instance in batch],
            batch_first=False, padding_value=self.pad_id,
        )
        trg_mask, src_key_padding_mask, trg_key_padding_mask, memory_key_padding_mask = get_masks(
            src, trg, PAD=self.pad_id)
        if torch.cuda.is_available():
            src, trg = src.cuda(), trg.cuda()
            trg_mask = trg_mask.cuda()
            src_key_padding_mask = src_key_padding_mask.cuda()
            trg_key_padding_mask = trg_key_padding_mask.cuda()
            memory_key_padding_mask = memory_key_padding_mask.cuda()
        return src, trg, trg_mask, src_key_padding_mask, trg_key_padding_mask, memory_key_padding_mask
        
class Seq2SeqDataset(Dataset):
    
    '''
    Seq2Seq-attn DataLoader
    '''
    
    def __init__(self, path, mode='train', lang='zh', max_length=256, n_vocab=50000):
        self.mode = mode
        # both test and train load the train_s2s.pt dataset
        self.pp_path = f'{os.path.split(path)[0]}/train_s2s.pt'
        if os.path.exists(self.pp_path):
            self.vocab, self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed vocab file from {self.pp_path}')
            self.pad_id = self.vocab.vocab.stoi['[PAD]']
            if self.mode == 'train':
                return None
        
        with open(path, 'r', encoding='utf-8') as f:
            dataset = read_text_data(path)
            # dataset = random.sample(dataset, 500000)
            if self.mode == 'train':
                if lang == 'zh':
                    self.vocab = ChineseTokenizer(dataset, n_vocab=n_vocab)
                    self.pad_id = self.vocab.vocab.stoi['[PAD]']
                else:
                    self.vocab = BertTokenizer.from_pretrained('bert-base-uncased')
                    self.pad_id = self.vocab.convert_tokens_to_ids('[PAD]')
                print('[!] init the vocabulary over')
            responses = [i[-1] for i in dataset]
            contexts = ['[SEP]'.join(i[:-1]) for i in dataset]
            self.data = []
            for context, response in tqdm(list(zip(contexts, responses))):
                if lang == 'zh':
                    rid = self.vocab.encode(response.strip(), max_length)
                    cid = self.vocab.encode(context.strip(), max_length)
                else:
                    rid = self.vocab.encode(response.strip())[-max_length:]
                    cid = self.vocab.encode(context.strip())[-max_length:]
                bundle = {'cid': cid, 'rid': rid}
                self.data.append(bundle)
            print(f'[!] collect {len(self.data)} samples for training')
        if mode == 'train':
            torch.save([self.vocab, self.data], self.pp_path)
            print(f'[!] save the vocab and processed data into {self.pp_path}')
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        return self.data[i]
    
    def collate(self, batch):
        src = pad_sequence(
            [torch.tensor(instance['cid'], dtype=torch.long) for instance in batch],
            batch_first=False, padding_value=self.pad_id,
        )
        src_l = torch.LongTensor([len(instance['cid']) for instance in batch])
        trg = pad_sequence(
            [torch.tensor(instance['rid'], dtype=torch.long) for instance in batch],
            batch_first=False, padding_value=self.pad_id,
        )
        trg_l = torch.LongTensor([len(instance['rid']) for instance in batch])
        if torch.cuda.is_available():
            src, trg = src.cuda(), trg.cuda()
            src_l, trg_l = src_l.cuda(), trg_l.cuda()
        return src, src_l, trg, trg_l
        
class BERTNADataset(Dataset):
    
    def __init__(self, path, mode='train', lang='zh', max_size=16):
        self.mode = mode
        self.vocab = BertTokenizer.from_pretrained('bert-base-chinese')
        self.pad_id = self.vocab.convert_tokens_to_ids('[PAD]')
        self.mask_id = self.vocab.convert_tokens_to_ids('[MASK]')
        self.sep_id = self.vocab.convert_tokens_to_ids('[SEP]')
        self.cls_id = self.vocab.convert_tokens_to_ids('[CLS]')
        self.max_size = max_size
        # keywords parameters
        # self.allowPOS = ['n', 'nr', 'nz', 'PER', 'LOC', 'ORG', 
        #                  'ns', 'nt', 'nw', 'vn', 's']
        self.topk = 5
        
        self.pp_path = f'{os.path.splitext(path)[0]}_na.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None
        
        with open(path, 'r', encoding='utf-8') as f:
            dataset = read_text_data(path)
            sample_size = 500000 if self.mode == 'train' else 10000
            dataset = random.sample(dataset, sample_size)
            responses = [' [SEP] '.join(analyse.extract_tags(i[-1], topK=self.topk)) for i in tqdm(dataset)]
            contexts = [i[0] for i in dataset]
            self.data = []
            for context, response in tqdm(list(zip(contexts, responses))):
                rid = self.vocab.encode(response)[1:-1]    # ignore the [CLS] and [SEP]
                cid = self.vocab.encode(context)[1:]    # ignore the [CLS]
                if len(rid) >= self.max_size:
                    labels = [self.cls_id] + rid[:self.max_size] + [self.sep_id] + [self.pad_id] * len(cid)
                else:
                    labels = [self.cls_id] + rid + [self.sep_id] + [self.pad_id] * (self.max_size - len(rid)) + [self.pad_id] * len(cid)
                ids = [self.cls_id] + [self.mask_id] * self.max_size + [self.sep_id] + cid
                bundle = {
                    'ids': ids[:512],
                    'labels': labels[:512],
                }
                self.data.append(bundle)
            print(f'[!] collect {len(self.data)} samples for training')
            torch.save(self.data, self.pp_path)
            print(f'[!] process the dataset and write it into {self.pp_path}')
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        return self.data[i]
    
    def collate(self, batch):
        inpt_ids = pad_sequence(
            [torch.tensor(instance['ids'], dtype=torch.long) for instance in batch],
            batch_first=True, padding_value=self.pad_id,
        )
        labels = pad_sequence(
            [torch.tensor(instance['labels'], dtype=torch.long) for instance in batch],
            batch_first=True, padding_value=self.pad_id,
        )
        return inpt_ids, labels
        

if __name__ == "__main__":
    # ========== PONE ========== #
    # train_data = PONEDataset('data/dailydialog/train.txt', mode='train', lang='en', bert=False)
    # train_data.save_pickle()
    # ========== KWGPT2 ========== #
    # train_data = KWGPT2Dataset('./data/train_generative/train.txt', mode='train')
    # train_iter = DataLoader(train_data, shuffle=True, batch_size=10, collate_fn=gpt2_train_collate_fn)
    # ========== GPT2Retrieval ========== #
    # train_data = GPT2Dataset('./data/train_generative/train.txt', mode='train', ensemble=True, candidates_k=2)
    # train_data.save_pickle()
    # train_iter = DataLoader(train_data, shuffle=True, batch_size=10, collate_fn=gpt2retrieval_train_collate_fn)
    # ========== When2Talk ========== #
    # train_data = When2talkDataset('./data/when2talk/train.txt', mode='train')
    # train_iter = DataLoader(train_data, shuffle=True, batch_size=10, collate_fn=gpt2_train_collate_fn)
    # ========== BERTLOGIN ========== #
    # train_data = BERTLOGICDataset('data/train_retrieval/train.txt')
    # train_iter = DataLoader(train_data, shuffle=True, batch_size=10, collate_fn=bert_ir_train_collate_fn)
    # ========== GPT2RL ========== #
    # train_data = DecoupleGPT2RLDataset('data/decouple_rl/train.txt')
    # train_iter = DecoupleGPT2RLDataLoader(train_data)
    # ========== BERTIRMulti ========== #
    # train_data = BERTIRMultiDataset('data/train_retrieval/train.txt')
    # train_iter = BERTIRMultiDataLoader(train_data)
    # ========== GPT2Dataset ========== #
    # train_data = GPT2Dataset('./data/zh50w/train.txt', mode='train')
    # train_iter = DataLoader(train_data, batch_size=10, shuffle=False, collate_fn=gpt2_train_collate_fn)
    # ========== MultiGPT2 ========== #
    # train_data = MultiGPT2Dataset('./data/zh50w/train.csv', mode='train')
    # train_iter = DataLoader(train_data, shuffle=True, batch_size=10, collate_fn=multigpt2_train_collate_fn)
    # ========== BERTIRDataset ========== #
    # train_data = BERTIRDataset('data/zh50w/train.txt', mode='train', samples=9, negative_aspect='overall')
    # train_iter = DataLoader(train_data, shuffle=True, batch_size=10, collate_fn=bert_ir_train_collate_fn)
    # ========== BERTIRCLDataset ========== #
    # train_data = BERTIRCLDataset('data/zh50w/train.txt', mode='train', samples=1)
    # train_data.save_pickle()
    # train_iter = BERTIRCLDataLoader(train_data, int(len(train_data) * 5 / 10), 
    #                                 batch_size=10)
    # train_iter.forLoss = True
    # ========== BERTIR ========== #
    # train_data = BERTNLIDataset('data/NLI/cnsd_snli_v1.0.train.jsonl', mode='train')
    # train_iter = DataLoader(train_data, shuffle=True, batch_size=10, collate_fn=nli_collate_fn)
    # ========== LCCC ========== #
    # train_data = UNIDataset('/data/lantian/data/LCCD_GPT', 'data/LCCC/LCCC-base.json')
    # train_iter = DataLoader(train_data, collate_fn=train_data.collate, batch_size=10)
    # ========== Seq2Seq ========== #
    # train_data = Seq2SeqDataset('data/zh50w/train.txt', mode='train')
    # train_iter = DataLoader(train_data, collate_gn=train_data.collate, batch_size=8, shuffle=True)
    # ========== GPT2V2 ========== #
    # train_data = GPT2V2Dataset('./data/zh50w/train.txt', mode='train')
    # train_iter = DataLoader(train_data, collate_gn=train_data.collate, batch_size=8, shuffle=True)
    # ========== BERTIRBI ========== #
    train_data = BERTIRBIDataset('data/zh50w/train.txt', mode='train')
    train_iter = DataLoader(train_data, collate_gn=train_data.collate, batch_size=8, shuffle=True)
    # ========= ITERATE ========= # 
    for batch in tqdm(train_iter):
        ipdb.set_trace()
    # ========== CUSTOM ITERATOR ========== #
    # with tqdm(total=len(train_iter)) as pbar:
    #     for batch in train_iter:
    #         ipdb.set_trace()
    #         pbar.update(len(batch[0]))
