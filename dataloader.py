from header import *
from utils import *
from data import *

'''
The Dataset Object can handle the single-turn and multi-turn (<eou> seperator) dialog format.
'''

class vocabulary:
    
    '''
    Only for RNN based model
    '''

    def __init__(self, corpus, n_vocab=50000, min_freq=1, lang='zh'):
        if n_vocab == -1:
            n_vocab = None
        self.allowPOS = ['n', 'nr', 'nz', 'PER', 'LOC', 'ORG', 'ns', 'nt', 'nw', 'vn', 's']
        self.topk = 10
        # <res> is the first token for response
        # <ctx> is the firat token for context
        # <eou> is the spearator between two utterances
        reversed_tokens = ['<pad>', '<unk>', '<sos>', '<eos>']
        self.lang = lang
        self.vocab = vocab.Vocab(self._build_keywords(corpus), max_size=n_vocab,
                min_freq=min_freq, specials=reversed_tokens)
        # make sure the <pad> token id is 0
        assert self.vocab.stoi['<pad>'] == 0, f'<pad> id should be 0, but got {self.vocab.stoi["<pad>"]}'
        print(f'[!] init the vocabulary over, vocab size: {len(self.vocab)}')

    def __len__(self):
        return len(self.vocab)

    @property
    def size(self):
        return len(self.vocab)

    def idx2str(self, idx_seq, spliter=''):
        # chinese spliter: ''; english spliter: ' '
        words = self.idx2toks(idx_seq)
        return spliter.join(words)

    def toks2idx(self, tok_seq, len_size_limit):
        first_token = self.vocab.stoi['<sos>']
        sentence = list(map(lambda i: self.vocab.stoi[i] if i in self.vocab.stoi else self.vocab.stoi['<unk>'], tok_seq))[-(len_size_limit-2):]
        idxs = [first_token] + sentence + [self.vocab.stoi['<eos>']]
        return idxs

    def idx2toks(self, idx_seq):
        toks = ['<sos>'] + list(map(lambda i: self.vocab.itos[i], idx_seq)) + ['<eos>']
        return toks

    def _build_vocab(self, corpus):
        vocab_counter = Counter()
        for words in corpus[0]:
            vocab_counter.update(words)
        for words in corpus[1]:
            vocab_counter.update(words)
        print(f'[!] whole vocab size: {len(vocab_counter)}')
        return vocab_counter

    def _build_keywords(self, corpus):
        keywords = Counter()
        for dialog in tqdm(corpus):
            for utterance in dialog:
                words = jieba.analyse.extract_tags(
                        utterance, 
                        topK=self.topk, 
                        allowPOS=self.allowPOS)
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

class GPT2Dataset(Dataset):
    
    '''
    Training GPT2 model doesn't need the target sentence, just training the Lnaguage Model
    GPT2 model can leverage the ability for all the pure text information, which is better than Seq2Seq architecture

    Train mode: only a long pure text
    Test/Dev mode: The long context and ground-truth

    :reversed parameter: for the reversed gpt2 language model (DialoGPT MMI Model)
    :ensemble parameter: for the gpt2retrieval model (need the TestAgent)
        ensemble mode maybe vyer time-consuming because of the Elasticsearch is used
    :candidates_k: candides_k will be used when ensemble model is on
    '''

    def __init__(self, path, mode='train', lang='zh',
                 min_length=20, src_len_size=512, tgt_len_size=128,
                 reversed=False, ensemble=False, candidates_k=2):
        if lang == 'zh':
            vocab_file = 'data/vocab/vocab_small'
        else:
            vocab_file = 'data/vocab/vocab_english'
        self.mode = mode
        self.pad = '[PAD]'
        self.vocab = BertTokenizer(vocab_file=vocab_file)
        self.pad_id = self.vocab.convert_tokens_to_ids(self.pad)
        self.src_len_size, self.tgt_len_size = src_len_size, tgt_len_size
        # load and process the data
        # CHECK WHETHER EXIST THE PREPROCESSED FILE
        assert not (reversed and ensemble), f'[!] reversed and ensemble model cannot be used both time'
        if reversed:
            self.pp_path = f'{os.path.splitext(path)[0]}_mmi.pkl'
        elif ensemble:
            self.pp_path = f'{os.path.splitext(path)[0]}_es.pkl'
        else:
            self.pp_path = f'{os.path.splitext(path)[0]}.pkl'

        if os.path.exists(self.pp_path):
            with open(self.pp_path, 'rb') as f:
                self.data = pickle.load(f)
            print(f'[!] load preprocessed file from {self.pp_path}')
            # Dataset object must return None
            return None 

        # PREPROCESSED
        if reversed:
            data = read_text_data_reversed(path)
        elif ensemble:
            # load the chinese word2vector embedding
            w2v = load_w2v('data/chinese_w2v')
            from models import TestAgent
            data = read_text_data(path)
            # search all the candidates of the samples in the dataset
            # default batch size is 1024 
            irmodel = TestAgent()
            batch_size, data_candidates, index = 256, [], 0
            with tqdm(total=len(data)) as pbar:
                while index < len(data):
                    batch = data[index:index+batch_size]
                    # query = [i[0] for i in batch]
                    # be careful, the very long context will raise the retrieval error for elasticsearch
                    query = [i[0][-300:] for i in batch]
                    rest = irmodel.MultiSearch(query, samples=candidates_k)
                    data_candidates.extend(rest)    # dataset_size*[k]
                    c_batch_size = len(batch)
                    pbar.update(c_batch_size)
                    index += c_batch_size
            print(f'[!] obtain all the retrieval samples')
        else:
            data = read_text_data(path)
        self.data = []
        if self.mode in ['train', 'dev']:
            contexts = []
            for sample in data:
                contexts.append(' [SEP] '.join(sample))
            if ensemble:
                for sample, candidates in tqdm(list(zip(contexts, data_candidates))):
                    bundle = dict()
                    # obtain the retrieval samples
                    bundle['retrieval_text'] = candidates 
                    embeddings = convert_text_embedding(w2v, candidates)
                    bundle['retrieval_embedding'] = embeddings    # k*[300]
                    # 
                    bundle['context_text'] = sample
                    ids = self.vocab.encode(sample)
                    if len(ids) < min_length:
                        continue
                    # length size of the context
                    ids = ids[-self.src_len_size:]
                    bundle['context_id'] = torch.LongTensor(ids)
                    self.data.append(bundle)
            # NOTE: sort the self.data based on the length for miniminzing the padding tokens
            else:
                for sample in tqdm(contexts):
                    bundle = dict()
                    bundle['context_text'] = sample
                    ids = self.vocab.encode(sample)
                    if len(ids) < min_length:
                        continue
                    # length size of the context
                    ids = ids[-self.src_len_size:]
                    bundle['context_id'] = torch.LongTensor(ids)
                    self.data.append(bundle)
            # NOTE: sort the self.data based on the length for miniminzing the padding tokens
            self.data = sorted(self.data, key=lambda x: len(x['context_id']))
        else:
            contexts, responses = [], []
            for sample in data:
                contexts.append(' [SEP] '.join(sample[:-1]))
                responses.append(sample[-1])
            if ensemble:
                for c, r, candidates in tqdm(list(zip(contexts, responses, data_candidates))):
                    bundle = dict()
                    bundle['retrieval_text'] = candidates 
                    rest = []
                    for candidate in candidates:
                        batch = torch.LongTensor(
                                    self.vocab.encode(candidate)[-self.src_len_size:]
                                )
                        rest.append(batch)
                    bundle['retrieval_ids'] = rest    # k*[seq]
                    bundle['context_text'] = c
                    bundle['reply_text'] = r
                    ids = self.vocab.encode(c)
                    if len(ids) < min_length:
                        continue
                    # length size of the context
                    ids = ids[-self.src_len_size:]
                    bundle['context_id'] = torch.LongTensor(ids)
                    ids = self.vocab.encode(r)
                    # length size of the context
                    ids = ids[:self.tgt_len_size]
                    bundle['reply_id'] = torch.LongTensor(ids)
            else:
                for c, r in tqdm(list(zip(contexts, responses))):
                    bundle = dict()
                    bundle['context_text'] = c
                    bundle['reply_text'] = r
                    ids = self.vocab.encode(c)
                    # NOTE: Data Augmentation Delete these two lines
                    # if len(ids) < min_length:
                    #     continue
                    # length size of the context
                    ids = ids[-self.src_len_size:]
                    bundle['context_id'] = torch.LongTensor(ids)
                    ids = self.vocab.encode(r)
                    # length size of the context
                    ids = ids[:self.tgt_len_size]
                    bundle['reply_id'] = torch.LongTensor(ids)
                    self.data.append(bundle)
        print(f'[!] read and process raw data from {path} over')

    def save_pickle(self):
        with open(self.pp_path, 'wb') as f:
            pickle.dump(self.data, f)
        print(f'[!] save dataset into {self.pp_path}')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

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
                 model_type='mc'):
        self.mode = mode
        self.max_len = max_len 
        data = read_text_data(path)
        responses = [i[1] for i in data]
        self.vocab = BertTokenizer.from_pretrained('bert-base-chinese')
        
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
            # max is better than min
            self.data = sorted(self.data, key=lambda i: max(len(i['ids'][0]), len(i['ids'][1])))
        else:
            for context, response in tqdm(d_):
                context_id = self.vocab.encode(context)
                # delete the [CLS] token of the response sequence for combining
                ids = [context_id + self.vocab.encode(r)[1:] for r in response]
                bundle = dict()
                bundle['ids'] = ids
                bundle['label'] = 0
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

class BERTIRDataset(Dataset):

    '''
    BERT IR Dataset
    1. train and dev mode only need the binary classification
    2. test mode needs to consider about the measurement
    
    Aspects:
    1. fluency
    2. coherence
    3. diversity
    4. naturalness: 
        * maybe the BM25 model is used for selecting the topic-close but unnatural response for the given context (intuitively, the BM25 retrieval-based dialog systems always unnatural for the given context, so maybe this operation is very helpful)
        * the human annotation should be statistised to analyze whether the responses retrieved by the BM25 is the unnatural.
        * naturalness may need the label filter algorithm to ignore the noises
    5. relatedness
    6. overall: coherence sampling with only 1 negative samples for training the aggration head. Overall may need the `naturalness` negative samples for training (harder negative samples is better).
    '''

    def __init__(self, path, mode='train', src_min_length=20, tgt_min_length=15, 
                 turn_size=3, max_len=300, samples=9, vocab_file='data/vocab/vocab_small',
                 negative_aspect='coherence', reduce=False, reduce_num=50000):
        self.mode = mode
        self.max_len = max_len 
        # data = read_csv_data(path)
        data = read_text_data(path)
        # context and response are all the negative samples 
        responses = [i[1] for i in data]
        # self.vocab = BertTokenizer(vocab_file=vocab_file)
        self.vocab = BertTokenizer.from_pretrained('bert-base-chinese')
        self.pp_path = f'{os.path.splitext(path)[0]}_{negative_aspect}.pkl'
        if os.path.exists(self.pp_path):
            with open(self.pp_path, 'rb') as f:
                self.data = pickle.load(f)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None
        self.data = []

        # collect the data samples
        d_ = []
        if negative_aspect == 'diversity':
            # PROCESS ALL THE DATASET AND FIND THE TOP-10000 in-diversity samnples
            diversity_path = f'{os.path.split(path)[0]}/diversity_scores.pkl'
            if os.path.exists(diversity_path):
                with open(diversity_path, 'rb') as f:
                    diversity_scores, responses_ = pickle.load(f)
                print(f'[!] load pre-trained diversity scores')
            else:
                NIDF = NIDF_TF()
                responses_ = random.sample(responses, 50000)
                idx, bsz_diveristy, diversity_scores = 0, 512, []
                with tqdm(total=len(responses_)) as pbar:
                    while idx < len(responses_):
                        samples_ = responses_[idx:idx+bsz_diveristy]
                        bsz_ = len(samples_)
                        diversity_scores.extend(NIDF.scores(samples_, topk=5))
                        pbar.update(bsz_)
                        idx += bsz_
                with open(diversity_path, 'wb') as f:
                    pickle.dump((diversity_scores, responses_), f)
                print(f'[!] save the pre-trained diversity scores into {diversity_path}')
            sort_index = np.argsort(diversity_scores)[:1000]
            diversity_negative = [responses_[i] for i in sort_index]
        elif negative_aspect == 'fluency':
            vocab_path = f'{os.path.split(path)[0]}/fluency_vocab.pkl'
            if os.path.exists(vocab_path):
                with open(vocab_path, 'rb') as f:
                    vocabs = pickle.load(f)
                print('[!] load preprosed vocab file for fluency perturbation')
            else:
                print(f'[!] begin to collect the vocabs for the fluency perturbation')
                vocabs = make_vocabs(responses)
                with open(vocab_path, 'wb') as f:
                    pickle.dump(vocabs, f)
                print(f'[!] save the vocabs in {vocab_path}')
        elif negative_aspect in ['relatedness', 'naturalness']:
            eschator = ESChat('zh50w_database', kb=False)
            if negative_aspect == 'relatedness':
                w2v = load_w2v('data/chinese_w2v')
        elif negative_aspect in ['coherence', 'overall']:
            pass
        else:
            raise Exception(f'[!] got unknow negative aspect {negative_aspect}')
            
        # reduce for fine tuning
        if reduce:
            random_idx = random.sample(range(len(data)), reduce_num)
            data = [data[i] for i in random_idx]
            
        if negative_aspect in ['naturalness', 'relatedness']:
            # too slow, use multisearch for speeding up
            with tqdm(total=len(data)) as pbar:
                idx, bsz = 0, 128
                while idx < len(data):
                    items = data[idx:idx+bsz]
                    bsz_ = len(items)
                    contexts_, responses_ = [i[0] for i in items], [i[1] for i in items]
                    if negative_aspect == 'naturalness':
                        negatives = generate_negative_samples_naturalness(contexts_, samples=samples, bm25Model=eschator, pool_size=128)
                    elif negative_aspect == 'relatedness':
                        # or "Semantic relatedness"
                        negatives = generate_negative_samples_relatedness(contexts_, samples=samples, bm25Model=eschator, pool_size=64, w2v=w2v, embedding_function=convert_text_embedding)
                    idx += bsz_
                    for c, r, n in zip(contexts_, responses_, negatives):
                        d_.append((c, [r] + n))
                    pbar.update(bsz_)
        else:
            for i in tqdm(data):
                context, response = i[0], i[1]
                if negative_aspect in ['coherence', 'overall']:
                    negative = generate_negative_samples(response, responses, samples=samples)
                elif negative_aspect == 'fluency':
                    negative = generate_negative_samples_fluency(response, samples=samples, vocab=vocabs)
                elif negative_aspect == 'diversity':
                    negative = generate_negative_samples_diversity(response, diversity_negative, samples=samples)
                else:
                    raise Exception(f'[!] got unkonow negative aspect {negative_aspect}')
                d_.append((context, [response] + negative))
        print(f'[!] collect the dataset over, prepare to process them')

        if mode in ['train', 'dev']:
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

class KWGPT2Dataset(Dataset):

    def __init__(self, path, mode='train', min_length=25, lang='zh', src_len_size=512, tgt_len_size=128):
        if lang == 'zh':
            vocab_file = 'data/vocab/vocab_small'
        else:
            vocab_file = 'data/vocab/vocab_english'
        self.mode = mode
        # tokenizer with addtional tokens
        self.vocab = BertTokenizer(vocab_file=vocab_file)
        additional_tokens = {'additional_special_tokens': ['[STP]']}
        self.vocab.add_special_tokens(additional_tokens)
        self.src_len_size, self.tgt_len_size = src_len_size, tgt_len_size
        #
        self.pp_path = f'{os.path.splitext(path)[0]}_kw.pkl'
        if os.path.exists(self.pp_path):
            with open(self.pp_path, 'rb') as f:
                self.data = pickle.load(f)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None
        # 
        data = read_text_data(path)
        self.data = []
        if self.mode in ['train', 'dev']:
            contexts = [i[0] for i in data]
            responses = [i[1] for i in data]
            for ctx, res in tqdm(list(zip(contexts, responses))):
                bundle = dict()
                # obtain the keywords
                keywords = kwgpt2_utils(res)
                if keywords is None:
                    continue
                sample = f'{ctx} [STP] {keywords} [STP] {res} [STP]'
                bundle['context_text'] = sample
                ids = self.vocab.encode(sample)[:-1]
                if len(ids) < min_length:
                    continue
                ids = ids[-self.src_len_size:]
                bundle['context_id'] = torch.LongTensor(ids)
                self.data.append(bundle)
            self.data = sorted(self.data, key=lambda x: len(x['context_id']))
        else:
            contexts = [i[0] for i in data]
            responses = [i[1] for i in data]
            for ctx, res in tqdm(list(zip(contexts, responses))):
                # DO NOT ADD THE Keywords for the test running mode
                bundle = dict()
                # obtain the keywords
                keywords = kwgpt2_utils(res)
                if keywords is None:
                    continue
                bundle['context_text'] = ctx
                ctx = f'{ctx} [STP]'
                ids = self.vocab.encode(ctx)[:-1]    # ignore the final [SEP]
                if len(ids) < min_length:
                    continue
                ids = ids[-self.src_len_size:]
                bundle['context_id'] = torch.LongTensor(ids)
                ids = self.vocab.encode(f'{keywords} [STP] {res} [STP]')[1:]   # ignore the [CLS]
                bundle['reply_id'] = torch.LongTensor(ids)
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
    # ========== Traditional IR ========= #
    # train_data = IRDataset(
    #         'data/zh50w/train.csv', 
    #         'data/zh50w/train.pkl', 
    #         mode='train', 
    #         samples=9)
    # train_iter = DataLoader(train_data, shuffle=True, batch_size=10, collate_fn=ir_collate_fn)
    # ========== GPT2Dataset ========== #
    # train_data = GPT2Dataset('./data/zh50w/train.txt', mode='train')
    # train_iter = DataLoader(train_data, batch_size=10, shuffle=False, collate_fn=gpt2_train_collate_fn)
    # ========== MultiGPT2 ========== #
    # train_data = MultiGPT2Dataset('./data/zh50w/train.csv', mode='train')
    # train_iter = DataLoader(train_data, shuffle=True, batch_size=10, collate_fn=multigpt2_train_collate_fn)
    # ========== BERTIRDataset ========== #
    # train_data = BERTIRDataset('data/zh50w/train.txt', mode='train', samples=9, negative_aspect='coherence')
    # train_iter = DataLoader(train_data, shuffle=True, batch_size=10, collate_fn=bert_ir_test_collate_fn)
    # ========== BERTIRCLDataset ========== #
    train_data = BERTIRCLDataset('data/zh50w/train.txt', mode='train', samples=1)
    train_data.save_pickle()
    train_iter = BERTIRCLDataLoader(train_data, int(len(train_data) * 5 / 10), 
                                    batch_size=10)
    train_iter.forLoss = True
    # ========== BERTIR ========== #
    # train_data = BERTNLIDataset('data/NLI/cnsd_snli_v1.0.train.jsonl', mode='train')
    # train_iter = DataLoader(train_data, shuffle=True, batch_size=10, collate_fn=nli_collate_fn)
    
    # ========= ITERATE ========= # 
    # for batch in tqdm(train_iter):
    #     ipdb.set_trace()
    # ========== CUSTOM ITERATOR ========== #
    with tqdm(total=len(train_iter)) as pbar:
        for batch in train_iter:
            ipdb.set_trace()
            pbar.update(len(batch[0]))
