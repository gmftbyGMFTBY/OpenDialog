from .header import *
from .bert_retrieval import BERTRetrieval

class GPT2RL(nn.Module):

    '''
    past mechanism can speed up and decrease the capacity of the cuda memory
    In order to run the batch version, the GPT2RL contains the generator and discriminator
    Rollout with top_k_top_p mechanism are used
    
    In the debug mode, the attribute `memory` is not NoneType
    '''

    def __init__(self, min_token, rollout_samples,
                 generative_path, topk, topp,
                 config_path='data/config/model_config_dialogue_small.json',
                 vocab_path='data/vocab/vocab_small',
                 memory=None):
        super(GPT2RL, self).__init__()
        # vocab
        self.vocab = BertTokenizer(vocab_file=vocab_path)
        vocab_size = len(self.vocab)
        self.debug = True if memory else False
        self.memory = memory

        # generator
        self.model_config = GPT2Config.from_json_file(config_path)
        self.generator = GPT2LMHeadModel(config=self.model_config)
        self.generator.resize_token_embeddings(vocab_size)
        # discriminator
        self.discriminator = BERTRetrieval()

        # essential parameters
        self.n_ctx = self.generator.config.to_dict().get('n_ctx')
        self.sample_topk, self.rl_topk = topk
        self.sample_topp, self.rl_topp = topp
        self.unk_id = self.vocab.convert_tokens_to_ids('[UNK]')
        self.sep_id = self.vocab.convert_tokens_to_ids('[SEP]')
        self.cls_id = self.vocab.convert_tokens_to_ids('[CLS]')
        self.pad_id = self.vocab.convert_tokens_to_ids('[PAD]')
        self.rollout_samples = rollout_samples
        self.min_token_to_keep = min_token

        # load the pretrained model
        self.load_model(self.generator, generative_path)
        print(f'[!] load the pretrained model (MLE) over')
    
    def load_model(self, model, path):
        state_dict = torch.load(path)
        try:
            model.load_state_dict(state_dict)
        except:
            current_module = True if 'model' in [i[0] for i in model.state_dict().items()][0] else False
            saved_module = True if 'model' in [i[0] for i in state_dict.items()][0] else False
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if saved_module and not current_module:
                    name = k[6:]
                    new_state_dict[name] = v
                elif not saved_module and current_module:
                    name = f'model.{k}'
                    new_state_dict[name] = v
                else:
                    pass
            model.load_state_dict(new_state_dict)
        print(f'[!] load model from {path}')

    @torch.no_grad()
    def generative_predict(self, inpt_ids, max_len):
        '''
        Generate the fake data for the discriminator
        inpt_ids: [batch, seq]
        return: [max_len, batch]
        '''
        generated = []    # [max_len, batch], a list of tensor
        prev, past = inpt_ids.clone().detach(), None
        sep_batch_index = [0] * len(inpt_ids)
        sep_batch_flag = [0] * len(inpt_ids)
        for i in range(max_len):
            outputs = self.generator(input_ids=prev, past=past)
            outputs, past = outputs[:2]
            next_token_logits = outputs[:, -1, :]    # [batch, vocab]
            # ignore the [UNK] token
            next_token_logits[:, self.unk_id] = -np.inf
            filtered_logits = top_k_top_p_filtering_batch(
                    next_token_logits, 
                    top_k=self.sample_topk, 
                    top_p=self.sample_topp,
                    min_token_to_keep=self.min_token_to_keep)
            next_token = torch.multinomial(
                    F.softmax(filtered_logits, dim=-1),
                    num_samples=1)    # [batch, 1]
            for idx, token_ in enumerate(next_token.squeeze(1)):
                if sep_batch_flag[idx] == 0 and token_ == self.sep_id:
                    sep_batch_index[idx] = i + 1
                    sep_batch_flag[idx] = 1
            prev = next_token
            generated.append(next_token.squeeze(1).tolist())
        for i in range(len(sep_batch_flag)):
            if sep_batch_flag[i] == 0:
                sep_batch_index[i] = max_len
        return self.process_generated(generated, sep_batch_index, max_len)

    def process_generated(self, generated, index, max_len):
        '''
        Process the generated samples for the training the discriminator
        1. filter tokens after [SEP]
        2. pad and return
        3. the last token of the sequence maybe the [PAD], [SEP] and other tokens

        :generated: [max_len, batch]
        :index: [batch]
        '''
        rest = []
        for idx in range(len(index)):
            r = [item[idx] for item in generated]
            r = r[:index[idx]]
            if len(r) == 0 or (len(r) < max_len and r[-1] != self.sep_id):
                r.append(self.sep_id)
            # if len(r) == max_len and r[-1] != self.sep_id:
            #     r[-1] = self.sep_id
            rest.append(torch.LongTensor(r))
        rest = pad_sequence(rest, batch_first=True, padding_value=self.pad_id)
        rest_len, batch_size = len(rest[0]), len(rest)
        if rest_len < max_len:
            add_pad = torch.LongTensor([[self.pad_id] * (max_len-rest_len)]*batch_size)
            rest = torch.cat((rest, add_pad), dim=1)    # [batch, max_len]
        if torch.cuda.is_available():
            rest = rest.cuda()
        return rest    # [batch, seq]

    @torch.no_grad()
    def rollout_batch(self, past, current_token, max_len):
        '''
        Batch predict mode
        return :generated: [batch, max_len]
        Rollout speed up with `past` mechanism

        Also need to use `sep_batch_index`
        '''
        batch_size = len(current_token)
        sep_batch_index = [[0] * batch_size for _ in range(self.rollout_samples)]
        rollout_rest = []
        for rollout_idx in range(self.rollout_samples):
            response = []
            sep_batch_flag = [0] * batch_size
            past_ = tuple([i.clone().detach() for i in past]) 
            current_token_ = current_token.clone().detach()
            for lid in range(max_len):
                outputs = self.generator(input_ids=current_token_, past=past_)
                outputs, past_ = outputs[:2]
                next_token_logits = outputs[:, -1, :]    # [batch, vocab]
                next_token_logits[:, self.unk_id] = -np.inf
                # NOTE
                # filtered_logits = top_k_top_p_filtering_batch(
                #         next_token_logits, 
                #         top_k=self.sample_topk, 
                #         top_p=self.sample_topp,
                #         min_token_to_keep=self.min_token_to_keep)
                # next_token = torch.multinomial(
                #         F.softmax(filtered_logits, dim=-1),
                #         num_samples=1)    # [batch, 1]
                next_token = torch.multinomial(
                        F.softmax(next_token_logits, dim=-1),
                        num_samples=1)    # [batch, 1]
                current_token_ = next_token
                # Action: next_token
                for idx, token_ in enumerate(next_token.squeeze(1)):
                    if sep_batch_flag[idx] == 0 and token_ == self.sep_id:
                        sep_batch_index[rollout_idx][idx] = lid+1
                        sep_batch_flag[idx] = 1
                response.append(next_token.squeeze(1))    # [batch]
            # re check sep_batch_index
            for idx, v_ in enumerate(sep_batch_flag):
                if v_ == 0:
                    sep_batch_index[rollout_idx][idx] = max_len
            response = torch.stack(response).transpose(0, 1)    # [batch, max_len]
            # replace the token after [SEP] with the [PAD]
            for iidx, idx in enumerate(sep_batch_index[rollout_idx]):
                response[iidx][idx:] = self.pad_id
            rollout_rest.append(response)    # rollout_samples*[(batch, max_len)]
        return rollout_rest 

    @torch.no_grad()
    def obtain_rewards(self, rollout_samples):
        '''
        Use discriminator to generate the rewards (scores) for the rollout samples
        :rollout_samples: self.rollout_samples*[batch, max_len]
        return :rewards: [batch] (average strategy are used)

        In order to use the DataParallel, cannot use the `.cuda` method

        Rewards range: [0, 100]
        '''
        rewards = torch.zeros(len(rollout_samples[0]))    # [batch]
        for samples in rollout_samples:
            output = F.softmax(self.discriminator(samples), dim=-1)[:, 1].cpu()
            rewards += output
        return 100 * rewards / len(rollout_samples)

    def forward_mle(self, cid):
        '''
        Train Generator with MLE
        '''
        self.generator.train()
        attn_mask = generate_attention_mask(cid)
        logits = self.generator(cid, attention_mask=attn_mask)
        return logits[0]

    def forward_dis(self, inpt):
        '''
        Train Discrimintator
        '''
        self.discriminator.train()
        output = self.discriminator(inpt)
        # output: [batch, 2]
        return output 

    def forward_rl(self, inpt_ids, max_len):
        '''
        Train Generator with RL
        inpt_ids: [batch, seq]
        return actions and corresponding probs:
            torch tensor object: [batch, max_len], [batch, max_len]; [batch]
        '''
        self.generator.train()
        generated, probs = [], []
        rewards = []    # max_len*[batch]
        inpt_ids_ = inpt_ids.clone().detach()
        sep_batch_index = torch.LongTensor([0] * len(inpt_ids))
        sep_batch_flag = [0] * len(inpt_ids)
        prev, past, history = inpt_ids_, None, inpt_ids_
        for i in range(1, max_len+1):
            outputs = self.generator(input_ids=prev, past=past)
            outputs, past = outputs[:2]
            next_token_logits = outputs[:, -1, :]    # [batch, vocab]
            next_token_logits[:, self.unk_id] = -np.inf
            # NOTE: using all the tokens
            # filtered_logits = top_k_top_p_filtering_batch(
            #         next_token_logits,
            #         top_k=self.rl_topk,
            #         top_p=self.rl_topp,
            #         min_token_to_keep=self.min_token_to_keep)    # [batch, vocab]
            # filtered_logits = F.softmax(filtered_logits, dim=-1)    # [batch, vocab]
            filtered_logits = F.softmax(next_token_logits, dim=-1)    # [batch, vocab]
            # Action: next_token
            next_token = torch.multinomial(
                    filtered_logits,
                    num_samples=1)    # [batch, 1]
            next_token = next_token.squeeze(1)    # [batch]
            for idx, token_ in enumerate(next_token):
                if sep_batch_flag[idx] == 0 and token_ == self.sep_id:
                    sep_batch_index[idx] = i
                    sep_batch_flag[idx] = 1
            # prob
            prob = filtered_logits[range(len(next_token)), next_token]   # [batch]
            generated.append(next_token)
            probs.append(prob)
            # prev
            prev = next_token.unsqueeze(1)
            # rollout
            # rest: rollout_samples*[batch, seq]
            if i < max_len:
                # rest: rollout_samples*[batch, max_len]
                rest = self.rollout_batch(past, next_token.unsqueeze(1), max_len-i)
                generated_ = torch.stack(generated).transpose(0, 1)    # [batch, seq]
                rest = [torch.cat((inpt_ids, generated_, rollout_item), dim=1) for rollout_item in rest]
            else:
                # reset the sep_batch_index to the max_len
                for idx in range(len(sep_batch_index)):
                    if sep_batch_flag[idx] == 0:
                        sep_batch_index[idx] = max_len 
                generated_ = torch.stack(generated).transpose(0, 1)    # [batch, seq]
                # [PAD]
                for idx, index in enumerate(sep_batch_index):
                    generated_[idx][index:] = self.pad_id
                rest = [torch.cat((inpt_ids, generated_), dim=1)]    # 1*[batch, seq]
            # obtain the rewards
            rewards.append(self.obtain_rewards(rest))
        rewards = torch.stack(rewards).transpose(0, 1).cuda()    # [batch, max_len]
        probs = torch.stack(probs).transpose(0, 1)    # [batch, max_len]
        # sep_batch_index: [batch]
        sep_batch_index = sep_batch_index.cuda()
        return_data = {
                'probs': probs, 
                'rewards': rewards, 
                'sep_batch_index': sep_batch_index}
        return return_data

    def save_memory(self, path):
        '''
        If debug mode is True, save the memory in the ckpt folder
        '''
        with open(path, 'wb') as f:
            pickle.dump(self.memory, f)
        print(f'[!] save the debug memory into {path}')

    def forward(self, cid, rid=None, max_size=None, mode='gen_rl'):
        '''
        Compatible for DataParallel
        '''
        if mode == 'gen_rl':
            # rid won't be used
            assert max_size, f'[!] max_size must not be NoneType for gen_rl mode'
            data = self.forward_rl(cid, max_size)
            return data
        elif mode == 'gen_mle':
            if rid is None:
                raise Exception('[!] rid must not be NoneType for gen_mle mode')
            cid = torch.cat((cid, rid[:, 1:]), dim=1)    # [batch, seq]
            shift_logits = self.forward_mle(cid)
            return shift_logits
        elif mode == 'dis':
            if rid is None:
                raise Exception('[!] rid must not be NoneType for gen_mle mode')
            cid = torch.cat((cid, rid), dim=1)    # [batch, c_len+r_len]
            output = self.forward_dis(cid)    # [batch, 2]
            return output
        elif mode == 'gen_predict':
            f_rid = self.generative_predict(cid, max_size)
            return f_rid
        else:
            raise Exception(f'[!] Except to get [gen_rl; gen_mle; dis] mode, but got {mode}')

class GPT2RLAgent(BaseAgent):

    '''
    1. action space: action space (topk, topp) will decrease by the time
    '''

    def __init__(self, multi_gpu, vocab_file='data/vocab/vocab_small', run_mode='train', lang='zh', local_rank=0):
        super(GPT2RLAgent, self).__init__()
        # hyperparameters
        try:
            self.gpu_ids = list(range(len(multi_gpu.split(','))))
        except:
            raise Exception(f'[!] except to obtain the multiple gpu ids, but got {multi_gpu}')
        vocab_file = 'data/vocab/vocab_small' if lang == 'zh' else 'data/vocab/vocab_english'
        assert run_mode in ['train', 'test'], f'[!] runing mode must be train or test, but got {run_mode}'
        self.args = {
            'gen_lr': 1e-5,
            'gen_mle_lr': 1.5e-5, 
            'dis_lr': 1e-6,
            'gen_step': 1,
            'dis_step': 1, 
            'grad_clip': 3.0,
            'pad': 0,
            'tgt_len_size': 20,
            'test_tgt_len_size': 50,
            'sample_topk': 20,
            'sample_topp': 1.0,
            'rl_topk': 500,
            'rl_topp': 1.0,
            'config_path': 'data/config/model_config_dialogue_big.json',
            'vocab_path': vocab_file,
            'gen_pretrained_model': 'ckpt/zh50w/gpt2/best.pt',
            'rollout_samples': 2,
            'min_token_to_keep': 10,
            'multi_gpu': self.gpu_ids,
            'run_mode': run_mode,
            'debug': False,
            'memory': 1000,
            'memory_batch_size': 16,
            'memory_ckpt_path': 'ckpt/train_generative_rl/gpt2gan/memory.pkl',
            'lang': lang,
            'amp_level': 'O2',
            'local_rank': 0,
        }
        # hyperparameters
        
        # debug memory: save the generated samples during RL training
        if self.args['debug']:
            self.debug_memory = ReplayMemory(
                    self.args['memory'],
                    BertTokenizer(vocab_file=self.args['vocab_path']),
                    self.args['memory_batch_size'])
        else:
            self.debug_memory = None

        # Generator
        self.model = GPT2RL(
            self.args['min_token_to_keep'],
            self.args['rollout_samples'],
            self.args['gen_pretrained_model'],
            (self.args['sample_topk'], self.args['rl_topk']), 
            (self.args['sample_topp'], self.args['rl_topp']),
            config_path=self.args['config_path'],
            vocab_path=self.args['vocab_path'],
            memory=self.debug_memory,
        )
        # CUDA and DataParallel
        if torch.cuda.is_available():
            self.model.cuda()
        
        # optimizer
        self.gen_optimizer = optim.Adam(
                self.model.generator.parameters(),
                lr=self.args['gen_lr'])
        self.gen_mle_optimizer = transformers.AdamW(
                self.model.generator.parameters(),
                lr=self.args['gen_mle_lr'],
                correct_bias=True)
        self.dis_optimizer = transformers.AdamW(
                self.model.discriminator.parameters(),
                lr=self.args['dis_lr'])
        # self.model, [self.gen_optimizer, self.gen_mle_optimizer, self.dis_optimizer] = amp.initialize(
        #     self.model, 
        #     [self.gen_optimizer, self.gen_mle_optimizer, self.dis_optimizer], 
        #     opt_level=self.args['amp_level']
        # )
        # DataParallel
        if self.args['run_mode'] == 'train':
            self.model = DataParallel(self.model, device_ids=[local_rank], output_device=local_rank)
        # criterion
        self.dis_criterion = nn.CrossEntropyLoss()
        self.mle_criterion = nn.CrossEntropyLoss(
                ignore_index=self.args['pad'], reduction='sum')
        self.show_parameters(self.args)

    def train_discriminator(self, cid, rid, label):
        self.dis_optimizer.zero_grad()
        output = self.model(cid, rid=rid, mode='dis')    # [batch, 2]
        loss = self.dis_criterion(output, label)
        with amp.scale_loss(loss, self.dis_optimizer) as scaled_loss:
            scaled_loss.backward()
        clip_grad_norm_(amp.master_params(self.dis_optimizer), self.args['grad_clip'])
        # loss.backward()
        # clip_grad_norm_(
        #         self.model.module.discriminator.parameters(), 
        #         self.args['grad_clip'])
        self.dis_optimizer.step()
        
        now_correct = torch.max(F.softmax(output, dim=-1), dim=-1)[1]
        now_correct = torch.sum(now_correct == label).item()
        loss = round(loss.item(), 4)
        now_correct = round(now_correct / len(cid), 4)
        return loss, now_correct

    def train_generator_rl(self, cid, rid, max_size):
        # probs/rewards: [batch, max_len]; sep_batch_index: [batch]
        self.gen_optimizer.zero_grad()
        return_data = self.model(cid, rid=rid, max_size=max_size, mode='gen_rl')
        probs = return_data['probs']
        rewards = return_data['rewards']
        sep_batch_index = return_data['sep_batch_index']
        # Batch Policy Gradient
        rl_loss, stepR = 0, 0
        batch_size, seq_length = len(probs), len(probs[0])
        for r, p, idx in zip(rewards, probs, sep_batch_index):
            rewards_, probs_ = r[:idx], p[:idx]
            for r, p in list(zip(rewards_, probs_)):
                rl_loss += -torch.log(p) * r
            stepR += np.mean(rewards_.tolist())
        with amp.scale_loss(rl_loss, self.gen_optimizer) as scaled_loss:
            scaled_loss.backward()
        clip_grad_norm_(amp.master_params(self.gen_optimizer), self.args['grad_clip'])
        # rl_loss.backward()
        # clip_grad_norm_(
        #         self.model.module.generator.parameters(), 
        #         self.args['grad_clip'])
        self.gen_optimizer.step()
        stepR /= batch_size
        avg_step = round(np.mean(sep_batch_index.tolist()), 4) 
        return rl_loss.item()/batch_size, avg_step, stepR

    def train_generator_mle(self, cid, rid):
        '''
        cid/rid: [batch, seq]; [CLS] begin and [SEP] end
        '''
        self.gen_mle_optimizer.zero_grad()
        shift_logits = self.model(cid, rid=rid, mode='gen_mle')
        cid_ = torch.cat((cid, rid[:, 1:]), dim=1)
        shift_logits = shift_logits[..., :-1, :].contiguous()
        shift_labels = cid_[..., 1:].contiguous()
        loss = self.mle_criterion(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1))
        _, preds = shift_logits.max(dim=-1)
        not_ignore = shift_labels.ne(self.args['pad'])
        num_targets = not_ignore.long().sum().item()
        correct = (shift_labels == preds) & not_ignore
        correct = correct.float().sum()
        accuracy = correct / num_targets
        loss = loss / num_targets
        with amp.scale_loss(loss, self.gen_mle_optimizer) as scaled_loss:
            scaled_loss.backward()
        clip_grad_norm_(amp.master_params(self.gen_mle_optimizer), self.args['grad_clip'])
        # loss.backward()
        # clip_grad_norm_(
        #        self.model.module.generator.parameters(), 
        #         self.args['grad_clip'])
        self.gen_mle_optimizer.step()
        return loss.item(), accuracy.item() 

    def prepare_discriminator_data(self, cid, rid, max_size):
        '''
        sample and prepare the data for training the discriminator (generate; shuffle; cuda)
        cid: [batch, seq]
        rid: [batch, seq]; contains the [CLS] token
        '''
        # f_rid: [batch, seq] without the [CLS] token
        # generate the fake data by sampling from the generator
        batch_size = len(cid)
        f_rid = self.model(cid, max_size=max_size, mode='gen_predict')
        # combine the data (fake and true)
        cid = torch.cat((cid, cid), dim=0)    # [2*batch, seq]
        # NOTE: the seq_len of `rid` maybe different from the f_rid (max_len)
        # pad is 0 for `transformers`
        rid = rid[:, 1:]
        if f_rid.shape[-1] != rid.shape[-1]:
            pad4rid = torch.zeros(batch_size, f_rid.shape[-1] - rid.shape[-1]).long()
            if torch.cuda.is_available():
                pad4rid = pad4rid.cuda()
            rid = torch.cat((rid, pad4rid), dim=1)    # [batch, max_len]
        rid = torch.cat((rid, f_rid), dim=0)    # [2*batch, seq]
        label = torch.LongTensor([1] * batch_size + [0] * batch_size)
        if torch.cuda.is_available():
            label = label.cuda()
        # shuffle
        random_idx = list(range(2*batch_size))
        random.shuffle(random_idx)
        cid = cid[random_idx]
        rid = rid[random_idx]
        label = label[random_idx]
        # return: [2*batch, seq] (cid/rid) | [2*batch] (label)
        return cid, rid, label

    def train_model(self, train_iter, mode='train', recoder=None, idx_=0):
        '''
        Core logic of the GPT2RL Model
        1. train discriminator with binary classification
        2. train generator with Policy Graident
        3. train generator with MLE
        '''
        dataparallel_error, dataparallel_error_samples = 0, 0
        oom_time, oom_time_samples = 0, 0
        dis_acc, dis_loss = [], []
        rl_loss_, rl_reward_, rl_step_ = [], [], []
        # ========== D step ========== #
        for didx in range(self.args['dis_step']):
            with tqdm(total=len(train_iter)) as pbar:
                for idx, batch in enumerate(train_iter):
                    try:
                        cid, rid, _ = batch    # [batch, seq]
                        max_size = min(rid.shape[-1], self.args['tgt_len_size'])
                        batch_size = len(cid)
                        cid, rid, label = self.prepare_discriminator_data(cid, rid, max_size)
                        loss, acc = self.train_discriminator(cid, rid, label)
                        log_str = f'[!] {didx+1}/{self.args["dis_step"]}; dis-loss: {loss}; dis-acc: {acc}'
                        dis_acc.append(acc)
                        dis_loss.append(loss)
                        recoder.add_scalar(f'train-epoch-{idx_}/DisRunLoss', loss, idx)
                        recoder.add_scalar(f'train-epoch-{idx_}/DisRunAcc', acc, idx)
                        pbar.set_description(log_str)
                        pbar.update(batch_size)
                        # print(log_str, file=open(recoder, 'a'))
                    except RuntimeError as exception:
                        if 'out of memory' in str(exception):
                            oom_time += 1
                            oom_time_samples += batch_size
                            torch.cuda.empty_cache()
                            pbar.set_description(f'[!] occur OOM Error: {oom_time}|{oom_time_samples}')
                        else:
                            raise exception
                    except TypeError as exception:
                        # the Error made by the DataParallel
                        ipdb.set_trace()
                        dataparallel_error += 1
                        dataparallel_error_samples += batch_size
                        pbar.set_description(f'[!] occur DataParallel Error: {dataparallel_error}|{dataparallel_error_samples}')
                recoder.add_scalar(f'train-whole/DisAcc', np.mean(dis_acc), idx_)
                recoder.add_scalar(f'train-whole/DisLoss', np.mean(dis_loss), idx_)
        # ========== G step ========== #
        for gidx in range(self.args['gen_step']):
            with tqdm(total=len(train_iter)) as pbar:
                for batch in train_iter:
                    try:
                        # cid, rid = batch    # [batch, seq]
                        cid, rid, rid_length = batch
                        # rid_length is the average length of the ground-truth responses (reference)
                        rid_length = round(np.mean(rid_length), 2)
                        max_size = min(rid.shape[-1], self.args['tgt_len_size'])
                        batch_size = len(cid)
                        # train the generator with RL(Policy Gradient)
                        rl_loss, avg_step, stepR = self.train_generator_rl(
                                cid, None, max_size)
                        rl_loss, stepR = round(rl_loss, 4), round(stepR, 4)
                        rl_loss_.append(rl_loss)
                        rl_reward_.append(stepR)
                        rl_step_.append(avg_step)
                        
                        # NOTE:
                        # MLE Training
                        # mle_loss, mle_acc = self.train_generator_mle(cid, rid)
                        # mle_loss, mle_acc = round(mle_loss, 4), round(mle_acc, 4)
                        # log_str = f'[!] {gidx+1}/{self.args["gen_step"]}; loss(rl/mle): {rl_loss}/{mle_loss}; rl_inf(step/reward): {avg_step}({rid_length})/{stepR}; mle_acc: {mle_acc}'
                        log_str = f'[!] {gidx+1}/{self.args["gen_step"]}; loss(rl): {rl_loss}; rl_inf(step/reward): {avg_step}({rid_length})/{stepR};'
                        recoder.add_scalar(f'train-epoch-{idx_}/RLRunLoss', rl_loss, idx)
                        recoder.add_scalar(f'train-epoch-{idx_}/Reward', stepR, idx)
                        recoder.add_scalar(f'train-epoch-{idx_}/Step', avg_step, idx)
                        pbar.set_description(log_str)
                        pbar.update(batch_size)
                        # rint(log_str, file=open(recoder, 'a'))
                        # debug mode
                        if self.args['debug']:
                            self.model.module.save_memory(self.args['memory_ckpt_path'])
                    except RuntimeError as exception:
                        if 'out of memory' in str(exception):
                            oom_time += 1
                            oom_time_samples += batch_size
                            torch.cuda.empty_cache()
                            pbar.set_description(f'[!] occur OOM Error: {oom_time}|{oom_time_samples}')
                        else:
                            raise exception
                    except TypeError as exception:
                        # the Error made by the DataParallel
                        dataparallel_error += 1
                        dataparallel_error_samples += batch_size
                        pbar.set_description(f'[!] occur DataParallel Error: {dataparallel_error}|{dataparallel_error_samples}')
                recoder.add_scalar(f'train-whole/RLLoss', np.mean(rl_loss_), idx_)
                recoder.add_scalar(f'train-whole/RLReward', np.mean(rl_reward_), idx_)
                recoder.add_scalar(f'train-whole/RLStep', np.mean(rl_step_), idx_)
                        
        print(f'[!] DataParallel Error: {dataparallel_error}|{dataparallel_error_samples}')
        print(f'[!] OOM Error: {oom_time}|{oom_time_samples}')

    def save_model(self, path):
        '''
        save generator and discrimintor separately
        '''
        file_name = os.path.splitext(path)[0]
        dis_path = f'{file_name}_dis.pt'
        gen_path = f'{file_name}_gen.pt'
        state_dict = self.model.module.discriminator.state_dict()
        torch.save(state_dict, dis_path)
        state_dict = self.model.module.generator.state_dict()
        torch.save(state_dict, gen_path)
        print(f'[!] save discriminator model into {dis_path}')
        print(f'[!] save generator model into {gen_path}')

    def load_model(self, path):
        '''
        Load the generator and discriminator model
        No DataParallel
        path: ckpt/train_generative_rl/gpt2gan/best.pt
        '''
        path = os.path.splitext(path)[0]
        dis_path = f'{path}_dis.pt'
        gen_path = f'{path}_gen.pt'

        state_dict = torch.load(gen_path)
        self.model.generator.load_state_dict(state_dict)
        print(f'[!] load model from {gen_path}')
        
        state_dict = torch.load(dis_path)
        self.model.discriminator.load_state_dict(state_dict)
        print(f'[!] load model from {dis_path}')
    
    def test_model(self, test_iter, path):
        '''
        Generate the test dataset and measure the performance
        Test model NO DataParallel
        Batch_size is 1

        Test the generator model
        '''
        def filter(x):
            return x.replace('[PAD]', '')
        self.model.eval()
        with tqdm(total=len(test_iter)) as pbar:
            with open(path, 'w') as f:
                for batch in test_iter:
                    c, r, _ = batch    # [1, seq]
                    max_size = max(len(r), self.args['test_tgt_len_size'])
                    tgt = self.model.generative_predict(c, max_size)[0]
                    text = self.model.vocab.convert_ids_to_tokens(tgt)
                    tgt = filter(''.join(text))
                    c = c[0]
                    ctx = self.model.vocab.convert_ids_to_tokens(c)
                    ctx = filter(''.join(ctx))
                    r = r[0]
                    ref = self.model.vocab.convert_ids_to_tokens(r)
                    ref = filter(''.join(ref))
    
                    f.write(f'CTX: {ctx}\n')
                    f.write(f'REF: {ref}\n')
                    f.write(f'TGT: {tgt}\n\n')

                    pbar.update(len(c))
            print(f'[!] translate test dataset over, write into {path}')
        # measure the performance
        (b1, b2, b3, b4), ((r_max_l, r_min_l, r_avg_l), (c_max_l, c_min_l, c_avg_l)), (dist1, dist2, rdist1, rdist2), (average, extrema, greedy) = cal_generative_metric(path)
        print(f'[TEST] BLEU: {b1}/{b2}/{b3}/{b4}; Length(max, min, avg): {c_max_l}/{c_min_l}/{c_avg_l}|{r_max_l}/{r_min_l}/{r_avg_l}; Dist: {dist1}/{dist2}|{rdist1}/{rdist2}; Embedding(average/extrema/greedy): {average}/{extrema}/{greedy}')
