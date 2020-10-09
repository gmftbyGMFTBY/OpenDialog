from .header import *
from .bert_retrieval import BERTRetrieval

class GPT2V2RL(nn.Module):

    def __init__(self, vocab_size, unk_id, sep_id, cls_id, pad_id, topk, topp, 
                 repetition_penalty,
                 config_path='data/config/model_config_dialogue_small.json', 
                 embedding_size=300, policy_size=32, action_std=0.5, k_epochs=10, 
                 eps_clip=0.2):
        super(GPT2V2RL, self).__init__()
        self.model_config = GPT2Config.from_json_file(config_path)
        self.model = GPT2Model(config=self.model_config)
        self.model.resize_token_embeddings(vocab_size)
        self.n_ctx = self.model.config.to_dict().get('n_ctx')
        self.n_embd = self.model.config.to_dict().get('n_embd')
        self.topk, self.topp = topk, topp
        self.unk_id = unk_id
        self.sep_id = sep_id
        self.cls_id = cls_id
        self.pad_id = pad_id
        self.action_std, self.eps_clip = action_std, eps_clip
        self.k_epoch, self.embedding_size, self.policy_size = k_epochs, embedding_size, policy_size
        self.repetition_penalty = repetition_penalty
        
        self.agent = ActorCritic(policy_size, embedding_size, action_std=action_std)
        self.proj = nn.Linear(self.n_embd + policy_size, vocab_size)
    
    def update(self, memory, criterion, optimizer):
        '''update the policy network; return the loss'''
        # construct the rewards (one-step MDP without gamma ratio):
        rewards = torch.tensor(memory.rewards, dtype=torch.float)
        rewards = to_cuda((rewards - rewards.mean()) / (rewards.std() + 1e-5))    # Normalizing the rewards
        
        # construct the state, action, probability; [MEMORY_SIZE, S/A]
        old_states = to_cuda(torch.stack(memory.states)).detach()
        old_actions = to_cuda(torch.stack(memory.actions)).detach()
        old_logprobs = to_cuda(torch.stack(memory.logprobs)).detach()
        
        # optimizing for K epochs:
        losses = []
        for _ in range(self.k_epoch):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            ratios = torch.exp(logprobs - old_logprobs)
            advatanges = rewards - state_values.detach()
            surr1 = ratios * advatanges
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advatanges
            # actor loss
            loss = - torch.min(surr1, surr2) - 0.01 * dist_entropy
            # add the critic loss
            loss += 0.5 * criterion(state_values, rewards)
            losses.append(loss.item())
            
            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()
        return np.mean(losses)

    @torch.no_grad()
    def predict_batch(self, inpt_ids, attn_mask, position_ids, context_embd, response_embd, max_len):
        '''past parameter and position_ids parameters should be careful
        https://github.com/huggingface/transformers/issues/3021#issuecomment-681792104'''
        batch_size = inpt_ids.shape[0]
        generated = [[self.cls_id] * batch_size]
        prev, past = inpt_ids, None
        stop_flag = np.zeros(batch_size)
        policy_embd = self.agent.act(torch.cat([context_embd, response_embd], dim=-1))   # [batch, 32]
        for _ in range(max_len):
            outputs = self.model(
                input_ids=prev,
                attention_mask=attn_mask,
                position_ids=position_ids,
                past=past,
            )
            output, past = outputs[:2]
            output = output[:, -1, :]    # [batch, 768]
            output = torch.cat([output, policy_embd], dim=-1)    # [batch, 768+32]
            next_token_logits = self.proj(output)    # [batch, vocab]
            next_token_logits[:, self.unk_id] = -np.inf
            # repetition penalty
            for x in range(batch_size):
                y = [item[x] for item in generated]
                next_token_logits[x, y] /= self.repetition_penalty
            filtered_logits = top_k_top_p_filtering_batch(
                next_token_logits,
                top_k=self.topk, 
                top_p=self.topp,
            )
            next_token = torch.multinomial(
                F.softmax(filtered_logits, dim=-1),
                num_samples=1,
            )    # [batch, 1]
            for idx, i in enumerate(next_token.squeeze(1)):
                if i.item() == self.sep_id:
                    stop_flag[idx] = 1
            generated.append([token.item() for token in next_token.squeeze(1)])
            prev = next_token
            if sum(stop_flag) == batch_size:
                break
            attn_mask = torch.cat([attn_mask, torch.tensor([1] * batch_size).unsqueeze(1).cuda()], dim=1)
            if past:
                position_ids = (attn_mask.long().cumsum(-1) - 1)
                position_ids.masked_fill_(attn_mask == 0, 0)
                position_ids = position_ids[:, -1].unsqueeze(-1)    # [B, 1]
                
        ng, batch_size = [], len(generated[0])
        for i in range(batch_size):
            p, flag = [], False
            for g in generated:
                if flag:
                    p.append(self.pad_id)
                    continue
                if g[i] == self.sep_id:
                    flag = True
                p.append(g[i])
            ng.append(p)
        return ng

class GPT2V2RLAgent(BaseAgent):
    
    '''Due to the custom DataLoader, the GPT2V2RL cannot leverage the torch.distributed.launch to speed up, so sad.'''

    def __init__(self, multi_gpu, run_mode='train', lang='zh'):
        super(GPT2V2RLAgent, self).__init__()
        try:
            self.gpu_ids = list(range(len(multi_gpu.split(','))))
        except:
            raise Exception(f'[!] multi gpu ids are needed, but got: {multi_gpu}')
        vocab_file = 'data/vocab/vocab_small' if lang == 'zh' else 'data/vocab/vocab_english'
        self.args = {
            'lr': 1e-4,
            'dis_lr': 1e-4,
            'grad_clip': 1.0,
            'tgt_len_size': 30,
            'lr_gamma': 0.5,
            'warmup_steps': 8000,
            'topk': 2000,
            'topp': 0.97,
            'config_path': 'data/config/model_config_dialogue_small.json',
            'multi_gpu': self.gpu_ids,
            'run_mode': run_mode,
            'vocab_file': 'data/vocab/vocab_small',
            'lang': lang,
            'repetition_penalty': 1,
            'amp_level': 'O2',
            'policy_size': 32,
            'embedding_size': 300,
            'dis_step': 1,
            'gen_step': 1,
            'shuffle': True,
            # parameters of PPO
            'K_epochs': 80,
            'eps_clip': 0.2,
            'betas': (0.9, 0.999),
            'action_std': 0.5,
            'update_timestep': 2048,
        }
        self.vocab = BertTokenizer(vocab_file=self.args['vocab_file'])
        self.vocab_size = len(self.vocab)
        self.unk = self.vocab.convert_tokens_to_ids('[UNK]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        
        # generator: GPT-2 LM
        self.generator = GPT2V2RL(
            self.vocab_size, 
            self.unk, 
            self.sep,
            self.cls,
            self.pad,
            self.args['topk'], 
            self.args['topp'], 
            self.args['repetition_penalty'],
            config_path=self.args['config_path'],
            embedding_size=self.args['embedding_size'],
            action_std=self.args['action_std'],
            k_epochs=self.args['K_epochs'],
            eps_clip=self.args['eps_clip'],
        )
        # the old agent model (policy generator)
        self.agent_old = ActorCritic(
            self.args['policy_size'], 
            self.args['embedding_size'], 
            action_std=self.args['action_std'],
        )
        self.copy_parameters()
        # Memory for optimizing PPO Agent
        self.memory = Memory()
        
        # discriminator: Simple Bert binary classification ?
        self.discriminator = BERTRetrieval()
        
        # load the word2vec
        if lang == 'zh':
            self.w2v = load_w2v('data/chinese_w2v')
        else:
            self.w2v = gensim.models.KeyedVectors.load_word2vec_format('data/english_w2v.bin', binary=True)
        print('[!] load the word2vec over ...')
        
        self.gen_criterion = nn.MSELoss()
        self.dis_criterion = nn.CrossEntropyLoss()
        if torch.cuda.is_available():
            self.generator.cuda()
            self.agent_old.cuda()
            self.discriminator.cuda()
        # only the policy head of generator is the trainable parameters
        self.gen_optimizer = optim.Adam(
            self.generator.agent.parameters(), 
            lr=self.args['lr'], 
            betas=self.args['betas'],
        )
        self.dis_optimizer = transformers.AdamW(
            self.discriminator.parameters(),
            lr=self.args['dis_lr']
        )
        if run_mode == 'train':
            self.generator, self.gen_optimizer = amp.initialize(
                self.generator, 
                self.gen_optimizer, 
                opt_level=self.args['amp_level'],
            )
            self.discriminator, self.dis_optimizer = amp.initialize(
                self.discriminator,
                self.dis_optimizer,
                opt_level=self.args['amp_level']
            )
        self.show_parameters(self.args)
        
    def show_parameters(self, args):
        print('========== Model ==========')
        print(self.generator)
        print(self.agent_old)
        print(self.discriminator)
        print('========== Model ==========')
        print(f'========== Model Parameters ==========')
        for key, value in args.items():
            print(f'{key}: {value}')
        print(f'========== Model Parameters ==========')
        
    def origin_sentence(ids, front=True):
        rest = []
        for i in ids:
            count = i.nonzero().shape[0]
            if front:
                rest.append(i[count:])
            else:
                rest.append(i[:-count])
        return rest
    
    def copy_parameters(self):
        self.agent_old.load_state_dict(self.generator.agent.state_dict())
        print('[!] copy the parameters from self.agent to self.agent_old')
        
    def packup_discriminator(self, cid, rid, frid):
        '''packup the fake response id for training the discriminator
        :cid: [B, S];
        :rid: [B, L1]; [CLS] start
        :frid: a list of sequence, list length is batch_size; [CLS] start'''
        cid_origin = self.origin_sentence(cid, front=True)
        rid_origin = self.origin_sentence(rid, front=False)
        frid_origin  = self.origin_sentence(torch.LongTensor(frid), front=False)
        cid_rid, label = [], [], []
        for c, r, f in zip(cid_origin, rid_origin, frid_origin):
            cid_rid.append(torch.cat([c, r[1:]]))
            cid_rid.append(torch.cat([c, f[1:]]))
            label.extend([1, 0])
        label = torch.LongTensor(label).cuda()
        cid_rid = pad_sequence(cid_rid, batch_first=True, padding_value=self.pad_id).cuda()
        if self.args['shuffle']:
            random_idx = list(range(len(cid_rid)))
            random.shuffle(random_idx)
            cid_rid = [cid_rid[i] for i in random_idx]
            label = [label[i] for i in random_idx]
        return cid_rid, label
    
    def prepare_samples_for_reward(self, cid, frid):
        '''packup the fake response id for training the discriminator
        :cid: [B, S]; :frid: a list of sequence, list length is batch_size; [CLS] start'''
        cid_origin = self.origin_sentence(cid, front=True)
        frid_origin  = self.origin_sentence(torch.LongTensor(frid), front=False)
        cid_rid = [torch.cat([c, f[1:]]) for c, f in zip(cid_origin, frid_origin)]
        cid_rid = pad_sequence(cid_rid, batch_first=True, padding_value=self.pad_id).cuda()
        return cid_rid
        
    def train_discriminator(self, train_iter, mode='train', recoder=None, idx_=0, d_step_idx=0):
        '''use the real samples and the generated samples'''
        self.discriminator.train()
        total_loss, batch_num, correct, s = 0, 0, 0, 0
        pbar = tqdm(enumerate(train_iter))
        for idx, batch in pbar:
            cid, rid, attn_mask, position_ids, ctx_text, can_text = batch
            # generate the fake positive samples
            context_embd = convert_text_embedding(self.w2v, ctx_text)    # [B, 300]
            context_embd = torch.tensor(context_embd, dtype=torch.float).cuda()
            r_embd = []
            for j in range(len(can_text[0])):
                response_embd = convert_text_embedding(
                    self.w2v,
                    [i[j] for i in response_text],
                )    # [B, 300]
                r_embd.append(torch.tensor(response_embd, dtype=torch.float))
            r_embd = torch.stack(r_embd).mean(dim=0).cuda()    # [B, 300]
            max_len = max(self.args['tgt_len_size'], rid.shape[1])
            frid = self.generator.predict_batch(cid, attn_mask, position_ids, context_embd, r_embd, max_len)    # list, length is B
            cids, labels = packup_discriminator(cid, rid, frid)
            
            self.dis_optimizer.zero_grad()
            
            # feed forward
            output = self.generator(cids)    # [B, 2]
            loss = self.criterion(
                output,
                labels.view(-1),
            )
            
            with amp.scale_loss(loss, self.dis_optimizer) as scaled_loss:
                scaled_loss.backward()
            clip_grad_norm_(amp.master_params(self.dis_optimizer), self.args['grad_clip'])
            self.dis_optimizer.step()
            
            total_loss += loss.item()
            batch_num += 1
            
            now_correct = torch.max(F.softmax(output, dim=-1), dim=-1)[1]    # [B]
            now_correct = torch.sum(now_correct == label).item()
            correct += now_correct
            s += len(label)
            
            recoder.add_scalar(f'train-epoch-{idx_}-DStepIdx-{d_step_idx}/Loss', total_loss/batch_num, idx)
            recoder.add_scalar(f'train-epoch-{idx_}-DStepIdx-{d_step_idx}/RunLoss', loss.item(), idx)
            recoder.add_scalar(f'train-epoch-{idx_}-DStepIdx-{d_step_idx}/Acc', correct/s, idx)
            recoder.add_scalar(f'train-epoch-{idx_}-DStepIdx-{d_step_idx}/RunAcc', now_correct/len(label), idx)

            pbar.set_description(f'[!] loss: {round(loss.item(), 4)}|{round(total_loss/batch_num, 4)}; acc: {round(now_correct/len(label), 4)}|{round(correct/s, 4)}')
        if d_step_idx == 0:
            recoder.add_scalar(f'train-whole/Loss', total_loss/batch_num, idx_)
            recoder.add_scalar(f'train-whole/Acc', correct/s, idx_)
        return round(total_loss / batch_num, 4)
    
    def train_generator(self, train_iter, mode='train', recoder=None, idx_=0, g_step_idx=0):
        '''train generator(polilcy head parameters) by policy gradient; the agen-environment interaction loop'''
        self.generator.train()
        total_loss, s, time_step, running_rewards, batch_num = 0, 0, 0, 0, 0
        pbar = tqdm(enumerate(train_iter))
        for idx, batch in pbar:
            time_step += 1
            cid, rid, attn_mask, position_ids, ctx_text, can_text = batch
            # init the state
            context_embd = convert_text_embedding(self.w2v, context_text)    # [B, 300]
            context_embd = torch.tensor(context_embd, dtype=torch.float).cuda()
            r_embd = []
            for j in range(len(response_text[0])):
                response_embd = convert_text_embedding(
                    self.w2v,
                    [i[j] for i in response_text],
                )    # [B, 300]
                r_embd.append(torch.tensor(response_embd, dtype=torch.float))
            r_embd = torch.stack(r_embd).mean(dim=0).cuda()    # [B, 300]
            state = torch.cat([context_embd, r_embd], dim=-1)    # [B, 600]
            
            # select the action by policy_old
            action = self.agent_old.act(state, self.memory)    # [B, P]; P=policy_size
            
            # rewards (generate and discriminate)
            max_len = max(self.args['tgt_len_size'], rid.shape[1])
            generated = self.generator.predict_batch(cid, attn_mask, position_ids, context_embd, r_embd, max_len)
            samples = self.prepare_samples_for_reward(cid, generated)
            rewards = self.discriminator(samples)[:, 1].tolist()    # [B, 2] -> a list of rewards, list length is B
            
            # save rewards into the memory
            self.memory.rewards.expend(rewards)
            running_rewards += sum(rewards)
            
            # update the policy network
            if time_step % self.args['update_timestep'] == 0:
                losses = self.model.update(self.memory, self.gen_criterion, self.gen_optimizer)
                total_loss.append(losses)
                s += 1
                self.copy_parameters()
                memory.clear_memory()
                time_step = 0
                
            batch_num += 1
            recoder.add_scalar(f'train-epoch-{idx_}-GStepIdx-{g_step_idx}/Loss', total_loss/batch_num, idx)
            recoder.add_scalar(f'train-epoch-{idx_}-GStepIdx-{g_step_idx}/RunLoss', loss.item(), idx)
            recoder.add_scalar(f'train-epoch-{idx_}-GStepIdx-{g_step_idx}/Reward', running_rewards/batch_num, idx)
            recoder.add_scalar(f'train-epoch-{idx_}-GStepIdx-{g_step_idx}/RunReward', reward, idx)

            pbar.set_description(f'[!] loss: {round(loss.item(), 4)}|{round(total_loss/batch_num, 4)}; reward: {round(reward, 4)}|{round(running_rewards/batch_num, 4)}')
        if g_step_idx == 0:
            recoder.add_scalar(f'train-whole/Loss', total_loss/batch_num, idx_)
            recoder.add_scalar(f'train-whole/Reward', running_rewards/batch_num, idx_)
        return round(total_loss/s, 4)
        
    def train_model(self, train_iter, mode='train', recoder=None, idx_=0):
        '''Core logic of the GPT2V2RL Model:
        1. train discriminator with binary classification first
        2. train generator with Policy Gradient'''
        dis_acc, dis_loss = [], []
        rl_loss, rl_rewad = [], []
        # ========== D STEP ========== #
        for d_step_idx in range(self.args['dis_step']):
            self.train_discriminator(
                train_iter, mode=mode, recoder=recoder, idx_=idx_, d_step_idx=d_step_idx
            )
        # ========== G STEP ========== #
        for g_step_idx in range(self.args['gen_step']):
            self.train_generator(
                train_iter, mode=mode, recoder=recoder, idx_=idx_, g_step_idx=g_step_idx
            )
    
    def save_model(self, path):
        file_name = os.path.splitext(path)[0]
        dis_path = f'{file_name}_dis.pt'
        gen_path = f'{file_name}_gen.pt'
        state_dict = self.discriminator.module.discriminator.state_dict()
        torch.save(state_dict, dis_path)
        state_dict = self.generator.module.generator.state_dict()
        torch.save(state_dict, gen_path)
        print(f'[!] save discriminator model into {dis_path}')
        print(f'[!] save generator model into {gen_path}')