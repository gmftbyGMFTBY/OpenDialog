from .header import *
from .bert_retrieval import BERTRetrieval

class GPT2V2RL(nn.Module):

    def __init__(self, vocab_size, unk_id, sep_id, cls_id, topk, topp, 
                 repetition_penalty,
                 config_path='data/config/model_config_dialogue_small.json', 
                 embedding_size=300, policy_size=32, action_std=0.5, k_epoch=10, eps_clip=0.2):
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
        self.action_std, self.eps_clip = action_std, eps_clip
        self.k_epoch, self.embedding_size, self.policy_size = k_epoch, embedding_size, policy_size
        self.repetition_penalty = repetition_penalty
        
        self.agent = ActorCritic(policy_size, embedding_size, action_std=action_std)
        
        self.proj = nn.Linear(self.n_embd + policy_size, vocab_size)
        
    def _build_old_agent(self):
        '''must be called right after the GPT2V2RL Model is built'''
        self.agent_old = ActorCritic(self.policy_size, self.embedding_size, action_std=self.action_std)
        self.agent_old.load_state_dict(self.agent.state_dict())
        
    def select_action(self, state, memory):
        '''state from the pre-trained word embeddings, return the sampled old policy network action;
        Then save the action and corresponding information into memory'''
        return self.agent_old.act(state, memory).cpu().numpy().flatten()
    
    def update(self, memory, criterion, optimizer):
        '''update the policy network; return the loss'''
        # construct the rewards (one-step MDP without gamma ratio):
        rewards = torch.tensor(memory.rewards, dtype=torch.float)
        rewards = to_cuda((rewards - rewards.mean()) / (rewards.std() + 1e-5))    # Noemalizing the rewards
        
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
            
        self.agent_old.load_state_dict(self.agent.state_dict())
        return np.mean(losses)

    @torch.no_grad()
    def predict(self, inpt_ids, context_embd, response_embd, max_len):
        '''batch_size is 1; inpt_ids: [seq]; return a list of ids (generated)'''
        generated = [self.cls_id]
        policy_embd = self.agent(torch.cat([context_embd, response_embd], dim=-1))    # [32]
        for _ in range(max_len):
            outputs = self.model(
                input_ids=inpt_ids
            )
            output = outputs[0][-1, :]    # [768]
            output = torch.cat([output, policy_embd], dim=-1)    # [768+32]
            next_token_logits = self.proj(output)    # [vocab]
            # next_token_logits = outputs[0][-1, :]    # [V]
            next_token_logits[self.unk_id] = -np.inf
            if generated:
                next_token_logits[list(set(generated))] /= self.repetition_penalty
            filtered_logits = top_k_top_p_filtering(
                next_token_logits, 
                top_k=self.topk, 
                top_p=self.topp,
            )
            next_token = torch.multinomial(
                F.softmax(filtered_logits, dim=-1),
                num_samples=1,
            )
            generated.append(next_token.item())
            if next_token == self.sep_id:
                break
            inpt_ids = torch.cat((inpt_ids, next_token), dim=0)
            inpt_ids = inpt_ids[-self.n_ctx:]
        return generated
    
    @torch.no_grad()
    def prepare_fake_samples(self, inpt_ids, context_embd, response_embd, max_len):
        pass
    
    def forward(self):
        pass

class GPT2V2RLAgent(BaseAgent):
    
    '''Due to the custom DataLoader, the GPT2V2RL cannot leverage the torch.distributed.launch to speed up, so sad.'''

    def __init__(self, total_steps, multi_gpu, vocab_file='data/vocab/vocab_small', run_mode='train', lang='zh', lm=False):
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
            'total_steps': total_steps,
            'topk': 2000,
            'topp': 0.97, 
            'config_path': 'data/config/model_config_dialogue_small.json',
            'multi_gpu': self.gpu_ids,
            'run_mode': run_mode,
            'vocab_file': vocab_file,
            'lang': lang,
            'repetition_penalty': 1,
            'amp_level': 'O2',
            'policy_size': 32,
            'embedding_size': 300,
            'word2vec': 'data/chinese_w2v' if lang == 'zh' else 'data/english_w2v.bin',
            'dis_step': 1,
            'gen_step': 1,
            # parameters of PPO
            'K_epochs': 80,
            'eps_clip': 0.2,
            'betas': (0.9, 0.999),
            'action_std': 0.5,
            'update_timestep': 1000,
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
            self.args['topk'], 
            self.args['topp'], 
            self.args['repetition_penalty'],
            config_path=self.args['config_path'],
            embedding_size=self.args['embedding_size'],
            action_std=self.args['action_std'],
            k_epochs=self.args['K_epochs'],
            eps_clip=self.args['eps_clip'],
        )
        # load the pretrained parameters
        self.generator._build_old_agent()
        
        # discriminator: Simple Bert binary classification ?
        self.discriminator = BERTRetrieval()
        
        # Memory for optimizing PPO Agent
        self.memory = Memory()
        
        # load the word2vec
        if lang == 'zh':
            self.w2v = load_w2v('data/chinese_w2v')
        else:
            self.w2v = gensim.models.KeyedVectors.load_word2vec_format('data/english_w2v.bin', binary=True)
        
        self.gen_criterion = nn.MSELoss()
        self.dis_criterion = nn.CrossEntropyLoss()
        if torch.cuda.is_available():
            self.generator.cuda()
            self.discriminator.cuda()
        # only the policy head of generator is the trainable parameters
        self.gen_optimizer = optim.Adam(
            self.generator.parameters(), 
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
        
    def train_discriminator(self, train_iter, mode='train', recoder=None, idx_=0, d_step_idx=0):
        '''use the real samples and the generated samples'''
        self.discriminator.train()
        with tqdm(total=len(train_iter)) as pbar:
            for idx, batch in enumerate(train_iter):
                cid, rid, context_embed, response_embed = batch
                
    
    def train_generator(self, train_iter, mode='train', recoder=None, idx_=0, g_step_idx=0):
        self.generator.train()
        pass
        
    def train_model(self, train_iter, mode='train', recoder=None, idx_=0):
        '''Core logic of the GPT2V2RL Model:
        1. train discriminator with binary classification first
        2. train generator with Policy Gradient'''
        dis_acc, dis_loss = [], []
        rl_loss, rl_rewad = [], []
        # ========== D STEP ========== #
        for d_step_idx in range(self.args['dis_step']):
            self.train_discriminator(train_iter, mode='train', recoder=None, idx_=0, d_step_idx=d_step_idx)
        # ========== G STEP ========== #
        for g_step_idx in range(self.args['gen_step']):
            self.train_generator(train_iter, mode='train', recoder=None, idx_=0, g_step_idx=g_step_idx)
    
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
