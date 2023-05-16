# !/usr/bin/env python
# coding: utf-8

class A2CRModel(torch.nn.Module):
    def __init__(self, config_a2cr):
        super(A2CRModel, self).__init__()

        self.C, self.H, self.W = config_a2cr['env'].observation_space.shape
        assert self.C == 1 or 3  #check channel is 1 or 3(SuperMarioBros)
        
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(self.C, 32, kernel_size=8, stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1),
            torch.nn.ReLU()
        )
        
        feature_size = self.features(torch.zeros(1, *config_a2cr['env'].observation_space.shape)).view(1, -1).size(1)
        
        self.critic = torch.nn.Sequential(
            torch.nn.Linear(feature_size, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 1)
        )
        self.actor = torch.nn.Sequential(
            torch.nn.Linear(feature_size, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, config_a2cr['env'].action_space.n),
            torch.nn.Softmax(dim=-1)
        )

        
    def forward(self, x):
        if x.shape == (self.C, self.H, self.W):  #(H, W, C), if without batch, add batch = 1 
            x = x.reshape((-1,)+ x.shape)  #(batch_size=1, channel, H, W) 
        elif x.shape  == (x.shape[0],) + (self.C, self.H, self.W): #(batch_size, H, W, C) 
            x = x    #(batch_size, channel, H, W)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        value = self.critic(x)
        actions = self.actor(x)
        return value, actions
        
    def get_critic(self, x):
        if x.shape == (self.C, self.H, self.W):  #(H, W, C), if without batch, add batch = 1 
            x = x.reshape((-1,)+ x.shape)  #(batch_size=1, channel, H, W) 
        elif x.shape  == (x.shape[0],) + (self.C, self.H, self.W): #(batch_size, H, W, C) 
            x = x    #(batch_size, channel, H, W)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.critic(x)
    
    def evaluate_action(self, state, action):
        if state.shape == (self.C, self.H, self.W):  #(H, W, C), if without batch, add batch = 1 
            state = state.reshape((-1,)+ state.shape)  #(batch_size=1, channel, H, W) 
        elif state.shape  == (state.shape[0],) + (self.C, self.H, self.W): #(batch_size, H, W, C) 
            state = state    #(batch_size, channel, H, W)
        value, actor_features = self.forward(state)
        dist = torch.distributions.Categorical(actor_features)
        log_probs = dist.log_prob(action).view(-1, 1)
        entropy = dist.entropy().mean()
        return value, log_probs, entropy    
    
    def act(self, state):
        if state.shape == (self.C, self.H, self.W):  #(H, W, C), if without batch, add batch = 1 
            state = state.reshape((-1,)+ state.shape)  #(batch_size=1, channel, H, W) 
        elif state.shape  == (state.shape[0],) + (self.C, self.H, self.W): #(batch_size, H, W, C) 
            state = state    #(batch_size, channel, H, W)
        value, actor_features = self.forward(state)
            
        dist = torch.distributions.Categorical(actor_features)
        chosen_action = dist.sample()
        return chosen_action.item()



class A2CRModel_Reasoner(torch.nn.Module):
    def __init__(self, config_a2cr):
        super(A2CRModel_Reasoner, self).__init__()

        self.C, self.H, self.W = config_a2cr['env'].observation_space.shape
        assert self.C == 1 or 3  #check channel is 1 or 3(SuperMarioBros)
        
        self.features_reasoner = torch.nn.Sequential(
            torch.nn.Conv2d(self.C, 32, kernel_size=8, stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1),
            torch.nn.ReLU()
        )
        
        feature_size_reasoner = self.features_reasoner(torch.zeros(1, *config_a2cr['env'].observation_space.shape)).view(1, -1).size(1)
        self.reasoner = torch.nn.Sequential(
            torch.nn.Linear(feature_size_reasoner, 512), 
            torch.nn.ReLU(),
            torch.nn.Linear(512, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, config_a2cr['Reasoner_CLS']),
            torch.nn.Softmax(dim=-1)
        )
    def forward(self, diff_state):
        '''
        diff_state: (batch_size, C, H, W) 
        '''
        if diff_state.shape == (self.C, self.H, self.W):  #(H, W, C), if without batch, add batch = 1 
            diff_state = diff_state.reshape((-1,)+ diff_state.shape)  #(batch_size=1, channel, H, W) 
        elif diff_state.shape  == (diff_state.shape[0],) + (self.C, self.H, self.W): #(batch_size, H, W, C) 
            diff_state = diff_state    #(batch_size, channel, H, W)
        
        dx = self.features_reasoner(diff_state)
        dx = dx.view(dx.size(0), -1)
        return self.reasoner(dx)



import cv2
from skimage.filters import window
def stage_exploration(states2, states1, yUpperCut = 11, yBottomCut = 11, window_type = 'hann'):
    '''
    states2/states1: (channel * height * width), states2 is the new states and states1 is the old states
    yUpperCut: eliminate the influence of game title on the screen
    yBottomCut: eliminate the influence of game ground on the screen
    window: apply windowing function to reduce edge effects. 
    '''
    channel, height, width = states2.shape
    new_area_value = 0.
    common_area_values = 0.
    states1, states2 = np.array(states1.cpu()), np.array(states2.cpu())
        
    for c in range(channel):
        if window_type:
            image2 = states2[c,:,yUpperCut:].transpose() 
            image1 = states1[c,:,yUpperCut:].transpose()  
            wimage2 = image2 * window(window_type, image2.shape)
            wimage1 = image1 * window(window_type, image1.shape)
            (x, y), power = cv2.phaseCorrelate(wimage2, wimage1)
        else: 
            image2 = states2[c,:,yUpperCut:].transpose() 
            image1 = states1[c,:,yUpperCut:].transpose()        
            (x, y), power = cv2.phaseCorrelate(image2, image1)
        
        if abs(x) > 20:  #if the x-shift too large, should ignore
            x = 0
        if round(x) >= 0:
            appeared_area = torch.tensor(states2[c, width-round(x):width ,yUpperCut:height-yBottomCut].transpose())
            disappeared_area = torch.tensor(states1[c, :round(x) ,yUpperCut:height-yBottomCut].transpose())
            new_area_value += torch.norm(appeared_area, p='fro')  #F norm 
            new_area_value += torch.norm(disappeared_area, p='fro') #F norm 

            common_area1 = torch.tensor(states1[c, round(x):width , yUpperCut:height-yBottomCut].transpose())
            common_area2 = torch.tensor(states2[c, :width-round(x) ,yUpperCut:height-yBottomCut].transpose())
        
        else :
            appeared_area = torch.tensor(states2[c, :-round(x) ,yUpperCut:height-yBottomCut].transpose())
            disappeared_area = torch.tensor(states1[c, width+round(x):, yUpperCut:height-yBottomCut].transpose())
            new_area_value += torch.norm(appeared_area, p='fro')  #F norm 
            new_area_value += torch.norm(disappeared_area, p='fro') #F norm 

            common_area1 = torch.tensor(states1[c, -round(x):width , yUpperCut:height-yBottomCut].transpose())
            common_area2 = torch.tensor(states2[c, :width+round(x) ,yUpperCut:height-yBottomCut].transpose())
        
        common_area_values += torch.norm(common_area1 - common_area2, p='fro') #F norm 
        
    return new_area_value + common_area_values



class ExploringPool(object): 
    def __init__(self, capacity=100000):
        '''
        spool: mean of difference of states  
        rpool: reward 
        diffvpool: difference of values of states
        '''
        self.spool, self.rpool, self.diffvpool = [], [], []
        self.capacity = capacity
    
    def updatePool(self, sExplor, diffv, r):
        if len(self.spool) < self.capacity:
            self.spool.append(sExplor.item())
            self.rpool.append(r)
            self.diffvpool.append(diffv)
        else: 
            self.spool.pop(0)
            self.spool.append(sExplor.item())
            
            self.rpool.pop(0)
            self.rpool.append(r)
            
            self.diffvpool.pop(0)
            self.diffvpool.append(diffv)
            assert len(self.rpool) == self.capacity
    
    def get_label(self, stage_explor, diff_v, reward, w1):
        change_1 = w1 * diff_v + (1 - w1) * reward 
        change_2 = stage_explor
        
        random_index = torch.LongTensor(torch.randint(low=0, high=len(torch.tensor(self.rpool)),
                                           size=(int(len(torch.tensor(self.rpool))/2)+1,)))
        E0_rpool = torch.tensor(self.rpool).clone().index_select(0, random_index)
        E0_diffvpool = torch.tensor(self.diffvpool).clone().index_select(0, random_index)
        E0_spool = torch.tensor(self.spool).clone().index_select(0, random_index)
        
        Criteria_VR = change_1 > torch.mean( w1 * E0_rpool + (1 - w1) * E0_diffvpool)
        Criteria_S = change_2 > torch.mean(E0_spool)
        four_subspace = 2*Criteria_S - 1*Criteria_VR #Distinguish four subspaces 
        #(large s large r/v: 1, small s large r/v: -1, small s small r/v: 0, large s small r/v: 2)
        #{1:0, -1:1, 0:2, 2:3}

        #method 1:
        pseudo_GT = (four_subspace == 1)*0 + (four_subspace ==-1)*1 + (four_subspace == 0)*2 + (four_subspace == 2)*3 
    
        return LongTensor(pseudo_GT)



class Batch_Memory(object):
    def __init__(self):
        self.states, self.actions, self.true_values, self.diff_ss, self.diff_vs, self.rs, self.pseudo_GTs = [], [], [], [], [], [], []
        
    def pushWorkerData(self, state, action, true_value):  
        self.states.append(state)
        self.actions.append(action)
        self.true_values.append(true_value)
        
    def pushCollectorData(self, diff_sb, diff_vb, rb, pseudo_GTb): 
        self.diff_ss.append(diff_sb)
        self.diff_vs.append(diff_vb)
        self.rs.append(rb)
        self.pseudo_GTs.append(pseudo_GTb)
        
    def pop_all_worker(self):
        states = torch.stack(self.states)
        actions = LongTensor(self.actions)
        true_values = FloatTensor(self.true_values).unsqueeze(1)
        self.states, self.actions, self.true_values =[] , [], [] 
        return states, actions, true_values
    
    def pop_all_collector(self):        
        diff_ss = FloatTensor(torch.stack(self.diff_ss))
        diff_vs = FloatTensor(self.diff_vs)
        rs = FloatTensor(self.rs)
        pseudo_GTs = LongTensor(self.pseudo_GTs)
        self.diff_ss, self.diff_vs, self.rs, self.pseudo_GTs = [], [], [], []
        return diff_ss, diff_vs, rs, pseudo_GTs
    

def get_true_returns(states, rewards, dones):
    '''
    true discounted returns and estimate returns 
    '''
    true_values = []
    rewards = FloatTensor(rewards)
    dones = FloatTensor(dones)
    states = torch.stack(states)
    
    if dones[-1] == True:
        next_value = rewards[-1]
    else:
        next_value = model.get_critic(states[-1].unsqueeze(0))
        
    true_values.append(next_value) 
    for i in reversed(range(0, len(rewards) - 1)):
        if not dones[i]:
            next_value = rewards[i] + next_value * config_a2cr['gamma']
        else:
            next_value = rewards[i]
        true_values.append(next_value)
        
    true_values.reverse()
    
    return FloatTensor(true_values)


def update(batch_memory, optimizer, optimizer_reasoner, frame_idx, config_a2cr):
    
    states, actions, true_values =  batch_memory.pop_all_worker()
    
    values, log_probs, entropy = model.evaluate_action(states, actions)
    
    advantages =  true_values - values
    critic_loss = advantages.pow(2).mean()
    actor_loss = -(log_probs * advantages.detach()).mean()
    total_loss = (config_a2cr['critic_coef'] * critic_loss) + actor_loss - (config_a2cr['entropy_coef'] * entropy)
    
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()
    
    if config_a2cr['Rea_collector'] and frame_idx >= config_a2cr['Reasoner_Afterframes']:
        diff_ss, diff_vs, rs, pseudo_GTs = batch_memory.pop_all_collector()

        reason_cls_prob = model_reasoner.forward(diff_ss)
        onehot_GTs = torch.zeros(len(pseudo_GTs), config_a2cr['Reasoner_CLS']).to(device)
        onehot_GTs[range(onehot_GTs.shape[0]), pseudo_GTs]=1
        reasoner_loss = F.binary_cross_entropy_with_logits(reason_cls_prob, onehot_GTs).to(device)
        
        optimizer_reasoner.zero_grad()
        reasoner_loss.backward()
        torch.nn.utils.clip_grad_norm_(model_reasoner.parameters(), 0.5)
        optimizer_reasoner.step()    
        pseudo_GTs = pseudo_GTs.tolist()
    else:
        reasoner_loss = torch.tensor(0.).to(device)
        pseudo_GTs = []
            
    return values.mean().item(), critic_loss.item(), actor_loss.item(), total_loss.item(), reasoner_loss.item(), pseudo_GTs



class WorkerAndCollector(object):
    def __init__(self, config_a2cr, model, labor, w1):
        '''
        labor is in {'worker','collector'}
        '''
        self.env = Env_preprocess(config_a2cr['env_name'], movement = config_a2cr['movement'])
        self.episode_reward = 0
        self.state = FloatTensor(self.env.reset())
        self.fake_actions = list(range(config_a2cr['env'].action_space.n))
        self.labor = labor
        self.state_value_model = model.get_critic
        self.w1 = w1
            
    def get_batch(self, EXpool, frame_idx):

        states, actions, rewards, dones = [], [], [], []
        diff_sbb, diff_vbb,  rbb,  pseudo_GTbb = [], [], [], []  
        if self.labor == 'worker': 
            for i in range(config_a2cr['batch_size']):
                action = model.act(self.state)
                next_state, reward, done, _ = self.env.step(action)
                self.episode_reward += reward
                states.append(self.state)
                actions.append(action)
                rewards.append(reward)
                dones.append(done)

                if done:
                    self.state = FloatTensor(self.env.reset())
                    record['training_episode_rewards'].append(self.episode_reward)
                    record['episode'].append(len(record['episode'])) 
                    self.episode_reward = 0
                else:
                    self.state = FloatTensor(next_state)

            true_returns = get_true_returns(states, rewards, dones).unsqueeze(1)
            return states, actions, true_returns

        if self.labor == 'collector': 
            for i in range(config_a2cr['batch_size']):
                action = model.act(self.state)
                next_state, reward, done, _ = self.env.step(action)   
                next_state = FloatTensor(next_state)
                diff_s = next_state - self.state 
                stage_explor = stage_exploration(next_state, self.state )
                diff_v = self.state_value_model(next_state) - self.state_value_model(self.state)
                
                EXpool.updatePool(stage_explor, diff_v.detach().item(), reward)
                pseudo_GT = EXpool.get_label(stage_explor, diff_v, reward, self.w1)
                diff_sbb.append(diff_s)
                
                diff_vbb.append(diff_v.detach().item())
                rbb.append(reward)
                pseudo_GTbb.append(pseudo_GT.detach().item())

                if done:
                    self.state = FloatTensor(self.env.reset())
                else:
                    self.state = FloatTensor(next_state)

            return diff_sbb, diff_vbb, rbb, pseudo_GTbb


