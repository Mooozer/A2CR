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



