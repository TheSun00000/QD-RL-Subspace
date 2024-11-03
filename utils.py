import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

import copy

class Linear(nn.Module):
    def __init__(self, n_anchors, in_channels, out_channels, bias = True, same_init = False):
        super().__init__()
        self.n_anchors = n_anchors
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.is_bias = bias

        if same_init:
            anchor = nn.Linear(in_channels,out_channels,bias=self.is_bias)
            anchors = [copy.deepcopy(anchor) for _ in range(n_anchors)]
        else:
            anchors = [nn.Linear(in_channels,out_channels,bias=self.is_bias) for _ in range(n_anchors)]
        self.anchors = nn.ModuleList(anchors)

    def forward(self, x, alpha):
        xs = [anchor(x) for anchor in self.anchors]
        xs = torch.stack(xs,dim=-1)

        alpha = torch.stack([alpha] * self.out_channels, dim=-2)
        xs = (xs * alpha).sum(-1)
        return xs

class Sequential(nn.Sequential):
    def __init__(self,*args):
        super().__init__(*args)

    def forward(self, input, t):
        for module in self:
            input = module(input,t) if isinstance(module,Linear) else module(input)
        return input
    
    


class ActorCriticCategorical(nn.Module):
    def __init__(self, n_anchors, state_dim, action_dim, same_init, actor_hidden_layers, critic_hidden_layers):
        super(ActorCriticCategorical, self).__init__()
        
        # Define actor network
        actor_layers = []
        input_dim = state_dim
        for hidden_dim in actor_hidden_layers:
            actor_layers.append(Linear(n_anchors, input_dim, hidden_dim, same_init=same_init))
            actor_layers.append(nn.ReLU())
            input_dim = hidden_dim
        actor_layers.append(Linear(n_anchors, input_dim, action_dim, same_init=same_init))
        self.actor = Sequential(*actor_layers)
        self.n_anchors = n_anchors
        
        # Define critic network
        critic_layers = []
        input_dim = state_dim
        for hidden_dim in critic_hidden_layers:
            critic_layers.append(nn.Linear(input_dim, hidden_dim))
            critic_layers.append(nn.ReLU())
            input_dim = hidden_dim
        critic_layers.append(nn.Linear(input_dim, 1))
        self.critic = nn.Sequential(*critic_layers)


    def forward(self, x, alpha, action=None):
        action_probs = self.actor(x, alpha)
        dist = Categorical(logits=action_probs)
        
        if action is None:
            action = dist.sample()
            
        log_p = F.log_softmax(action_probs, dim=-1).gather(-1, action.unsqueeze(-1)).squeeze(-1)
        
        value = self.critic(x)
        
        return action, log_p, value, dist.entropy()
    
    
    
    
    
class ActorCriticContinuous(nn.Module):
    def __init__(self, n_anchors, state_dim, action_dim, same_init, actor_hidden_layers, critic_hidden_layers, action_std):
        super(ActorCriticContinuous, self).__init__()
        
        # Define actor network
        actor_layers = []
        input_dim = state_dim
        for hidden_dim in actor_hidden_layers:
            actor_layers.append(Linear(n_anchors, input_dim, hidden_dim, same_init=same_init))
            actor_layers.append(nn.ReLU())
            input_dim = hidden_dim
        actor_layers.append(Linear(n_anchors, input_dim, action_dim, same_init=same_init))
        self.actor = Sequential(*actor_layers)
        self.n_anchors = n_anchors
        
        # Define critic network
        critic_layers = []
        input_dim = state_dim
        for hidden_dim in critic_hidden_layers:
            critic_layers.append(nn.Linear(input_dim, hidden_dim))
            critic_layers.append(nn.ReLU())
            input_dim = hidden_dim
        critic_layers.append(nn.Linear(input_dim, 1))
        self.critic = nn.Sequential(*critic_layers)
        
        self.action_var = nn.Parameter(torch.full((action_dim,), action_std**2, requires_grad=True))


    def forward(self, x, alpha, action=None):
        action_mean = self.actor(x, alpha)
        cov_matrix = torch.exp(self.action_var)
        dist = Normal(loc=action_mean, scale=cov_matrix)
        
        if action is None:
            action = dist.sample()
            
        log_p = dist.log_prob(action).sum(dim=-1)
        
        value = self.critic(x)
        
        return action, log_p, value, dist.entropy()