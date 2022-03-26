import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from collections import OrderedDict
from torch.autograd import Variable
import  os
from torch import Tensor

class OUNoise:
    def __init__(self, action_dimension, scale=0.1, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state * self.scale


def weight_init(m):
    if isinstance(m, nn.Linear):
    
        stdv = 1. / math.sqrt(m.weight.size(1))
        m.weight.data.uniform_(-stdv, stdv)
        m.bias.data.uniform_(-stdv, stdv)
#        nn.init.xavier_normal_(m.weight)
#        nn.init.constant_(m.bias, 0)

def get_negative_expectation(q_samples, measure, average=True):

    log_2 = math.log(2.)

    if measure == 'GAN':
        Eq = F.softplus(-q_samples) + q_samples
    elif measure == 'JSD':
        #
        Eq = F.softplus(-q_samples) + q_samples - log_2  # Note JSD will be shifted
        #Eq = F.softplus(q_samples) #+ q_samples - log_2
    elif measure == 'X2':
        Eq = -0.5 * ((torch.sqrt(q_samples ** 2) + 1.) ** 2)
    elif measure == 'KL':
        q_samples = torch.clamp(q_samples,-1e6,9.5)
        
        #print("neg q samples ",q_samples.cpu().data.numpy())
        Eq = torch.exp(q_samples - 1.)
    elif measure == 'RKL':
        Eq = q_samples - 1.
    elif measure == 'H2':
        Eq = torch.exp(q_samples) - 1.
    elif measure == 'W1':
        Eq = q_samples
    else:
        assert 1==2

    if average:
        return Eq.mean()
    else:
        return Eq

def get_positive_expectation(p_samples, measure, average=True):

    log_2 = math.log(2.)

    if measure == 'GAN':
        Ep = - F.softplus(-p_samples)
    elif measure == 'JSD':
        Ep = log_2 - F.softplus(-p_samples)  # Note JSD will be shifted
        #Ep =  - F.softplus(-p_samples)
    elif measure == 'X2':
        Ep = p_samples ** 2
    elif measure == 'KL':
        Ep = p_samples
    
    elif measure == 'RKL':
    
        Ep = -torch.exp(-p_samples)
    elif measure == 'DV':
        Ep = p_samples
    elif measure == 'H2':
        Ep = 1. - torch.exp(-p_samples)
    elif measure == 'W1':
        Ep = p_samples
    else:
        assert 1==2

    if average:
        return Ep.mean()
    else:
        return Ep



def fenchel_dual_loss(l, m, measure=None):
    '''Computes the f-divergence distance between positive and negative joint distributions.
    Note that vectors should be sent as 1x1.
    Divergences supported are Jensen-Shannon `JSD`, `GAN` (equivalent to JSD),
    Squared Hellinger `H2`, Chi-squeared `X2`, `KL`, and reverse KL `RKL`.
    Args:
        l: Local feature map.
        m: Multiple globals feature map.
        measure: f-divergence measure.
    Returns:
        torch.Tensor: Loss.
    '''
    N, units = l.size()

    # Outer product, we want a N x N x n_local x n_multi tensor.
    u = torch.mm(m, l.t())
    
    # Since we have a big tensor with both positive and negative samples, we need to mask.
    mask = torch.eye(N).to(l.device)
    n_mask = 1 - mask
    # Compute the positive and negative score. Average the spatial locations.
    E_pos = get_positive_expectation(u, measure, average=False)
    E_neg = get_negative_expectation(u, measure, average=False)
    MI = (E_pos * mask).sum(1) #- (E_neg * n_mask).sum(1)/(N-1)
    # Mask positive and negative terms for positive and negative parts of loss
    E_pos_term = (E_pos * mask).sum(1)
    E_neg_term = (E_neg * n_mask).sum(1) /(N-1)
    loss = E_neg_term - E_pos_term
    return loss,MI

class NEW_MINE(nn.Module):
    def __init__(self,state_size,com_a_size,measure ="JSD"):

        super(NEW_MINE, self).__init__()
        self.measure = measure
        self.com_a_size = com_a_size
        self.state_size = state_size
        self.nonlinearity = F.leaky_relu
        self.l1 = nn.Linear(self.state_size, 32)
        self.l2 = nn.Linear(self.com_a_size, 32)

    def forward(self, state, joint_action,params =None):
        em_1 = self.nonlinearity(self.l1(state),inplace=True)
        em_2 = self.nonlinearity(self.l2(joint_action),inplace=True)
        two_agent_embedding = [em_1,em_2]
        loss, MI = fenchel_dual_loss(two_agent_embedding[0], two_agent_embedding[1], measure=self.measure)
        return loss ,MI


class CLUB(nn.Module):  # CLUB: Mutual Information Contrastive Learning Upper Bound
    '''
        This class provides the CLUB estimation to I(X,Y)
        Method:
            forward() :      provides the estimation with input samples
            loglikeli() :   provides the log-likelihood of the approximation q(Y|X) with input samples
        Arguments:
            x_dim, y_dim :         the dimensions of samples from X, Y respectively
            hidden_size :          the dimension of the hidden layer of the approximation network q(Y|X)
            x_samples, y_samples : samples from X and Y, having shape [sample_size, x_dim/y_dim]
    '''

    def __init__(self, x_dim, y_dim, hidden_size):
        super(CLUB, self).__init__()
        # p_mu outputs mean of q(Y|X)
        # print("create CLUB with dim {}, {}, hiddensize {}".format(x_dim, y_dim, hidden_size))
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size // 2),
                                  nn.ReLU(inplace=True),
                                  nn.Linear(hidden_size // 2, y_dim))
        # p_logvar outputs log of variance of q(Y|X)
        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size // 2),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(hidden_size // 2, y_dim),
                                      nn.Tanh())

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar

    def forward(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)

        # log of conditional probability of positive sample pairs
        positive = - (mu - y_samples) ** 2 / 2. / logvar.exp()

        prediction_1 = mu.unsqueeze(1)  # shape [nsample,1,dim]
        y_samples_1 = y_samples.unsqueeze(0)  # shape [1,nsample,dim]

        # log of conditional probability of negative sample pairs
        negative = - ((y_samples_1 - prediction_1) ** 2).mean(dim=1) / 2. / logvar.exp()

        return  positive.sum(dim=-1) - negative.sum(dim=-1)

    def loglikeli(self, x_samples, y_samples):  # unnormalized loglikelihood
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples) ** 2 / logvar.exp() - logvar).sum(dim=1).mean(dim=0)

    def learning_loss(self, x_samples, y_samples):
        return - self.loglikeli(x_samples, y_samples)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action,layer_sizes=None):
        super(Actor, self).__init__()
        if layer_sizes is None :
            layer_sizes = [state_dim,64,64]
        self.nonlinearity = F.relu
        self.num_layers = len(layer_sizes)
        for i in range(1, self.num_layers):
            self.add_module('layer{0}'.format(i),nn.Linear(layer_sizes[i - 1], layer_sizes[i]))
        self.a_layer = nn.Linear(layer_sizes[-1], action_dim)
        self.max_action = max_action
        
        

    def forward(self, x,params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())
        output = x

        output_list = []
        for i in range(1, self.num_layers):
            output = F.linear(output,
                              weight=params['layer{0}.weight'.format(i)],
                              bias=params['layer{0}.bias'.format(i)])
            output = self.nonlinearity(output,inplace=True)
            output_list.append(output)
        obs_embedding = output_list[0]
        policy_embedding = output_list[1]
        output = F.linear(output,weight=params['a_layer.weight'],bias=params['a_layer.bias'])
        output = self.max_action * torch.tanh(output)
        #logits = output
        #u = torch.rand(output.shape)
        #output = torch.nn.functional.softmax(output - torch.log(-torch.log(u)),dim=-1)
        return output,obs_embedding,policy_embedding

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, 64)
        self.l2 = nn.Linear(64, 64)
        self.l3 = nn.Linear(64, 1)

    def forward(self, x, u):
        xu = torch.cat([x, u], 1)

        x = F.relu(self.l1(xu),inplace=True)
        x = F.relu(self.l2(x),inplace=True)
        x = self.l3(x)
        return x 

class MA_T_DDPG(object):
    def __init__(self, num_agent, obs_dim, state_dim, action_dim_list, max_action, deivce ,max_mi_c,min_mi_c):
        self.device = deivce
        self.num_agent = num_agent
        self.total_action = sum(action_dim_list)

      #  self.pr_actor = Actor(state_dim, sum(action_dim_list), max_action).to(self.device)

        self.actor = [Actor(obs_dim[i], action_dim_list[i], max_action).to(self.device) for i in range(self.num_agent)]
        self.actor_target = [Actor(obs_dim[i], action_dim_list[i], max_action).to(self.device) for i in range(self.num_agent)]
        [self.actor_target[i].load_state_dict(self.actor[i].state_dict()) for i in range(self.num_agent)]

      #  self.pr_actor_optimizer = torch.optim.Adam([{'params': self.pr_actor.parameters()}],lr=3e-5)
        self.actor_optimizer = torch.optim.Adam([{'params': self.actor[i].parameters()} for i in range(num_agent)],
                                                lr=1e-4)
        #if max_mi_c > 0.0 :
        self.mine_policy=NEW_MINE(state_dim,self.total_action).to(self.device)
        self.mine_optimizer = torch.optim.Adam([{'params': self.mine_policy.parameters()}], lr=0.0001)
        
        self.club_policy = CLUB(state_dim,self.total_action,64).to(self.device)
        self.club_optimizer = torch.optim.Adam([{'params': self.club_policy.parameters()}], lr=0.0001)

        self.critic = Critic(state_dim, self.total_action).to(self.device)
        self.critic_target = Critic(state_dim, self.total_action).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),lr=1e-3)

        self.update_lr = 0.1
        self.exploration = OUNoise(action_dim_list[0])
    
    def scale_noise(self, scale):
        """
        Scale noise for each agent
        Inputs:
            scale (float): scale of noise
        """
        self.exploration.scale = scale
    def reset_noise(self):
        self.exploration.reset()
    
    def select_action(self, state, index,params=None):
      #  print("????????? ")
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
            action = self.actor[index](state,params)[0]
     #       print("ac ", action.shape,action )
            noise = Variable(Tensor(self.exploration.noise()),requires_grad=False)
     #       print("noise ",noise.shape,noise)
            action += noise
            
     #       print("final a ", action )
            action = action.clamp(-1, 1)
            return action.cpu().data.numpy().flatten()

    def train_mine(self, replay_buffer, iterations, batch_size=64):

        loss_list = []
        Mi_loss = []
        # replay_buffer.caculate_Q(self.actor,self.critic_target,device=self.device)
        for it in range(iterations):
            # Sample replay buffer
            pos_action, pos_obs, pos_state = replay_buffer.sample_pos(batch_size)
            if self.num_agent == 2:
                pos_action1 = []
                pos_action2 = []
                for i in range(len(pos_action)):
                    pos_action1.append(pos_action[i][0])
                    pos_action2.append(pos_action[i][1])
                pos_action1 = torch.FloatTensor(pos_action1).to(self.device)
                pos_action2 = torch.FloatTensor(pos_action2).to(self.device)

                pos_action = [pos_action1, pos_action2]
            elif self.num_agent == 3:
                pos_action1 = []
                pos_action2 = []
                pos_action3 = []
                for i in range(len(pos_action)):
                    pos_action1.append(pos_action[i][0])
                    pos_action2.append(pos_action[i][1])
                    pos_action3.append(pos_action[i][2])
                pos_action1 = torch.FloatTensor(pos_action1).to(self.device)
                pos_action2 = torch.FloatTensor(pos_action2).to(self.device)
                pos_action3 = torch.FloatTensor(pos_action3).to(self.device)
                pos_action = [pos_action1, pos_action2, pos_action3]
            elif self.num_agent == 4:
                pos_action1 = []
                pos_action2 = []
                pos_action3 = []
                pos_action4 = []
                for i in range(len(pos_action)):
                    pos_action1.append(pos_action[i][0])
                    pos_action2.append(pos_action[i][1])
                    pos_action3.append(pos_action[i][2])
                    pos_action4.append(pos_action[i][3])
                pos_action1 = torch.FloatTensor(pos_action1).to(self.device)
                pos_action2 = torch.FloatTensor(pos_action2).to(self.device)
                pos_action3 = torch.FloatTensor(pos_action3).to(self.device)
                pos_action4 = torch.FloatTensor(pos_action4).to(self.device)

                pos_action = [pos_action1, pos_action2, pos_action3, pos_action4]
            elif self.num_agent == 5:
                pos_action1 = []
                pos_action2 = []
                pos_action3 = []
                pos_action4 = []
                pos_action5 = []
                for i in range(len(pos_action)):
                    pos_action1.append(pos_action[i][0])
                    pos_action2.append(pos_action[i][1])
                    pos_action3.append(pos_action[i][2])
                    pos_action4.append(pos_action[i][3])
                    pos_action5.append(pos_action[i][4])

                pos_action1 = torch.FloatTensor(pos_action1).to(self.device)
                pos_action2 = torch.FloatTensor(pos_action2).to(self.device)
                pos_action3 = torch.FloatTensor(pos_action3).to(self.device)
                pos_action4 = torch.FloatTensor(pos_action4).to(self.device)
                pos_action5 = torch.FloatTensor(pos_action5).to(self.device)
                pos_action = [pos_action1, pos_action2, pos_action3, pos_action4, pos_action5]
            elif self.num_agent == 6:
                pos_action1 = []
                pos_action2 = []
                pos_action3 = []
                pos_action4 = []
                pos_action5 = []
                pos_action6 = []
                for i in range(len(pos_action)):
                    pos_action1.append(pos_action[i][0])
                    pos_action2.append(pos_action[i][1])
                    pos_action3.append(pos_action[i][2])
                    pos_action4.append(pos_action[i][3])
                    pos_action5.append(pos_action[i][4])
                    pos_action6.append(pos_action[i][5])
                pos_action1 = torch.FloatTensor(pos_action1).to(self.device)
                pos_action2 = torch.FloatTensor(pos_action2).to(self.device)
                pos_action3 = torch.FloatTensor(pos_action3).to(self.device)
                pos_action4 = torch.FloatTensor(pos_action4).to(self.device)
                pos_action5 = torch.FloatTensor(pos_action5).to(self.device)
                pos_action6 = torch.FloatTensor(pos_action6).to(self.device)
                pos_action = [pos_action1, pos_action2, pos_action3, pos_action4, pos_action5, pos_action6]
            elif self.num_agent == 9:
                pos_action1 = []
                pos_action2 = []
                pos_action3 = []
                pos_action4 = []
                pos_action5 = []
                pos_action6 = []
                pos_action7 = []
                pos_action8 = []
                pos_action9 = []
                for i in range(len(pos_action)):
                    pos_action1.append(pos_action[i][0])
                    pos_action2.append(pos_action[i][1])
                    pos_action3.append(pos_action[i][2])
                    pos_action4.append(pos_action[i][3])
                    pos_action5.append(pos_action[i][4])
                    pos_action6.append(pos_action[i][5])
                    pos_action7.append(pos_action[i][6])
                    pos_action8.append(pos_action[i][7])
                    pos_action9.append(pos_action[i][8])
                pos_action1 = torch.FloatTensor(pos_action1).to(self.device)
                pos_action2 = torch.FloatTensor(pos_action2).to(self.device)
                pos_action3 = torch.FloatTensor(pos_action3).to(self.device)
                pos_action4 = torch.FloatTensor(pos_action4).to(self.device)
                pos_action5 = torch.FloatTensor(pos_action5).to(self.device)
                pos_action6 = torch.FloatTensor(pos_action6).to(self.device)
                pos_action7 = torch.FloatTensor(pos_action7).to(self.device)
                pos_action8 = torch.FloatTensor(pos_action8).to(self.device)
                pos_action9 = torch.FloatTensor(pos_action9).to(self.device)
                pos_action = [pos_action1, pos_action2, pos_action3, pos_action4, pos_action5, pos_action6,pos_action7, pos_action8, pos_action9]
            elif self.num_agent == 12:
                pos_action1 = []
                pos_action2 = []
                pos_action3 = []
                pos_action4 = []
                pos_action5 = []
                pos_action6 = []
                pos_action7 = []
                pos_action8 = []
                pos_action9 = []
                pos_action10 = []
                pos_action11 = []
                pos_action12 = []
                for i in range(len(pos_action)):
                    pos_action1.append(pos_action[i][0])
                    pos_action2.append(pos_action[i][1])
                    pos_action3.append(pos_action[i][2])
                    pos_action4.append(pos_action[i][3])
                    pos_action5.append(pos_action[i][4])
                    pos_action6.append(pos_action[i][5])
                    pos_action7.append(pos_action[i][6])
                    pos_action8.append(pos_action[i][7])
                    pos_action9.append(pos_action[i][8])
                    pos_action10.append(pos_action[i][9])
                    pos_action11.append(pos_action[i][10])
                    pos_action12.append(pos_action[i][11])
                pos_action1 = torch.FloatTensor(pos_action1).to(self.device)
                pos_action2 = torch.FloatTensor(pos_action2).to(self.device)
                pos_action3 = torch.FloatTensor(pos_action3).to(self.device)
                pos_action4 = torch.FloatTensor(pos_action4).to(self.device)
                pos_action5 = torch.FloatTensor(pos_action5).to(self.device)
                pos_action6 = torch.FloatTensor(pos_action6).to(self.device)
                pos_action7 = torch.FloatTensor(pos_action7).to(self.device)
                pos_action8 = torch.FloatTensor(pos_action8).to(self.device)
                pos_action9 = torch.FloatTensor(pos_action9).to(self.device)
                pos_action10 = torch.FloatTensor(pos_action10).to(self.device)
                pos_action11 = torch.FloatTensor(pos_action11).to(self.device)
                pos_action12 = torch.FloatTensor(pos_action12).to(self.device)
                pos_action = [pos_action1, pos_action2, pos_action3, pos_action4, pos_action5, pos_action6,pos_action7, pos_action8, pos_action9,pos_action10, pos_action11, pos_action12]
            elif self.num_agent == 24:
                pos_action1 = []
                pos_action2 = []
                pos_action3 = []
                pos_action4 = []
                pos_action5 = []
                pos_action6 = []
                pos_action7 = []
                pos_action8 = []
                pos_action9 = []
                pos_action10 = []
                pos_action11 = []
                pos_action12 = []
                pos_action13 = []
                pos_action14 = []
                pos_action15 = []
                pos_action16 = []
                pos_action17 = []
                pos_action18 = []
                pos_action19 = []
                pos_action20 = []
                pos_action21 = []
                pos_action22 = []
                pos_action23 = []
                pos_action24 = []
                for i in range(len(pos_action)):
                    pos_action1.append(pos_action[i][0])
                    pos_action2.append(pos_action[i][1])
                    pos_action3.append(pos_action[i][2])
                    pos_action4.append(pos_action[i][3])
                    pos_action5.append(pos_action[i][4])
                    pos_action6.append(pos_action[i][5])
                    pos_action7.append(pos_action[i][6])
                    pos_action8.append(pos_action[i][7])
                    pos_action9.append(pos_action[i][8])
                    pos_action10.append(pos_action[i][9])
                    pos_action11.append(pos_action[i][10])
                    pos_action12.append(pos_action[i][11])
                    pos_action13.append(pos_action[i][12])
                    pos_action14.append(pos_action[i][13])
                    pos_action15.append(pos_action[i][14])
                    pos_action16.append(pos_action[i][15])
                    pos_action17.append(pos_action[i][16])
                    pos_action18.append(pos_action[i][17])
                    pos_action19.append(pos_action[i][18])
                    pos_action20.append(pos_action[i][19])
                    pos_action21.append(pos_action[i][20])
                    pos_action22.append(pos_action[i][21])
                    pos_action23.append(pos_action[i][22])
                    pos_action24.append(pos_action[i][23])
                pos_action1 = torch.FloatTensor(pos_action1).to(self.device)
                pos_action2 = torch.FloatTensor(pos_action2).to(self.device)
                pos_action3 = torch.FloatTensor(pos_action3).to(self.device)
                pos_action4 = torch.FloatTensor(pos_action4).to(self.device)
                pos_action5 = torch.FloatTensor(pos_action5).to(self.device)
                pos_action6 = torch.FloatTensor(pos_action6).to(self.device)
                pos_action7 = torch.FloatTensor(pos_action7).to(self.device)
                pos_action8 = torch.FloatTensor(pos_action8).to(self.device)
                pos_action9 = torch.FloatTensor(pos_action9).to(self.device)
                pos_action10 = torch.FloatTensor(pos_action10).to(self.device)
                pos_action11 = torch.FloatTensor(pos_action11).to(self.device)
                pos_action12 = torch.FloatTensor(pos_action12).to(self.device)
                pos_action13 = torch.FloatTensor(pos_action13).to(self.device)
                pos_action14 = torch.FloatTensor(pos_action14).to(self.device)
                pos_action15 = torch.FloatTensor(pos_action15).to(self.device)
                pos_action16 = torch.FloatTensor(pos_action16).to(self.device)
                pos_action17 = torch.FloatTensor(pos_action17).to(self.device)
                pos_action18 = torch.FloatTensor(pos_action18).to(self.device)
                pos_action19 = torch.FloatTensor(pos_action19).to(self.device)
                pos_action20 = torch.FloatTensor(pos_action20).to(self.device)
                pos_action21 = torch.FloatTensor(pos_action21).to(self.device)
                pos_action22 = torch.FloatTensor(pos_action22).to(self.device)
                pos_action23 = torch.FloatTensor(pos_action23).to(self.device)
                pos_action24 = torch.FloatTensor(pos_action24).to(self.device)
                pos_action = [pos_action1, pos_action2, pos_action3, pos_action4, pos_action5, pos_action6,pos_action7, pos_action8, pos_action9,pos_action10, pos_action11, pos_action12,pos_action13, pos_action14, pos_action15, pos_action16, pos_action17, pos_action18,pos_action19, pos_action20, pos_action21,pos_action22, pos_action23, pos_action24]
            
            
            pos_state = torch.FloatTensor(pos_state).to(self.device)


            if self.num_agent == 2:
                tp_pos_action = [pos_action[0], pos_action[1]]
            elif self.num_agent == 3:
                tp_pos_action = [pos_action[0], pos_action[1], pos_action[2]]
            elif self.num_agent == 4:
                tp_pos_action = [pos_action[0], pos_action[1], pos_action[2], pos_action[3]]
            elif self.num_agent == 5:
                tp_pos_action = [pos_action[0], pos_action[1], pos_action[2], pos_action[3], pos_action[4]]
            elif self.num_agent == 6:
                tp_pos_action = [pos_action[0], pos_action[1], pos_action[2], pos_action[3], pos_action[4], pos_action[5]]
            elif self.num_agent == 9:
                tp_pos_action = [pos_action[0], pos_action[1], pos_action[2], pos_action[3], pos_action[4], pos_action[5],pos_action[6], pos_action[7], pos_action[8]]
            elif self.num_agent == 12:
                tp_pos_action = [pos_action[0], pos_action[1], pos_action[2], pos_action[3], pos_action[4], pos_action[5],pos_action[6], pos_action[7], pos_action[8],pos_action[9], pos_action[10], pos_action[11]]
            elif self.num_agent == 24:
                tp_pos_action = [pos_action[0], pos_action[1], pos_action[2], pos_action[3], pos_action[4], pos_action[5],pos_action[6], pos_action[7], pos_action[8],pos_action[9], pos_action[10], pos_action[11],pos_action[12], pos_action[13], pos_action[14], pos_action[15], pos_action[16], pos_action[17],pos_action[18], pos_action[19], pos_action[20],pos_action[21], pos_action[22], pos_action[23]]
                
            pos_loaa, pos_MI = self.mine_policy(pos_state, torch.cat(tp_pos_action, -1))

            loss = pos_loaa.mean()
            loss_list.append(loss.cpu().data.numpy())
            Mi_loss.append(pos_MI.cpu().data.numpy())
            self.mine_optimizer.zero_grad()
            loss.backward()
            self.mine_optimizer.step()

        return np.mean(loss_list), np.mean(Mi_loss)



    def train_club(self,replay_buffer,iterations,batch_size=64):
        min_MI_loss_list =[]
        for it in range(iterations):
            pos_action, pos_obs, pos_state = replay_buffer.sample_pos(batch_size)
            pos_action1 = []
            pos_action2 = []
            pos_action3 = []
            pos_action4 = []
            pos_action5 = []
            pos_action6 = []
            if self.num_agent == 2:
                for i in range(len(pos_action)):
                    pos_action1.append(pos_action[i][0])
                    pos_action2.append(pos_action[i][1])
                pos_action1 = torch.FloatTensor(pos_action1).to(self.device)
                pos_action2 = torch.FloatTensor(pos_action2).to(self.device)
                pos_action = [pos_action1, pos_action2]
                tp_pos_action = [pos_action[0], pos_action[1]]

            elif self.num_agent == 3:
                for i in range(len(pos_action)):
                    pos_action1.append(pos_action[i][0])
                    pos_action2.append(pos_action[i][1])
                    pos_action3.append(pos_action[i][2])
                pos_action1 = torch.FloatTensor(pos_action1).to(self.device)
                pos_action2 = torch.FloatTensor(pos_action2).to(self.device)
                pos_action3 = torch.FloatTensor(pos_action3).to(self.device)
                pos_action = [pos_action1, pos_action2,pos_action3]
                tp_pos_action = [pos_action[0],pos_action[1],pos_action[2]]

            elif self.num_agent == 4:
                for i in range(len(pos_action)):
                    pos_action1.append(pos_action[i][0])
                    pos_action2.append(pos_action[i][1])
                    pos_action3.append(pos_action[i][2])
                    pos_action4.append(pos_action[i][3])
                pos_action1 = torch.FloatTensor(pos_action1).to(self.device)
                pos_action2 = torch.FloatTensor(pos_action2).to(self.device)
                pos_action3 = torch.FloatTensor(pos_action3).to(self.device)
                pos_action4 = torch.FloatTensor(pos_action4).to(self.device)
                pos_action = [pos_action1, pos_action2,pos_action3,pos_action4]
                tp_pos_action = [pos_action[0], pos_action[1],pos_action[2],pos_action[3]]
                
            elif self.num_agent == 5:
                for i in range(len(pos_action)):
                    pos_action1.append(pos_action[i][0])
                    pos_action2.append(pos_action[i][1])
                    pos_action3.append(pos_action[i][2])
                    pos_action4.append(pos_action[i][3])
                    pos_action5.append(pos_action[i][4])

                pos_action1 = torch.FloatTensor(pos_action1).to(self.device)
                pos_action2 = torch.FloatTensor(pos_action2).to(self.device)
                pos_action3 = torch.FloatTensor(pos_action3).to(self.device)
                pos_action4 = torch.FloatTensor(pos_action4).to(self.device)
                pos_action5 = torch.FloatTensor(pos_action5).to(self.device)
                tp_pos_action = [pos_action1, pos_action2, pos_action3, pos_action4, pos_action5]

            elif self.num_agent == 6 :
                for i in range(len(pos_action)):
                    pos_action1.append(pos_action[i][0])
                    pos_action2.append(pos_action[i][1])
                    pos_action3.append(pos_action[i][2])
                    pos_action4.append(pos_action[i][3])
                    pos_action5.append(pos_action[i][4])
                    pos_action6.append(pos_action[i][5])
                pos_action1 = torch.FloatTensor(pos_action1).to(self.device)
                pos_action2 = torch.FloatTensor(pos_action2).to(self.device)
                pos_action3 = torch.FloatTensor(pos_action3).to(self.device)
                pos_action4 = torch.FloatTensor(pos_action4).to(self.device)
                pos_action5 = torch.FloatTensor(pos_action5).to(self.device)
                pos_action6 = torch.FloatTensor(pos_action6).to(self.device)
                pos_action = [pos_action1, pos_action2,pos_action3, pos_action4,pos_action5, pos_action6]
                tp_pos_action = [pos_action[0], pos_action[1],pos_action[2], pos_action[3],pos_action[4], pos_action[5]]
            elif self.num_agent == 9:
                pos_action1 = []
                pos_action2 = []
                pos_action3 = []
                pos_action4 = []
                pos_action5 = []
                pos_action6 = []
                pos_action7 = []
                pos_action8 = []
                pos_action9 = []
                for i in range(len(pos_action)):
                    pos_action1.append(pos_action[i][0])
                    pos_action2.append(pos_action[i][1])
                    pos_action3.append(pos_action[i][2])
                    pos_action4.append(pos_action[i][3])
                    pos_action5.append(pos_action[i][4])
                    pos_action6.append(pos_action[i][5])
                    pos_action7.append(pos_action[i][6])
                    pos_action8.append(pos_action[i][7])
                    pos_action9.append(pos_action[i][8])
                pos_action1 = torch.FloatTensor(pos_action1).to(self.device)
                pos_action2 = torch.FloatTensor(pos_action2).to(self.device)
                pos_action3 = torch.FloatTensor(pos_action3).to(self.device)
                pos_action4 = torch.FloatTensor(pos_action4).to(self.device)
                pos_action5 = torch.FloatTensor(pos_action5).to(self.device)
                pos_action6 = torch.FloatTensor(pos_action6).to(self.device)
                pos_action7 = torch.FloatTensor(pos_action7).to(self.device)
                pos_action8 = torch.FloatTensor(pos_action8).to(self.device)
                pos_action9 = torch.FloatTensor(pos_action9).to(self.device)
                pos_action = [pos_action1, pos_action2, pos_action3, pos_action4, pos_action5, pos_action6,pos_action7, pos_action8, pos_action9]
                tp_pos_action = [pos_action[0], pos_action[1],pos_action[2], pos_action[3],pos_action[4], pos_action[5], pos_action[6],pos_action[7], pos_action[8]]
            elif self.num_agent == 12:
                pos_action1 = []
                pos_action2 = []
                pos_action3 = []
                pos_action4 = []
                pos_action5 = []
                pos_action6 = []
                pos_action7 = []
                pos_action8 = []
                pos_action9 = []
                pos_action10 = []
                pos_action11 = []
                pos_action12 = []
                for i in range(len(pos_action)):
                    pos_action1.append(pos_action[i][0])
                    pos_action2.append(pos_action[i][1])
                    pos_action3.append(pos_action[i][2])
                    pos_action4.append(pos_action[i][3])
                    pos_action5.append(pos_action[i][4])
                    pos_action6.append(pos_action[i][5])
                    pos_action7.append(pos_action[i][6])
                    pos_action8.append(pos_action[i][7])
                    pos_action9.append(pos_action[i][8])
                    pos_action10.append(pos_action[i][9])
                    pos_action11.append(pos_action[i][10])
                    pos_action12.append(pos_action[i][11])
                pos_action1 = torch.FloatTensor(pos_action1).to(self.device)
                pos_action2 = torch.FloatTensor(pos_action2).to(self.device)
                pos_action3 = torch.FloatTensor(pos_action3).to(self.device)
                pos_action4 = torch.FloatTensor(pos_action4).to(self.device)
                pos_action5 = torch.FloatTensor(pos_action5).to(self.device)
                pos_action6 = torch.FloatTensor(pos_action6).to(self.device)
                pos_action7 = torch.FloatTensor(pos_action7).to(self.device)
                pos_action8 = torch.FloatTensor(pos_action8).to(self.device)
                pos_action9 = torch.FloatTensor(pos_action9).to(self.device)
                pos_action10 = torch.FloatTensor(pos_action10).to(self.device)
                pos_action11 = torch.FloatTensor(pos_action11).to(self.device)
                pos_action12 = torch.FloatTensor(pos_action12).to(self.device)
                pos_action = [pos_action1, pos_action2, pos_action3, pos_action4, pos_action5, pos_action6,pos_action7, pos_action8, pos_action9,pos_action10, pos_action11, pos_action12]
                tp_pos_action = [pos_action[0], pos_action[1],pos_action[2], pos_action[3],pos_action[4], pos_action[5], pos_action[6],pos_action[7], pos_action[8], pos_action[9],pos_action[10], pos_action[11]]
            elif self.num_agent == 24:
                pos_action1 = []
                pos_action2 = []
                pos_action3 = []
                pos_action4 = []
                pos_action5 = []
                pos_action6 = []
                pos_action7 = []
                pos_action8 = []
                pos_action9 = []
                pos_action10 = []
                pos_action11 = []
                pos_action12 = []
                pos_action13 = []
                pos_action14 = []
                pos_action15 = []
                pos_action16 = []
                pos_action17 = []
                pos_action18 = []
                pos_action19 = []
                pos_action20 = []
                pos_action21 = []
                pos_action22 = []
                pos_action23 = []
                pos_action24 = []
                for i in range(len(pos_action)):
                    pos_action1.append(pos_action[i][0])
                    pos_action2.append(pos_action[i][1])
                    pos_action3.append(pos_action[i][2])
                    pos_action4.append(pos_action[i][3])
                    pos_action5.append(pos_action[i][4])
                    pos_action6.append(pos_action[i][5])
                    pos_action7.append(pos_action[i][6])
                    pos_action8.append(pos_action[i][7])
                    pos_action9.append(pos_action[i][8])
                    pos_action10.append(pos_action[i][9])
                    pos_action11.append(pos_action[i][10])
                    pos_action12.append(pos_action[i][11])
                    pos_action13.append(pos_action[i][12])
                    pos_action14.append(pos_action[i][13])
                    pos_action15.append(pos_action[i][14])
                    pos_action16.append(pos_action[i][15])
                    pos_action17.append(pos_action[i][16])
                    pos_action18.append(pos_action[i][17])
                    pos_action19.append(pos_action[i][18])
                    pos_action20.append(pos_action[i][19])
                    pos_action21.append(pos_action[i][20])
                    pos_action22.append(pos_action[i][21])
                    pos_action23.append(pos_action[i][22])
                    pos_action24.append(pos_action[i][23])
                pos_action1 = torch.FloatTensor(pos_action1).to(self.device)
                pos_action2 = torch.FloatTensor(pos_action2).to(self.device)
                pos_action3 = torch.FloatTensor(pos_action3).to(self.device)
                pos_action4 = torch.FloatTensor(pos_action4).to(self.device)
                pos_action5 = torch.FloatTensor(pos_action5).to(self.device)
                pos_action6 = torch.FloatTensor(pos_action6).to(self.device)
                pos_action7 = torch.FloatTensor(pos_action7).to(self.device)
                pos_action8 = torch.FloatTensor(pos_action8).to(self.device)
                pos_action9 = torch.FloatTensor(pos_action9).to(self.device)
                pos_action10 = torch.FloatTensor(pos_action10).to(self.device)
                pos_action11 = torch.FloatTensor(pos_action11).to(self.device)
                pos_action12 = torch.FloatTensor(pos_action12).to(self.device)
                pos_action13 = torch.FloatTensor(pos_action13).to(self.device)
                pos_action14 = torch.FloatTensor(pos_action14).to(self.device)
                pos_action15 = torch.FloatTensor(pos_action15).to(self.device)
                pos_action16 = torch.FloatTensor(pos_action16).to(self.device)
                pos_action17 = torch.FloatTensor(pos_action17).to(self.device)
                pos_action18 = torch.FloatTensor(pos_action18).to(self.device)
                pos_action19 = torch.FloatTensor(pos_action19).to(self.device)
                pos_action20 = torch.FloatTensor(pos_action20).to(self.device)
                pos_action21 = torch.FloatTensor(pos_action21).to(self.device)
                pos_action22 = torch.FloatTensor(pos_action22).to(self.device)
                pos_action23 = torch.FloatTensor(pos_action23).to(self.device)
                pos_action24 = torch.FloatTensor(pos_action24).to(self.device)
                pos_action = [pos_action1, pos_action2, pos_action3, pos_action4, pos_action5, pos_action6,pos_action7, pos_action8, pos_action9,pos_action10, pos_action11, pos_action12,pos_action13, pos_action14, pos_action15, pos_action16, pos_action17, pos_action18,pos_action19, pos_action20, pos_action21,pos_action22, pos_action23, pos_action24]
                tp_pos_action = [pos_action[0], pos_action[1],pos_action[2], pos_action[3],pos_action[4], pos_action[5], pos_action[6],pos_action[7], pos_action[8], pos_action[9],pos_action[10], pos_action[11],pos_action[12], pos_action[13],pos_action[14], pos_action[15],pos_action[16], pos_action[17], pos_action[18],pos_action[19], pos_action[20], pos_action[21],pos_action[22], pos_action[23]]
                
            copy_action = torch.cat(tp_pos_action, -1)

            pos_state = torch.FloatTensor(pos_state).to(self.device)
            club_loss = self.club_policy.learning_loss(pos_state, copy_action)

            min_MI_loss_list.append(club_loss.cpu().data.numpy())
            self.club_optimizer.zero_grad()
            club_loss.backward()
            self.club_optimizer.step()

        return np.mean(min_MI_loss_list)


    def train_actor_with_mine(self, replay_buffer, iterations, batch_size=64, discount=0.99, tau=0.001,max_mi_c = 0.0,min_mi_c =0.0,min_adv_c=0.0,max_adv_c =0.0):
        Q_loss_list = []
        min_mi_list = []
        max_mi_list = []
        Q_grads_weight = []

        MI_grads_weight = []
        for it in range(iterations):
            o, x, y, o_, u, r, d ,ep_reward= replay_buffer.sample(batch_size)

            obs = torch.FloatTensor(o).to(self.device)
            next_obs = torch.FloatTensor(o_).to(self.device)
            state = torch.FloatTensor(x).to(self.device)
            action = torch.FloatTensor(u).to(self.device)
            next_state = torch.FloatTensor(y).to(self.device)
            done = torch.FloatTensor(1 - d).to(self.device)
            reward = torch.FloatTensor(r).to(self.device)

            #### TODO update Q
            next_action_list = []
            for i in range(self.num_agent):
                next_action_list.append(self.actor_target[i](next_obs[:, i])[0])
            with torch.no_grad():
                target_Q = self.critic_target(next_state, torch.cat(next_action_list, 1))
            if min_adv_c > 0.0 :
                with torch.no_grad():
                    MI_upper_bound = self.club_policy(state, action).detach()
                MI_upper_bound = MI_upper_bound.reshape(reward.shape)
            else :
                MI_upper_bound = 0.0
            if max_adv_c > 0.0 :
                with torch.no_grad():
                    neg_MI, half_MI= self.mine_policy(state, action)
                    neg_MI = neg_MI.detach()
                    half_MI = half_MI.detach()
                    MI_lower_bound = - neg_MI
                MI_lower_bound = MI_lower_bound.reshape(reward.shape)
            else :
                MI_lower_bound = 0.0
            
            
            #print(reward.shape, MI_upper_bound.shape, MI_lower_bound.shape)
            #assert reward.shape[0] == MI_upper_bound.shape[0] == MI_lower_bound.shape[0]
            #assert reward.shape[1] == MI_upper_bound.shape[1] == MI_lower_bound.shape[1]
            
            
            target_Q = reward + (done * discount * target_Q ).detach() - min_adv_c*MI_upper_bound + max_adv_c * MI_lower_bound
            
            # Get current Q estimate
            current_Q = self.critic(state, action)
            # Compute critic loss
            critic_loss = F.mse_loss(current_Q, target_Q)
            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)

            self.critic_optimizer.step()



            gen_old_action = []
          
            for i in range(self.num_agent):
                a, _,_ = self.actor[i](obs[:, i])
                gen_old_action.append(a)
            
            pol_loss = (torch.cat(gen_old_action, 1)**2).mean() * 1e-3
            actor_Q_loss = -self.critic(state, torch.cat(gen_old_action, 1)).mean()  + 1e-3*pol_loss 
            
            
            mi_sum_loss = 0.0

            if min_mi_c > 0.0 :
                min_upper_bound= self.club_policy(state, torch.cat(gen_old_action, 1))
                min_mi_loss = min_mi_c * torch.mean(min_upper_bound)
                min_mi_list.append(min_upper_bound.cpu().data.numpy())
            else :
                min_mi_list.append(0.0)
                min_mi_loss = 0.0
            if max_mi_c > 0.0:
                neg_max_lower_bound,_ = self.mine_policy(state, torch.cat(gen_old_action, 1))
                max_mi_loss = - max_mi_c * torch.mean(neg_max_lower_bound)
                max_mi_list.append(-neg_max_lower_bound.cpu().data.numpy())
            else :
                max_mi_loss = 0.0
                max_mi_list.append(0.0)
            mi_sum_loss+= min_mi_loss -  max_mi_loss
            
            
            

            Q_loss_list.append(actor_Q_loss.cpu().data.numpy())
            
            if max_mi_c == 0 and min_mi_c == 0 :
                self.actor_optimizer.zero_grad()
                actor_Q_loss.backward()
                for i in range(self.num_agent):
                    torch.nn.utils.clip_grad_norm_(self.actor[i].parameters(), 0.5)
                self.actor_optimizer.step()
                
            else :
                self.actor_optimizer.zero_grad()
                agent_1_fast_weights = OrderedDict(self.actor[0].named_parameters())
                actor_Q_loss.backward(retain_graph = True)
                for name, param in agent_1_fast_weights.items():
                    if name == "a_layer.weight":
                        Q_grads_weight.append(param.grad.cpu().data.numpy())
                mi_sum_loss.backward()
                for name, param in agent_1_fast_weights.items():
                    if name == "a_layer.weight":
                        MI_grads_weight.append(param.grad.cpu().data.numpy() - Q_grads_weight[0])
                for i in range(self.num_agent):
                    torch.nn.utils.clip_grad_norm_(self.actor[i].parameters(), 0.5)
                
                self.actor_optimizer.step()            
            

            ### TODO  replace ...........
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            for i in range(self.num_agent):
                for param, target_param in zip(self.actor[i].parameters(), self.actor_target[i].parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        return np.mean(Q_loss_list),np.mean(min_mi_list),np.mean(max_mi_list),0.0,0.0


    def train(self, replay_buffer, iterations, batch_size=64, discount=0.99, tau=0.001):

        Q_loss_list = []
        for it in range(iterations):
            # Sample replay buffer
            o, x, y, o_, u, r, d ,_= replay_buffer.sample(batch_size)

            obs = torch.FloatTensor(o).to(self.device)
            next_obs = torch.FloatTensor(o_).to(self.device)
            state = torch.FloatTensor(x).to(self.device)
            action = torch.FloatTensor(u).to(self.device)
            next_state = torch.FloatTensor(y).to(self.device)
            done = torch.FloatTensor(1 - d).to(self.device)
            reward = torch.FloatTensor(r).to(self.device)

            # Compute the target Q value
            next_action_list = []
            for i in range(self.num_agent):
                next_action_list.append(self.actor_target[i](next_obs[:, i])[0])
                #next_action_list.append(self.actor_target(next_obs[:, i])[0])

            target_Q = self.critic_target(next_state, torch.cat(next_action_list, 1))
            target_Q = reward + (done * discount * target_Q).detach()

            # Get current Q estimate
            current_Q = self.critic(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q, target_Q)

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
          
            self.critic_optimizer.step()

            # Compute actor loss
            gen_action = []
         
            for i in range(self.num_agent):
              #  gen_action.append(self.actor(obs[:, i])[0])
           #     print("???",obs[:, i].shape)
                action , _ , _ = self.actor[i](obs[:, i])
                gen_action.append(action)
                
            pol_loss = (torch.cat(gen_action, 1)**2).mean() * 1e-3
            actor_loss = -self.critic(state, torch.cat(gen_action, 1)).mean() +1e-3*pol_loss
            
            Q_loss_list.append(actor_loss.cpu().data.numpy())
            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            for i in range(self.num_agent):
                torch.nn.utils.clip_grad_norm_(self.actor[i].parameters(), 0.5)
         #   torch.nn.utils.clip_grad_norm(self.actor[1].parameters(), 0.5)
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                

            for i in range(self.num_agent):
                for param, target_param in zip(self.actor[i].parameters(), self.actor_target[i].parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        return np.mean(Q_loss_list)
            

    def save(self, filename, directory):
        if os.path.exists(directory) is False:
            os.makedirs(directory)
            
        for i in range(self.num_agent):
            torch.save(self.actor[i].state_dict(), '%s/%s_%s_actor.pth' % (directory, filename, str(i)))
        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))
        torch.save(self.mine_policy.state_dict(), '%s/%s_mine_policy.pth' % (directory, filename))

    def load(self, filename, directory):
        for i in range(self.num_agent):
            self.actor[i].load_state_dict(torch.load('%s/%s_%s_actor.pth' % (directory, filename, str(i))))
            
            
        #directory = "model/Org_Action_1_JSD_mi_0.05_100000_10_mine_MA_MINE_DDPG_HalfCheetah-v2_recon_100_0_1_0"
    
        self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.mine_policy.load_state_dict(torch.load('%s/%s_mine_policy.pth' % (directory, filename)))