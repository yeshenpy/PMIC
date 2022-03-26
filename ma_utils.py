
import numpy as np
import scipy.signal
import torch
# from mpi_tools import mpi_statistics_scalar


# Code based on: 
# https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py

# Expects tuples of (state, next_state, action, reward, done)
class ReplayBuffer(object):
    def __init__(self, max_size=1e6):
        self.storage = []
        self.eposide_rewad =[]
        self.max_size = max_size
        self.ptr = 0
        self.max_reward = -1000000
        self.min_reward = 1000000
    def set_max_reward(self,max_reward):
        if max_reward >= self.max_reward:
            self.max_reward = max_reward
        return  bool(max_reward >= self.max_reward)

    def set_min_reward(self, min_reward):
        if min_reward <= self.min_reward:
            self.min_reward = min_reward
        return bool(min_reward <= self.min_reward)


    def save(self,file_name):
        numpy_array = np.array(self.storage)
        print("total len ", len(numpy_array))
        np.save(file_name+'_val_buffer.npy', numpy_array)

    def load(self,file_name):
        data = np.load(file_name+'_val_buffer.npy')
        print("data len :",len(data))
        self.storage=list(data)
    def add(self, data,eposide_reward):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.eposide_rewad[int(self.ptr)] = eposide_reward
            self.ptr = (self.ptr + 1) % self.max_size

        else:
            self.storage.append(data)
            self.eposide_rewad.append(eposide_reward)
            self.ptr = (self.ptr + 1) % self.max_size
    def sample_recently(self,batch_size,recent_num = 4000):

        if self.ptr == 0:
            ind = np.random.randint(int(self.max_size-recent_num), int(self.max_size), size=batch_size)
        else :
            ind = np.random.randint(max(int(self.ptr-recent_num),0), self.ptr, size=batch_size)
        o, x, y, o_, u, r, d = [], [], [], [], [],[],[]
        for i in ind:
            O, X, Y, O_, U, R, D = self.storage[i]
            o.append(np.array(O, copy=False))
            o_.append(np.array(O_, copy=False))
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))

        return np.array(o), np.array(x), np.array(y), np.array(o_), np.array(u), np.array(r).reshape(-1, 1), np.array(
            d).reshape(-1, 1)
    def sample_ago(self,batch_size):

        if self.ptr < 10000:
            ind = np.random.randint(self.ptr,int(self.max_size - 8000), size=batch_size)
        else:
            ind = np.random.randint(0, self.ptr-8000,size=batch_size)
        o, x, y, o_, u, r, d = [], [], [], [], [],[],[]
        for i in ind:
            O, X, Y, O_, U, R, D = self.storage[i]
            o.append(np.array(O, copy=False))
            o_.append(np.array(O_, copy=False))
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))

        return np.array(o), np.array(x), np.array(y), np.array(o_), np.array(u), np.array(r).reshape(-1, 1), np.array(
            d).reshape(-1, 1)


    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        o,x, y,o_, u, r, d ,ep_reward=[], [], [], [], [], [],[],[]

        for i in ind:
            O,X, Y,O_, U, R, D = self.storage[i]
            ep_r = self.eposide_rewad[i]
            ep_reward.append(np.array(ep_r, copy=False))
            o.append(np.array(O, copy=False))
            o_.append(np.array(O_, copy=False))
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))

        return np.array(o),np.array(x), np.array(y),np.array(o_), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1),np.array(ep_reward)


class embedding_Buffer(object):
    def __init__(self, max_size=1e6):
        self.pos_storage = []
        self.pos_storage_reward = []

        self.pos_epoch_reward = []
        self.neg_storage = []
        self.neg_storage_reward = []
        self.neg_epoch_reward = []
        self.worse_storage = []
        self.worse_storage_reward = []
        self.storage = []

        self.max_max_size = max_size
        self.max_size = max_size
        self.total_max = max_size
        self.pos_ptr = 0 
        self.neg_ptr = 0
        self.ptr = 0
        self.worse_ptr = 0
        self.max_reward = -100000000
        self.min_reward = 1000000000

    def rank_storage(self, current_policy_reward):
        
        
        #print("step 0:", self.pos_storage_reward)
        index = np.argsort(self.pos_storage_reward)
        
        
        #print("test 1:",index)
        #print("test 2:",self.pos_storage_reward)
        self.pos_storage_reward = list(np.array(self.pos_storage_reward)[index])
        
        self.pos_storage = list(np.array(self.pos_storage)[index])
        #print("test 3:",self.pos_storage)
        start_index = 0 
        for i , d in enumerate(self.pos_storage_reward):
            if d > current_policy_reward:
                break;
            else :
                start_index = i+1 
        self.pos_storage_reward = self.pos_storage_reward[start_index::]
        self.pos_storage = self.pos_storage[start_index::]
        
        if len(self.pos_storage) == self.max_size :
            self.pos_ptr = 0
        else :
            self.pos_ptr = self.pos_ptr - start_index 
            
        return  self.pos_storage_reward[0]

    def greater(self,reward):
        if reward > self.max_reward:
            self.max_reward = reward
            return  True
        else :
            return  False
    def can_save(self,reward):
        if reward > self.max_reward *0.8 :
            return True
        else :
            return False
    def get_baseline(self):
        return  self.max_reward*0.8

    def get_MI(self,policy,deafult = 0):
        pos_mean_mi_list =[]
        neg_mena_mi_list =[]
        for index in range(0,len(self.pos_storage),100):
            temp_list = []
            temp_obs =[]
            temp_state =[]
            for i in range(100):
                if index+i < len(self.pos_storage):
                    temp_list.append(self.pos_storage[index+i][0])
                    temp_obs.append(self.pos_storage[index+i][1])
                    temp_state.append(self.pos_storage[index+i][2])
            mean_mi = policy.get_mi_from_a(np.array(temp_obs),np.array(temp_list),deafult)
            pos_mean_mi_list.append(mean_mi)
        for index in range(0, len(self.neg_storage), 100):
            temp_list = []
            temp_state =[]
            temp_obs =[]
            for i in range(100):
                if index + i < len(self.neg_storage):
                    temp_list.append(self.neg_storage[index + i][0])
                    temp_obs.append(self.neg_storage[index+i][1])
                    temp_state.append(self.neg_storage[index + i][2])
            mean_mi = policy.get_mi_from_a(np.array(temp_obs),np.array(temp_list),deafult)
            neg_mena_mi_list.append(mean_mi)
        return np.mean(pos_mean_mi_list),np.mean(neg_mena_mi_list)

    def clear_pos(self):
        self.pos_storage_reward =[]
        self.pos_storage = []
        self.pos_ptr = 0
    def get_mean_pos_reward(self):
        return np.mean(self.pos_epoch_reward)
    def get_mean_neg_reward(self):
        return np.mean(self.neg_epoch_reward)

    def add_pos(self, data, rollout_reward):
    
        #print("rollout_reward",rollout_reward)
        if len(self.pos_storage) == self.max_size:
            self.pos_storage_reward[int(self.pos_ptr)] = rollout_reward
            self.pos_storage[int(self.pos_ptr)] = data
            self.pos_ptr = (self.pos_ptr + 1) % self.max_size
        else:
            self.pos_storage.append(data)
            self.pos_storage_reward.append(rollout_reward)
            self.pos_ptr = (self.pos_ptr + 1) % self.max_size

    def add(self, data):
        if len(self.storage) == self.total_max:

            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.total_max
        else:
            self.storage.append(data)
            self.ptr = (self.ptr + 1) % self.total_max

    def add_neg(self,data,rollout_reward):
        if len(self.neg_storage) == self.max_max_size:
            #print("int(self.neg_ptr) ",int(self.neg_ptr))
            self.neg_storage[int(self.neg_ptr)] = data
            self.neg_storage_reward[int(self.neg_ptr)] = rollout_reward
            self.neg_ptr = (self.neg_ptr + 1) % self.max_max_size
        else:
            self.neg_storage_reward.append(rollout_reward)
            self.neg_storage.append(data)
            self.neg_ptr = (self.neg_ptr + 1) % self.max_max_size


    # def caculate_Q(self,actor,critic,device):
    #     q_list = []
    #     obs_list =[]
    #     state_list =[]
    #     for d in self.pos_storage:
    #         obs_list.append(d[1])
    #         state_list.append(d[2])
    #
    #     for i in range(0,len(self.pos_storage),100):
    #         pos_obs = np.array(obs_list)[i:i+100]
    #         pos_state = np.array(state_list)[i:i + 100]
    #
    #         pos_obs = torch.FloatTensor(pos_obs).to(device)
    #         pos_state = torch.FloatTensor(pos_state).to(device)
    #         action_list =[]
    #
    #         #print("2222 ",pos_obs[:, 0].shape)
    #         #print("3333 ",pos_obs[:, 1].shape)
    #         for i in range(2):
    #             action_list.append(actor[i](pos_obs[:, i])[0])
    #         current_Q = critic(pos_state, torch.cat(action_list, 1))
    #         q_list.extend(list(current_Q.cpu().data.numpy()))
    #     self.pos_Q = np.reshape(q_list,[-1])
    #     #self.pos_adv = (self.pos_storage_reward - self.pos_Q) / np.abs(np.max(self.pos_storage_reward - self.pos_Q))
    #
    #     obs_list = []
    #     state_list = []
    #     for d in self.neg_storage:
    #         obs_list.append(d[1])
    #         state_list.append(d[2])
    #     q_list = []
    #     for i in range(0, len(self.neg_storage), 100):
    #         neg_obs = np.array(obs_list)[i:i + 100]
    #         neg_state = np.array(state_list)[i:i + 100]
    #         neg_obs = torch.FloatTensor(neg_obs).to(device)
    #         neg_state = torch.FloatTensor(neg_state).to(device)
    #         action_list = []
    #         for i in range(2):
    #             action_list.append(actor[i](neg_obs[:, i])[0])
    #         current_Q = critic(neg_state, torch.cat(action_list, 1))
    #         q_list.extend(list(current_Q.cpu().data.numpy()))
    #     self.neg_Q =  np.reshape(q_list,[-1])
    #
    #     #self.neg_adv = (self.neg_Q - self.neg_storage_reward)/np.abs(np.max(self.neg_Q - self.neg_storage_reward))
    #
    #     self.pos_adv = (self.pos_storage_reward) / np.abs(np.max(self.pos_storage_reward ))
    #     self.neg_adv = (self.neg_storage_reward) / np.abs(np.max(self.neg_storage_reward))

    def sample_pos(self, batch_size):
        ind = np.random.randint(0, len(self.pos_storage) , size=batch_size)
        embedding_list,obs_list,state_list =[],[], []
        for i in ind:
            embedding,obs,state= self.pos_storage[i]
            #discount_reward = self.pos_storage_reward[i]
            # Q_list.append(self.pos_Q[i])
            # adv.append(self.pos_adv[i])
            state_list.append(np.array(state, copy=False))
            embedding_list.append(np.array(embedding, copy=False))
            obs_list.append(np.array(obs,copy=False))
            #discount_reward_list.append(discount_reward)
        return np.array(embedding_list),np.array(obs_list),np.array(state_list)

    def sample_neg(self, batch_size):
        ind = np.random.randint(0, len(self.neg_storage) , size=batch_size)
        embedding_list , discount_reward_list ,obs_list,state_list,Q_list,adv=[],[],[],[], [],[]
        for i in ind:
            embedding ,obs,state= self.neg_storage[i]
            discount_reward = self.neg_storage_reward[i]
            # adv.append(self.neg_adv[i])
            # Q_list.append(self.neg_Q[i])
            state_list.append(np.array(state, copy=False))
            embedding_list.append(np.array(embedding, copy=False))
            obs_list.append(np.array(obs,copy=False))
            discount_reward_list.append(discount_reward)
        return np.array(embedding_list),np.array(discount_reward_list),np.array(obs_list),np.array(state_list)

    def sample_rencently(self,batch_size,recent_num=10000):

        if self.ptr == 0:
            ind = np.random.randint(int(self.max_size - recent_num), int(self.max_size), size=batch_size)
        else:
            ind = np.random.randint(max(int(self.ptr - recent_num), 0), self.ptr, size=batch_size)
        #ind = np.random.randint(0, len(self.storage), size=batch_size)
        embedding_list, obs_list, state_list = [], [], []
        for i in ind:
            embedding, obs, state = self.storage[i]

            state_list.append(np.array(state, copy=False))
            embedding_list.append(np.array(embedding, copy=False))
            obs_list.append(np.array(obs, copy=False))

        return np.array(embedding_list), np.array(obs_list), np.array(state_list)
    def sample(self,batch_size):

        ind = np.random.randint(0, len(self.storage), size=batch_size)
        #ind = np.random.randint(0, len(self.storage), size=batch_size)
        embedding_list, obs_list, state_list = [], [], []
        for i in ind:
            embedding, obs, state = self.storage[i]

            state_list.append(np.array(state, copy=False))
            embedding_list.append(np.array(embedding, copy=False))
            obs_list.append(np.array(obs, copy=False))

        return np.array(embedding_list), np.array(obs_list), np.array(state_list)

    def refresh_pos_data(self,mean_reward):
        new_pos_storage = []
        new_pos_storage_reward = []
        for index,reward in enumerate(self.pos_storage_reward):
            if reward > mean_reward:
                new_pos_storage.append(self.pos_storage[index])
                new_pos_storage_reward.append(reward)
        self.pos_storage = new_pos_storage
        self.pos_storage_reward = new_pos_storage_reward
        self.pos_ptr = len(new_pos_storage)%self.max_max_size

    def refresh_neg_data(self,mean_reward):
        new_neg_storage = []
        new_neg_storage_reward = []
        for index, reward in enumerate(self.neg_storage_reward):
            if reward < mean_reward:
                new_neg_storage.append(self.neg_storage[index])
                new_neg_storage_reward.append(reward)
        self.neg_storage = new_neg_storage
        self.neg_storage_reward = new_neg_storage_reward
        self.neg_ptr = len(new_neg_storage)%self.max_max_size


class Buffer(object):
    def __init__(self, max_size=1e3):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def add(self, data):
        self.storage[int(self.ptr)] = data

    def clear(self):
        self.storage.clear()

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


class ReplayBufferPPO(object):
    """
    original from: https://github.com/bluecontra/tsallis_actor_critic_mujoco/blob/master/spinup/algos/ppo/ppo.py
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.size = size
        self.gamma, self.lam = gamma, lam
        self.ptr = 0
        self.path_start_idx, self.max_size = 0, size

        self.reset()

    def reset(self):
        self.obs_buf = np.zeros([self.size, self.obs_dim], dtype=np.float32)
        self.act_buf = np.zeros([self.size, self.act_dim], dtype=np.float32)
        self.adv_buf = np.zeros(self.size, dtype=np.float32)
        self.rew_buf = np.zeros(self.size, dtype=np.float32)
        self.ret_buf = np.zeros(self.size, dtype=np.float32)
        self.val_buf = np.zeros(self.size, dtype=np.float32)
        self.logp_buf = np.zeros(self.size, dtype=np.float32)

    def add(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size  # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount(deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = discount(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        assert self.ptr == self.max_size  # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        # adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        adv_mean = np.mean(self.adv_buf)
        adv_std = np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        return [self.obs_buf, self.act_buf, self.adv_buf,
                self.ret_buf, self.logp_buf]


class ReplayBuffer_MC(object):
    def __init__(self, max_size=1e6):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def add(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        x, u, r = [], [], []

        for i in ind:
            X, U, R = self.storage[i]
            x.append(np.array(X, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))

        return np.array(x), np.array(u), np.array(r).reshape(-1, 1)


class ReplayBuffer_VDFP(object):
    def __init__(self, max_size=1e5):
        self.storage = []
        self.max_size = int(max_size)
        self.ptr = 0

    def add(self, data):
        if len(self.storage) == self.max_size:
            self.storage[self.ptr] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        s, a, u, x = [], [], [], []

        for i in ind:
            S, A, U, X = self.storage[i]
            s.append(np.array(S, copy=False))
            a.append(np.array(A, copy=False))
            u.append(np.array(U, copy=False))
            x.append(np.array(X, copy=False))

        return np.array(s), np.array(a), np.array(u).reshape(-1, 1), np.array(x)

    def sample_traj(self, batch_size, offset=0):
        ind = np.random.randint(0, len(self.storage) - int(offset), size=batch_size)
        if len(self.storage) == self.max_size:
            ind = (self.ptr + self.max_size - ind) % self.max_size
        else:
            ind = len(self.storage) - ind - 1
        # ind = (self.ptr - ind + len(self.storage)) % len(self.storage)
        s, a, x = [], [], []

        for i in ind:
            S, A, _, X = self.storage[i]
            s.append(np.array(S, copy=False))
            a.append(np.array(A, copy=False))
            x.append(np.array(X, copy=False))

        return np.array(s), np.array(a), np.array(x)

    def sample_traj_return(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        u, x = [], []

        for i in ind:
            _, _, U, X = self.storage[i]
            u.append(np.array(U, copy=False))
            x.append(np.array(X, copy=False))

        return np.array(u).reshape(-1, 1), np.array(x)


def store_experience(replay_buffer, trajectory, s_dim, a_dim,
                     sequence_length, min_sequence_length=0, is_padding=False, gamma=0.99,
                     ):
    s_traj, a_traj, r_traj = trajectory

    # for the convenience of manipulation
    arr_s_traj = np.array(s_traj)
    arr_a_traj = np.array(a_traj)
    arr_r_traj = np.array(r_traj)

    zero_pads = np.zeros(shape=[sequence_length, s_dim + a_dim])

    # for i in range(len(s_traj) - self.sequence_length):
    for i in range(len(s_traj) - min_sequence_length):
        tmp_s = arr_s_traj[i]
        tmp_a = arr_a_traj[i]
        tmp_soff = arr_s_traj[i:i + sequence_length]
        tmp_aoff = arr_a_traj[i:i + sequence_length]
        tmp_saoff = np.concatenate([tmp_soff, tmp_aoff], axis=1)

        tmp_saoff_padded = np.concatenate([tmp_saoff, zero_pads], axis=0)
        tmp_saoff_padded_clip = tmp_saoff_padded[:sequence_length, :]

        tmp_roff = arr_r_traj[i:i + sequence_length]
        tmp_u = np.matmul(tmp_roff, np.power(gamma, [j for j in range(len(tmp_roff))]))

        replay_buffer.add((tmp_s, tmp_a, tmp_u, tmp_saoff_padded_clip))


def discount(x, gamma):
    """ Calculate discounted forward sum of a sequence at each point """
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input: 
        vector x, 
        [x0, 
         x1, 
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    # return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]
    return scipy.signal.lfilter([1.0], [1.0, -gamma], x[::-1])[::-1]


class Scaler(object):
    """ Generate scale and offset based on running mean and stddev along axis=0
        offset = running mean
        scale = 1 / (stddev + 0.1) / 3 (i.e. 3x stddev = +/- 1.0)
    """

    def __init__(self, obs_dim):
        """
        Args:
            obs_dim: dimension of axis=1
        """
        self.vars = np.zeros(obs_dim)
        self.means = np.zeros(obs_dim)
        self.m = 0
        self.n = 0
        self.first_pass = True

    def update(self, x):
        """ Update running mean and variance (this is an exact method)
        Args:
            x: NumPy array, shape = (N, obs_dim)
        see: https://stats.stackexchange.com/questions/43159/how-to-calculate-pooled-
               variance-of-two-groups-given-known-group-variances-mean
        """
        if self.first_pass:
            self.means = np.mean(x, axis=0)
            self.vars = np.var(x, axis=0)
            self.m = x.shape[0]
            self.first_pass = False
        else:
            n = x.shape[0]
            new_data_var = np.var(x, axis=0)
            new_data_mean = np.mean(x, axis=0)
            new_data_mean_sq = np.square(new_data_mean)
            new_means = ((self.means * self.m) + (new_data_mean * n)) / (self.m + n)
            self.vars = (((self.m * (self.vars + np.square(self.means))) +
                          (n * (new_data_var + new_data_mean_sq))) / (self.m + n) -
                         np.square(new_means))
            self.vars = np.maximum(0.0, self.vars)  # occasionally goes negative, clip
            self.means = new_means
            self.m += n

    def get(self):
        """ returns 2-tuple: (scale, offset) """
        return 1/(np.sqrt(self.vars) + 0.1)/3, self.means
