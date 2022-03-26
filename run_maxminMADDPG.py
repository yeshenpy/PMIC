'''
    Demo run file for different algorithms and Mujoco tasks.
'''

import numpy as np
import torch
import gym
import argparse

import ma_utils
import algorithms.mpe_new_maxminMADDPG as MA_MINE_DDPG
import math
import os 


from tensorboardX import SummaryWriter


from multiprocessing import cpu_count
from maddpg.utils.env_wrappers import SubprocVecEnv, DummyVecEnv
import time 



cpu_num = 1
os.environ ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)


def make_parallel_env(env_id, n_rollout_threads, seed, discrete_action):
    def get_env_fn(rank):
        def init_env():
            env = make_env(env_id, discrete_action=discrete_action)
            env.seed(seed + rank * 1000)
            np.random.seed(seed + rank * 1000)
            return env
        return init_env
    if n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)])


def natural_exp_inc(init_param, max_param, global_step, current_step, inc_step=1000, inc_rate=0.5, stair_case=False):

    p = (global_step - current_step) / inc_step
    if stair_case:
        p = math.floor(p)
    increased_param = min((max_param - init_param) * math.exp(-inc_rate * p) + init_param, max_param)
    return increased_param

def make_env(scenario_name, benchmark=False,discrete_action=False):
    '''
    Creates a MultiAgentEnv object as env. This can be used similar to a gym
    environment by calling env.reset() and env.step().
    Use env.render() to view the environment on the screen.

    Input:
        scenario_name   :   name of the scenario from ./scenarios/ to be Returns
                            (without the .py extension)
        benchmark       :   whether you want to produce benchmarking data
                            (usually only done during evaluation)

    Some useful env properties (see environment.py):
        .observation_space  :   Returns the observation space for each agent
        .action_space       :   Returns the action space for each agent
        .n                  :   Returns the number of Agents
    '''
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation )
    return env


# Runs policy for X episodes and returns average reward
def evaluate_policy(policy, eval_episodes=10):
    avg_reward = 0.
    
    
    num_step = 0
    for _ in range(eval_episodes):
        obs = env.reset()
        done = False
        ep_step_num = 0

        #print("obs ",obs)
        while ep_step_num < 50:
            t = ep_step_num / 1000
            obs_t = []
            for i in range(n_agents):
                obs_t_one = list(obs[i])
                obs_t.append(obs_t_one)
            obs_t = np.array(obs_t)
            num_step +=1
            scaled_a_list = []
            for i in range(n_agents):
                #print("obs_t[i]",obs_t[i])
                a = policy.select_action(obs_t[i], i)
                scaled_a = np.multiply(a, 1.0)
                scaled_a_list.append(scaled_a)
                
            action_n = [[0, a[0], 0, a[1], 0] for a in scaled_a_list]
            next_obs, reward, done, _ = env.step(action_n)
        
            next_state = next_obs[0]
            obs = next_obs
            ep_step_num += 1
            avg_reward += reward[0]

    avg_reward /= eval_episodes
  
    print("---------------------------------------")
    print("Evaluation over %d episodes: %f" % (eval_episodes, avg_reward))
    print("---------------------------------------")
    return avg_reward,num_step/eval_episodes


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_name", default="DDPG")					# Policy name
    parser.add_argument("--env_name", default="HalfCheetah-v1")			# OpenAI gym environment name
    parser.add_argument("--seed", default=1, type=int)					# Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--batch_size", default=1024, type=int)			# Batch size for both actor and critic
    parser.add_argument("--discount", default=0.95, type=float)			# Discount factor
    parser.add_argument("--tau", default=0.01, type=float)				# Target network update rate
    parser.add_argument("--policy_freq", default=2, type=int)			# Frequency of delayed policy updates
    parser.add_argument("--freq_test_mine", default=5e3, type=float)
    parser.add_argument("--gpu-no", default='-1', type=str)				# GPU number, -1 means CPU
    parser.add_argument("--MI_update_freq", default=1, type=int)
    parser.add_argument("--max_adv_c", default=0.0, type=float)
    parser.add_argument("--min_adv_c", default=0.0, type=float)
    parser.add_argument("--discrete_action",action='store_true')
    args = parser.parse_args()
     
    file_name = "PMIC_%s_%s_%s_%s_%s"% (args.MI_update_freq,args.max_adv_c,args.min_adv_c,args.env_name,args.seed)

    writer = SummaryWriter(log_dir="./tensorboard/" + file_name)
    
    
    output_dir="./output/" + file_name
    model_dir = "./model/" + file_name
    if os.path.exists(output_dir) is False :
        os.makedirs(output_dir)
    if os.path.exists(model_dir) is False :
        os.makedirs(model_dir)  
    
    
    print("---------------------------------------")
    print("Settings: %s" % (file_name))
    print("---------------------------------------")


    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_no
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    env = make_parallel_env(args.env_name, 1, args.seed,
                            args.discrete_action)
    
    env = make_env(args.env_name)
    env = env.unwrapped
    #env = gym.make(args.env_name)
    # Set seeds
    env.seed(args.seed)

    n_agents = env.n
    obs_shape_n = [env.observation_space[i].shape[0] for i in range(env.n)]
    print("obs_shape_n ",obs_shape_n)
    ac = [env.action_space[i].shape for i in range(env.n)]
   
    action_shape_n = [2 for i in range(env.n)]
        
    print("env .n ",env.n)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    policy = MA_MINE_DDPG.MA_T_DDPG(n_agents,obs_shape_n,sum(obs_shape_n), action_shape_n, 1.0, device,0.0,0.0)

    replay_buffer = ma_utils.ReplayBuffer(1e6)
    
    good_data_buffer = ma_utils.embedding_Buffer(1e3) 
    bad_data_buffer = ma_utils.embedding_Buffer(1e3)  

    
    total_timesteps = 0
    timesteps_since_eval = 0
    episode_num = 0
    episode_reward = 0
    done = True

    get_epoch_Mi = False
    Mi_list = []
    data_recorder = []
    replay_buffer_recorder =[]
    moving_avg_reward_list = []
    embedding_recorder=[]

    best_reward_start = -1
    reward_list =[]
    eposide_reward_list =[]
    recorder_reward = 0
    best_reward  = -100000000000
    eposide_num = -1
    current_policy_performance = -1000
    episode_timesteps = 25
    start_time = time.time()
    
    t1 = time.time()
    while total_timesteps < 40e5:
        if episode_timesteps == 25:
         
            eposide_num +=1
            for d in replay_buffer_recorder:
                replay_buffer.add(d,episode_reward)
            if total_timesteps != 0:

                
         
                  
                  
                # walker 50 
                if len(good_data_buffer.pos_storage_reward) != 0:
                    if len(moving_avg_reward_list) >=10:
                        move_avg = np.mean(moving_avg_reward_list[-1000:])
                        current_policy_performance = move_avg
                    
                    lowest_reward = good_data_buffer.rank_storage(-100000)
                    mean_reward = np.mean(good_data_buffer.pos_storage_reward)
                    
                    if len(moving_avg_reward_list) >=10:
                        #writer.add_scalar("data/tp_lowest_reward", lowest_reward, total_timesteps)
                        if lowest_reward < move_avg:
                            lowest_reward = move_avg
                else :
                    lowest_reward = -500
                    mean_reward = 0 



                if lowest_reward < episode_reward:
                    Obs_list = []
                    State_list = []
                    Action_list = []
                    for d in data_recorder:
                        obs = d[0]
                        state = d[1]
                        Action_list.append(d[2])
                        Obs_list.append(obs)
                        State_list.append(state)
                    for index in range(len(Action_list)):
                        good_data_buffer.add_pos((Action_list[index], Obs_list[index], State_list[index]),episode_reward)
                else :
                    Obs_list = []
                    State_list = []
                    Action_list = []
                    for d in data_recorder:
                        obs = d[0]
                        state = d[1]
                        Action_list.append(d[2])
                        Obs_list.append(obs)
                        State_list.append(state)
                    for index in range(len(Action_list)):
                        bad_data_buffer.add_pos((Action_list[index], Obs_list[index], State_list[index]),episode_reward)

          
                # org 10
                if len(moving_avg_reward_list) % 10 == 0 :
                    writer.add_scalar("data/reward", np.mean(moving_avg_reward_list[-1000:]), total_timesteps)
                    writer.add_scalar("data/data_in_pos", np.mean(good_data_buffer.pos_storage_reward), total_timesteps)
                    
                    
                if episode_num%1000 == 0 :
                    print('Total T:', total_timesteps, 'Episode Num:', episode_num, 'Episode T:', episode_timesteps, 'Reward:', np.mean(moving_avg_reward_list[-1000:])/3.0, " time cost:", time.time() - t1)
                    t1 = time.time()
                    
                
           
            if total_timesteps >= 1024 and total_timesteps%100 == 0:
                sp_actor_loss_list =[]
                process_Q_list = []
                process_min_MI_list = []
                process_max_MI_list = []
                process_min_MI_loss_list = []
                process_max_MI_loss_list = []
                Q_grads_list =[]
                MI_grads_list =[]

                for i in range(1):
                    if  len(good_data_buffer.pos_storage)< 500:
                        process_Q = policy.train(replay_buffer, 1, args.batch_size, args.discount, args.tau)

                        process_min_MI = 0
                        process_min_MI_loss = 0
                        min_mi = 0.0
                        min_mi_loss = 0.0
                        process_max_MI = 0
                        pr_sp_loss = 0.0
                        Q_grads = 0.0
                        MI_grads =0.0
                        process_max_MI_loss = 0.0
                    else :
                        if total_timesteps % (args.MI_update_freq*100) == 0 :
                            if args.min_adv_c > 0.0 :
                                process_min_MI_loss = policy.train_club(bad_data_buffer, 1, batch_size=args.batch_size)
                            else :
                                process_min_MI_loss = 0.0
                            
                            if args.max_adv_c > 0.0 :
                                process_max_MI_loss,_ = policy.train_mine(good_data_buffer, 1, batch_size=args.batch_size)
                            else:
                                process_max_MI_loss = 0.0

                        else :
                            process_min_MI_loss = 0.0
                            process_max_MI_loss = 0.0
                        process_Q,process_min_MI ,process_max_MI, Q_grads,MI_grads = policy.train_actor_with_mine(replay_buffer, 1, args.batch_size,args.discount, args.tau,max_mi_c=0.0,min_mi_c=0.0 ,min_adv_c=args.min_adv_c, max_adv_c=args.max_adv_c )

                    process_max_MI_list.append(process_max_MI)
                    process_Q_list.append(process_Q)
                    Q_grads_list.append(Q_grads)
                    MI_grads_list.append(MI_grads)
                    process_max_MI_loss_list.append(process_max_MI_loss)
                    process_min_MI_list.append(process_min_MI)

                    process_min_MI_loss_list.append(process_min_MI_loss)
                if len(moving_avg_reward_list) % 10 == 0 :
                    writer.add_scalar("data/MINE_lower_bound_loss", np.mean(process_max_MI_loss_list), total_timesteps)
    
                    writer.add_scalar("data/process_Q", np.mean(process_Q_list), total_timesteps)
                    writer.add_scalar("data/club_upper_bound_loss", np.mean(process_min_MI_loss_list), total_timesteps)
              


            obs = env.reset()
            state =  np.concatenate(obs, -1)
            
            #print("ep reward ", episode_reward)
            moving_avg_reward_list.append(episode_reward)
            #obs = env.reset()
            
            #writer.add_scalar("data/run_reward", episode_reward, total_timesteps)

            done = False
            
            explr_pct_remaining = max(0, 25000 - episode_num) / 25000
            policy.scale_noise( 0.3 * explr_pct_remaining)
            policy.reset_noise()
            episode_reward = 0
            reward_list = []
            eposide_reward_list = []
            episode_timesteps = 0
            episode_num += 1
            data_recorder = []
            replay_buffer_recorder =[]
            best_reward_start = -1
            best_reward = -1000000
            # FIXME 1020
            Mi_list = []



        
      
        # Select action randomly or according to policy
        scaled_a_list = []
  
        for i in range(n_agents):
            a = policy.select_action(obs[i], i)
            scaled_a = np.multiply(a, 1.0)
            scaled_a_list.append(scaled_a)
            
       
        if args.env_name == "simple_reference":
            action_n = np.array([[0, a[0], 0, a[1],0, a[2],a[3]] for a in scaled_a_list])
            # Perform action
        elif args.env_name == "simple":
            action_n = np.array([[a[0], a[1]] for a in scaled_a_list])
        else :
            action_n = np.array([[0, a[0], 0, a[1],0] for a in scaled_a_list])
        #print(action_n)
        next_obs, reward, done, _ = env.step(action_n)
        reward = reward[0]
        next_state = np.concatenate(next_obs, -1)
        
        done = all(done)
     
        terminal = (episode_timesteps + 1 >= 25)  
        done_bool = float(done or terminal)
        
        episode_reward += reward
        eposide_reward_list.append(reward)
        
        # Store data in replay buffer
        replay_buffer_recorder.append((obs, state,next_state,next_obs, np.concatenate(scaled_a_list,-1), reward, done))
        data_recorder.append([obs, state, scaled_a_list])
        
        obs = next_obs
        state = next_state

        episode_timesteps += 1
        total_timesteps += 1
        timesteps_since_eval += 1
    
    print("total time ",time.time()-start_time)
    policy.save(total_timesteps, "model/" + file_name)
 
    writer.close()
