# Demo run command

nohup python -u run_maxminMADDPG.py  --MI_update_freq=1 --min_adv_c=0.1 --max_adv_c=0.01  --env_name=simple_tag_coop_partial_obs_3_agents  --policy_name=MA_MINE_DDPG --seed=1 --gpu-no=-1 > ./logs/t1.log 2>&1 &
nohup python -u run_maxminMADDPG.py  --MI_update_freq=1 --min_adv_c=0.1 --max_adv_c=0.01  --env_name=simple_tag_coop_partial_obs_3_agents  --policy_name=MA_MINE_DDPG --seed=2 --gpu-no=-1 > ./logs/t2.log 2>&1 &
nohup python -u run_maxminMADDPG.py  --MI_update_freq=1 --min_adv_c=0.1 --max_adv_c=0.01  --env_name=simple_tag_coop_partial_obs_3_agents  --policy_name=MA_MINE_DDPG --seed=3 --gpu-no=-1 > ./logs/t3.log 2>&1 &
nohup python -u run_maxminMADDPG.py  --MI_update_freq=1 --min_adv_c=0.1 --max_adv_c=0.01  --env_name=simple_tag_coop_partial_obs_3_agents  --policy_name=MA_MINE_DDPG --seed=4 --gpu-no=-1 > ./logs/t3.log 2>&1 &
nohup python -u run_maxminMADDPG.py  --MI_update_freq=1 --min_adv_c=0.1 --max_adv_c=0.01  --env_name=simple_tag_coop_partial_obs_3_agents  --policy_name=MA_MINE_DDPG --seed=5 --gpu-no=-1 > ./logs/t3.log 2>&1 &