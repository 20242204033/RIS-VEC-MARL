import math
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import numpy as np
import matplotlib.pyplot as plt
import Environment
from sac_agent import Agent
from buffer import ReplayBuffer
# 【已修改】导入与训练时一致的 Global_SAC_Critic
from global_sac_critic import Global_SAC_Critic


# ################## SETTINGS ######################
# RIS coordination
RIS_x, RIS_y, RIS_z = 220, 220, 25

# BS coordination
BS_x, BS_y, BS_z = 0, 0, 25

up_lanes = [i/2.0 for i in [400+3.5/2, 400+3.5+3.5/2, 800+3.5/2, 800+3.5+3.5/2]]
down_lanes = [i/2.0 for i in [400-3.5-3.5/2, 400-3.5/2, 800-3.5-3.5/2, 800-3.5/2]]
left_lanes = [i/2.0 for i in [400+3.5/2, 400+3.5+3.5/2, 800+3.5/2, 800+3.5+3.5/2]]
right_lanes = [i/2.0 for i in [400-3.5-3.5/2, 400-3.5/2, 800-3.5-3.5/2, 800-3.5/2]]

print('------------- lanes are -------------')
print('up_lanes :', up_lanes)
print('down_lanes :', down_lanes)
print('left_lanes :', left_lanes)
print('right_lanes :', right_lanes)
print('------------------------------------')

width = 800/2
height = 800/2

# 【已修改】测试时应设为0，因为我们不进行训练和存储
IS_TRAIN = 0

n_veh = 8
M = 36
control_bit = 3

###----------------环境设置------------###
env = Environment.Environ(down_lanes, up_lanes, left_lanes, right_lanes, width, height, n_veh, M, control_bit)

env.make_new_game()  # 添加车辆

n_episode_test = 10
n_step_per_episode = 100

# 【新增】将NOMA协商函数复制到此，以供调用
def pairing_negotiation(intent_actions, n_veh, channel_gains, ccs_threshold=0.1):
    partners = [-1] * n_veh
    for i in range(n_veh):
        intent_j = intent_actions[i]
        if intent_j != i:
            if intent_actions[intent_j] == i:
                gain_i = channel_gains[i]
                gain_j = channel_gains[intent_j]
                if abs(gain_i - gain_j) > ccs_threshold:
                    partners[i] = intent_j
                    partners[intent_j] = i
    final_groups = []
    paired_users = set()
    for i in range(n_veh):
        if i not in paired_users:
            partner_i = partners[i]
            if partner_i != -1:
                final_groups.append(sorted([i, partner_i]))
                paired_users.add(i)
                paired_users.add(partner_i)
            else:
                final_groups.append([i])
                paired_users.add(i)
    return final_groups

def marl_get_state(idx):
    """ Get state from the environment """
    list = []

    Data_Buf = env.DataBuf[idx] / 10

    data_t = env.data_t[idx] / 10

    data_p = env.data_p[idx] / 10

    over_data = env.over_data[idx] / 10

    rate = env.vehicle_rate[idx] / 20

    list.append(Data_Buf)
    list.append(data_t)
    list.append(data_p)
    list.append(over_data)
    list.append(rate)

    return list

marl_n_input = len(marl_get_state(0))
# 【已修改】定义与SAC Agent匹配的动作维度
marl_n_continuous_actions = 2
marl_n_discrete_actions = 1
marl_n_output = marl_n_continuous_actions + marl_n_discrete_actions

##---Initializations networks parameters---##
batch_size = 64
memory_size = 1000000
gamma = 0.99
alpha = 0.0001
beta = 0.001
update_actor_interval = 2
noise = 0.2
C_fc1_dims = 1024
C_fc2_dims = 512
C_fc3_dims = 256
A_fc1_dims = 512
A_fc2_dims = 256
tau = 0.005

# 【已修改】初始化与训练时一致的SAC智能体
agents = []
for index_agent in range(n_veh):
    print("Initializing agent", index_agent)
    agent = Agent(alpha, beta, marl_n_input, tau,
                  n_continuous_actions=marl_n_continuous_actions,
                  n_discrete_actions=marl_n_discrete_actions,
                  gamma=gamma,
                  c1=C_fc1_dims, c2=C_fc2_dims, c3=C_fc3_dims,
                  a1=A_fc1_dims, a2=A_fc2_dims,
                  batch_size=batch_size, n_agents=n_veh,
                  agent_name=index_agent, noise=noise)
    agents.append(agent)

# 【已修改】初始化与训练时一致的Global_SAC_Critic
print("Initializing Global SAC critic for testing...")
global_agent = Global_SAC_Critic(beta, marl_n_input, tau, marl_n_output, gamma, C_fc1_dims, C_fc2_dims, C_fc3_dims,
                 batch_size, n_veh, update_actor_interval)

# 【已修改】测试循环，以兼容SAC智能体并包含联合优化逻辑
# 定义RIS优化频率
K_STEPS_FOR_RIS_OPTIMIZATION = 100 # 与训练时保持一致

# 加载您通过 marl_train_bcd.py 训练好的SAC模型
print("Loading SAC models for testing...")
global_agent.load_models()
for i in range(n_veh):
    agents[i].load_models()

record_reward_ = np.zeros([n_veh, n_episode_test])
record_global_reward_average = []
Sum_DataBuf_length = []
Sum_Power = []
Sum_Power_local = []
Sum_Power_offload = []
Vehicle_positions_x0 = []
Vehicle_positions_y0 = []
Vehicle_positions_x1 = []
Vehicle_positions_y1 = []
Vehicle_positions_x2 = []
Vehicle_positions_y2 = []
Vehicle_positions_x3 = []
Vehicle_positions_y3 = []
Vehicle_positions_x4 = []
Vehicle_positions_y4 = []
Vehicle_positions_x5 = []
Vehicle_positions_y5 = []
Vehicle_positions_x6 = []
Vehicle_positions_y6 = []
Vehicle_positions_x7 = []
Vehicle_positions_y7 = []

for i_episode in range(n_episode_test):
    done = 0
    print("-------------------------------- Test Episode:", i_episode, "-------------------------------------------")
    record_reward = np.zeros([n_veh, n_step_per_episode], dtype=np.float16)
    record_global_reward = np.zeros(n_step_per_episode)
    DataBuf_length = []
    Power = []
    Power_local = []
    Power_offload = []

    env.renew_positions()
    env.compute_parms()
    Vehicle_positions_x0.append(env.vehicles[0].position[0])
    Vehicle_positions_y0.append(env.vehicles[0].position[1])
    Vehicle_positions_x1.append(env.vehicles[1].position[0])
    Vehicle_positions_y1.append(env.vehicles[1].position[1])
    Vehicle_positions_x2.append(env.vehicles[2].position[0])
    Vehicle_positions_y2.append(env.vehicles[2].position[1])
    Vehicle_positions_x3.append(env.vehicles[3].position[0])
    Vehicle_positions_y3.append(env.vehicles[3].position[1])
    Vehicle_positions_x4.append(env.vehicles[4].position[0])
    Vehicle_positions_y4.append(env.vehicles[4].position[1])
    Vehicle_positions_x5.append(env.vehicles[5].position[0])
    Vehicle_positions_y5.append(env.vehicles[5].position[1])
    Vehicle_positions_x6.append(env.vehicles[6].position[0])
    Vehicle_positions_y6.append(env.vehicles[6].position[1])
    Vehicle_positions_x7.append(env.vehicles[7].position[0])
    Vehicle_positions_y7.append(env.vehicles[7].position[1])

    marl_state_old_all = [marl_get_state(i) for i in range(n_veh)]

    for i_step in range(n_step_per_episode):
        
        # --- 1. 按频率优化RIS并更新信道状态 ---
        if i_step % K_STEPS_FOR_RIS_OPTIMIZATION == 0:
            env.optimize_phase_shift()
            env.update_channel_gains()

        # --- 2. 智能体观察最新的环境状态 ---
        current_channel_gains = env.get_channel_gains()
        marl_state_old_all = [marl_get_state(i) for i in range(n_veh)]

        # --- 3. 智能体决策并进行NOMA协商 ---
        marl_power_actions = []
        marl_intent_actions = []
        for i in range(n_veh):
            # SAC Agent的choose_action返回两个值
            power_action, intent_action = agents[i].choose_action(marl_state_old_all[i])
            marl_power_actions.append(power_action)
            marl_intent_actions.append(intent_action)
        
        # 在测试时同样需要NOMA协商
        noma_groups = pairing_negotiation(marl_intent_actions, n_veh, current_channel_gains)

        # --- 4. 准备功率动作并与环境交互 ---
        action_for_env = np.zeros([2, n_veh], dtype=float)
        for i in range(n_veh):
            clipped_power = np.clip(marl_power_actions[i], -0.999, 0.999)
            action_for_env[0, i] = (clipped_power[0] + 1) / 2
            action_for_env[1, i] = (clipped_power[1] + 1) / 2

        per_user_reward, global_reward, databuf, data_trans, data_local, over_power, over_data = env.step(action_for_env, noma_groups)
        
        Power.append(np.sum(action_for_env))
        Power_offload.append(np.sum(action_for_env[0]))
        Power_local.append(np.sum(action_for_env[1]))
        DataBuf_length.append(np.sum(databuf))

        record_global_reward[i_step] = global_reward
        for i in range(n_veh):
            record_reward[i, i_step] = per_user_reward[i]

        marl_state_old_all = [marl_get_state(i) for i in range(n_veh)]

    for i in range(n_veh):
        record_reward_[i, i_episode] = np.mean(record_reward[i])
        print('user', i, record_reward_[i, i_episode], end='      ')
    
    average_global_reward = np.mean(record_global_reward)
    record_global_reward_average.append(average_global_reward)

    Power_episode = np.mean(Power)
    Power_local_episode = np.mean(Power_local)
    Power_offload_episode = np.mean(Power_offload)
    DataBuf_length_episode = np.mean(DataBuf_length)

    Sum_Power.append(Power_episode)
    Sum_Power_local.append(Power_local_episode)
    Sum_Power_offload.append(Power_offload_episode)
    Sum_DataBuf_length.append(DataBuf_length_episode)
    print('Global reward:', average_global_reward)
    print('Average Total Power:', Power_episode)
    print('Average local power:', Power_local_episode, '   Average offload power:',
          Power_offload_episode)

print('------------SAC-NOMA-RIS Test Done-------------')
for i in range(n_veh):
    print('Average User Reward:', i, ':', np.mean(record_reward_[i, :]))
print('Average Global Reward:', np.mean(record_global_reward_average),
      '   Sum Average Power:', np.mean(Sum_Power), '   Sum Average DataBuf:', np.mean(Sum_DataBuf_length))
print('Sum average local power:', np.mean(Sum_Power_local), '   Sum Average offload power:', np.mean(Sum_Power_offload))

plt.figure(1)
# fig = plt.figure(figsize=(width/100, height/100))
plt.plot(BS_x, BS_y, 'o', markersize=5, color='black', label='BS')
plt.plot(RIS_x, RIS_y, 'o', markersize=5, color='brown', label='RIS')
plt.plot(Vehicle_positions_x0, Vehicle_positions_y0)
plt.plot(Vehicle_positions_x1, Vehicle_positions_y1)
plt.plot(Vehicle_positions_x2, Vehicle_positions_y2)
plt.plot(Vehicle_positions_x3, Vehicle_positions_y3)
plt.plot(Vehicle_positions_x4, Vehicle_positions_y4)
plt.plot(Vehicle_positions_x5, Vehicle_positions_y5)
plt.plot(Vehicle_positions_x6, Vehicle_positions_y6)
plt.plot(Vehicle_positions_x7, Vehicle_positions_y7)

plt.show()
