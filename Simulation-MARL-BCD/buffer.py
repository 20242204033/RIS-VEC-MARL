import numpy as np

class ReplayBuffer():
    def __init__(self, max_size, input_shape, n_actions, n_agents):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, input_shape * n_agents), dtype=np.float32)
        self.action_memory = np.zeros((self.mem_size, n_actions * n_agents), dtype=np.float32)
        self.reward_global_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.reward_local_memory =  np.zeros((self.mem_size, n_agents), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, input_shape * n_agents), dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)
        # 新增：按行展平的可行性掩码
        self.mask_memory = np.zeros((self.mem_size, n_agents * n_agents), dtype=np.float32)

    def store_transition(self, state, action, reward_g, reward_l, state_, done, mask_flat):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_global_memory[index] = reward_g
        self.reward_local_memory[index] = reward_l
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = done
        self.mask_memory[index] = mask_flat
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)
        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards_g = self.reward_global_memory[batch]
        rewards_l = self.reward_local_memory[batch]
        states_ = self.new_state_memory[batch]
        dones = self.terminal_memory[batch]
        masks = self.mask_memory[batch]
        return states, actions, rewards_g, rewards_l, states_, dones, masks

