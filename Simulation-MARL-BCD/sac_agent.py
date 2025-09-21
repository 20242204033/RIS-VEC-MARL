import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from networks import CriticNetwork


class PolicyNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, n_actions, n_agents, name, agent_label,
                 chkpt_dir='model2/ris_sac_model'):
        super(PolicyNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.n_agents = n_agents  # Store n_agents
        self.name = name + '_' + str(agent_label)
        self.checkpoint_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), chkpt_dir)
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.name + '_policy')
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)

        # --- Head for continuous actions (power) ---
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        self.log_std = nn.Linear(self.fc2_dims, self.n_actions)

        # --- Head for discrete actions (intention) ---
        self.intent_logits = nn.Linear(self.fc2_dims, self.n_agents)

        self.bn1 = nn.LayerNorm(self.fc1_dims)
        self.bn2 = nn.LayerNorm(self.fc2_dims)

        f1 = 1. / np.sqrt(self.fc1.weight.data.size()[0])
        self.fc1.weight.data.uniform_(-f1, f1)
        self.fc1.bias.data.uniform_(-f1, f1)
        f2 = 1. / np.sqrt(self.fc2.weight.data.size()[0])
        self.fc2.weight.data.uniform_(-f2, f2)
        self.fc2.bias.data.uniform_(-f2, f2)
        f3 = 0.003
        self.mu.weight.data.uniform_(-f3, f3)
        self.mu.bias.data.uniform_(-f3, f3)
        self.log_std.weight.data.uniform_(-f3, f3)
        self.log_std.bias.data.uniform_(-f3, f3)
        # --- Weight initialization for the new layer ---
        self.intent_logits.weight.data.uniform_(-f3, f3)
        self.intent_logits.bias.data.uniform_(-f3, f3)

        # ---- Stage-1: Gumbel-Softmax 控制参数 ----
        # 温度（会在训练循环里退火）
        self.register_buffer("tau", T.tensor(2.0))
        # 是否用 straight-through（one-hot + 直通估计）
        self.gumbel_hard = False

        self.optimizer = T.optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)


    def forward(self, state):
        x = self.fc1(state)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)

        # --- Continuous action outputs ---
        mu = self.mu(x)
        log_std = self.log_std(x)
        log_std = T.clamp(log_std, min=-20, max=2)

        # --- Discrete action outputs (return logits; sampling done in sample_normal) ---
        intent_logits = self.intent_logits(x)

        return mu, log_std, intent_logits

    def sample_normal(self, state, reparameterize=True, mask=None):
        mu, log_std, intent_logits = self.forward(state)
        # --- 连续动作部分（原样保留） ---
        std = log_std.exp()
        normal_dist = T.distributions.Normal(mu, std)
        x_t = normal_dist.rsample() if reparameterize else normal_dist.sample()
        power_action = T.tanh(x_t)
        log_prob_power = normal_dist.log_prob(x_t) - T.log(1 - power_action.pow(2) + 1e-6)
        log_prob_power = log_prob_power.sum(1, keepdim=True)

        # --- 阶段4：在 logits 上应用 mask（1=可选，0=屏蔽） ---
        masked_logits = intent_logits

        if mask is not None:
            # 与 logits 同 device/shape；若给的是[1,N]或[N]均可
            m = mask.to(intent_logits.device)
            if m.dim() == 1:
                m = m.unsqueeze(0).expand_as(intent_logits)
            # 全零行保护：若某行全 0，放开该行，避免无可行动作
            rows_all_zero = (m.sum(dim=-1) == 0)
            if rows_all_zero.any():
                m = m.clone()
                m[rows_all_zero] = 1.0

            # ✅ 关键修复：按当前 dtype 选择不会溢出的负大数（fp16 下 -1e9 会溢出）
            neg_large = T.finfo(intent_logits.dtype).min / 2  # fp16≈-3.2e4；fp32≈-1.7e38
            masked_logits = intent_logits.masked_fill(m <= 0, neg_large)

        # --- Discrete Action (Intention) via Gumbel-Softmax（对 masked_logits 采样） ---
        # --- Discrete Action (Intention) via Gumbel-Softmax（对 masked_logits 采样） ---
        y = F.gumbel_softmax(
            masked_logits.float(), tau=float(self.tau.item()),
            hard=bool(self.gumbel_hard), dim=-1
        )

        # ✅ 正确的离散 logπ 计算：对“掩码后的 logits”取 log_softmax
        log_probs = F.log_softmax(masked_logits.float(), dim=-1)  # [B, N_agents]

        if self.gumbel_hard:
            # ST one-hot：按采样到的类别取对应 log 概率
            idx = y.argmax(dim=1, keepdim=True)  # [B, 1]
            log_prob_intent = log_probs.gather(1, idx)  # [B, 1]
        else:
            # 软样本：期望 log 概率，低方差近似 E[logπ]
            log_prob_intent = (y * log_probs).sum(dim=1, keepdim=True)  # [B, 1]

        total_log_prob = log_prob_power + log_prob_intent
        return power_action, y, log_prob_power, log_prob_intent, total_log_prob

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file, map_location='cpu'))


class Agent:
    def __init__(self, alpha, beta, input_dims, tau,
                 n_continuous_actions, n_discrete_actions, gamma, c1, c2, c3,
                 a1, a2, batch_size, n_agents, agent_name, noise,
                 chkpt_root=None):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.number_agents = n_agents
        self.n_continuous_actions = n_continuous_actions
        self.n_discrete_actions = n_discrete_actions
        self.n_total_actions = n_continuous_actions + n_discrete_actions
        self.number_states = input_dims
        self.agent_name = agent_name
        self.noise = noise
        self.alpha = alpha
        self.beta = beta
        self.local_critic_loss = []

        # —— 把本地网络的权重都存到这个新目录 ——
        # —— 把本地网络的权重保存到 run_dir/agent（由外部 marl_train_bcd 传入） ——
        if chkpt_root is None:
            new_dir_agent = 'runs/exp_prob2N/agent'  # 兼容旧逻辑
        else:
            import os
            new_dir_agent = os.path.join(chkpt_root, 'agent')
            os.makedirs(new_dir_agent, exist_ok=True)

        self.policy = PolicyNetwork(alpha, input_dims, a1, a2, n_continuous_actions, n_agents,
                                    name='policy', agent_label=agent_name,
                                    chkpt_dir=new_dir_agent)

        # 本地 critic 需要看到“功率(2)+意图概率(N)”
        critic_action_dim = self.n_continuous_actions + self.number_agents

        self.q1 = CriticNetwork(beta, input_dims, c1, c2, c3, n_agents,
                                action_dim=critic_action_dim, name='q1', agent_label=agent_name,
                                chkpt_dir=new_dir_agent)
        self.q2 = CriticNetwork(beta, input_dims, c1, c2, c3, n_agents,
                                action_dim=critic_action_dim, name='q2', agent_label=agent_name,
                                chkpt_dir=new_dir_agent)
        self.target_q1 = CriticNetwork(beta, input_dims, c1, c2, c3, n_agents,
                                       action_dim=critic_action_dim, name='target_q1', agent_label=agent_name,
                                       chkpt_dir=new_dir_agent)
        self.target_q2 = CriticNetwork(beta, input_dims, c1, c2, c3, n_agents,
                                       action_dim=critic_action_dim, name='target_q2', agent_label=agent_name,
                                       chkpt_dir=new_dir_agent)

        self.update_network_parameters(tau=1)

    # --- sac_agent.py ---
    def choose_action(self, observation, mask=None):
        """
        推理/交互阶段：返回 连续动作(功率)、离散意图概率、意图one-hot。
        支持 Stage-4 的可行性掩码：mask 在策略内部对 logits 生效。
        """
        self.policy.eval()
        import torch as T
        import torch.nn.functional as F
        import numpy as np

        with T.no_grad():
            # 1) 规范化 observation -> Tensor[1, obs_dim]
            if isinstance(observation, np.ndarray):
                state = T.tensor(observation, dtype=T.float32, device=self.policy.device).unsqueeze(0)
            elif isinstance(observation, (list, tuple)):
                state = T.tensor(observation, dtype=T.float32, device=self.policy.device).unsqueeze(0)
            elif isinstance(observation, T.Tensor):
                state = observation.to(self.policy.device).unsqueeze(0) if observation.dim() == 1 \
                    else observation.to(self.policy.device)
            else:
                raise TypeError(f"Unsupported observation type: {type(observation)}")

            # 2) 一次性采样（内置掩码），得到功率 + 意图概率 + 各种 logp（后两者此处主要用于日志/训练）
            out = self.policy.sample_normal(state, reparameterize=False, mask=mask)
            if not isinstance(out, (tuple, list)) or len(out) < 2:
                raise RuntimeError("policy.sample_normal 返回类型异常")
            power_action, intent_probs, _, _, _ = out

            # 3) one-hot（用于下游）
            idx = T.argmax(intent_probs, dim=-1)  # [1]
            intent_onehot = F.one_hot(idx, num_classes=self.number_agents).float()  # [1, N]

            # 4) to numpy
            power_action = power_action.squeeze(0).detach().cpu().numpy()
            intent_probs = intent_probs.squeeze(0).detach().cpu().numpy()
            intent_onehot = intent_onehot.squeeze(0).detach().cpu().numpy()

        self.policy.train()
        return power_action, intent_probs, intent_onehot

    def save_models(self):
        self.policy.save_checkpoint()
        self.q1.save_checkpoint()
        self.q2.save_checkpoint()
        self.target_q1.save_checkpoint()
        self.target_q2.save_checkpoint()

    def load_models(self):
        self.policy.load_checkpoint()
        self.q1.load_checkpoint()
        self.q2.load_checkpoint()
        self.target_q1.load_checkpoint()
        self.target_q2.load_checkpoint()

    def local_learn(self, state, action, reward_l, state_, terminal, entropy_coeff_power, entropy_coeff_intent):

        """
        【已重构】此函数现在只负责更新本地的 Critic（q1/q2/target_q*）。
        Actor 的更新改在 Global_SAC_Critic 中集中进行。
        参数形状：
          state, state_ : [B, obs_dim]
          # action : [B, n_actions] # 连续2 + 概率N
          reward_l      : [B] 或 [B,1]
          terminal      : [B] (bool)
        """
        import torch as T
        import torch.nn.functional as F

        device = self.q1.device
        states  = state.to(device)
        states_ = state_.to(device)
        actions = action.to(device)
        rewards = reward_l if reward_l.dim() > 1 else reward_l.unsqueeze(-1)
        rewards = rewards.to(device)
        done    = terminal.to(device)

        with T.no_grad():
            # 现在拿到拆分后的三项
            next_power_actions, next_intent_probs, next_logp_power, next_logp_intent, next_total_log_probs = \
                self.policy.sample_normal(states_)

            # 用 logits 重算下一步离散类别，并屏蔽 self，保证与策略约束一致
            mu_fw, logstd_fw, intent_logits_fw = self.policy.forward(states_)
            masked_logits_fw = intent_logits_fw.clone()
            masked_logits_fw[:, self.agent_name] = T.finfo(masked_logits_fw.dtype).min / 2  # 负大数屏蔽自配
            idx = T.argmax(masked_logits_fw, dim=-1)  # [B]
            next_intent_onehot = F.one_hot(idx, num_classes=self.number_agents).float()  # [B, N]
            next_actions_concatenated = T.cat([next_intent_onehot, next_power_actions], dim=1)

            q1_next = self.target_q1.forward(states_, next_actions_concatenated)
            q2_next = self.target_q2.forward(states_, next_actions_concatenated)

            # 分离温度： α_c * logp_power + α_d * logp_intent
            # 若你仍想走“单α”，上层会把两个系数设成同一个值。
            entropy_term = entropy_coeff_power * next_logp_power + entropy_coeff_intent * next_logp_intent
            min_q_next = T.min(q1_next, q2_next) - entropy_term

            target = rewards + (1 - done.float()).unsqueeze(-1) * self.gamma * min_q_next

        # critic 前向
        actions_for_critic = actions.view(actions.size(0), self.n_total_actions)
        q1 = self.q1.forward(states, actions_for_critic)
        q2 = self.q2.forward(states, actions_for_critic)

        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)

        self.q1.optimizer.zero_grad(set_to_none=True)
        self.q2.optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        T.nn.utils.clip_grad_norm_(self.q1.parameters(), 1.0)
        T.nn.utils.clip_grad_norm_(self.q2.parameters(), 1.0)
        self.q1.optimizer.step()
        self.q2.optimizer.step()

        self.local_critic_loss.append(critic_loss.detach().cpu().numpy())
        # 软更新 target
        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        for target_param, param in zip(self.target_q1.parameters(), self.q1.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        for target_param, param in zip(self.target_q2.parameters(), self.q2.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
