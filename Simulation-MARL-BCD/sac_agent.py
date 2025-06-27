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
        self.name = name + '_' + str(agent_label)
        self.checkpoint_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), chkpt_dir)
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.name + '_policy')

        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        self.log_std = nn.Linear(self.fc2_dims, self.n_actions)

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
        mu = self.mu(x)
        log_std = self.log_std(x)
        log_std = T.clamp(log_std, min=-20, max=2)
        return mu, log_std

    def sample_normal(self, state):
        mu, log_std = self.forward(state)
        std = log_std.exp()
        normal = T.distributions.Normal(mu, std)
        x_t = normal.rsample()
        y_t = T.tanh(x_t)
        action = y_t
        log_prob = normal.log_prob(x_t)
        log_prob -= T.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        action = action.clamp(-0.999, 0.999)
        return action, log_prob

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file, map_location='cpu'))

class Agent:
    def __init__(self, alpha, beta, input_dims, tau, n_actions, gamma, c1, c2, c3,
                 a1, a2, batch_size, n_agents, agent_name, noise):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.number_agents = n_agents
        self.number_actions = n_actions
        self.number_states = input_dims
        self.agent_name = agent_name
        self.noise = noise
        self.alpha = alpha
        self.beta = beta
        self.local_critic_loss = []

        self.policy = PolicyNetwork(alpha, input_dims, a1, a2, n_actions, n_agents,
                                    name='policy', agent_label=agent_name)
        self.q1 = CriticNetwork(beta, input_dims, c1, c2, c3, n_agents,
                                n_actions=n_actions, name='q1', agent_label=agent_name)
        self.q2 = CriticNetwork(beta, input_dims, c1, c2, c3, n_agents,
                                n_actions=n_actions, name='q2', agent_label=agent_name)
        self.target_q1 = CriticNetwork(beta, input_dims, c1, c2, c3, n_agents,
                                       n_actions=n_actions, name='target_q1', agent_label=agent_name)
        self.target_q2 = CriticNetwork(beta, input_dims, c1, c2, c3, n_agents,
                                       n_actions=n_actions, name='target_q2', agent_label=agent_name)
        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        state = T.tensor([observation], dtype=T.float).to(self.policy.device)
        actions, _ = self.policy.sample_normal(state)
        return actions.cpu().detach().numpy()[0]

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

    def local_learn(self, global_loss, state, action, reward_l, state_, terminal, entropy_coefficient):
        states = state
        states_ = state_
        actions = action
        rewards = reward_l
        done = terminal
        self.global_loss = global_loss

        with T.no_grad():
            next_actions, next_log_probs = self.policy.sample_normal(states_)
            q1_next = self.target_q1.forward(states_, next_actions)
            q2_next = self.target_q2.forward(states_, next_actions)
            min_q_next = T.min(q1_next, q2_next) - entropy_coefficient * next_log_probs
            target = rewards + self.gamma * min_q_next.view(-1)
            target[done] = rewards[done]
            target = target.view(self.batch_size, 1)

        q1 = self.q1.forward(states, actions)
        q2 = self.q2.forward(states, actions)
        q1_loss = F.mse_loss(q1, target)
        q2_loss = F.mse_loss(q2, target)
        critic_loss = q1_loss + q2_loss
        self.q1.optimizer.zero_grad()
        self.q2.optimizer.zero_grad()
        critic_loss.backward()
        T.nn.utils.clip_grad_norm_(self.q1.parameters(), 1.0)
        T.nn.utils.clip_grad_norm_(self.q2.parameters(), 1.0)
        self.q1.optimizer.step()
        self.q2.optimizer.step()
        self.local_critic_loss.append(critic_loss.detach().cpu().numpy())

        new_actions, log_probs = self.policy.sample_normal(states)
        q1_new = self.q1.forward(states, new_actions)
        q2_new = self.q2.forward(states, new_actions)
        critic_value = T.min(q1_new, q2_new)
        actor_loss = (entropy_coefficient * log_probs - critic_value).mean() + T.mean(self.global_loss)
        self.policy.optimizer.zero_grad()
        actor_loss.backward()
        T.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.policy.optimizer.step()

        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        for target_param, param in zip(self.target_q1.parameters(), self.q1.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        for target_param, param in zip(self.target_q2.parameters(), self.q2.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
