import os
import numpy as np
import torch as T
import torch.nn.functional as F
from networks import CriticNetwork

class Global_SAC_Critic:
    def __init__(self, beta, input_dims, tau, n_actions, gamma, c1, c2, c3,
                 batch_size, n_agents, update_actor_interval, alpha_lr=3e-4,
                 target_entropy=None, entropy_scale=1.0):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.beta = beta
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.n_states = input_dims
        self.update_actor_iter = update_actor_interval
        # coefficient applied to the entropy term to encourage exploration
        self.entropy_scale = entropy_scale
        self.learn_step_counter = 0
        self.Global_Loss = []

        # use the same names as the DDPG global critic so the network
        # interprets these as global networks and expands the input/output
        # dimensions to n_agents times the per-agent dimensions
        self.global_critic1 = CriticNetwork(beta, input_dims, c1, c2, c3, n_agents,
                                           n_actions=n_actions, name='global_critic1', agent_label='global')
        self.global_critic2 = CriticNetwork(beta, input_dims, c1, c2, c3, n_agents,
                                           n_actions=n_actions, name='global_critic2', agent_label='global')
        self.global_target_critic1 = CriticNetwork(beta, input_dims, c1, c2, c3, n_agents,
                                                  n_actions=n_actions, name='global_target_critic1', agent_label='global')
        self.global_target_critic2 = CriticNetwork(beta, input_dims, c1, c2, c3, n_agents,
                                                  n_actions=n_actions, name='global_target_critic2', agent_label='global')
        self.update_global_network_parameters(tau=1)

        if target_entropy is None:
            self.target_entropy = -float(n_actions*n_agents)
        else:
            self.target_entropy = target_entropy
        self.log_alpha = T.zeros(1, requires_grad=True, device=self.global_critic1.device)
        self.alpha_optimizer = T.optim.Adam([self.log_alpha], lr=alpha_lr)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def global_learn(self, agents_nets, state, action, reward_g, reward_l, state_, terminal):
        states = T.tensor(state, dtype=T.float).to(self.global_critic1.device)
        states_ = T.tensor(state_, dtype=T.float).to(self.global_critic1.device)
        actions = T.tensor(action, dtype=T.float).to(self.global_critic1.device)
        rewards_g = T.tensor(reward_g, dtype=T.float).to(self.global_critic1.device)
        rewards_l = T.tensor(reward_l, dtype=T.float).to(self.global_critic1.device)
        done = T.tensor(terminal).to(self.global_critic1.device)

        self.global_target_critic1.eval()
        self.global_target_critic2.eval()
        self.global_critic1.eval()
        self.global_critic2.eval()

        next_actions = T.zeros(self.batch_size, self.n_actions*self.n_agents).to(self.global_critic1.device)
        next_log_probs = T.zeros(self.batch_size, 1).to(self.global_critic1.device)
        for i in range(self.n_agents):
            a, log_p = agents_nets[i].policy.sample_normal(states_[:, i*self.n_states:(i+1)*self.n_states])
            next_actions[:, i*self.n_actions:(i+1)*self.n_actions] = a
            next_log_probs += log_p

        with T.no_grad():
            q1_next = self.global_target_critic1.forward(states_, next_actions)
            q2_next = self.global_target_critic2.forward(states_, next_actions)
            min_q_next = T.min(q1_next, q2_next) - self.alpha * self.entropy_scale * next_log_probs
        target = rewards_g + self.gamma*min_q_next.view(-1)
        target[done] = rewards_g[done]
        target = target.view(self.batch_size, 1)

        q1 = self.global_critic1.forward(states, actions)
        q2 = self.global_critic2.forward(states, actions)
        q1_loss = F.mse_loss(q1, target)
        q2_loss = F.mse_loss(q2, target)
        critic_loss = q1_loss + q2_loss
        self.global_critic1.optimizer.zero_grad()
        self.global_critic2.optimizer.zero_grad()
        critic_loss.backward()
        T.nn.utils.clip_grad_norm_(self.global_critic1.parameters(), 1.0)
        T.nn.utils.clip_grad_norm_(self.global_critic2.parameters(), 1.0)
        self.global_critic1.optimizer.step()
        self.global_critic2.optimizer.step()
        self.Global_Loss.append(critic_loss.detach().cpu().numpy())

        self.update_global_network_parameters()
        self.learn_step_counter += 1
        if self.learn_step_counter % self.update_actor_iter != 0:
            return

        actions_pi = T.zeros(self.batch_size, self.n_actions*self.n_agents).to(self.global_critic1.device)
        log_pis = T.zeros(self.batch_size, 1).to(self.global_critic1.device)
        for i in range(self.n_agents):
            a, log_p = agents_nets[i].policy.sample_normal(states[:, i*self.n_states:(i+1)*self.n_states])
            actions_pi[:, i*self.n_actions:(i+1)*self.n_actions] = a
            log_pis += log_p

        actor_global_loss = (self.alpha * self.entropy_scale * log_pis -
                             self.global_critic1.forward(states, actions_pi)).mean()
        # update temperature
        alpha_loss = -(self.log_alpha * (log_pis + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        T.nn.utils.clip_grad_norm_([self.log_alpha], 1.0)
        self.alpha_optimizer.step()

        for i in range(self.n_agents):
            global_loss_copy = actor_global_loss.detach().clone().repeat(self.batch_size, 1)
            agents_nets[i].local_learn(global_loss_copy, states[:, i*self.n_states:(i+1)*self.n_states],
                                        actions[:, i*self.n_actions:(i+1)*self.n_actions], rewards_l[:, i],
                                        states_[:, i*self.n_states:(i+1)*self.n_states], done,
                                        self.alpha.detach() * self.entropy_scale)

    def update_global_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau
        for target_param, param in zip(self.global_target_critic1.parameters(), self.global_critic1.parameters()):
            target_param.data.copy_(tau * param.data + (1-tau) * target_param.data)
        for target_param, param in zip(self.global_target_critic2.parameters(), self.global_critic2.parameters()):
            target_param.data.copy_(tau * param.data + (1-tau) * target_param.data)


    def save_models(self):
        self.global_critic1.save_checkpoint()
        self.global_critic2.save_checkpoint()
        self.global_target_critic1.save_checkpoint()
        self.global_target_critic2.save_checkpoint()

    def load_models(self):
        self.global_critic1.load_checkpoint()
        self.global_critic2.load_checkpoint()
        self.global_target_critic1.load_checkpoint()
        self.global_target_critic2.load_checkpoint()
