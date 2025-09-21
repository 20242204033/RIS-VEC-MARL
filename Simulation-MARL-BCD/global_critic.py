
import torch as T
import torch.nn.functional as F
from networks import CriticNetwork
from torch.amp import GradScaler
import math

class Global_SAC_Critic:
    def __init__(self, beta, input_dims, tau, n_actions, gamma, c1, c2, c3,
                 batch_size, n_agents, update_actor_interval, alpha_lr=3e-4,
                 target_entropy=None, entropy_scale=1.0, separate_alpha=False,
                 chkpt_root=None, use_amp=False, warmup_actor_steps=2000,
                 update_alpha_interval=4):

        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.beta = beta
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.n_states = input_dims
        self.entropy_scale = entropy_scale
        self.learn_step_counter = 0
        self.Global_Loss = []
        # === Running stats for actor/entropy（新增）===
        self.entropy_ema = None  # 避免 AttributeError
        self.entropy_ema_beta = 0.98  # 指数滑动平均的衰减因子（可调）
        self.last_actor_stats = {}  # 便于调试/可视化时缓存最近一次统计

        self.global_action_dim = n_actions * n_agents
        self.warmup_actor_steps = warmup_actor_steps
        self.update_actor_iter  = update_actor_interval
        self.update_alpha_interval = update_alpha_interval
        self.use_amp = use_amp
        self.scaler  = GradScaler(enabled=self.use_amp)


        # —— 把全局网络的权重都存到这个新目录 ——
        # —— 把全局网络的权重保存到 run_dir/global（由外部 marl_train_bcd 传入） ——
        if chkpt_root is None:
            new_dir_global = 'runs/exp_prob2N/global'  # 兼容旧逻辑
        else:
            import os
            new_dir_global = os.path.join(chkpt_root, 'global')
            os.makedirs(new_dir_global, exist_ok=True)

        self.global_critic1 = CriticNetwork(beta, input_dims, c1, c2, c3, n_agents,
                                            action_dim=self.global_action_dim, name='global_critic1',
                                            agent_label='global',
                                            chkpt_dir=new_dir_global)
        self.global_critic2 = CriticNetwork(beta, input_dims, c1, c2, c3, n_agents,
                                            action_dim=self.global_action_dim, name='global_critic2',
                                            agent_label='global',
                                            chkpt_dir=new_dir_global)
        self.global_target_critic1 = CriticNetwork(beta, input_dims, c1, c2, c3, n_agents,
                                                   action_dim=self.global_action_dim, name='global_target_critic1',
                                                   agent_label='global',
                                                   chkpt_dir=new_dir_global)
        self.global_target_critic2 = CriticNetwork(beta, input_dims, c1, c2, c3, n_agents,
                                                   action_dim=self.global_action_dim, name='global_target_critic2',
                                                   agent_label='global',
                                                   chkpt_dir=new_dir_global)

        self.update_global_network_parameters(tau=1)

        self.separate_alpha = bool(separate_alpha)

        if not self.separate_alpha:
            # 旧：单α
            self.log_alpha = T.zeros(1, requires_grad=True, device=self.global_critic1.device)
            self.alpha_optimizer = T.optim.Adam([self.log_alpha], lr=alpha_lr)
        else:
            # 新：两套α（连续/离散）
            self.log_alpha_cont = T.zeros(1, requires_grad=True, device=self.global_critic1.device)
            self.log_alpha_disc = T.zeros(1, requires_grad=True, device=self.global_critic1.device)
            self.alpha_opt_cont = T.optim.Adam([self.log_alpha_cont], lr=alpha_lr)
            self.alpha_opt_disc = T.optim.Adam([self.log_alpha_disc], lr=alpha_lr)

        if target_entropy is None:
            # 按你现有尺度来：每个agent有 n_cont=2 个连续维度；离散K=N
            n_cont = 2
            K = max(2, self.n_agents)
            target_ent_cont = -0.5 * (self.n_agents * n_cont)  # H_c = -1.0 · dim(a_c)
            target_ent_disc = -0.5 * (self.n_agents * math.log(K))  # H_d = -0.7 · log K

            if not self.separate_alpha:
                self.target_entropy = target_ent_cont + target_ent_disc
            else:
                # 分离目标熵
                self.target_entropy_cont = target_ent_cont
                self.target_entropy_disc = target_ent_disc
        else:
            if not self.separate_alpha:
                self.target_entropy = target_entropy
            else:
                # 如果外部传了标量，你也可以按比例切；这里简单按原默认切法
                self.target_entropy_cont = target_entropy * 0.5
                self.target_entropy_disc = target_entropy * 0.5

    @property
    def alpha(self):
        # 兼容旧调用：需要一个标量时，返回“合并后”的等效α（用于老路径）
        if not self.separate_alpha:
            return self.log_alpha.exp()
        else:
            # 用作回退或日志显示：简单相加（也可取加权平均，这里与local的用法一致）
            return self.log_alpha_cont.exp() + self.log_alpha_disc.exp()

    @property
    def alpha_cont(self):
        return self.log_alpha_cont.exp() if self.separate_alpha else self.log_alpha.exp()

    @property
    def alpha_disc(self):
        return self.log_alpha_disc.exp() if self.separate_alpha else self.log_alpha.exp()

    def centralized_actor_update(self, agents, states_batch, learn_step_counter, masks_batch=None):
        """
        【新增】集中式 Actor 更新（中心化一次反传）。
        """
        from torch.amp import autocast

        if learn_step_counter < self.warmup_actor_steps:
            return None

        device = self.global_critic1.device
        obs_dim = self.n_states

        obs_list = [states_batch[:, i*obs_dim:(i+1)*obs_dim].to(device)
                    for i in range(self.n_agents)]

        joint_actions = []
        device_type = self.global_critic1.device.type
        with autocast(device_type=device_type, enabled=self.use_amp):
            logp_power_list = []
            logp_int_list = []
            # === [NEW] Actor 联合采样：支持 mask，与 Critic 目标一致 ===
            for i, ag in enumerate(agents):
                obs_i = obs_list[i]

                mask_i = None
                if masks_batch is not None:
                    if masks_batch.dim() == 3:
                        mask_i = masks_batch[:, i, :]  # [B, n_agents]
                    elif masks_batch.dim() == 2:
                        mask_i = masks_batch  # [B, n_agents]
                    else:
                        raise RuntimeError("masks_batch 的形状不合法，应为 [B,N] 或 [B,N,N]")

                out = ag.policy.sample_normal(obs_i, mask=mask_i)
                if isinstance(out, (tuple, list)) and len(out) == 5:
                    a_pwr, a_probs, logp_power, logp_int, logp_total = out
                elif isinstance(out, (tuple, list)) and len(out) == 3:
                    a_pwr, a_probs, logp_total = out
                    if self.separate_alpha:
                        raise RuntimeError(
                            "policy.sample_normal 只返回3个值；当 separate_alpha=True 需要同时返回 logp_power 与 logp_int"
                        )
                    logp_power = logp_total
                    logp_int = T.zeros_like(logp_total)  # 占位
                else:
                    raise RuntimeError("policy.sample_normal 返回值数量不符合预期(应为3或5)")

                joint_actions.append(T.cat([a_probs, a_pwr], dim=-1))
                logp_power_list.append(logp_power)  # [B,1]
                logp_int_list.append(logp_int)  # [B,1]

            actions_pi = T.cat(joint_actions, dim=-1).to(device)
            logp_power_mat = T.cat(logp_power_list, dim=-1)  # [B, N]
            logp_int_mat = T.cat(logp_int_list, dim=-1)  # [B, N]
            sum_logp_power = logp_power_mat.sum(dim=-1, keepdim=True)
            sum_logp_int = logp_int_mat.sum(dim=-1, keepdim=True)

            q1_pi = self.global_critic1.forward(states_batch.to(device), actions_pi)
            q2_pi = self.global_critic2.forward(states_batch.to(device), actions_pi)
            q_min = T.min(q1_pi, q2_pi)

            # === 修正：使用连续+离散两部分 logπ 的和 ===
            sum_logp_total = sum_logp_power + sum_logp_int
            batch_entropy = -sum_logp_total.detach()
            cur_ent = float(batch_entropy.mean().item())

            if self.entropy_ema is None:
                self.entropy_ema = cur_ent
            else:
                beta = float(self.entropy_ema_beta)
                self.entropy_ema = beta * self.entropy_ema + (1.0 - beta) * cur_ent
            if (learn_step_counter % self.update_alpha_interval) == 0:
                if not self.separate_alpha:
                    # 旧：单α路径（沿用 sum of total logp：= sum_logp_power + sum_logp_int）
                    sum_logp_total = sum_logp_power + sum_logp_int
                    alpha_loss = -(self.log_alpha * (-sum_logp_total.detach() - self.target_entropy)).mean()
                    self.alpha_optimizer.zero_grad(set_to_none=True)
                    if self.use_amp:
                        self.scaler.scale(alpha_loss).backward()
                        self.scaler.step(self.alpha_optimizer)
                        self.scaler.update()
                    else:
                        alpha_loss.backward()
                        self.alpha_optimizer.step()
                else:
                    # 新：分离α路径——各自目标熵与各自logp
                    alpha_loss_cont = -(
                                self.log_alpha_cont * (-sum_logp_power.detach() - self.target_entropy_cont)).mean()
                    alpha_loss_disc = -(
                                self.log_alpha_disc * (-sum_logp_int.detach() - self.target_entropy_disc)).mean()

                    self.alpha_opt_cont.zero_grad(set_to_none=True)
                    self.alpha_opt_disc.zero_grad(set_to_none=True)
                    if self.use_amp:
                        self.scaler.scale(alpha_loss_cont + alpha_loss_disc).backward()
                        self.scaler.step(self.alpha_opt_cont)
                        self.scaler.step(self.alpha_opt_disc)
                        self.scaler.update()
                    else:
                        (alpha_loss_cont + alpha_loss_disc).backward()
                        self.alpha_opt_cont.step()
                        self.alpha_opt_disc.step()

            # 构造 actor 损失
            if not self.separate_alpha:
                sum_logp_total = sum_logp_power + sum_logp_int
                alpha_pi = (self.log_alpha.exp() * self.entropy_scale).detach()
                actor_global_loss = (alpha_pi * sum_logp_total - q_min).mean()
            else:
                alpha_c = (self.log_alpha_cont.exp() * self.entropy_scale).detach()
                alpha_d = (self.log_alpha_disc.exp() * self.entropy_scale).detach()
                # 分别加权两部分熵
                actor_global_loss = (alpha_c * sum_logp_power + alpha_d * sum_logp_int - q_min).mean()

        uniq = {id(ag.policy): ag.policy for ag in agents}
        uniq_pols = list(uniq.values())

        for pol in uniq_pols:
            pol.optimizer.zero_grad(set_to_none=True)

        if self.use_amp:
            self.scaler.scale(actor_global_loss).backward()
            for pol in uniq_pols:
                self.scaler.unscale_(pol.optimizer)
                T.nn.utils.clip_grad_norm_(pol.parameters(), 1.0)
            for pol in uniq_pols:
                self.scaler.step(pol.optimizer)
            self.scaler.update()
        else:
            actor_global_loss.backward()
            for pol in uniq_pols:
                T.nn.utils.clip_grad_norm_(pol.parameters(), 1.0)
                pol.optimizer.step()

        ret = {
            "actor_global_loss": float(actor_global_loss.detach().cpu().item()),
            "entropy_ema": float(self.entropy_ema),
            "q_min_mean": float(q_min.detach().mean().cpu().item()),
        }
        if not self.separate_alpha:
            ret["alpha"] = float(self.alpha.detach().cpu().item())
        else:
            ret["alpha_cont"] = float(self.alpha_cont.detach().cpu().item())
            ret["alpha_disc"] = float(self.alpha_disc.detach().cpu().item())
        return ret

    # 新签名：增加 masks 形参；并把它规范为 [B, N, N] 或 None
    def global_learn(self, agents_nets, state, action, reward_g, reward_l, state_, terminal, masks=None):
        states = T.tensor(state, dtype=T.float).to(self.global_critic1.device)
        states_ = T.tensor(state_, dtype=T.float).to(self.global_critic1.device)
        actions = T.tensor(action, dtype=T.float).to(self.global_critic1.device)
        rewards_g = T.tensor(reward_g, dtype=T.float).to(self.global_critic1.device)
        rewards_l = T.tensor(reward_l, dtype=T.float).to(self.global_critic1.device)
        done = T.tensor(terminal, dtype=T.bool, device=self.global_critic1.device)


        # 统一 batch 掩码的形状
        masks_batch = None
        if masks is not None:
            m = T.tensor(masks, dtype=T.float32, device=self.global_critic1.device)
            if m.dim() == 2 and m.size(1) == self.n_agents * self.n_agents:
                masks_batch = m.view(-1, self.n_agents, self.n_agents)  # [B,N,N]
            elif m.dim() == 3 and m.size(1) == self.n_agents and m.size(2) == self.n_agents:
                masks_batch = m  # [B,N,N]
            elif m.dim() == 2 and m.size(1) == self.n_agents:
                # 也兼容传进来就是 [B,N] 的“行掩码”，后面会按行取用
                masks_batch = m  # [B,N]
            else:
                raise RuntimeError("masks 的形状应为 [B,N,N] 或 [B,N] 或 [B,N*N]")

        # ---- 1) 更新全局 Critic ----
        self.global_target_critic1.eval()
        self.global_target_critic2.eval()

        B = states_.size(0)  # 实际 batch 大小
        next_actions = T.zeros(B, self.n_actions * self.n_agents, device=self.global_critic1.device)
        next_logp_power_sum = T.zeros(B, 1, device=self.global_critic1.device)
        next_logp_int_sum = T.zeros(B, 1, device=self.global_critic1.device)
        # === [NEW] 目标Q动作采样：支持 mask，与 Actor 一致 ===
        for j, agent in enumerate(agents_nets):
            obs_j = states_[:, j * self.n_states:(j + 1) * self.n_states]

            # 从 batch 级 mask 里取出第 j 个 agent 的可选配对（形状 [B, n_agents] 或 [n_agents]）
            mask_j = None
            if 'masks_batch' in locals() and masks_batch is not None:
                # 允许两种形状：
                #   1) [B, n_agents, n_agents]：取第 j 行 => [B, n_agents]
                #   2) [B, n_agents]：直接使用
                if masks_batch.dim() == 3:
                    mask_j = masks_batch[:, j, :]  # [B, n_agents]
                elif masks_batch.dim() == 2:
                    mask_j = masks_batch  # [B, n_agents]
                else:
                    raise RuntimeError("masks_batch 的形状不合法，应为 [B,N] 或 [B,N,N]")

            out = agent.policy.sample_normal(obs_j, mask=mask_j)
            if isinstance(out, (tuple, list)) and len(out) == 5:
                power_a, intent_probs, logp_power, logp_int, logp_total = out
            elif isinstance(out, (tuple, list)) and len(out) == 3:
                power_a, intent_probs, logp_total = out
                if self.separate_alpha:
                    raise RuntimeError(
                        "policy.sample_normal 只返回3个值；当 separate_alpha=True 需要 logp_power 与 logp_int"
                    )
                logp_power = logp_total
                logp_int = T.zeros_like(logp_total)
            else:
                raise RuntimeError("policy.sample_normal 返回值数量不符合预期(应为3或5)")

            start = j * self.n_actions
            # TD 目标：把离散意图概率压成 one-hot(argmax)
            idx_j = T.argmax(intent_probs, dim=-1)  # [B]
            onehot_j = F.one_hot(idx_j, num_classes=self.n_agents).float()  # [B, N]

            # 写回全局 next_actions 的该 agent 段
            next_actions[:, start:start + self.n_agents] = onehot_j
            next_actions[:, start + self.n_agents:start + self.n_actions] = power_a

            next_logp_power_sum = next_logp_power_sum + logp_power
            next_logp_int_sum = next_logp_int_sum + logp_int

        with T.no_grad():
            q1_next = self.global_target_critic1.forward(states_, next_actions)
            q2_next = self.global_target_critic2.forward(states_, next_actions)
            if not self.separate_alpha:
                # 单α：复用原来总logp = 两者相加
                next_logp_total = next_logp_power_sum + next_logp_int_sum
                min_q_next = T.min(q1_next, q2_next) - self.alpha * self.entropy_scale * next_logp_total
            else:
                # 分离α：分别加权
                alpha_c = self.alpha_cont * self.entropy_scale
                alpha_d = self.alpha_disc * self.entropy_scale
                min_q_next = T.min(q1_next, q2_next) - (alpha_c * next_logp_power_sum + alpha_d * next_logp_int_sum)

        target = rewards_g + self.gamma * min_q_next.view(-1)
        target[done] = rewards_g[done]
        target = target.view(B, 1)

        self.global_critic1.train()
        self.global_critic2.train()
        q1 = self.global_critic1.forward(states, actions)
        q2 = self.global_critic2.forward(states, actions)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)

        self.global_critic1.optimizer.zero_grad(set_to_none=True)
        self.global_critic2.optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        T.nn.utils.clip_grad_norm_(self.global_critic1.parameters(), 1.0)
        T.nn.utils.clip_grad_norm_(self.global_critic2.parameters(), 1.0)
        self.global_critic1.optimizer.step()
        self.global_critic2.optimizer.step()
        self.Global_Loss.append(critic_loss.detach().cpu().numpy())

        self.update_global_network_parameters()
        self.learn_step_counter += 1

        # ---- 2) 间歇触发：中心化 Actor 更新 + 本地 Critic 更新 ----
        if self.learn_step_counter % self.update_actor_iter == 0:
            self.last_actor_stats = self.centralized_actor_update(
                agents_nets, states, self.learn_step_counter, masks_batch
            )
            for i, agent in enumerate(agents_nets):
                action_slice = actions[:, i*self.n_actions:(i+1)*self.n_actions]
                state_slice  = states[:,  i*self.n_states:(i+1)*self.n_states]
                next_slice   = states_[:, i*self.n_states:(i+1)*self.n_states]
                reward_slice = rewards_l[:, i]
                if not self.separate_alpha:
                    agent.local_learn(state_slice, action_slice, reward_slice, next_slice,
                                      done,
                                      self.alpha.detach() * self.entropy_scale,  # power
                                      self.alpha.detach() * self.entropy_scale)  # intent
                else:
                    agent.local_learn(state_slice, action_slice, reward_slice, next_slice,
                                      done,
                                      self.alpha_cont.detach() * self.entropy_scale,
                                      self.alpha_disc.detach() * self.entropy_scale)

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
        # === save alpha parameters (stage-3) ===
        try:
            alpha_dict = {}
            if not self.separate_alpha:
                alpha_dict["log_alpha"] = self.log_alpha.detach().cpu()
            else:
                alpha_dict["log_alpha_cont"] = self.log_alpha_cont.detach().cpu()
                alpha_dict["log_alpha_disc"] = self.log_alpha_disc.detach().cpu()
            import os, torch as T
            alpha_path = os.path.join(self.global_critic1.checkpoint_dir, "alpha.pt")
            T.save(alpha_dict, alpha_path)
        except Exception as _e:
            print("[WARN] save alpha failed:", _e)


    def load_models(self):
        self.global_critic1.load_checkpoint()
        self.global_critic2.load_checkpoint()
        self.global_target_critic1.load_checkpoint()
        self.global_target_critic2.load_checkpoint()
        # === load alpha parameters (stage-3) ===
        try:
            import os, torch as T
            alpha_path = os.path.join(self.global_critic1.checkpoint_dir, "alpha.pt")
            if os.path.exists(alpha_path):
                alpha_dict = T.load(alpha_path, map_location=self.global_critic1.device)
                if not self.separate_alpha and "log_alpha" in alpha_dict:
                    with T.no_grad():
                        self.log_alpha.copy_(alpha_dict["log_alpha"].to(self.global_critic1.device))
                else:
                    if "log_alpha_cont" in alpha_dict:
                        with T.no_grad():
                            self.log_alpha_cont.copy_(alpha_dict["log_alpha_cont"].to(self.global_critic1.device))
                    if "log_alpha_disc" in alpha_dict:
                        with T.no_grad():
                            self.log_alpha_disc.copy_(alpha_dict["log_alpha_disc"].to(self.global_critic1.device))
        except Exception as _e:
            print("[WARN] load alpha failed:", _e)
