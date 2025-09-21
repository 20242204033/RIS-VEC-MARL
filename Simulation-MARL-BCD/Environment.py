import numpy as np
import time
import random
import math
import cmath

#np.random.seed(1234)
def _log_normal_shadow(std_db: float):
    """ 以 dB 标准差生成对数正态阴影衰落的线性倍率 """
    x_db = np.random.normal(loc=0.0, scale=std_db)
    return 10 ** (x_db / 10.0)

def _small_scale_power(rician_K_dB: float):
    """ 小尺度衰落功率；K=0 => 瑞利 |h|^2~Exp(1)，否则 Rice(K) """
    if rician_K_dB <= 1e-6:
        # Rayleigh: h~CN(0,1) => |h|^2 ~ Exp(1)，期望为 1
        return np.random.exponential(scale=1.0)
    else:
        K = 10 ** (rician_K_dB / 10.0)
        # Rice: h = sqrt(K/(K+1)) + (CN(0,1)/sqrt(K+1))，功率期望为 1
        s = np.sqrt(K / (K + 1.0))
        sigma = 1.0 / np.sqrt(2.0 * (K + 1.0))
        h_real = np.random.normal(loc=s, scale=sigma)
        h_imag = np.random.normal(loc=0.0, scale=sigma)
        return h_real * h_real + h_imag * h_imag


# RIS coordination
RIS_x, RIS_y, RIS_z = 220, 220, 25

# BS coordination
BS_x, BS_y, BS_z = 0, 0, 25

ro = 10 ** -2  # 参考距离d0 = 1m处的平均路径损耗功率增益 10dBm = 0.01w

cascaded_gain = 0
lamb = 1  # 载波长度
d = 0.5  # RIS元素之间的距离

sigma = 10 ** (-7)
alpha1 = 2.2  # 公式中的alpha
alpha2 = 2.5


class Vehicle:
    """Vehicle simulator: include all the information for a Vehicle"""

    def __init__(self, start_position, start_direction, velocity):
        self.position = start_position
        self.direction = start_direction
        self.velocity = velocity
        self.neighbors = []
        self.destinations = []


class Environ:
    def __init__(self, down_lane, up_lane, left_lane, right_lane, width, height, n_veh, M, control_bit):

        self.down_lanes = down_lane
        self.up_lanes = up_lane
        self.left_lanes = left_lane
        self.right_lanes = right_lane
        self.width = width
        self.height = height



        self.n_veh = n_veh
        # ===== 物理层配置=====
        self.channel_model = getattr(self, "channel_model", "free")  # {"free","3gpp_umi","3gpp_uma"}
        self.fc_GHz = getattr(self, "fc_GHz", 3.5)  # 载频，默认3.5GHz
        self.bandwidth = 1.0  # 仍以 "MHz" 记，保持后续公式兼容
        self.bandwidth_hz = self.bandwidth * 1e6  # Hz
        self.N0_dBm_per_Hz = -174  # 噪声谱密度(dBm/Hz)
        self.N0_W_per_Hz = 10 ** ((self.N0_dBm_per_Hz - 30) / 10)  # W/Hz
        self.noise_power = self.N0_W_per_Hz * self.bandwidth_hz  # W = N0 * B
        self.P_max = getattr(self, "P_max", 1.0)  # 每用户两路功率和的上限（W），可调
        # ---- QoS (NEW) ----
        self.qos_enable = True
        self.R_min_bpsHz = 0.20
        self.D_max_s = 0.10
        self.qos_penalty = 5.0

        self.vehicle_rate = np.zeros(n_veh)

        self.Decorrelation_distance = 10
        self.V2I_Shadowing = np.zeros(n_veh)
        self.V2I_pathloss = np.zeros(n_veh)
        self.V2I_channels_abs = np.zeros(n_veh)
        self.delta_distance = []
        self.sig2_dB = -110
        self.sig2 = 10 ** (self.sig2_dB / 10)

        self.bsAntGain = 8  # 基站天线增益
        self.bsNoiseFigure = 5  # 基站接收器噪声系数
        self.vehAntGain = 3  # 车辆天线增益
        self.vehNoiseFigure = 9  # 车辆接收器噪声增益

        self.vehicles = []

        self.time_slow = 0.1
        self.time_fast = 0.001
        # self.t = 0.02
        self.k = 1e-28
        self.L = 500
        # ===== [MEC & Computing – NEW] =====
        # CPU 上限
        self.f_local_max = 1.0e9  # 1 GHz，本地 CPU 最大频率（可由 YAML 覆盖）
        self.f_edge_max = 2.0e9  # 10 GHz，MEC 服务器 CPU 频率（单服务器）
        # 计算强度：每 bit 需要的 CPU cycles（和你原来的 L 保持同量纲）
        self.cycles_per_bit = float(self.L)  # 默认沿用 L=500，你也可以在 YAML 覆盖
        # —— 本地 CPU 份额最小下限（避免 f_local ≈ 0 导致排队/计算时延爆炸）——
        self.cpu_share_floor = 0.10  # 可按需调成 0.05~0.15；也可被外部脚本覆盖

        # MEC 单队列：以 “cycles” 计量，FCFS 近似
        self.mec_queue_cycles = 0.0
        # —— 在线归一化所需的运行统计 ——
        self.delay_mean = 0.0
        self.delay_var = 1.0
        self.energy_mean = 0.0
        self.energy_var = 1.0
        self.reward_norm_beta = getattr(self, "reward_norm_beta", 0.99)  # EMA平滑

        # ---- Day 3–5: 发射功率总上限（每用户两路功率之和 ≤ P_max）----
        self.P_max = 1.0  # 单位“W”，先设为 1.0；将来可在 cfg 里参数化
        # ===== Reward weights (for r = -(w_d * delay + w_e * energy)) =====
        # 支持两种模式：固定值 / 训练时按区间随机采样
        self.w_d = 1.0  # 默认权重（可被外部覆盖）
        self.w_e = 1.0  # 默认权重（可被外部覆盖）

        # 奖励权重与采样（让能耗更“重”）
        self.sample_weights = True  # 开启采样
        self.w_d_range = (0.2, 1.0)  # delay 权重采样范围（更保守）
        self.w_e_range = (2.0, 6.0)  # energy 权重采样范围（更重，原先偏小）
        self.w_fair_range = (0.2, 1.0)

        # 若不采样时的默认权重（也同步加重能耗）
        self.w_d = 0.5
        self.w_e = 3.0
        self.w_fair = 0.5
        # --- Reward scaling/clipping (for numerical stability) ---
        self.reward_scale = 10.0  # 把 delay+energy 缩一档
        self.reward_clip = 50.0  # 单用户奖励裁剪范围 [-50, 50]
        # 训练时用于日志的额外暴露
        self.last_off_kbit_sum = 0.0
        self.last_local_kbit_sum = 0.0
        self.last_mec_queue_cycles = 0.0

        # 也可以在外部脚本里通过 env.w_d = ... / env.sample_weights = True 去改

        self.DataBuf = np.zeros(self.n_veh)
        self.over_data = np.zeros(self.n_veh)
        self.data_p = np.zeros(self.n_veh)
        self.data_t = np.zeros(self.n_veh)

        self.rate = 3
        self.data_r = np.zeros(self.n_veh)  # 定义所有车辆用户的任务到达率
        self.data_buf_size = 10  # bytes


        ###--------------RIS-------------###
        self.phases_R_i = np.zeros([n_veh, M], dtype=complex)  # RIS-vehicle

        self.distances_R_i = np.zeros(n_veh)
        self.angles_R_i = np.zeros(n_veh)

        self.M = M  # 元素数量
        self.control_bit = control_bit  # 元素相移控制比特
        self.possible_angles = np.linspace(0, 2 * math.pi, 2 ** self.control_bit, endpoint=False)

        self.elements_phase_shift_complex = np.zeros(self.M, dtype=complex)  # 复数形式 RIS元素的相移，这个是后面要强化学习的动作
        self.phase_R = np.zeros(self.M, dtype=complex)  # RIS到BS
        # self.phases_R_i = np.zeros([self.number_of_vehicles, self.M], dtype=complex) #车辆到RIS

        self.distance_B_R = math.sqrt(
            (BS_x - RIS_x) ** 2 + (BS_y - RIS_y) ** 2 + (BS_z - RIS_z) ** 2)
        self.angle_B_R = (RIS_x - BS_x) / self.distance_B_R  # 从RIS到BS的角度
        for m in range(self.M):
            self.phase_R[m] = cmath.exp(2 * (math.pi / lamb) * d * self.angle_B_R * m * 1j)

        # self.Random_phase()
        self.elements_phase_shift_real = np.zeros(M)
        self.channel_gains = np.zeros(self.n_veh)  # Add a class member to store channel gains
        # === 3GPP 38.901 简化开关与参数（新增） ===
        self.channel_model = "free"   # 可选: "free", "3gpp_umi", "3gpp_uma"
        self.fc_GHz = 3.5             # 载频 (GHz)，可按需改成 2.6/28 等
        self.shadow_std_los = 4.0     # dB，UMi/UMa LOS 典型
        self.shadow_std_nlos = 7.0    # dB，UMi/UMa NLOS 典型
        self.rician_K_dB = 0.0        # 简化：0 表示瑞利；>0 表示Rice
        self.compute_parms()

    def get_path_loss(self, position_A):
        d1 = abs(position_A[0] - BS_x)
        d2 = abs(position_A[1] - BS_y)
        distance = math.hypot(d1, d2)
        return 128.1 + 37.6 * np.log10(math.sqrt(distance ** 2 + (BS_z - 1.5) ** 2) / 1000)

    def get_shadowing(self, delta_distance, vehicle):
        self.R = np.sqrt(0.5 * np.ones([1, 1]) + 0.5 * np.identity(1))
        return np.multiply(np.exp(-1 * (delta_distance / self.Decorrelation_distance)), self.V2I_Shadowing[vehicle]) \
            + np.sqrt(1 - np.exp(-2 * (delta_distance / self.Decorrelation_distance))) * np.random.normal(0, 8, 1)

    def Random_phase(self):
        self.elements_phase_shift_real = [random.choice(self.possible_angles) for x1 in range(self.M)]  # 随机产生RIS的相移
        for m in range(self.M):
            self.elements_phase_shift_complex[m] = cmath.exp(self.elements_phase_shift_real[m] * 1j)

    def optimize_phase_shift(self, ):
        for m in range(self.M):
            best = 0
            best_phase = 0
            for phase in self.possible_angles:
                self.elements_phase_shift_complex[m] = cmath.exp(phase * 1j)

                x = self.optimize_compute_objective_function()
                if (best < x):
                    best = x
                    best_phase = cmath.exp(phase * 1j)

            self.elements_phase_shift_complex[m] = best_phase

    def optimize_compute_objective_function(self, ):
        sum_snr = 0
        for vehicle in range(self.n_veh):
            img = 0
            img = np.sum(np.multiply(np.multiply(self.elements_phase_shift_complex, self.phases_R_i), self.phase_R))
            cascaded_gain = (ro * img) / (
                    math.sqrt(self.distances_R_i[vehicle] ** alpha1) * math.sqrt(self.distance_B_R ** alpha2))

            sum_snr += (np.abs(cascaded_gain) ** 2) / sigma ** 2
        return sum_snr

    def get_next_phase(self, action_phase):
        """for i in range(self.M):
            index = i % n_veh
            self.elements_phase_shift_real[i] = action_phase[index]"""
        self.elements_phase_shift_real = action_phase
        for m in range(self.M):
            self.elements_phase_shift_complex[m] = cmath.exp(self.elements_phase_shift_real[m] * 1j)

    def compute_parms(self):
        # Calculate vehicle to RIS distance and angles

        for vehicle in range(len(self.vehicles)):
            d_R_i = math.sqrt((self.vehicles[vehicle].position[0] - RIS_x) ** 2 + (
                        self.vehicles[vehicle].position[1] - RIS_y) ** 2 + (1.5 - RIS_z) ** 2)
            self.distances_R_i[vehicle] = d_R_i
            self.angles_R_i[vehicle] = ((self.vehicles[vehicle].position[0] - RIS_x) / d_R_i)

        # Calculate phase shift with vehicles
        for m in range(len(self.elements_phase_shift_real)):
            for vehicle in range(len(self.vehicles)):
                self.phases_R_i[vehicle][m] = cmath.exp(-2 * (math.pi / lamb) * d * self.angles_R_i[vehicle] * m * 1j)

    def update_channel_gains(self):
        """
        更新 self.channel_gains：
          - "free": 保留你原来的 RIS 级联模型（默认）
          - "3gpp_umi": 3GPP TR 38.901 UMi（简化 LOS/NLOS + 对数正态 + 小尺度）
          - "3gpp_uma": 3GPP TR 38.901 UMa（简化 LOS/NLOS + 对数正态 + 小尺度）
        说明：此处仅做“链路增益”的生成，不改你的 NOMA 计算流程。
        """
        if self.channel_model == "free":
            # —— 原始 RIS 级联（保持不变）——
            for i in range(self.n_veh):
                img = 0
                for m in range(self.M):
                    comp = self.elements_phase_shift_complex[m] * self.phases_R_i[i][m] * self.phase_R[m]
                    img += comp
                cascaded_gain = (ro * img) / (
                        math.sqrt(self.distances_R_i[i] ** alpha1) * math.sqrt(self.distance_B_R ** alpha2))
                self.channel_gains[i] = np.abs(cascaded_gain) ** 2
            return

        # —— 以下是 3GPP 38.901 的简化实现（统一用 3D 距离，单位米；fc 用 GHz）——
        fc = float(self.fc_GHz)
        lam = 3e8 / (fc * 1e9)  # 仅用于需要时的直观检查，不直接用

        def _pl_umi_los_dB(d3d_m):
            # 38.901 UMi-Street LOS 近距离公式（简化版）
            return 32.4 + 21.0 * np.log10(fc) + 20.0 * np.log10(max(d3d_m, 1.0))

        def _pl_umi_nlos_dB(d3d_m):
            # 38.901 UMi NLOS（取 LOS 与 NLOS 中较大者的近似做法，这里直接给一条常用式）
            return 36.7 + 22.7 * np.log10(fc) + 26.0 * np.log10(max(d3d_m, 1.0))

        def _pl_uma_los_dB(d3d_m):
            # 38.901 UMa LOS（简化）
            return 28.0 + 22.0 * np.log10(fc) + 20.0 * np.log10(max(d3d_m, 1.0))

        def _pl_uma_nlos_dB(d3d_m):
            # 38.901 UMa NLOS（简化）
            return 13.54 + 39.08 * np.log10(max(d3d_m, 1.0)) + 20.0 * np.log10(fc) - 0.6 * (self.vehAntGain)

        # 简单 LOS 判决：距离越近越可能 LOS（可以换成更精细的地图逻辑）
        def _is_los(d2d_m):
            # 200m 内 70% 概率 LOS，之后快速衰减
            p_los = 0.7 * np.exp(-d2d_m / 200.0)
            return np.random.rand() < p_los

        for i in range(self.n_veh):
            # 与 BS 的水平距离（米）
            dx = abs(self.vehicles[i].position[0] - BS_x)
            dy = abs(self.vehicles[i].position[1] - BS_y)
            dz = abs(BS_z - 1.5)  # UE 高约 1.5m
            d2d = math.hypot(dx, dy)
            d3d = math.sqrt(d2d * d2d + dz * dz)

            los = _is_los(d2d)

            if self.channel_model == "3gpp_umi":
                pl_db = _pl_umi_los_dB(d3d) if los else _pl_umi_nlos_dB(d3d)
            elif self.channel_model == "3gpp_uma":
                pl_db = _pl_uma_los_dB(d3d) if los else _pl_uma_nlos_dB(d3d)
            else:
                # 容错：遇到未知关键字，退回 free
                pl_db = 0.0

            # 路损线性
            large_scale = 10 ** (-pl_db / 10.0)
            # 阴影：LOS 用 4dB，NLOS 用 7dB（可在 __init__ 配置）
            shadow = _log_normal_shadow(self.shadow_std_los if los else self.shadow_std_nlos)
            # 小尺度：瑞利/Rice
            small = _small_scale_power(self.rician_K_dB)

            # 最终“功率增益”= 路损 * 阴影 * 小尺度
            self.channel_gains[i] = large_scale * shadow * small

    # power是一个2 * n_veh的数组，第一行是卸载的功率，第二行是本地执行功率
    # noma_groups 是一个动态的分组列表, e.g., [[0, 3], [5, 2], [1], [4], [6], [7]]
    def compute_data_rate(self, power, noma_groups):
        # 1. 【已修改】直接使用在别处已更新的信道增益
        channel_gains = self.channel_gains

        # 2. 初始化所有用户的速率为0
        vehicle_rates = np.zeros(self.n_veh)

        # 3. 遍历传入的NOMA分组，逐组计算速率
        for group in noma_groups:
            # === NEW: 组带宽分摊（FDM/TDM 等价处理）===
            G = max(1, len(noma_groups))
            group_frac = 1.0 / G  # 每组仅占系统带宽的1/G

            if len(group) == 1:
                # --- 单用户传输 (OMA) ---
                user_index = group[0]
                signal_power = power[0, user_index] * channel_gains[user_index]
                sinr = signal_power / (self.noise_power)
                vehicle_rates[user_index] = group_frac * math.log2(1 + sinr)

            elif len(group) == 2:
                # --- NOMA 配对 ---
                user1_idx, user2_idx = group[0], group[1]
                gain1, gain2 = channel_gains[user1_idx], channel_gains[user2_idx]
                if gain1 > gain2:
                    near_user_idx, far_user_idx = user1_idx, user2_idx
                    near_gain, far_gain = gain1, gain2
                else:
                    near_user_idx, far_user_idx = user2_idx, user1_idx
                    near_gain, far_gain = gain2, gain1

                far_user_signal = power[0, far_user_idx] * far_gain
                near_user_interf_far = power[0, near_user_idx] * far_gain
                far_user_sinr = far_user_signal / (near_user_interf_far + self.noise_power)
                vehicle_rates[far_user_idx] = group_frac * math.log2(1 + far_user_sinr)

                near_user_signal = power[0, near_user_idx] * near_gain
                near_user_sinr = near_user_signal / (self.noise_power)
                vehicle_rates[near_user_idx] = group_frac * math.log2(1 + near_user_sinr)


        return vehicle_rates

    def get_channel_gains(self):
        """ A helper function to return the latest calculated channel gains. """
        return self.channel_gains

    def add_new_vehicles(self, start_position, start_direction, start_velocity):
        self.vehicles.append(Vehicle(start_position, start_direction, start_velocity))

    def add_new_vehicles_by_number(self, n):
        string = 'dulr'
        for i in range(n):
            ind = np.random.randint(0, len(self.down_lanes))

            start_position = [self.down_lanes[ind], np.random.randint(220, 230)]
            start_direction = 'd'  # velocity: 10 ~ 15 m/s, random
            self.add_new_vehicles(start_position, start_direction, np.random.randint(10, 15))

            start_position = [self.up_lanes[0], np.random.randint(170, 180)]
            start_direction = 'u'
            self.add_new_vehicles(start_position, start_direction, np.random.randint(10, 15))

            start_position = [np.random.randint(220, 230), self.left_lanes[0]]
            start_direction = 'l'
            self.add_new_vehicles(start_position, start_direction, np.random.randint(10, 15))

            start_position = [np.random.randint(170, 180), self.right_lanes[0]]
            start_direction = 'r'
            self.add_new_vehicles(start_position, start_direction, np.random.randint(10, 15))

        for j in range(int(self.n_veh % 4)):  # 当车辆数不是4的倍数时，按照这个添加车辆
            ind = np.random.randint(0, len(self.down_lanes))
            str = random.choice(string)
            start_position = [self.down_lanes[ind], np.random.randint(0, self.height)]
            start_direction = str  # velocity: 10 ~ 15 m/s, random
            self.add_new_vehicles(start_position, start_direction, np.random.randint(15, 20))

        self.V2I_Shadowing = np.random.normal(0, 8, len(self.vehicles))
        self.delta_distance = np.asarray([c.velocity * self.time_slow for c in self.vehicles])

    def renew_positions(self):
        # ===============
        # This function updates the position of each vehicle
        # ===============

        i = 0
        while (i < len(self.vehicles)):
            delta_distance = self.vehicles[i].velocity * self.time_slow
            change_direction = False
            if self.vehicles[i].direction == 'u':
                # print ('len of position', len(self.position), i)
                for j in range(len(self.left_lanes)):

                    if (self.vehicles[i].position[1] <= self.left_lanes[j]) and (
                            (self.vehicles[i].position[1] + delta_distance) >= self.left_lanes[j]):  # came to an cross
                        if (np.random.uniform(0, 1) < 0.4):
                            self.vehicles[i].position = [self.vehicles[i].position[0] - (
                                        delta_distance - (self.left_lanes[j] - self.vehicles[i].position[1])),
                                                         self.left_lanes[j]]
                            self.vehicles[i].direction = 'l'
                            change_direction = True
                            break
                if change_direction == False:
                    for j in range(len(self.right_lanes)):
                        if (self.vehicles[i].position[1] <= self.right_lanes[j]) and (
                                (self.vehicles[i].position[1] + delta_distance) >= self.right_lanes[j]):
                            if (np.random.uniform(0, 1) < 0.4):
                                self.vehicles[i].position = [self.vehicles[i].position[0] + (
                                            delta_distance + (self.right_lanes[j] - self.vehicles[i].position[1])),
                                                             self.right_lanes[j]]
                                self.vehicles[i].direction = 'r'
                                change_direction = True
                                break
                if change_direction == False:
                    self.vehicles[i].position[1] += delta_distance
            if (self.vehicles[i].direction == 'd') and (change_direction == False):
                # print ('len of position', len(self.position), i)
                for j in range(len(self.left_lanes)):
                    if (self.vehicles[i].position[1] >= self.left_lanes[j]) and (
                            (self.vehicles[i].position[1] - delta_distance) <= self.left_lanes[j]):  # came to an cross
                        if (np.random.uniform(0, 1) < 0.4):
                            self.vehicles[i].position = [self.vehicles[i].position[0] - (
                                        delta_distance - (self.vehicles[i].position[1] - self.left_lanes[j])),
                                                         self.left_lanes[j]]
                            # print ('down with left', self.vehicles[i].position)
                            self.vehicles[i].direction = 'l'
                            change_direction = True
                            break
                if change_direction == False:
                    for j in range(len(self.right_lanes)):
                        if (self.vehicles[i].position[1] >= self.right_lanes[j]) and (
                                self.vehicles[i].position[1] - delta_distance <= self.right_lanes[j]):
                            if (np.random.uniform(0, 1) < 0.4):
                                self.vehicles[i].position = [self.vehicles[i].position[0] + (
                                            delta_distance + (self.vehicles[i].position[1] - self.right_lanes[j])),
                                                             self.right_lanes[j]]
                                # print ('down with right', self.vehicles[i].position)
                                self.vehicles[i].direction = 'r'
                                change_direction = True
                                break
                if change_direction == False:
                    self.vehicles[i].position[1] -= delta_distance
            if (self.vehicles[i].direction == 'r') and (change_direction == False):
                # print ('len of position', len(self.position), i)
                for j in range(len(self.up_lanes)):
                    if (self.vehicles[i].position[0] <= self.up_lanes[j]) and (
                            (self.vehicles[i].position[0] + delta_distance) >= self.up_lanes[j]):  # came to an cross
                        if (np.random.uniform(0, 1) < 0.4):
                            self.vehicles[i].position = [self.up_lanes[j], self.vehicles[i].position[1] + (
                                        delta_distance - (self.up_lanes[j] - self.vehicles[i].position[0]))]
                            change_direction = True
                            self.vehicles[i].direction = 'u'
                            break
                if change_direction == False:
                    for j in range(len(self.down_lanes)):
                        if (self.vehicles[i].position[0] <= self.down_lanes[j]) and (
                                (self.vehicles[i].position[0] + delta_distance) >= self.down_lanes[j]):
                            if (np.random.uniform(0, 1) < 0.4):
                                self.vehicles[i].position = [self.down_lanes[j], self.vehicles[i].position[1] - (
                                            delta_distance - (self.down_lanes[j] - self.vehicles[i].position[0]))]
                                change_direction = True
                                self.vehicles[i].direction = 'd'
                                break
                if change_direction == False:
                    self.vehicles[i].position[0] += delta_distance
            if (self.vehicles[i].direction == 'l') and (change_direction == False):
                for j in range(len(self.up_lanes)):

                    if (self.vehicles[i].position[0] >= self.up_lanes[j]) and (
                            (self.vehicles[i].position[0] - delta_distance) <= self.up_lanes[j]):  # came to an cross
                        if (np.random.uniform(0, 1) < 0.4):
                            self.vehicles[i].position = [self.up_lanes[j], self.vehicles[i].position[1] + (
                                        delta_distance - (self.vehicles[i].position[0] - self.up_lanes[j]))]
                            change_direction = True
                            self.vehicles[i].direction = 'u'
                            break
                if change_direction == False:
                    for j in range(len(self.down_lanes)):
                        if (self.vehicles[i].position[0] >= self.down_lanes[j]) and (
                                (self.vehicles[i].position[0] - delta_distance) <= self.down_lanes[j]):
                            if (np.random.uniform(0, 1) < 0.4):
                                self.vehicles[i].position = [self.down_lanes[j], self.vehicles[i].position[1] - (
                                            delta_distance - (self.vehicles[i].position[0] - self.down_lanes[j]))]
                                change_direction = True
                                self.vehicles[i].direction = 'd'
                                break
                    if change_direction == False:
                        self.vehicles[i].position[0] -= delta_distance

            # if it comes to an exit
            if (self.vehicles[i].position[0] < 0) or (self.vehicles[i].position[1] < 0) or (
                    self.vehicles[i].position[0] > self.width) or (self.vehicles[i].position[1] > self.height):
                # delete
                #    print ('delete ', self.position[i])
                if (self.vehicles[i].direction == 'u'):
                    self.vehicles[i].direction = 'r'
                    self.vehicles[i].position = [self.vehicles[i].position[0], self.right_lanes[-1]]
                else:
                    if (self.vehicles[i].direction == 'd'):
                        self.vehicles[i].direction = 'l'
                        self.vehicles[i].position = [self.vehicles[i].position[0], self.left_lanes[0]]
                    else:
                        if (self.vehicles[i].direction == 'l'):
                            self.vehicles[i].direction = 'u'
                            self.vehicles[i].position = [self.up_lanes[0], self.vehicles[i].position[1]]
                        else:
                            if (self.vehicles[i].direction == 'r'):
                                self.vehicles[i].direction = 'd'
                                self.vehicles[i].position = [self.down_lanes[-1], self.vehicles[i].position[1]]

            i += 1

    def localProcRev(self, b):
        return np.power(b * 1000 * self.L / self.time_fast, 3.0) * self.k

    def step(self, action_power, noma_groups):
        # 计算执行卸载和本地部分
        per_user_reward = np.zeros(self.n_veh)
        # self.optimize_phase_shift() # 已移除：RIS相位应由学习决定或作为固定环境
        # self.Random_phase()
        # ---- Day 3–5：功率投影（两路功率非负，且每个用户两路之和 ≤ P_max）----
        # 约定：action_power 形状为 [2, n_veh]，上层已做 [-1,1]→[0,1] 映射，但这里再做“严格投影”
        # 先裁负，再对每个用户列做“若和>1则归一化至1”，最终再乘 P_max 得到物理功率
        power_scale = getattr(self, "power_scale", 0.7)
        proj = np.clip(action_power, 0.0, None)* power_scale  # 先把负数裁成0
        for i in range(self.n_veh):
            s = proj[:, i].sum()
            if s > 1.0:
                proj[:, i] /= (s + 1e-12)  # 列归一化，使 Pu+Pv = 1
        power_W = proj * self.P_max  # 物理发射功率（W）


        # 一次性计算所有用户的NOMA速率
        all_vehicle_rates = self.compute_data_rate(power_W, noma_groups)

        # ===== [MEC + CPU model – NEW] =====
        # 速率与可传输数据（kbit）
        self.vehicle_rate[:] = all_vehicle_rates
        self.data_t[:] = self.vehicle_rate * self.time_fast * self.bandwidth * 1000.0  # kbit 可卸载
        # 新（用“未缩放动作”决定 CPU 频率；仅发射功率继续用 proj）
        cpu_share = np.clip(action_power[1, :], 0.0, 1.0)
        # === RVM-guard: 防止本地 CPU 份额趋近于 0，避免 delay_local 爆炸 ===
        _floor = float(getattr(self, "cpu_share_floor", 0.10))
        # 保险：把 floor 限制在 [0, 0.95]，避免极端配置
        if not np.isfinite(_floor): _floor = 0.10
        _floor = max(0.0, min(_floor, 0.95))
        cpu_share = np.maximum(cpu_share, _floor)

        f_local = cpu_share * self.f_local_max

        Cpb = float(self.cycles_per_bit)

        # 本地待处理 backlog（以 cycles 计）
        backlog_kbit_before = self.DataBuf.copy()
        backlog_cycles_before = backlog_kbit_before * 1000.0 * Cpb

        # 本地本步最大可处理 cycles 与实际处理 cycles
        local_cycles_cap = f_local * self.time_fast
        local_cycles_used = np.minimum(local_cycles_cap, backlog_cycles_before)
        local_done_kbit = local_cycles_used / (Cpb * 1000.0)
        self.data_p[:] = local_done_kbit  # 维持 data_p 语义 = “本地完成的 kbit”

        # 卸载（kbit）：不能超过剩余 backlog，也不能超过链路能传的 data_t
        remaining_kbit = np.maximum(0.0, backlog_kbit_before - self.data_p)
        off_kbit = np.minimum(self.data_t, remaining_kbit)
        # ===== 传输时延（Tx time）=====
        # 吞吐率( kb/s ) = rate(bit/s/Hz) * 带宽(MHz) * 1000
        throughput_kbps = self.vehicle_rate * self.bandwidth * 1000.0
        # t_tx(秒) = 传输的任务量(kbit) / 吞吐率(kb/s)；对 0 速率做稳定保护
        t_tx = np.divide(off_kbit, throughput_kbps + 1e-12)

        # MEC 入队（cycles）
        edge_cycles_in = off_kbit * 1000.0 * Cpb
        edge_queue_before = self.mec_queue_cycles
        self.mec_queue_cycles += edge_cycles_in.sum()

        # MEC 服务器服务（单服务器，FCFS 近似）
        edge_service_cycles = min(self.f_edge_max * self.time_fast, self.mec_queue_cycles)
        self.mec_queue_cycles -= edge_service_cycles
        # === 日志暴露（供 TB 用） ===
        self.last_off_kbit_sum = float(off_kbit.sum())
        self.last_local_kbit_sum = float(local_done_kbit.sum())
        self.last_mec_queue_cycles = float(self.mec_queue_cycles)

        # 更新每用户任务 backlog（kbit）
        self.DataBuf -= (self.data_p + off_kbit)
        self.DataBuf = np.maximum(0.0, self.DataBuf)  # 不允许负
        self.last_backlog_kbit_mean = float(self.DataBuf.mean())

        # ===== 时延拆分：本地排队+计算、边缘排队+计算（修正：本地扣除已卸载部分） =====
        eps = 1e-12
        local_cycles_after_offload = np.maximum(
            0.0, backlog_cycles_before - edge_cycles_in
        )
        delay_local = local_cycles_after_offload / (f_local + eps)

        # MEC：队列按入队占比分摊，计算按 f_edge_max
        share = edge_cycles_in / (edge_cycles_in.sum() + eps)
        delay_edge_q = share * (edge_queue_before / (self.f_edge_max + eps))
        delay_edge_c = edge_cycles_in / (self.f_edge_max + eps)

        per_user_delay = delay_local + t_tx + delay_edge_q + delay_edge_c

        # 同步更新四项均值指标（若你原先就有相应统计字段，则保留命名）
        self.last_delay_local_mean = float(np.mean(delay_local))
        self.last_delay_edge_q_mean = float(np.mean(delay_edge_q))
        self.last_delay_edge_c_mean = float(np.mean(delay_edge_c))
        self.last_t_tx_mean = float(np.mean(t_tx))

        per_user_delay = delay_local + t_tx + delay_edge_q + delay_edge_c
        # === 新增：供训练脚本读取的分项均值（本步） ===
        self.last_delay_local_mean = float(np.mean(delay_local))
        self.last_delay_edge_q_mean = float(np.mean(delay_edge_q))
        self.last_delay_edge_c_mean = float(np.mean(delay_edge_c))
        self.last_t_tx_mean = float(np.mean(t_tx))

        # 平均 backlog（kbit）：动作前（用于辅助判断系统负荷）
        self.last_backlog_kbit_mean = float(np.mean(backlog_kbit_before))

        # MEC 利用率（本步服务的 cycles / 本步可服务容量 cycles）
        mec_capacity_cycles = self.f_edge_max * self.time_fast
        self.last_mec_utilization = float(edge_service_cycles / (mec_capacity_cycles + 1e-12))

        # 本地 CPU 利用率（每车本步用掉的 cycles / 可用 cycles），取均值
        self.last_local_util_mean = float(np.mean(local_cycles_used / (local_cycles_cap + 1e-12)))

        # ===== 能耗：Tx + 本地 CPU（E = k * f^2 * cycles）=====
        E_tx = power_W[0, :] * t_tx
        E_loc = self.k * (f_local ** 2) * local_cycles_used
        per_user_energy = E_tx + E_loc  # 仅物理能量，不掺入QoS罚

        # === 统一功率口径：全部按“等效功率 = 本步能量 / 步长（W）”上报 ===
        P_tx_eq_W = E_tx / self.time_fast
        P_loc_eq_W = E_loc / self.time_fast
        self.last_power_W = np.vstack([P_tx_eq_W, P_loc_eq_W])

        # QoS 违约罚作为“独立罚项”，不再污染能耗
        qos_penalty_vec = np.zeros(self.n_veh, dtype=float)
        if getattr(self, "qos_enable", False):
            rate = np.asarray(self.vehicle_rate, dtype=float)
            delay = np.asarray(per_user_delay, dtype=float)
            viol = (rate < float(self.R_min_bpsHz)) | (delay > float(self.D_max_s))
            if np.any(viol):
                qos_penalty_vec = float(self.qos_penalty) * viol.astype(float)
            # 可选：暴露违约率用于日志
            self.last_qos_violation = float(np.mean(viol.astype(float)))

        # --- Balanced normalization (Method A) ---
        # 在线 EMA 统计（首次调用时初始化）
        if not hasattr(self, "reward_norm_beta"):
            self.reward_norm_beta = float(getattr(self, "reward_norm_beta", 0.9))
        if not hasattr(self, "delay_mean"):
            self.delay_mean, self.delay_var = 0.0, 1.0
        if not hasattr(self, "energy_mean"):
            self.energy_mean, self.energy_var = 0.0, 1.0

        b = float(self.reward_norm_beta)

        # 本批均值（标量，用于 EMA）
        _d = float(np.mean(per_user_delay))
        _e = float(np.mean(per_user_energy))

        # ===== New: train directly on the physical objective (no Z-normalization) =====
        # r_i = - ( w_d * delay_i + w_e * energy_i ) - qos_penalty_i
        wd = float(getattr(self, "w_d", 1.0))
        we = float(getattr(self, "w_e", 1.0))
        _cost = wd * per_user_delay + we * per_user_energy
        per_user_reward = -_cost - qos_penalty_vec

        # 数值稳定（不改变最优解，只限幅）
        clipv = float(getattr(self, "reward_clip", 50.0))
        per_user_reward = np.clip(per_user_reward, -clipv, clipv)

        # === 仍然记录“物理口径”的均值，供训练脚本打日志 ===
        self.last_delay_mean = float(np.mean(per_user_delay))
        self.last_energy_mean = float(np.mean(per_user_energy))

        # === 日志口径（保持对“物理能量”的跟踪）===
        self.last_delay_mean = float(np.mean(per_user_delay))
        self.last_energy_mean = float(np.mean(per_user_energy))

        '''for i in range(self.n_veh):
            per_user_reward[i] = -((self.t_factor1 * (action_power[0, i] + action_power[1, i]))) \
                                 - ((self.t_factor2 * self.DataBuf[i]))'''

        for k in range(self.n_veh):
            self.data_r[k] = np.random.poisson(self.rate)  # unit: mbit 任务到达率
            self.DataBuf[k] += self.data_r[k] * self.time_fast * 1000

        global_reward = np.mean(per_user_reward)
        """self.Reward = -((self.t_factor * np.sum(action_power) * 10) / self.n_veh)\
                              - (((1-self.t_factor) * np.sum(self.DataBuf)) / self.n_veh) - (self.penalty * np.sum(self.over_data) / (self.n_veh * 40))"""

        # === New: define over_power for compatibility & logging ===
        # per-user total transmit+local power (W)
        _per_user_sum_W = power_W[0, :] + power_W[1, :]
        # exceed amount over the per-user cap P_max; with projection it should be zeros
        over_power = np.maximum(0.0, _per_user_sum_W - self.P_max)

        return per_user_reward, global_reward, self.DataBuf, self.data_t, self.data_p, over_power, self.over_data

    def make_new_game(self):
        self.vehicles = []
        self.add_new_vehicles_by_number(int(self.n_veh / 4))

        self.DataBuf = np.random.randint(5, self.data_buf_size - 1) / 2.0 * np.ones(self.n_veh)
