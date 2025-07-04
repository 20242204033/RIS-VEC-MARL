import numpy as np
import time
import random
import math
import cmath

np.random.seed(1234)

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
        #self.t = 0.02
        self.bandwidth = 1#MHz
        self.k = 1e-28
        self.L = 500

        self.DataBuf = np.zeros(self.n_veh)
        self.over_data = np.zeros(self.n_veh)
        self.data_p = np.zeros(self.n_veh)
        self.data_t = np.zeros(self.n_veh)

        self.rate = 3
        self.data_r = np.zeros(self.n_veh)  # 定义所有车辆用户的任务到达率
        self.data_buf_size = 10 #bytes

        self.t_factor1 = 1
        self.t_factor2 = 0.6
        self.penalty1 = 2
        self.penalty2 = 2

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

        #self.Random_phase()
        self.elements_phase_shift_real = np.zeros(M)
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
        self.elements_phase_shift_real = [random.choice(self.possible_angles) for x1 in range(self.M)] #随机产生RIS的相移
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

    def optimize_compute_objective_function(self,):
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
            d_R_i = math.sqrt((self.vehicles[vehicle].position[0] - RIS_x) ** 2 + (self.vehicles[vehicle].position[1] - RIS_y) ** 2 + (1.5 - RIS_z) ** 2)
            self.distances_R_i[vehicle] = d_R_i
            self.angles_R_i[vehicle] = ((self.vehicles[vehicle].position[0] - RIS_x) / d_R_i)

        #Calculate phase shift with vehicles
        for m in range(len(self.elements_phase_shift_real)):
            for vehicle in range(len(self.vehicles)):
                self.phases_R_i[vehicle][m] = cmath.exp(-2 * (math.pi / lamb) * d * self.angles_R_i[vehicle] * m * 1j)

    # 计算一个车辆的信息传输速率, power是功率分配的动作
    #power是一个2 * n_veh的数组，第一行是卸载的功率，第二行是本地执行功率
    def compute_data_rate(self, vehicle, power):
        img = 0
        rate_ris = 0
        rate_direct = 0
        for m in range(self.M):
            comp = self.elements_phase_shift_complex[m] * self.phases_R_i[vehicle][m] * self.phase_R[m]
            img += comp

        cascaded_gain = (ro * img) / (
                math.sqrt(self.distances_R_i[vehicle] ** alpha1) * math.sqrt(self.distance_B_R ** alpha2))

        rate_ris = math.log(1 + power[0, vehicle] * (np.abs(cascaded_gain) ** 2) / sigma ** 2)

        '''# 直接通信链路
        V2I_Shadowing = self.get_shadowing(self.delta_distance[vehicle], vehicle)
        self.V2I_pathloss[vehicle] = self.get_path_loss((self.vehicles[vehicle].position))
        self.V2I_channels_abs[vehicle] = self.V2I_pathloss[vehicle] + V2I_Shadowing[0]
        power_dbm = 10 * math.log10(power[0, vehicle] * 1000)
        V2I_signal = 10 ** ((power_dbm - self.V2I_channels_abs[
            vehicle] + self.vehAntGain + self.bsAntGain - self.bsNoiseFigure) / 10)
        rate_direct = np.log2(1 + np.divide(V2I_signal, self.sig2))'''

        return rate_ris

    def add_new_vehicles(self, start_position, start_direction, start_velocity):
        self.vehicles.append(Vehicle(start_position, start_direction, start_velocity))

    def add_new_vehicles_by_number(self, n):
        string = 'dulr'
        for i in range(n):
            ind = np.random.randint(0, len(self.down_lanes))

            start_position = [self.down_lanes[0], np.random.randint(220, 230)]
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

        for j in range(int(self.n_veh % 4)): #当车辆数不是4的倍数时，按照这个添加车辆
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

                    if (self.vehicles[i].position[1] <= self.left_lanes[j]) and ((self.vehicles[i].position[1] + delta_distance) >= self.left_lanes[j]):  # came to an cross
                        if (np.random.uniform(0, 1) < 0.4):
                            self.vehicles[i].position = [self.vehicles[i].position[0] - (delta_distance - (self.left_lanes[j] - self.vehicles[i].position[1])), self.left_lanes[j]]
                            self.vehicles[i].direction = 'l'
                            change_direction = True
                            break
                if change_direction == False:
                    for j in range(len(self.right_lanes)):
                        if (self.vehicles[i].position[1] <= self.right_lanes[j]) and ((self.vehicles[i].position[1] + delta_distance) >= self.right_lanes[j]):
                            if (np.random.uniform(0, 1) < 0.4):
                                self.vehicles[i].position = [self.vehicles[i].position[0] + (delta_distance + (self.right_lanes[j] - self.vehicles[i].position[1])), self.right_lanes[j]]
                                self.vehicles[i].direction = 'r'
                                change_direction = True
                                break
                if change_direction == False:
                    self.vehicles[i].position[1] += delta_distance
            if (self.vehicles[i].direction == 'd') and (change_direction == False):
                # print ('len of position', len(self.position), i)
                for j in range(len(self.left_lanes)):
                    if (self.vehicles[i].position[1] >= self.left_lanes[j]) and ((self.vehicles[i].position[1] - delta_distance) <= self.left_lanes[j]):  # came to an cross
                        if (np.random.uniform(0, 1) < 0.4):
                            self.vehicles[i].position = [self.vehicles[i].position[0] - (delta_distance - (self.vehicles[i].position[1] - self.left_lanes[j])), self.left_lanes[j]]
                            # print ('down with left', self.vehicles[i].position)
                            self.vehicles[i].direction = 'l'
                            change_direction = True
                            break
                if change_direction == False:
                    for j in range(len(self.right_lanes)):
                        if (self.vehicles[i].position[1] >= self.right_lanes[j]) and (self.vehicles[i].position[1] - delta_distance <= self.right_lanes[j]):
                            if (np.random.uniform(0, 1) < 0.4):
                                self.vehicles[i].position = [self.vehicles[i].position[0] + (delta_distance + (self.vehicles[i].position[1] - self.right_lanes[j])), self.right_lanes[j]]
                                # print ('down with right', self.vehicles[i].position)
                                self.vehicles[i].direction = 'r'
                                change_direction = True
                                break
                if change_direction == False:
                    self.vehicles[i].position[1] -= delta_distance
            if (self.vehicles[i].direction == 'r') and (change_direction == False):
                # print ('len of position', len(self.position), i)
                for j in range(len(self.up_lanes)):
                    if (self.vehicles[i].position[0] <= self.up_lanes[j]) and ((self.vehicles[i].position[0] + delta_distance) >= self.up_lanes[j]):  # came to an cross
                        if (np.random.uniform(0, 1) < 0.4):
                            self.vehicles[i].position = [self.up_lanes[j], self.vehicles[i].position[1] + (delta_distance - (self.up_lanes[j] - self.vehicles[i].position[0]))]
                            change_direction = True
                            self.vehicles[i].direction = 'u'
                            break
                if change_direction == False:
                    for j in range(len(self.down_lanes)):
                        if (self.vehicles[i].position[0] <= self.down_lanes[j]) and ((self.vehicles[i].position[0] + delta_distance) >= self.down_lanes[j]):
                            if (np.random.uniform(0, 1) < 0.4):
                                self.vehicles[i].position = [self.down_lanes[j], self.vehicles[i].position[1] - (delta_distance - (self.down_lanes[j] - self.vehicles[i].position[0]))]
                                change_direction = True
                                self.vehicles[i].direction = 'd'
                                break
                if change_direction == False:
                    self.vehicles[i].position[0] += delta_distance
            if (self.vehicles[i].direction == 'l') and (change_direction == False):
                for j in range(len(self.up_lanes)):

                    if (self.vehicles[i].position[0] >= self.up_lanes[j]) and ((self.vehicles[i].position[0] - delta_distance) <= self.up_lanes[j]):  # came to an cross
                        if (np.random.uniform(0, 1) < 0.4):
                            self.vehicles[i].position = [self.up_lanes[j], self.vehicles[i].position[1] + (delta_distance - (self.vehicles[i].position[0] - self.up_lanes[j]))]
                            change_direction = True
                            self.vehicles[i].direction = 'u'
                            break
                if change_direction == False:
                    for j in range(len(self.down_lanes)):
                        if (self.vehicles[i].position[0] >= self.down_lanes[j]) and ((self.vehicles[i].position[0] - delta_distance) <= self.down_lanes[j]):
                            if (np.random.uniform(0, 1) < 0.4):
                                self.vehicles[i].position = [self.down_lanes[j], self.vehicles[i].position[1] - (delta_distance - (self.vehicles[i].position[0] - self.down_lanes[j]))]
                                change_direction = True
                                self.vehicles[i].direction = 'd'
                                break
                    if change_direction == False:
                        self.vehicles[i].position[0] -= delta_distance

            # if it comes to an exit
            if (self.vehicles[i].position[0] < 0) or (self.vehicles[i].position[1] < 0) or (self.vehicles[i].position[0] > self.width) or (self.vehicles[i].position[1] > self.height):
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

    def step(self, action_power):
        # 计算执行卸载和本地部分
        per_user_reward = np.zeros(self.n_veh)
        self.optimize_phase_shift()
        #self.Random_phase()
        for i in range(self.n_veh):
            rate = self.compute_data_rate(i, action_power)
            self.vehicle_rate[i] = rate
            self.data_t[i] = rate * self.time_fast * self.bandwidth * 1000 #unit:kbits
            self.data_p[i] = np.power(action_power[1, i] / self.k, 1.0/3.0) * self.time_fast/self.L/1000

        over_power = np.zeros(self.n_veh)
        self.DataBuf -= self.data_t + self.data_p
        for j in range(self.n_veh):
            if self.DataBuf[j] < 0:
                over_power[j] = action_power[1, j] - self.localProcRev(np.fmax(0, self.DataBuf[j] + self.data_p[j])) #这里在奖励计算时应该加一个惩罚来减小过载
                self.over_data[j] = -self.DataBuf[j]
                self.DataBuf[j] = 0
            else:
                self.over_data[j] = 0

        for i in range(self.n_veh):
            if self.DataBuf[i]>0:
                per_user_reward[i] = -((self.t_factor1 * (action_power[0, i] + action_power[1, i])))\
                      - ((self.t_factor2 * self.DataBuf[i])) - self.penalty1
            elif self.over_data[i]>2:
                per_user_reward[i] = -((self.t_factor1 * (action_power[0, i] + action_power[1, i]))) \
                                     - ((self.t_factor2 * self.DataBuf[i])) - self.penalty2
            else:
                per_user_reward[i] = -((self.t_factor1 * (action_power[0, i] + action_power[1, i]))) \
                                     - ((self.t_factor2 * self.DataBuf[i]))

        '''for i in range(self.n_veh):
            per_user_reward[i] = -((self.t_factor1 * (action_power[0, i] + action_power[1, i]))) \
                                 - ((self.t_factor2 * self.DataBuf[i]))'''

        for k in range(self.n_veh):
            self.data_r[k] = np.random.poisson(self.rate)  # unit: mbit 任务到达率
            self.DataBuf[k] += self.data_r[k] * self.time_fast * 1000

        global_reward = np.mean(per_user_reward)
        """self.Reward = -((self.t_factor * np.sum(action_power) * 10) / self.n_veh)\
                      - (((1-self.t_factor) * np.sum(self.DataBuf)) / self.n_veh) - (self.penalty * np.sum(self.over_data) / (self.n_veh * 40))"""

        return per_user_reward, global_reward, self.DataBuf, self.data_t, self.data_p, over_power, self.over_data

    def make_new_game(self):
        self.vehicles = []
        self.add_new_vehicles_by_number(int(self.n_veh / 4))

        self.DataBuf = np.random.randint(5, self.data_buf_size - 1) / 2.0 * np.ones(self.n_veh)

