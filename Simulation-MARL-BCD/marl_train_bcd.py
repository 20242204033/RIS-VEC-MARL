import math
import Environment
from sac_agent import Agent
from buffer import ReplayBuffer
from global_sac_critic import Global_SAC_Critic
import matplotlib.pyplot as plt
import pathlib, yaml
from torch.utils.tensorboard import SummaryWriter
import os, time
# log_root = "/tmp/tb_logs"
# os.makedirs(log_root, exist_ok=True)
# run_name = time.strftime("run_%Y%m%d_%H%M%S")
# writer = SummaryWriter(log_dir=os.path.join(log_root, run_name))

import time
# ============= Reproducibility + CLI + RunDir tools =============
import argparse
from pathlib import Path
from datetime import datetime
import torch
torch.set_num_threads(1)  # 单机可复现建议，避免 BLAS/MKL 线程争用


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--exp", type=str, default="default_exp", help="实验名（用于 runs/ 目录）")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    p.add_argument("--log", type=str, choices=["tb","none"], default="tb", help="是否写 TensorBoard")
    p.add_argument("--notes", type=str, default="", help="备注，会写入 run 目录")
    return p.parse_args()

def set_seed(seed: int):
    import os, random, torch, numpy as np
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

def make_run_dir(exp: str, seed: int) -> Path:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    rd = Path("runs") / exp / f"seed{seed:03d}-{ts}"
    (rd / "artifacts").mkdir(parents=True, exist_ok=True)
    return rd

def get_git_hash() -> str:
    # 在非 git 仓库下直接返回；并静音 git 的 stderr
    try:
        from pathlib import Path
        import subprocess, os

        # 不是 git 仓库就直接返回
        if not (Path(".git").exists() and Path(".git").is_dir()):
            return "no-git"

        h = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()

        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL
        ).decode().strip()

        return h + ("+dirty" if dirty else "")
    except Exception:

        return "no-git"

# === Stage-6: 读取训练基线（末尾窗口均值） ===
from pathlib import Path
import numpy as np

def _load_training_baseline(exp: str, tail: int = 20):
    """
    从 runs/<exp>/seed*-*/artifacts/episode_metrics.npy 读取训练基线：
      - global_reward 的尾窗口均值
      - min_user 的尾窗口均值
    返回 dict，如：
      {"global_mean": 1.23, "min_user_mean": 0.45}；若找不到返回 None
    """
    root = Path("runs") / exp
    if not root.exists():
        return None
    # 取最新的 seed-run
    cands = sorted(root.glob("seed*-*/artifacts/episode_metrics.npy"))
    if not cands:
        return None
    p = cands[-1]
    try:
        d = np.load(p, allow_pickle=True).item()
        out = {}
        if "global_reward" in d:
            g = np.asarray(d["global_reward"], dtype=np.float32)
            if g.size > 0:
                out["global_mean"] = float(np.mean(g[-tail:] if g.size >= tail else g))
        if "min_user" in d:
            m = np.asarray(d["min_user"], dtype=np.float32)
            if m.size > 0:
                out["min_user_mean"] = float(np.mean(m[-tail:] if m.size >= tail else m))
        return out if out else None
    except Exception:
        return None


# ---- Metrics helpers (Stage-0) ----
def _jain_index(x: np.ndarray) -> float:
    """Jain's fairness index: (Σx)^2 / (n * Σx^2). 处理全零/近零时做数值保护。"""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return 0.0
    s = x.sum()
    denom = (x.size * np.square(x).sum()) + 1e-12
    return float((s * s) / denom)


def _safe_mean(arr, default: float = 0.0) -> float:
    """对可能为空或含非数的列表做稳健均值，避免 RuntimeWarning。"""
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0 or not np.isfinite(arr).all():
        return float(default)
    return float(arr.mean())
def _anneal_topk(i_ep: int, n_agents: int, k_start: int, k_end: int, T_ep: int) -> int:
    """线性退火：episode 从 0→T_ep，把 K 从 k_start 收紧到 k_end。"""
    i = max(0, min(i_ep, T_ep))
    k = round(k_end + (k_start - k_end) * (1.0 - i / max(1, T_ep)))
    return int(np.clip(k, 1, n_agents - 1))

def _build_feasible_mask_from_delta_g(channel_gains, tau: float, K: int):
    g = np.asarray(channel_gains, dtype=float); N = g.size
    eps = 1e-15
    g = 10.0 * np.log10(np.maximum(g, eps))  # 统一用 dB 域
    mask = np.ones((N, N), dtype=np.float32); np.fill_diagonal(mask, 0.0)

    # 先按阈值筛
    for i in range(N):
        for j in range(N):
            if i != j and abs(g[i] - g[j]) < tau:
                mask[i, j] = 0.0

    # 再做每行 Top-K
    for i in range(N):
        cand = np.where(mask[i] > 0)[0]
        if cand.size > K:
            diffs = np.abs(g[i] - g[cand])
            keep = cand[np.argsort(-diffs)[:K]]
            drop = np.setdiff1d(cand, keep, assume_unique=False)
            mask[i, drop] = 0.0

    mask = mask * mask.T
    return mask
# ===== [NOMA-V3] helpers: 评分矩阵 / 黏滞互选 / 贪心补配 / 一次放宽 =====
import math, random
import numpy as np
from typing import Optional, Set, Tuple, List


# ===== [NOMA-V3] helpers: 评分矩阵 / 黏滞互选 / 贪心补配 / 一次放宽 =====
def _score_matrix_from_gain_and_history(
    gain_linear: np.ndarray,
    feasible_mask: np.ndarray,
    hist_affinity: np.ndarray,
    w_delta_db: float = 1.0,
    w_hist: float = 0.3,
    abs_gain_min_db: float = -math.inf,
    qos_soft_mask: Optional[np.ndarray] = None,   # NEW
    qos_soft_penalty: float = 6.0                # NEW：默认打 -6 “分”
) -> np.ndarray:
    """S_ij = wΔ * |Δg_dB| + wH * H_ij，并进行质量过滤（弱-弱屏蔽 + 可选QoS软惩罚）"""
    N = gain_linear.shape[0]
    g_db = 10.0 * np.log10(np.maximum(gain_linear, 1e-12))
    delta_db = np.abs(g_db[:, None] - g_db[None, :])
    # 质量过滤：至少一端绝对增益高于门槛
    abs_ok = ((g_db[:, None] >= abs_gain_min_db) | (g_db[None, :] >= abs_gain_min_db)).astype(np.float32)

    S = w_delta_db * delta_db + w_hist * hist_affinity
    if not np.any((feasible_mask > 0) & (abs_ok > 0)):
        abs_ok = np.ones_like(abs_ok, dtype=np.float32)

    # 先按硬可行 + 质量屏蔽到 -inf
    S = np.where((feasible_mask > 0) & (abs_ok > 0), S, -np.inf)

    # 再加 QoS 软惩罚：对“QoS不通过但未被不可行硬屏蔽”的边扣一个固定分
    if qos_soft_mask is not None:
        # 仅对还不是 -inf 的项施加惩罚
        S = np.where((qos_soft_mask <= 0) & np.isfinite(S), S - float(qos_soft_penalty), S)

    np.fill_diagonal(S, -np.inf)
    return S

def _mutual_then_greedy_pairs(
    S: np.ndarray,
    feasible: np.ndarray,
    min_pairs: int,
    sticky_pairs: Optional[Set[Tuple[int, int]]] = None,
    stick_tau_db: float = 1.0,
    gain_linear: Optional[np.ndarray] = None,
    unpaired_priority: Optional[np.ndarray] = None,
) -> List[List[int]]:

    """黏滞保持 -> 互选（带公平偏置）-> 贪心补配；返回 [[i,j], ...]"""
    N = S.shape[0]
    used, result = set(), []

    # 公平性微偏置：对连续落单多的用户，互选时给小偏置（不压倒 |Δg|）
    bias = np.zeros((N,), dtype=np.float32)
    if unpaired_priority is not None and np.max(unpaired_priority) > 0:
        bias = 0.2 * (unpaired_priority / (np.max(unpaired_priority) + 1e-6))

    # 1) 黏滞保持
    if sticky_pairs and gain_linear is not None:
        g_db = 10.0 * np.log10(np.maximum(gain_linear, 1e-12))
        for (i, j) in list(sticky_pairs):
            if i < N and j < N and feasible[i, j] > 0 and np.isfinite(S[i, j]):
                if abs(g_db[i] - g_db[j]) >= stick_tau_db and (i not in used) and (j not in used):
                    used.update([i, j]); result.append([i, j])

    # 2) 互选（行加公平偏置）
    top_choice = np.full((N,), -1, dtype=np.int32)
    for i in range(N):
        if i in used:
            continue
        row = S[i].copy()
        if not np.isfinite(row).any():
            continue
        row = row + bias[i]
        j = int(np.nanargmax(row))
        if np.isfinite(row[j]):
            top_choice[i] = j
    for i in range(N):
        j = top_choice[i]
        if j >= 0 and (top_choice[j] == i) and (i not in used) and (j not in used) and (i != j):
            used.update([i, j]); result.append([min(i, j), max(i, j)])

    # 3) 贪心补配
    if len(result) < min_pairs:
        edges = []
        for i in range(N):
            if i in used:
                continue
            for j in range(i+1, N):
                if j in used:
                    continue
                if feasible[i, j] > 0 and np.isfinite(S[i, j]):
                    edges.append((S[i, j], i, j))
        edges.sort(key=lambda x: x[0], reverse=True)
        for score, i, j in edges:
            if (i in used) or (j in used):
                continue
            used.update([i, j]); result.append([i, j])
            if len(result) >= min_pairs:
                break
    return result

def _relax_mask_once(mask_mat: np.ndarray,
                     gain_linear: np.ndarray,
                     tau_db: float,
                     topk: int) -> np.ndarray:
    """在现有 mask 基础上，按 (tau_db, topk) 放宽若干可行边（OR 到 mask）"""
    N = gain_linear.shape[0]
    g_db = 10.0 * np.log10(np.maximum(gain_linear, 1e-12))
    delta_db = np.abs(g_db[:, None] - g_db[None, :])
    cand = (delta_db >= tau_db).astype(np.uint8); np.fill_diagonal(cand, 0)
    topk_mask = np.zeros_like(cand)
    if topk >= 1:
        idx = np.argsort(-delta_db, axis=1)[:, :min(topk, N-1)]
        rows = np.arange(N)[:, None]
        topk_mask[rows, idx] = 1
    new_mask = ((mask_mat > 0) | (cand > 0) | (topk_mask > 0)).astype(np.uint8)
    return new_mask
def _mwm_completion(
    S: np.ndarray,
    feasible: np.ndarray,
    pairs_now: List[Tuple[int, int]],
    min_pairs: int,
    allow_singles: bool = True,
    completion_min_quantile: float = 0.30  # 可用 YAML 覆盖：completion_min_quantile
) -> List[Tuple[int, int]]:
    """
    只在“合格边”上做贪心补配；合格= (feasible>0) & (S有限) & (S>=分位阈值)。
    若合格边不足以补到 min_pairs，且 allow_singles=True，则停止补配（不硬配）。
    """
    N = S.shape[0]
    used = set()
    for a, b in pairs_now:
        used.add(a); used.add(b)

    # 取可行且 S 有限的边
    finite = np.isfinite(S) & (feasible > 0)
    if not np.any(finite):
        return pairs_now

    # 动态阈值：只在高于 completion_min_quantile 的边上补
    s_vals = S[finite]
    thr = np.nanquantile(s_vals, float(getattr(config, "completion_min_quantile", completion_min_quantile)))

    cand = []
    for i in range(N):
        for j in range(i + 1, N):
            if finite[i, j] and (S[i, j] >= thr):
                cand.append((S[i, j], i, j))

    # 分数从高到低贪心
    cand.sort(reverse=True)
    new_pairs = []
    occupied = set(used)
    for s, i, j in cand:
        if (i in occupied) or (j in occupied):
            continue
        new_pairs.append((i, j))
        occupied.add(i); occupied.add(j)
        if len(pairs_now) + len(new_pairs) >= int(min_pairs):
            break

    # 若仍不足目标对数，允许保留单播（不硬配）
    if allow_singles and (len(pairs_now) + len(new_pairs) < int(min_pairs)):
        return pairs_now + new_pairs

    return pairs_now + new_pairs

def _mwm_primary(
    S: np.ndarray,
    feasible: np.ndarray,
    accept_quantile: float = 0.20,
    allow_singles: bool = True
) -> List[List[int]]:
    """
    直接在全集图上做“最大权匹配”，以 S 为边权；
    - 仅选择权值 >= 分位数(accept_quantile) 的边，避免“硬配对”；
    - 允许保留单播（allow_singles=True）；
    - 返回 [[i,j], ...]，不含单播；单播在调用处通过全集补全。
    """
    N = S.shape[0]
    mask_edge = (np.asarray(feasible) > 0) & np.isfinite(S)
    cand_vals = S[mask_edge]
    if cand_vals.size == 0:
        return []  # 没有任何可行边

    # 接纳“Top-q”高质量边：q=0.20 表示仅保留 top 20% 的大权重边
    q = float(np.clip(accept_quantile, 0.0, 1.0))
    thr = float(np.quantile(cand_vals, 1.0 - q))  # 注意 1 - q
    # 仅保留得分 ≥ 上分位阈值的候选边
    W = np.where((S >= thr) & mask_edge, S, -np.inf)

    # 构建子问题的边权表（全集）
    edges = {}
    for i in range(N):
        for j in range(i + 1, N):
            if np.isfinite(W[i, j]):
                edges[(i, j)] = float(W[i, j])

    from functools import lru_cache
    ALL = N

    @lru_cache(None)
    def dp(mask: int):
        # 返回 (best_weight, list_of_pairs)
        if mask == (1 << ALL) - 1:
            return (0.0, [])

        # 找到第一个未用点
        for x in range(ALL):
            if not (mask & (1 << x)):
                i = x
                break

        best_w = -float("inf")
        best_pairs = []

        # 选项1：把 i 留成单播（允许时）
        if allow_singles:
            w1, P1 = dp(mask | (1 << i))
            if w1 > best_w:
                best_w, best_pairs = w1, P1

        # 选项2：把 i 与 j 配成一对
        for j in range(i + 1, ALL):
            if (mask & (1 << j)) == 0:
                w_edge = edges.get((i, j), -float("inf"))
                if not np.isfinite(w_edge):
                    continue
                w2, P2 = dp(mask | (1 << i) | (1 << j))
                if np.isfinite(w2) and (w_edge + w2 > best_w):
                    best_w = w_edge + w2
                    best_pairs = P2 + [(i, j)]
        if best_w == -float("inf"):
            return (0.0, [])
        return (best_w, best_pairs)

    total_w, pairs_sub = dp(0)
    out = [[min(i, j), max(i, j)] for (i, j) in pairs_sub]
    out.sort(key=lambda x: (x[0], x[1]))
    return out






# 解析参数 + 设种子（run_dir 稍后统一创建）
args = parse_args()
set_seed(args.seed)
writer = None  # 先给默认值，避免未定义
# ==========================================================




os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


class Config:
    def __init__(self):
        # ################## SETTINGS ######################
        # Environment Parameters
        self.n_veh = 8
        self.M = 40
        self.control_bit = 3
        self.width = 400
        self.height = 400
        # ===== Reward weights for r = -(w_d * delay + w_e * energy) =====
        self.use_weight_sampling = False  # True: 每个episode采样；False: 用固定值
        self.w_d_fixed = 1.0  # 固定值时使用
        self.w_e_fixed = 2.0

        # 采样区间（经验上 w_d 控制在 0.1~10，w_e 取决于能耗量级，先给 0.001~1 做对齐）
        self.w_d_range = (0.1, 10.0)
        self.w_e_range = (0.001, 1.0)
        # ===== NOMA-V3：MWM 补全（最大权匹配）开关 =====
        self.use_mwm_completion = True
        self.mwm_allow_singles = True
        # ===== NOMA-V3：MWM 作为主配对（新增）=====
        self.use_mwm_primary = True
        self.mwm_accept_quantile = 0.10  # 仅在高于该分位的边上做匹配，避免硬配
        self.mwm_backoff_rounds = 5  # 不足目标对数时的最多放宽轮数
        self.mwm_accept_q_step = 0.05  # 每轮把 accept_quantile 再降低一些
        # 兼容旧版回退：默认关闭。若设为 True，将启用互选/稳定匹配的旧回退分支
        self.legacy_pairing_backoff = False

        # Lane settings
        self.up_lanes = [i / 2.0 for i in [400 + 3.5 / 2, 400 + 3.5 + 3.5 / 2, 800 + 3.5 / 2, 800 + 3.5 + 3.5 / 2]]
        self.down_lanes = [i / 2.0 for i in [400 - 3.5 - 3.5 / 2, 400 - 3.5 / 2, 800 - 3.5 - 3.5 / 2, 800 - 3.5 / 2]]
        self.left_lanes = [i / 2.0 for i in [400 + 3.5 / 2, 400 + 3.5 + 3.5 / 2, 800 + 3.5 / 2, 800 + 3.5 + 3.5 / 2]]
        self.right_lanes = [i / 2.0 for i in [400 - 3.5 - 3.5 / 2, 400 - 3.5 / 2, 800 - 3.5 - 3.5 / 2, 800 - 3.5 / 2]]

        # Training Parameters
        self.n_episode = 300
        self.n_step_per_episode = 100
        self.IS_TRAIN = 1
        self.env_refresh_every = 5  # 每多少个 episode 刷新一次车辆位置/信道

        # SAC Algorithm Parameters
        self.batch_size = 256
        self.memory_size = 1000000
        self.gamma = 0.99
        self.alpha = 3e-4  # Actor learning rate
        self.beta =  3e-4   # Critic learning rate
        self.tau = 0.005  # Target network update factor
        self.update_actor_interval = 1
        self.noise = 0.2

        # ---- Stage-1: Gumbel-Softmax 退火参数 ----
        self.gumbel_tau_start = 2.0
        self.gumbel_tau_end = 0.3
        # 前50%训练进度退火完成：
        self.gumbel_anneal_steps = int(0.5 * self.n_episode * self.n_step_per_episode)

        self.gumbel_hard = False          # 训练前期建议 False，后期可改 True


        # Network Model Parameters
        self.C_fc1_dims = 1024
        self.C_fc2_dims = 512
        self.C_fc3_dims = 256
        self.A_fc1_dims = 512
        self.A_fc2_dims = 256

        # Reward Engineering Parameters
        self.FAIRNESS_LAMBDA = 0.01  # 公平项先降，避免主导
        self.LOW_REWARD_THRESHOLD = -4.0  # Z<-4 再视为“极低”
        self.LOW_REWARD_PENALTY = 80  # 大幅降权，配合上面的“幅度×封顶”
        self.VARIANCE_PENALTY = 0.02
        self.use_stable_matching = True  # 开启稳定匹配
        self.min_pair_target = max(1, self.n_veh // 4)  # 至少保证 1/4 NOMA 对数
        self.pairing_intent_weight_max = 0.5  # 意图权重的最大值
        self.pairing_intent_warmup_episodes = 200  # 权重达到最大值所需的回合数
        self.pairing_threshold_quantile = 0.5 # 自适应阈值采用 |Δg| 的分位数（50% 分位）
        # ---- Stage-3: centralized actor warmup & AMP ----
        self.warmup_actor_steps = 2000   # 中央 actor 更新前的预热步数
        self.update_alpha_interval = 4   # α 更新频率
        self.use_amp = True              # 是否启用自动混合精度
        # ---- Stage-4: mask curriculum ----
        self.mask_enable = True
        self.mask_topk_start = self.n_veh - 1  # 早期几乎不裁剪（除去自配）
        self.mask_topk_end = max(4, self.n_veh // 2)  # 后期收紧
        self.mask_tau_q_start = 0.2  # QoS 阈值：分位数从宽松到严格
        self.mask_tau_q_end = 0.4
        self.mask_warmup_episodes = 200  # 线性从 start→end
        # ---- Stage-4 (QoS) ----
        self.qos_enable = True                 # 开关
        self.qos_R_min_bpsHz = 0.15            # 每用户最低频谱效率阈值 R_min
        self.qos_D_max_s = 0.12                 # 每用户最大端到端时延阈值 D_max
        self.qos_penalty = 5.0                  # 违规惩罚强度（用于环境内的惩罚项）
        # ---- Stage-5: mutual selection pairing ----
        self.use_mutual_pairing = True        # True=启用互选配对(5B)，False=沿用稳定匹配(5A)
        self.mutual_q_quantile = 0.4          # |Δg| 自适应阈值的分位数（与稳定匹配保持一致）
        self.mutual_use_mask = True           # 是否与阶段4的可行性掩码联动
        # ---- Stage-6: 公平性与协同增强（奖励塑形精修）----
        # 小额“配对成功奖励”（仅引导，不喧宾夺主；随训练线性上调）
        self.COOP_PAIR_BONUS = 0.02

        # 测试阶段阈值（用于快速 PASS/FAIL 打印）
        self.test_target_jain = 0.90            # Jain 指数达标线
        self.test_max_global_drop = 0.03        # 全局回报允许的最大下降比例（≤3%）
        self.test_check_min_user_non_drop = True# 最小用户不下降的检查开关

        # 测试是否复用与训练一致的可行性掩码（建议 True）
        self.TEST_USE_MASK = True




# Instantiate the config
config = Config()

# 定义RIS优化频率变量
# 鉴于车辆每个Episode只移动一次，将此值设为100（即每Episode优化一次）可以极大加速训练且不损失性能。
K_STEPS_FOR_RIS_OPTIMIZATION = 100

# ################## ENVIRONMENT SETUP ######################
print('------------- lanes are -------------')
print('up_lanes :', config.up_lanes)
print('down_lanes :', config.down_lanes)
print('left_lanes :', config.left_lanes)
print('right_lanes :', config.right_lanes)
print('------------------------------------')

env = Environment.Environ(config.down_lanes, config.up_lanes, config.left_lanes, config.right_lanes,
                          config.width, config.height, config.n_veh, config.M, config.control_bit)
env.make_new_game()

# ===== 从 config.yaml 读取并覆盖关键参数=====
cfg_path = pathlib.Path('config.yaml')
if cfg_path.exists():
    try:
        with open(cfg_path, 'r', encoding='utf-8') as f:
            y = yaml.safe_load(f) or {}
        mec = y.get('mec', {})
        phy = y.get('phy', {})
        rew = y.get('reward', {})
        # —— 新增：读取等权/固定权重与归一化 EMA 系数 ——
        # 若你要严格“时延=能耗”，就把 sample 关掉，并在 YAML 里给 fixed=1.0
        config.w_d_fixed = float(rew.get('w_d_fixed', getattr(config, 'w_d_fixed', 1.0)))
        config.w_e_fixed = float(rew.get('w_e_fixed', getattr(config, 'w_e_fixed', 1.0)))

        # 若选择固定权重，则直接下发到环境；若仍保留 sample，则训练循环里可按 episode 重置 env.w_d/e
        if not bool(rew.get('sample', config.use_weight_sampling)):
            env.w_d = float(config.w_d_fixed)
            env.w_e = float(config.w_e_fixed)

        # 归一化 EMA 衰减系数（传给 Environment，用于 delay/energy 的在线标准化）
        setattr(env, 'reward_norm_beta', float(rew.get('norm_beta', getattr(env, 'reward_norm_beta', 0.9))))

        # 新增：从 YAML 顶层的 env: 节注入到达强度 rate
        env_cfg = y.get('env', {})
        env.rate = float(env_cfg.get('rate', env.rate))
        #从 YAML 注入 cpu_share_floor，未配置则保持 env 里的默认值
        setattr(env, 'cpu_share_floor', float(env_cfg.get('cpu_share_floor', getattr(env, 'cpu_share_floor', 0.10))))

        config.pairing_threshold_quantile = float(
            rew.get("pairing_threshold_quantile", config.pairing_threshold_quantile)
        )

        # MEC/CPU
        env.f_local_max = float(mec.get('f_local_max', env.f_local_max))
        env.f_edge_max = float(mec.get('f_edge_max', env.f_edge_max))
        env.cycles_per_bit = float(mec.get('cycles_per_bit', env.cycles_per_bit))
        env.k = float(mec.get('k_cpu', env.k))  # 动态功耗系数
        # 最小 CPU 占比下限（防饿死）
        env.cpu_share_floor = float(mec.get('cpu_share_floor', getattr(env, 'cpu_share_floor', 0.02)))

        # 物理层/功率总约束
        env.P_max = float(phy.get('P_max', env.P_max))
        env.bandwidth = float(phy.get('bandwidth_MHz', env.bandwidth))
        # 与带宽保持一致（MHz→Hz），并刷新噪声功率
        env.bandwidth_hz = env.bandwidth * 1e6
        env.noise_power = env.N0_W_per_Hz * env.bandwidth_hz
        env.channel_model = str(phy.get('channel_model', env.channel_model))
        env.fc_GHz = float(phy.get('fc_GHz', env.fc_GHz))


        # === NEW: robust bool parser (accepts "true"/"ture"/"yes"/1) ===
        def _as_bool(v):
            if isinstance(v, bool): return v
            if isinstance(v, (int, float)): return v != 0
            if isinstance(v, str):
                return v.strip().lower() in {"true", "ture", "yes", "y", "on", "1"}
            return bool(v)


        # --- Optional: action→power scaling (default=0.7 if not set) ---
        if "power_scale" in y:
            try:
                env.power_scale = float(y.get("power_scale"))
            except Exception:
                pass
        else:
            # ensure exists for downstream code
            env.power_scale = float(getattr(env, "power_scale", 0.7))

        # --- Soft power penalty (from YAML top-level) ---
        soft_keys = (
            "soft_power_beta", "soft_power_floor", "soft_power_margin",
            "soft_power_coef_linear", "soft_power_coef_quad", "soft_power_cap_per_step"
        )
        for k in soft_keys:
            if k in y:
                try:
                    setattr(config, k, float(y[k]))
                except Exception:
                    pass

        # tolerant boolean for the on/off switch (accepts the misspelling "ture")
        if "use_soft_power_penalty" in y:
            config.use_soft_power_penalty = _as_bool(y["use_soft_power_penalty"])

        # also mirror to env for logging/inspection (training logic still reads config)
        env.use_soft_power_penalty = bool(getattr(config, "use_soft_power_penalty", False))

        # 奖励权重策略
        config.use_weight_sampling = bool(rew.get('sample', config.use_weight_sampling))
        config.w_d_range = tuple(rew.get('w_d_range', list(config.w_d_range)))
        config.w_e_range = tuple(rew.get('w_e_range', list(config.w_e_range)))
        config.mask_enable = bool(rew.get("mask_enable", config.mask_enable))
        config.mask_topk_start = int(rew.get("mask_topk_start", config.mask_topk_start))
        config.mask_topk_end = int(rew.get("mask_topk_end", config.mask_topk_end))
        config.mask_tau_q_start = float(rew.get("mask_tau_q_start", config.mask_tau_q_start))
        config.mask_tau_q_end = float(rew.get("mask_tau_q_end", config.mask_tau_q_end))
        config.mask_warmup_episodes = int(rew.get("mask_warmup_episodes", config.mask_warmup_episodes))

        config.gumbel_tau_start = float(rew.get("gumbel_tau_start", config.gumbel_tau_start))
        config.gumbel_tau_end = float(rew.get("gumbel_tau_end", config.gumbel_tau_end))
        config.gumbel_hard = bool(rew.get("gumbel_hard", config.gumbel_hard))
        # ---- Stage-5/6：互选配对 + 公平/协同 奖励 + QoS + 环境刷新频率 ----
        config.use_mutual_pairing = bool(y.get("use_mutual_pairing", config.use_mutual_pairing))
        config.mutual_q_quantile = float(y.get("mutual_q_quantile", config.mutual_q_quantile))
        config.mutual_use_mask = bool(y.get("mutual_use_mask", config.mutual_use_mask))
        config.use_mwm_completion = bool(y.get("use_mwm_completion", getattr(config, "use_mwm_completion", True)))
        config.mwm_allow_singles = bool(y.get("mwm_allow_singles", getattr(config, "mwm_allow_singles", True)))
        # —— MWM 主配对（新增）——
        config.use_mwm_primary = bool(y.get("use_mwm_primary", getattr(config, "use_mwm_primary", True)))
        config.mwm_accept_quantile = float(y.get("mwm_accept_quantile", getattr(config, "mwm_accept_quantile", 0.20)))
        config.min_pair_target = int(y.get("min_pair_target", getattr(config, "min_pair_target", config.n_veh // 4)))
        config.mwm_backoff_rounds = int(y.get("mwm_backoff_rounds", getattr(config, "mwm_backoff_rounds", 3)))
        config.mwm_accept_q_step = float(y.get("mwm_accept_q_step", getattr(config, "mwm_accept_q_step", 0.05)))
        config.legacy_pairing_backoff = bool(
            y.get("legacy_pairing_backoff", getattr(config, "legacy_pairing_backoff", False)))

        config.COOP_PAIR_BONUS = float(y.get("COOP_PAIR_BONUS", getattr(config, "COOP_PAIR_BONUS", 0.0)))
        # 如果你有公平项/方差项（可选），同样从顶层读（没有就保持默认）
        config.FAIRNESS_LAMBDA = float(y.get("FAIRNESS_LAMBDA", getattr(config, "FAIRNESS_LAMBDA", 0.0)))
        config.VARIANCE_PENALTY = float(y.get("VARIANCE_PENALTY", getattr(config, "VARIANCE_PENALTY", 0.0)))
        # === 新增：re-shaping 总开关（默认 False）===
        config.reward_use_shaping = bool(y.get("reward_use_shaping", False))

        # QoS（你原先只从 reward 下读了 qos_penalty，这里再加对顶层兜底）
        config.qos_enable = bool(y.get("qos_enable", getattr(config, "qos_enable", True)))
        config.qos_penalty = float(y.get("qos_penalty", getattr(config, "qos_penalty", 5.0)))

        # 环境刷新频率（默认 5）
        config.env_refresh_every = int(y.get("env_refresh_every", getattr(config, "env_refresh_every", 5)))
        # --- 新增：把 gumbel_tau_warmup_episodes 映射为 gumbel_anneal_steps ---
        warm = (y.get('gumbel_tau_warmup_episodes', None))
        if warm is not None:
            # 支持“比例（0~1）”或“绝对episode数”
            if isinstance(warm, (int, float)) and warm <= 1.0:
                config.gumbel_anneal_steps = int(float(warm) * config.n_episode * config.n_step_per_episode)
            else:
                # 解释为绝对 episode 数
                config.gumbel_anneal_steps = int(float(warm) * config.n_step_per_episode)
        # === 顶层训练/算法超参接入（仅当 YAML 提供时覆盖） ===
        # 训练规模
        config.n_episode           = int(y.get('total_episodes',       config.n_episode))
        config.n_step_per_episode  = int(y.get('n_step_per_episode',   config.n_step_per_episode))
        config.batch_size          = int(y.get('batch_size',           config.batch_size))
        config.memory_size         = int(y.get('replay_size',          getattr(config, 'memory_size', 1_000_000)))

        # SAC 关键超参
        config.gamma               = float(y.get('gamma',               config.gamma))
        # tau_soft_update 也接收别名 'tau'
        config.tau                 = float(y.get('tau_soft_update',     y.get('tau', config.tau)))
        config.alpha               = float(y.get('lr_actor',            config.alpha))   # Actor LR
        config.beta                = float(y.get('lr_critic',           config.beta))    # Critic LR
        config.update_actor_interval = int(y.get('update_actor_interval', config.update_actor_interval))
        config.warmup_actor_steps    = int(y.get('warmup_actor_steps',    getattr(config, 'warmup_actor_steps', 2000)))
        config.update_alpha_interval = int(y.get('update_alpha_interval', getattr(config, 'update_alpha_interval', 4)))
        config.use_amp               = bool(y.get('use_amp',              getattr(config, 'use_amp', True)))

        # Gumbel 退火端点允许从顶层覆盖（reward 下已有，就以顶层为准）
        if 'gumbel_tau_start' in y:
            config.gumbel_tau_start = float(y['gumbel_tau_start'])
        if 'gumbel_tau_end' in y:
            config.gumbel_tau_end   = float(y['gumbel_tau_end'])
        if 'gumbel_hard_train' in y:
            config.gumbel_hard      = bool(y['gumbel_hard_train'])

        # 互选-掩码联动的“延后启用”阈（供你在互选阶段按 episode 分段用）
        # 若 YAML 未提供，则默认 300
        setattr(config, 'mutual_mask_after', int(y.get('mutual_mask_after', 300)))
        # —— V3 打分与稳定性（允许 YAML 覆盖；否则走默认）——
        setattr(config, 'score_w_delta_db', float(y.get('score_w_delta_db', 1.0)))
        setattr(config, 'score_w_history', float(y.get('score_w_history', 0.3)))
        setattr(config, 'stickiness_tau_db', float(y.get('stickiness_tau_db', 1.0)))
        setattr(config, 'pair_hist_decay', float(y.get('pair_hist_decay', 0.97)))
        setattr(config, 'pair_unstick_prob', float(y.get('pair_unstick_prob', 0.02)))

        # —— 质量过滤（在评分阶段做绝对门限；默认关闭）——
        setattr(config, 'abs_gain_min_db', float(y.get('abs_gain_min_db', float('-inf'))))

        # —— 受限回退策略（Backoff）——
        setattr(config, 'max_backoff_rounds', int(y.get('max_backoff_rounds', 3)))
        setattr(config, 'relax_q_step', float(y.get('relax_q_step', 0.02)))
        setattr(config, 'relax_topk_step', int(y.get('relax_topk_step', 1)))
        setattr(config, 'relax_tau_factor_per_round', float(y.get('relax_tau_factor_per_round', 0.95)))
        # —— 兼容旧键名：如果 YAML 只给了 max_backoff_rounds，则同步到 mwm_backoff_rounds ——
        if ('max_backoff_rounds' in y) and ('mwm_backoff_rounds' not in y):
            config.mwm_backoff_rounds = int(y.get('max_backoff_rounds', getattr(config, 'mwm_backoff_rounds', 3)))

        # 目标熵缩放（B4 会读取这两个比例并下发给 global_agent）
        setattr(config, 'target_entropy_cont_scale', float(y.get('target_entropy_cont_scale', 1.0)))
        setattr(config, 'target_entropy_disc_scale', float(y.get('target_entropy_disc_scale', 1.0)))
        # —— Freeze-in-episode 配置（可不写，默认生效）——
        config.freeze_group_in_episode  = bool(y.get("freeze_group_in_episode", True))
        config.freeze_recalc_every      = int(y.get("freeze_recalc_every", 0))
        config.freeze_unstick_prob      = float(y.get("freeze_unstick_prob", 0.0))
        config.freeze_reward_drop_ratio = float(y.get("freeze_reward_drop_ratio", 0.05))





    except Exception as e:
        print('[WARN] YAML load failed:', e)
    # ---- Sync QoS & soft-power flags into env ----
    env.qos_enable = bool(getattr(config, "qos_enable", True))
    env.R_min_bpsHz = float(getattr(config, "qos_R_min_bpsHz", env.R_min_bpsHz))
    env.D_max_s = float(getattr(config, "qos_D_max_s", env.D_max_s))
    env.qos_penalty = float(getattr(config, "qos_penalty", 1.5))
    env.use_soft_power_penalty = bool(getattr(config, "use_soft_power_penalty", False))

# # —— Day 3–5：环境物理开关（可按需要修改）——
# # 信道模型：可切 "free" | "3gpp_umi" | "3gpp_uma"
# env.channel_model = "free"     # 先保留“free”，做A/B对照时改为 "3gpp_umi" 或 "3gpp_uma"
#
# # 载频（GHz）：进入 38.901 路损公式的 fc 项
# env.fc_GHz = 3.5               # 例如 3.5GHz；你也可以设为 2.6 / 28 等
#
# # 发射功率总上限：每个用户两路功率的和 ≤ P_max
# # 注意：这要求你已在 Environment.step() 做了“功率投影”，用到 env.P_max
# env.P_max = 1.0                # 单位 W（后续要做真实功率量纲时用它）

# 带宽：保持你当前用法（MHz）。data_t = rate(bit/s/Hz) * time * bandwidth(MHz) * 1000
# 如需做带宽消融，改这里即可
# env.bandwidth = 1.0          # 默认已在 Environment 里设为 1.0（MHz），一般不用重复设
# ---- (可选) 若你不用采样而用固定值，可在这里直接固定 ----
if not config.use_weight_sampling:
    env.w_d = config.w_d_fixed
    env.w_e = config.w_e_fixed
# —— 同步 reward 配置到环境 ——（紧跟在固定权重分支之后）
env.sample_weights = bool(config.use_weight_sampling)
if hasattr(config, "w_d_range"):  # 让环境知道区间，采样才有来源
    env.w_d_range = tuple(config.w_d_range)
if hasattr(config, "w_e_range"):
    env.w_e_range = tuple(config.w_e_range)


# === 统一：创建本次运行目录（含 artifacts 子目录） ===
run_dir = make_run_dir(args.exp, args.seed)

# === 统一：把“最终生效配置”写盘（便于复现） ===
try:
    eff = dict(
        exp=args.exp, seed=args.seed, notes=args.notes, git=get_git_hash(),
        env=dict(
            n_veh=config.n_veh,
            bandwidth_MHz=getattr(env, "bandwidth", None),
            channel_model=getattr(env, "channel_model", None),
            fc_GHz=getattr(env, "fc_GHz", None),
            P_max=getattr(env, "P_max", None)
        ),
        reward=dict(
            use_weight_sampling=config.use_weight_sampling,
            w_d_fixed=getattr(config, "w_d_fixed", None),
            w_e_fixed=getattr(config, "w_e_fixed", None),
            w_d_range=list(getattr(config, "w_d_range", [])),
            w_e_range=list(getattr(config, "w_e_range", []))
        )
    )
    with open(run_dir / "effective_config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(eff, f, sort_keys=False, allow_unicode=True)
except Exception as _e:
    print("[WARN] dump effective_config.yaml failed:", _e)

# === 统一：初始化 TensorBoard（若要求记录） ===
if args.log == "tb":
    try:
        writer = SummaryWriter(log_dir=str(run_dir / "tb"))
        print("[TB] logdir:", run_dir / "tb")
    except Exception as _e:
        print("[WARN] TensorBoard init failed:", _e)



def marl_get_state(idx):
    """ Get state from the environment """
    state_list = []
    state_list.append(env.DataBuf[idx] / 10)
    state_list.append(env.data_t[idx] / 10)
    state_list.append(env.data_p[idx] / 10)
    state_list.append(env.over_data[idx] / 10)
    state_list.append(env.vehicle_rate[idx] / 20)
    return state_list


# def _adaptive_threshold_from_delta_g(channel_gains, q=0.6):
#     g = np.asarray(channel_gains, dtype=float)
#     if g.size < 2:
#         return 0.0
#     order = np.argsort(g)  # 弱→强
#     n = g.size
#     weak_idx = order[: n // 2]
#     strong_idx = order[n // 2:]
#     if weak_idx.size == 0 or strong_idx.size == 0:
#         return 0.0
#     diffs = np.abs(g[strong_idx][:, None] - g[weak_idx][None, :]).ravel()
#     return float(np.quantile(diffs, q))  # 自适应阈值（分位数）
def _adaptive_threshold_from_delta_g(channel_gains, q=0.6):
    g = np.asarray(channel_gains, dtype=float)
    if g.size < 2:
        return 0.0
    eps = 1e-15
    g_db = 10.0 * np.log10(np.maximum(g, eps))  # 线性域 -> dB 域
    order = np.argsort(g_db)  # 弱→强（按 dB）
    n = g_db.size
    weak_idx = order[: n // 2]
    strong_idx = order[n // 2:]
    if weak_idx.size == 0 or strong_idx.size == 0:
        return 0.0
    diffs = np.abs(g_db[strong_idx][:, None] - g_db[weak_idx][None, :]).ravel()
    return float(np.quantile(diffs, q))


def _qos_pair_feasible(i, j, g, p01, noise_power, P_max, R_min):
    """
    近似评估 (i,j) 作为 NOMA 对时，两端频谱效率是否都 >= R_min。
    g: 增益数组；p01: [0,1] 归一化卸载功率；noise_power/P_max: 来自 env
    """
    import numpy as np
    # 归一化卸载功率 -> 物理功率
    pi = float(p01[i]) * float(P_max)
    pj = float(p01[j]) * float(P_max)
    gi, gj = float(g[i]), float(g[j])

    # 近/远用户按增益决定
    if gi >= gj:
        g_near, g_far = gi, gj; p_near, p_far = pi, pj
    else:
        g_near, g_far = gj, gi; p_near, p_far = pj, pi

    # 远端：干扰=近端功率 * g_far
    sinr_far  = (p_far * g_far) / (p_near * g_far + float(noise_power) + 1e-12)
    r_far  = np.log2(1.0 + max(0.0, sinr_far))
    # 近端：SIC后仅受噪声
    sinr_near = (p_near * g_near) / (float(noise_power) + 1e-12)
    r_near = np.log2(1.0 + max(0.0, sinr_near))
    return (r_far >= float(R_min)) and (r_near >= float(R_min))



def _build_preferences(channel_gains, offload_power, strong_idx, weak_idx,
                       intent_probs=None, lam=0.0,
                       qos_R_min=None, noise_power=None, P_max=None):
    g = np.asarray(channel_gains, dtype=float)
    p = np.asarray(offload_power, dtype=float)
    eps = 1e-15
    g_db = 10.0 * np.log10(np.maximum(g, eps))  # 统一用 dB 计算差值
    S, W = len(strong_idx), len(weak_idx)
    score = np.zeros((S, W), dtype=float)
    for a, i in enumerate(strong_idx):
        for b, j in enumerate(weak_idx):

            # (NEW) QoS 预判：任一端 < R_min 则直接置 -inf
            if (qos_R_min is not None) and (noise_power is not None) and (P_max is not None):
                ok = _qos_pair_feasible(i, j, g, p, noise_power, P_max, qos_R_min)
                if not ok:
                    score[a, b] = -np.inf
                    continue

            g = np.asarray(channel_gains, dtype=float)
            p = np.asarray(offload_power, dtype=float)


            # 先验分
            delta_db = abs(g_db[i] - g_db[j])
            prior_score = delta_db * (p[i] + p[j] + 1e-6)

            # 策略意图分
            learn_score = 0.0
            if intent_probs is not None and lam > 0:
                learn_score = float(intent_probs[i, j]) + float(intent_probs[j, i])
            score[a, b] = prior_score + lam * learn_score


    # 按 score 生成偏好顺序 (不变)
    prefs_strong = {i: [weak_idx[b] for b in np.argsort(-score[a])]


                    for a, i in enumerate(strong_idx)}
    prefs_weak = {j: [strong_idx[a] for a in np.argsort(-score[:, b])]
                  for b, j in enumerate(weak_idx)}
    return prefs_strong, prefs_weak


def _gale_shapley(strong_idx, weak_idx, prefs_strong, prefs_weak):
    free = list(strong_idx)
    next_choice_ptr = {i: 0 for i in strong_idx}
    engaged_to = {}  # weak -> strong
    match = {}  # strong -> weak
    while free:
        i = free.pop(0)
        prefs_i = prefs_strong.get(i, [])
        if next_choice_ptr[i] >= len(prefs_i):
            continue
        j = prefs_i[next_choice_ptr[i]]
        next_choice_ptr[i] += 1
        if j not in engaged_to:
            engaged_to[j] = i;
            match[i] = j
        else:
            i_old = engaged_to[j]
            better = i if prefs_weak[j].index(i) < prefs_weak[j].index(i_old) else i_old
            worse = i_old if better == i else i
            engaged_to[j] = better;
            match[better] = j
            if worse in match: del match[worse]
            if worse in strong_idx: free.append(worse)
    return match


def pairing_stable(channel_gains, offload_power, q=0.6, min_delta=None,intent_probs=None, lam=0.0,qos_R_min=None, noise_power=None, P_max=None):
    """
    返回 NOMA/OMA 分组：如 [[i,j],[k],[m,n],...]
    - channel_gains: 长度=n_veh 的当前信道增益
    - offload_power: 已映射到 [0,1] 的“卸载发射功率”，长度=n_veh
    - q: 分位数；min_delta: 如指定则用硬阈值
    """
    g = np.asarray(channel_gains, dtype=float)
    n = g.size
    eps = 1e-15
    g_db = 10.0 * np.log10(np.maximum(g, eps))  # 线性→dB
    order = np.argsort(g_db)  # 弱→强（按 dB）
    weak_idx = list(order[: n // 2])
    strong_idx = list(order[n // 2:])

    prefs_s, prefs_w = _build_preferences(
        g, offload_power, strong_idx, weak_idx, intent_probs, lam,
        qos_R_min=qos_R_min, noise_power=noise_power, P_max=P_max
    )
    match = _gale_shapley(strong_idx, weak_idx, prefs_s, prefs_w)

    tau = _adaptive_threshold_from_delta_g(g, q) if min_delta is None else float(min_delta)
    paired = set();
    groups = []
    for i_strong, j_weak in match.items():
        if abs(g_db[i_strong] - g_db[j_weak]) >= tau:  # 用 dB 差比较 τ
            groups.append(sorted([i_strong, j_weak]))
            paired.add(i_strong);
            paired.add(j_weak)
    for u in range(n):
        if u not in paired:
            groups.append([u])  # 余者走 OMA
    groups = sorted(groups, key=lambda x: (len(x), x[0], x[-1]))
    return groups

def pairing_mutual(channel_gains, intent_probs, q=0.6, feasible_mask=None, allow_refill: bool = True):
    """
    互选配对（O(N)一趟）：
      - 每个 i 只在可行候选里挑概率最大的 j（j != i）；
      - 若 i 选择 j 且 j 也选择 i，则组成一对 [min(i,j), max(i,j)]；
      - 其余未成对的用户单播；
      - 仅保留 |Δg| >= 自适应阈值 tau 的二人组（与稳定匹配保持同一阈值逻辑）。
    参数：
      channel_gains: 长度 N 的信道增益数组
      intent_probs : [N, N]、对角为0
      q            : 自适应阈值分位数
      feasible_mask: [N, N] 的(0/1)可行性掩码；None 表示不过滤
    返回：
      分组列表，如 [[i,j], [k], [m,n], ...]
    """
    import numpy as np

    probs = np.asarray(intent_probs, dtype=float)
    N = probs.shape[0]
    # 自适应阈值（与稳定匹配一致）
    g = np.asarray(channel_gains, dtype=float)
    eps = 1e-15
    g_db = 10.0 * np.log10(np.maximum(g, eps))  # 线性→dB
    tau = _adaptive_threshold_from_delta_g(g, q)  # τ 已是 dB 口径

    # 1) 屏蔽不可行候选与自配
    mask = np.ones_like(probs, dtype=bool)
    np.fill_diagonal(mask, False)  # 禁止自配
    if feasible_mask is not None:
        mask &= (np.asarray(feasible_mask) > 0)
    safe_probs = probs.copy()
    safe_probs[~mask] = -np.inf  # 不可选 → 负无穷

    # 2) 每行取“首选”
    top_choice = np.full(N, -1, dtype=int)
    for i in range(N):
        j = int(np.argmax(safe_probs[i]))
        if np.isneginf(safe_probs[i, j]):   # 全不可选
            top_choice[i] = -1
        else:
            # |Δg| 阈值：若不达标则视为不可选
            if abs(g_db[i] - g_db[j]) < tau:  # 用 dB 差来判阈值
                top_choice[i] = -1
            else:
                top_choice[i] = j

    # 3) 互选成对
    used = np.zeros(N, dtype=bool)
    # --- 互选扫描（只负责找所有互选成功的二人组；不在循环内追加单播/补配）---
    groups = []
    used = used.copy()  # 如果上面已有 used，这里确保是本层可写副本

    for i in range(N):
        if used[i]:
            continue
        j = top_choice[i]
        if j >= 0 and (top_choice[j] == i) and (not used[j]) and (i != j):
            a, b = int(i), int(j)  # 统一为 python int
            if a > b:
                a, b = b, a  # 规范顺序 (小的在前)
            groups.append([a, b])
            used[i] = used[j] = True
        else:
            # 不在循环内做任何“收尾”工作（避免重复补单播/补配）
            pass

    # --- 循环结束后：一次性做单播收尾 + 二次补配 ---
    # 4) 把余下未配对的用户设为单播（只执行一次）
    for u in range(N):
        if not used[u]:
            groups.append([int(u)])

    # 4.1) 对“单播子集”做一次放宽阈值的二次互选补配（不破坏已成对）
    # 只允许顶层调用执行一次补配；子调用不再二次补配，避免递归
    if allow_refill:
        singles = [u for u in range(N) if not used[u]]
        if len(singles) >= 2:
            # 截取子概率/子信道/子掩码
            sub_probs = probs[np.ix_(singles, singles)]
            sub_gains = g[singles]
            sub_mask = None
            if feasible_mask is not None:
                sub_mask = np.asarray(feasible_mask)[np.ix_(singles, singles)]

            # 阈值适当放宽：例如 q_lo = max(0.10, q - 0.20)
            q_lo = max(0.10, float(q) - 0.20)

            # 在“单播子集”上再跑一次互选，但禁止其内部再次补配（allow_refill=False）
            sub_groups = pairing_mutual(
                sub_gains,
                intent_probs=sub_probs,
                q=q_lo,
                feasible_mask=sub_mask,
                allow_refill=False
            )

            # 合并：优先保留“第一次互选”得到的配对，再加“补配”新配对，最后余者仍为单播
            used_global = set()
            merged = []

            # 先放入第一次互选得到的所有 2人组
            for g0 in groups:
                if len(g0) == 2:
                    a, b = int(g0[0]), int(g0[1])
                    if a > b:
                        a, b = b, a
                    merged.append([a, b])
                    used_global.update([a, b])

            # 把子问题中的配对映射回全局索引；只合入真正“新”的二人组
            for g1 in sub_groups:
                if len(g1) == 2:
                    a = int(singles[int(g1[0])])
                    b = int(singles[int(g1[1])])
                    if a > b:
                        a, b = b, a
                    if (a not in used_global) and (b not in used_global):
                        merged.append([a, b])
                        used_global.update([a, b])

            # 把仍未被任何配对覆盖的代理补成单播
            for u in range(N):
                if u not in used_global:
                    merged.append([int(u)])

            groups = merged  # 用合并后的结果替换

    # --- 可选：最终一次去重 + 排序，确保输出稳定 ---
    seen, dedup = set(), []
    for gpair in groups:
        tup = tuple(gpair)  # gpair 长度为 1 或 2
        if tup not in seen:
            seen.add(tup)
            dedup.append([int(x) for x in gpair])  # 再保险：统一为 int
    groups = sorted(dedup, key=lambda x: (len(x), x[0], x[-1]))

    return groups


marl_n_input = len(marl_get_state(0))
marl_n_continuous_actions = 2  # 两个连续功率
marl_n_discrete_actions = config.n_veh  # 离散“意图概率向量”的长度=N（车辆数）
marl_n_output = marl_n_continuous_actions + marl_n_discrete_actions

# ################## AGENT INITIALIZATION ######################
agents = []
for index_agent in range(config.n_veh):
    print("Initializing agent", index_agent)
    # Pass both continuous and discrete action dimensions to the agent
    agent_i = Agent(config.alpha, config.beta, marl_n_input, config.tau,
                    n_continuous_actions=marl_n_continuous_actions,
                    n_discrete_actions=marl_n_discrete_actions,
                    gamma=config.gamma,
                    c1=config.C_fc1_dims, c2=config.C_fc2_dims, c3=config.C_fc3_dims,
                    a1=config.A_fc1_dims, a2=config.A_fc2_dims,
                    batch_size=config.batch_size, n_agents=config.n_veh,
                    agent_name=index_agent, noise=config.noise,chkpt_root=str(run_dir))
    agents.append(agent_i)
# ---- 下发 Gumbel 初值到每个 actor ----
# --- Apply Gumbel straight-through policy based on the policy's current temperature ---
for ag in agents:
    # 从策略网络中读取当前温度（避免依赖外部变量）
    try:
        cur_tau_val = float(ag.policy.tau.detach().mean().item())
    except Exception:
        # 若策略里没有 tau（极少见），退回到一个安全默认值
        cur_tau_val = float(getattr(config, "gumbel_tau_default", 1.0))

    # 遵从 YAML 开关；兼容两种键名（你可能在 YAML 里写的是 gumbel_hard_train）
    want_hard = bool(getattr(config, "gumbel_hard_train", False) or
                     getattr(config, "gumbel_hard", False))

    # 阈值降到 0.3：温度足够低时才启用 straight-through（更稳）
    ag.policy.gumbel_hard = bool(want_hard and (cur_tau_val <= 0.3))




memory = ReplayBuffer(config.memory_size, marl_n_input, marl_n_output, config.n_veh)
# ################## 验收断言 ######################
assert marl_n_output == config.n_veh + 2, "Action dimension mismatch!"
assert memory.action_memory.shape[1] == config.n_veh * (config.n_veh + 2), "Memory action shape mismatch!"

print("Initializing Global critic ...")
global_agent = Global_SAC_Critic(
    config.beta, marl_n_input, config.tau, marl_n_output, config.gamma,
    config.C_fc1_dims, config.C_fc2_dims, config.C_fc3_dims,
    config.batch_size, config.n_veh, config.update_actor_interval,
    entropy_scale=getattr(config, "entropy_scale", 0.5),  # ← 默认给 0.5 的“刹车”
    separate_alpha=True,
    chkpt_root=str(run_dir),
    use_amp=config.use_amp,
    warmup_actor_steps=config.warmup_actor_steps,
    update_alpha_interval=config.update_alpha_interval
)

# === B4: override target entropy from YAML (after Global_SAC_Critic is created) ===
# 保持和类里默认的系数一致：连续 1.0、离散 0.7；你的 YAML 只做“比例”缩放
cont_scale = float(getattr(config, "target_entropy_cont_scale", 1.0))    # 1.0 为“与默认相同”
disc_scale = float(getattr(config, "target_entropy_disc_scale", 1.0))    # 1.0 为“与默认相同”
n_cont = 2                                   # 每个 agent 的连续维度=2（功率2维），与你当前架构一致
K = max(2, config.n_veh)                     # 离散类别数的 logK，避免 log(1)

# 分离 α 的情形（你现在 separate_alpha=True）
global_agent.target_entropy_cont = -(1.0 * cont_scale) * (config.n_veh * n_cont)
global_agent.target_entropy_disc = -(0.7 * disc_scale) * (config.n_veh * math.log(K))

# 若以后你把 separate_alpha 关掉，也可以用下面一行把总目标熵设置成两部分之和：
# global_agent.target_entropy = global_agent.target_entropy_cont + global_agent.target_entropy_disc



# Running statistics for reward normalization
running_reward_mean = 0.0
running_reward_std = 1.0

# ################## TRAINING LOOP ######################
if config.IS_TRAIN:
    record_reward_ = np.zeros([config.n_veh, config.n_episode])
    record_critics_loss_ = np.zeros([config.n_veh + 1, config.n_episode])
    record_global_reward_average = []
    record_global_reward_episode = []  # 新增：按 episode 聚合后的全局奖励
    Sum_Power = []
    Sum_Power_local = []
    Sum_Power_offload = []


    # --- Soft power penalty baseline (EMA) ---
    ema_power = None  # episode 内自适应基线

    record_global_reward_raw_average = []  # 未加惩罚的原始全局回报（用于调参可读）
    # ---- Stage-0 episode-level metrics ----
    episode_min_user = []
    episode_var_user = []
    episode_jain = []
    episode_elapsed_sec = []
    episode_delay_mean = []
    episode_energy_mean = []
    # === A4 可见化：回合级功率与 Gumbel 状态、权重（一次性定义，贯穿全训练） ===
    episode_power_total = []
    episode_power_local = []
    episode_power_offload = []
    episode_gumbel_tau = []
    episode_gumbel_hard = []
    episode_w_d = []
    episode_w_e = []

    # Store vehicle positions for plotting
    vehicle_positions = {i: {'x': [], 'y': []} for i in range(config.n_veh)}
    LOG_EVERY = 10  # 每10个episode记录一次日志
    for i_episode in range(config.n_episode):
        done = False
        t0 = time.time()  # ← 新增：本回合计时起点
        print(f"-------------------------------- Episode: {i_episode} -------------------------------------------")
        # --- reset per-episode accumulators ---
        ep_delay_sum = 0.0
        ep_energy_sum = 0.0
        ep_steps = 0
        # === 新增：分项时延/状态累计（按步累加，回合末取均值） ===
        ep_delay_local_sum = 0.0
        ep_delay_edge_q_sum = 0.0
        ep_delay_edge_c_sum = 0.0
        ep_delay_tx_sum = 0.0

        ep_backlog_kbit_sum = 0.0
        ep_mec_util_sum = 0.0
        ep_local_util_sum = 0.0
        ep_qos_viol_sum = 0.0  # 若你的 env 已有 last_qos_violation

        ep_off_kbit_sum = 0.0
        ep_local_kbit_sum = 0.0
        # —— 本回合的配对统计（用于日志） ——
        ep_pairs = 0
        ep_singles = 0
        ep_pair_calls = 0
        ep_tau_sum = 0.0
        ep_mask_zero_ratio_sum = 0.0

        refresh_every = int(getattr(config, "env_refresh_every", 5))
        if i_episode % max(1, refresh_every) == 0:
            env.renew_positions()
            env.compute_parms()
            for i in range(config.n_veh):
                vehicle_positions[i]['x'].append(env.vehicles[i].position[0])
                vehicle_positions[i]['y'].append(env.vehicles[i].position[1])

        marl_state_old_all = [marl_get_state(i) for i in range(config.n_veh)]

        Power = []
        Power_local = []
        Power_offload = []
        # === 缓存“可行性掩码”和阈值（只在信道更新步重算） ===
        last_mask_mat = None              # numpy 的 N×N 掩码
        last_tau_now = None
        last_K_now = None
        last_q_now = None
        last_feasible_mask_gpu = None     # torch.Tensor（和策略同 device）
        # === [NOMA-V3] episode 级缓存：配对热度/上一步成对/连续未配对计数 ===
        pair_affinity_hist = np.zeros((config.n_veh, config.n_veh), dtype=np.float32)  # (N,N) 对称，主对角为 0
        prev_pairs = set()  # 存 (min(i,j), max(i,j))
        unpaired_streak = np.zeros((config.n_veh,), dtype=np.int32)
        # —— Episode-level freeze of NOMA grouping + three safeties ——
        freeze_group_in_episode   = bool(getattr(config, "freeze_group_in_episode", True))
        freeze_recalc_every       = int(getattr(config, "freeze_recalc_every", 0))     # 0=整回合冻结
        freeze_unstick_prob       = float(getattr(config, "freeze_unstick_prob", 0.0)) # 小概率解冻
        freeze_reward_drop_ratio  = float(getattr(config, "freeze_reward_drop_ratio", 0.05))  # env.global 跌幅阈值

        episode_groups    = None   # 本回合冻结使用的分组
        last_env_global   = None   # 最近一步 env.global（用于触发式解冻）
        ep_env_best       = -1e18  # 本回合历史最好 env.global
        unstick_used_flag = False  # 本回合是否已经触发过一次“掉幅解冻”



        for i_step in range(config.n_step_per_episode):

            # --- 1. 按频率优化RIS并更新信道状态 ---
            if i_step % K_STEPS_FOR_RIS_OPTIMIZATION == 0:
                env.optimize_phase_shift()  # 调用BCD算法优化RIS
                env.update_channel_gains()  # 基于新的RIS配置，更新全局信道增益

            # --- 2. 智能体观察最新的环境状态 ---
            current_channel_gains = env.get_channel_gains()
            marl_state_old_all = [marl_get_state(i) for i in range(config.n_veh)]

            # ==== Stage-4：构造“可行性掩码”(N x N) —— 仅在信道更新步重算，其它步复用 ====
            feasible_mask_gpu = None
            mask_mat = None
            if config.mask_enable:
                need_recalc = (i_step % K_STEPS_FOR_RIS_OPTIMIZATION == 0) or (last_mask_mat is None)

                if need_recalc:
                    # 课程：K 与 τ 从“宽松→收紧”
                    prog = min(1.0, i_episode / max(1, config.mask_warmup_episodes))
                    K_now = _anneal_topk(
                        i_episode, config.n_veh,
                        config.mask_topk_start, config.mask_topk_end,
                        config.mask_warmup_episodes
                    )
                    q_now = float(
                        config.mask_tau_q_start
                        + (config.mask_tau_q_end - config.mask_tau_q_start) * prog
                    )
                    # 自适应阈值 τ（基于 |Δg| 的分位数）
                    tau_now = _adaptive_threshold_from_delta_g(current_channel_gains, q_now)

                    # 基于 |Δg| + Top-K 生成 N×N 掩码矩阵（1=可选, 0=屏蔽）
                    mask_mat = _build_feasible_mask_from_delta_g(current_channel_gains, tau_now, K_now)

                    # 缓存本次结果
                    last_mask_mat = mask_mat
                    last_tau_now = tau_now
                    last_K_now = K_now
                    last_q_now = q_now

                    # （降频打印，避免刷屏）
                    if (i_episode < 5) or (i_episode % 10 == 0 and i_step % 20 == 0):
                        print(f"[MASK] ep={i_episode:4d}  K={K_now}  q={q_now:.2f}  tau={tau_now:.3e}")

                    # 构造/更新 GPU 端 tensor
                    last_feasible_mask_gpu = torch.tensor(
                        last_mask_mat, dtype=torch.float32, device=agents[0].policy.device
                    )
                else:
                    # 直接复用上一次的 GPU 掩码
                    if last_feasible_mask_gpu is None:
                        # 极端容错：若没缓存到，就临时构造一次（理论上不会走到这里）
                        last_feasible_mask_gpu = torch.tensor(
                            last_mask_mat, dtype=torch.float32, device=agents[0].policy.device
                        )

                feasible_mask_gpu = last_feasible_mask_gpu

                # --- NEW: 统计本步掩码零元比例（忽略对角线） ---
                if mask_mat is not None:
                    m = mask_mat.copy().astype(np.float32)
                    # 忽略对角线（自配），对角线视作“可用”
                    np.fill_diagonal(m, 1.0)
                    zero_ratio_step = float(1.0 - m.mean())
                    ep_mask_zero_ratio_sum += zero_ratio_step
                    # 这里只统计一次即可；若你“每步只构一次 mask”，则沿用 ep_pair_calls 作为分母是合理的

            # --- 3. 智能体基于新状态决策 ---
            marl_power_actions = []
            intent_probs_list = []
            intent_onehot_list = []
            for i in range(config.n_veh):
                # 取第 i 个智能体对应的一行 mask（形状 [N]）；如未启用则为 None
                mask_row = feasible_mask_gpu[i] if feasible_mask_gpu is not None else None

                # 把 mask 传给 choose_action（sac_agent.py 已按阶段4改为支持 mask 参数）
                power_action, intent_probs, intent_onehot = agents[i].choose_action(
                    marl_state_old_all[i], mask=mask_row
                )
                marl_power_actions.append(power_action)
                intent_probs_list.append(intent_probs)
                intent_onehot_list.append(intent_onehot)

            # 将概率列表组装成N x N矩阵
            intent_probs_mat = np.stack(intent_probs_list, axis=0)
            np.fill_diagonal(intent_probs_mat, 0)  # 自己和自己配对的概率置为0

            # 也保留 one-hot 版本，供 Phase-2 写入回放
            intent_onehot_mat = np.stack(intent_onehot_list, axis=0)

            # --- 3.1 生成“用于配对评分”的卸载功率（从[-1,1]映射到[0,1]）
            offload_power_for_pairing = np.zeros(config.n_veh, dtype=float)
            for i in range(config.n_veh):
                clipped = np.clip(marl_power_actions[i], -0.999, 0.999)
                offload_power_for_pairing[i] = (clipped[0] + 1) / 2.0  # 与 env.step 一致

            # ===== [NOMA-V3] 分组主逻辑：黏滞 + 互选 + 贪心 + 多轮受限回退（含质量过滤） =====

            # 1) 热度衰减 + 偶发解锁，防止“粘死”
            gamma_decay = float(getattr(config, "pair_hist_decay", 0.97))  # 每步乘 γ
            eps_unstick = float(getattr(config, "pair_unstick_prob", 0.02))  # 1%~3% 概率解锁旧对
            pair_affinity_hist *= gamma_decay
            if prev_pairs:
                for (i_old, j_old) in list(prev_pairs):
                    if random.random() < eps_unstick:
                        prev_pairs.remove((i_old, j_old))  # 本步不强制黏滞

            # 2) 读取/准备参数
            min_pairs_target = max(1, getattr(config, "min_pair_target", config.n_veh // 4))
            max_backoff_rounds = int(getattr(config, "max_backoff_rounds", 3))
            stick_tau_db = float(getattr(config, "stickiness_tau_db", 1.0))
            w_delta_db = float(getattr(config, "score_w_delta_db", 1.0))
            w_hist = float(getattr(config, "score_w_history", 0.3))
            abs_gain_min_db = float(getattr(config, "abs_gain_min_db", -math.inf))

            gains_for_pairing = current_channel_gains  # 就用你构造掩码的这一份增益
            feasible_mask = (mask_mat.astype(np.uint8)  # 候选掩码（来自阶段4）
                             if (config.mask_enable and mask_mat is not None) else
                             np.ones((config.n_veh, config.n_veh), dtype=np.uint8) - np.eye(config.n_veh,
                                                                                            dtype=np.uint8))
            # --- NEW: 训练期构造 QoS 软掩码（仅在开启 QoS 时，用于评分扣分，不改变硬候选） ---
            qos_soft_mask = None
            if getattr(config, "qos_enable", False):
                qos_soft_mask = np.zeros_like(feasible_mask, dtype=np.uint8)
                g_lin = gains_for_pairing
                Rmin = float(getattr(config, "qos_R_min_bpsHz", 0.0))
                for i in range(config.n_veh):
                    for j in range(config.n_veh):
                        if i == j:
                            continue
                        ok = _qos_pair_feasible(
                            i, j,
                            g_lin, offload_power_for_pairing,
                            noise_power=env.noise_power, P_max=env.P_max, R_min=Rmin  # ← 正确关键字 R_min
                        )
                        qos_soft_mask[i, j] = 1 if ok else 0
            S0 = _score_matrix_from_gain_and_history(
                gain_linear=gains_for_pairing,
                feasible_mask=feasible_mask,
                hist_affinity=pair_affinity_hist,
                w_delta_db=w_delta_db,
                w_hist=w_hist,
                abs_gain_min_db=abs_gain_min_db,
                qos_soft_mask=qos_soft_mask,  # NEW
                qos_soft_penalty=float(getattr(config, "qos_soft_penalty_dbscore", 6.0)),  # NEW
            )

            # 4) 主配对：MWM + 软阈值（不硬配对）
            pairs = []
            accept_q = float(getattr(config, "mwm_accept_quantile", 0.20))
            if bool(getattr(config, "use_mwm_primary", True)):
                pairs = _mwm_primary(
                    S=S0,
                    feasible=feasible_mask,
                    accept_quantile=accept_q,
                    allow_singles=bool(getattr(config, "mwm_allow_singles", True))
                )
                # [PATCH][completion-train-main] 保底补配
                min_pairs_target = max(1, getattr(config, "min_pair_target", config.n_veh // 4))
                if len(pairs) < min_pairs_target:
                    pairs = _mwm_completion(S0, feasible_mask, pairs, min_pairs_target)

            else:
                # 兼容保留：若关闭主MWM，就走原先的互选+贪心
                pairs = _mutual_then_greedy_pairs(
                    S=S0,
                    feasible=feasible_mask,
                    min_pairs=min_pairs_target,
                    sticky_pairs=prev_pairs,
                    stick_tau_db=stick_tau_db,
                    gain_linear=gains_for_pairing,
                    unpaired_priority=unpaired_streak
                )

            # 5) 受限回退：若对数不足，小步放宽（降分位/放宽Top-K/降低τ），每轮重跑主MWM；仍不强配
            round_id = 0
            q_step = float(getattr(config, "relax_q_step", 0.02))  # 每轮分位数下降
            K_step = int(getattr(config, "relax_topk_step", 1))  # 每轮 Top-K +1
            tau_factor = float(getattr(config, "relax_tau_factor_per_round", 0.95))  # 每轮 τ×0.95

            q_back = float(last_q_now if last_q_now is not None else getattr(config, "pairing_threshold_quantile", 0.4))
            K_back = int(last_K_now if last_K_now is not None else
                         _anneal_topk(i_episode, config.n_veh, config.mask_topk_start, config.mask_topk_end,
                                      config.mask_warmup_episodes))
            tau_back = float(
                last_tau_now if last_tau_now is not None else _adaptive_threshold_from_delta_g(gains_for_pairing,
                                                                                               q_back))

            while (len(pairs) < min_pairs_target) and (round_id < int(getattr(config, "mwm_backoff_rounds", 3))):
                round_id += 1
                # 掩码放宽
                q_back = max(0.05, q_back - q_step)
                K_back = min(config.n_veh - 1, K_back + K_step)
                tau_back = max(float(getattr(config, "tau_back_floor_db", 3.0)), tau_back * tau_factor)
                feasible_mask = _relax_mask_once(feasible_mask, gains_for_pairing, tau_back, K_back)

                # 评分重算
                S_back = _score_matrix_from_gain_and_history(
                    gain_linear=gains_for_pairing,
                    feasible_mask=feasible_mask,
                    hist_affinity=pair_affinity_hist,
                    w_delta_db=w_delta_db,
                    w_hist=w_hist,
                    abs_gain_min_db=abs_gain_min_db,
                    qos_soft_mask=qos_soft_mask,
                    qos_soft_penalty=float(getattr(config, "qos_soft_penalty_dbscore", 6.0)),
                )
                # 软阈值也小步放宽
                accept_q = max(0.05, accept_q - float(getattr(config, "mwm_accept_q_step", 0.05)))

                pairs = _mwm_primary(
                    S=S_back,
                    feasible=feasible_mask,
                    accept_quantile=accept_q,
                    allow_singles=bool(getattr(config, "mwm_allow_singles", True))
                )
                # [PATCH][completion-train-backoff] 保底补配
                min_pairs_target = max(1, getattr(config, "min_pair_target", config.n_veh // 4))
                if len(pairs) < min_pairs_target:
                    pairs = _mwm_completion(S_back, feasible_mask, pairs, min_pairs_target)

            # ===== Freeze-in-episode with three safeties =====
            need_repair = False
            if freeze_group_in_episode and (episode_groups is not None):
                # A) 周期性重算
                if freeze_recalc_every > 0 and (i_step % freeze_recalc_every == 0):
                    need_repair = True
                # B) 触发式解冻：最近 env.global 明显低于本回合历史最好值
                if (not need_repair) and (last_env_global is not None) and (not unstick_used_flag):
                    if last_env_global < (ep_env_best * (1.0 - freeze_reward_drop_ratio)):
                        need_repair = True
                        unstick_used_flag = True
                # C) 小概率解冻（探索）
                if (not need_repair) and (freeze_unstick_prob > 0.0):
                    if np.random.rand() < freeze_unstick_prob:
                        need_repair = True

            if freeze_group_in_episode and (episode_groups is not None) and (not need_repair):
                # —— 不重算：复用本回合冻结分组，并同步 pairs/singles 以供后续统计使用 ——
                pairs = [(g[0], g[1]) for g in episode_groups if len(g) == 2]
                used_now = set(u for ab in pairs for u in ab)
                singles = [g[0] for g in episode_groups if len(g) == 1]
                noma_groups = [list(g) for g in episode_groups]
            else:
                # —— 正常路径：按当前评分/掩码求解出来的 pairs/singles，刷新冻结分组 ——
                used_now = set([u for ab in pairs for u in ab])
                singles = [u for u in range(config.n_veh) if u not in used_now]
                episode_groups = [[i, j] for (i, j) in pairs] + [[k] for k in singles]
                noma_groups = [list(g) for g in episode_groups]

            # 7) 更新热度/黏滞与连续落单计数
            for (i, j) in pairs:
                pair_affinity_hist[i, j] += 1.0
                pair_affinity_hist[j, i] += 1.0
            prev_pairs = set((min(i, j), max(i, j)) for (i, j) in pairs)
            for u in range(config.n_veh):
                unpaired_streak[u] = 0 if (u in used_now) else (unpaired_streak[u] + 1)

            # 统计：本步的配对数量、单播数量、阈值 tau
            pairs_count = sum(1 for g in noma_groups if len(g) == 2)
            singles_count = sum(1 for g in noma_groups if len(g) == 1)
            # —— Stage-6：为“配对成功奖励”保存当步信息（在奖励处理时使用）——
            pair_ratio_step = pairs_count / max(1, (pairs_count + singles_count))
            # 计算当步阈值（分位数）
            tau_step = _adaptive_threshold_from_delta_g(current_channel_gains, config.pairing_threshold_quantile)
            ep_tau_sum += float(tau_step)
            ep_pair_calls += 1
            # —— Backoff：若配对数过少，自动放宽阈值再配一次（训练阶段新增） ——
            min_pairs = max(1, getattr(config, "min_pair_target", config.n_veh // 4))  # 例如 8 用户 → 至少 2 对
            ep_pairs += pairs_count
            ep_singles += singles_count

            # ---- Debug：更宽松的逐步打印（两种配对路径都能看到）----
            # 想看更多轮：把 5/10 调大，例如 8/20
            if (i_episode < 50 and i_step < 10) or (i_episode % 10 == 0 and i_step % 20 == 0):
                tau_now = _adaptive_threshold_from_delta_g(
                    current_channel_gains, config.pairing_threshold_quantile
                )
                pairs = [g for g in noma_groups if len(g) == 2]
                singles = [g for g in noma_groups if len(g) == 1]
                # 把增益格式化成科学计数法字符串
                gains_sci = np.array2string(
                    np.asarray(current_channel_gains, dtype=float),
                    formatter={'float_kind': lambda x: f'{x:.3e}'}
                )

                print(
                    "[DEBUG] ep=", i_episode, " step=", i_step,
                    " tau=", f"{float(tau_now):.3e}",  # 阈值也顺便用科学计数法
                    " gains=", gains_sci,  # <- 用科学计数法字符串
                    " groups=", noma_groups,
                    " | pairs=", len(pairs), " singles=", len(singles),
                    flush=True
                )

            # --- 4. 准备功率动作并与环境交互 ---
            action_for_env = np.zeros([2, config.n_veh], dtype=float)
            for i in range(config.n_veh):
                clipped_power = np.clip(marl_power_actions[i], -0.999, 0.999)
                action_for_env[0, i] = (clipped_power[0] + 1) / 2
                action_for_env[1, i] = (clipped_power[1] + 1) / 2
                _floor = float(getattr(env, "cpu_share_floor", 0.10))
                _floor = max(0.0, min(_floor, 0.95))
                action_for_env[1, i] = max(action_for_env[1, i], _floor)

            # === 1) 强制对齐学习口径：learn_reward = ENV 物理口径 ===
            per_user_reward, global_reward, _, _, _, _, _ = env.step(action_for_env, noma_groups)

            env_global_reward = float(global_reward)  # ENV：物理口径的全局回报（负的 delay+energy + QoS 扣罚）
            learn_reward = env_global_reward  # 学习信号一律用 ENV
            # 记录“未加惩罚”的原始全局回报（仅用于调参/评估可读性）
            _raw_global = float(global_reward)  # 注意：这里等于 env.step 返回的全局回报（你的环境里等于 np.mean(per_user_reward)）
            # —— 更新触发式解冻指标（用于下一步是否“解冻重算”）——
            if last_env_global is None:
                ep_env_best = _raw_global
            else:
                if _raw_global > ep_env_best:
                    ep_env_best = _raw_global
            last_env_global = _raw_global

            record_global_reward_raw_average.append(_raw_global)
            # === NEW: accumulate step-wise delay/energy for this episode ===
            d = getattr(env, "last_delay_mean", None)
            e = getattr(env, "last_energy_mean", None)
            if d is not None:
                ep_delay_sum += float(d)
            if e is not None:
                ep_energy_sum += float(e)
            ep_steps += 1
            # === 新增：分项/状态的按步累计 ===
            v = getattr(env, "last_delay_local_mean", None)
            if v is not None: ep_delay_local_sum += float(v)

            v = getattr(env, "last_delay_edge_q_mean", None)
            if v is not None: ep_delay_edge_q_sum += float(v)

            v = getattr(env, "last_delay_edge_c_mean", None)
            if v is not None: ep_delay_edge_c_sum += float(v)

            v = getattr(env, "last_t_tx_mean", None)
            if v is not None: ep_delay_tx_sum += float(v)

            v = getattr(env, "last_backlog_kbit_mean", None)
            if v is not None: ep_backlog_kbit_sum += float(v)

            v = getattr(env, "last_mec_utilization", None)
            if v is not None: ep_mec_util_sum += float(v)

            v = getattr(env, "last_local_util_mean", None)
            if v is not None: ep_local_util_sum += float(v)

            v = getattr(env, "last_qos_violation", None)  # 若环境已计算 QoS 违约率
            if v is not None: ep_qos_viol_sum += float(v)

            off_sum = getattr(env, "last_off_kbit_sum", None)
            loc_sum = getattr(env, "last_local_kbit_sum", None)
            if off_sum is not None: ep_off_kbit_sum += float(off_sum)
            if loc_sum is not None: ep_local_kbit_sum += float(loc_sum)

            # === Re-shaping (safer) — gate by config.reward_use_shaping ===
            if bool(getattr(config, "reward_use_shaping", False)):
                running_reward_mean = 0.99 * running_reward_mean + 0.01 * np.mean(per_user_reward)
                running_reward_std = 0.99 * running_reward_std + 0.01 * np.std(per_user_reward)
                per_user_reward_norm = (per_user_reward - running_reward_mean) / (running_reward_std + 1e-7)

                T = int(config.n_episode)
                warm = int(0.3 * T);
                hold = int(0.4 * T);
                decay_start = warm + hold
                if i_episode < warm:
                    scaling = i_episode / max(1, warm)
                elif i_episode < decay_start:
                    scaling = 1.0
                else:
                    scaling = max(0.0, 1.0 - (i_episode - decay_start) / max(1, (T - decay_start)))

                # === 2) 整形项仅做监控，不改变 learn_reward / env_global_reward ===
                metrics = locals().get("metrics", {}) if isinstance(locals().get("metrics", None), dict) else {}

                # (A) 欠公平（用最弱用户的规范化回报作为观测）
                t_min = -0.5
                min_norm = float(np.min(per_user_reward_norm))
                fairness_deficit = max(0.0, t_min - min_norm)
                metrics['fairness_deficit'] = float(fairness_deficit)

                # (B) 方差观测
                var_norm = float(np.var(per_user_reward_norm))
                metrics['variance_norm'] = float(min(var_norm, 1.5))

                # (C) 协同配对观测
                metrics['coop_pair_ratio'] = float(locals().get("pair_ratio_step", 0.0))

                # （D）极低回报观测
                extreme_deficit = - (per_user_reward_norm - config.LOW_REWARD_THRESHOLD)
                extreme_deficit = np.clip(extreme_deficit, 0.0, None)
                if np.any(extreme_deficit > 0):
                    bad_frac = float((extreme_deficit > 0).sum()) / float(config.n_veh)
                    metrics['low_reward_frac'] = float(bad_frac)
                    metrics['low_reward_mean_def'] = float(extreme_deficit.mean())

                # 供后续统计显示
                per_user_reward_norm = np.clip(per_user_reward_norm, -5, 5)


            else:
                # 关闭 re-shaping：学习信号=env.step 的 global_reward
                # 为了局部统计稳定，保留一个“无信息”的 per_user_reward_norm（恒等映射）
                per_user_reward_norm = per_user_reward.copy()

            # 到这里，再对标准化后的 per-user 奖励做数值截断，供后续统计/显示更稳定
            per_user_reward_norm = np.clip(per_user_reward_norm, -5, 5)

            # === 总功率“自适应软阈惩罚”（不影响环境硬约束，仅提供可微梯度）+ 记录 ===
            _pw = getattr(env, "last_power_W", None)
            if _pw is not None:
                total_power_now = float(np.array(_pw).sum())

                # ---- 1) 更新 episode 内 EMA 基线 ----
                beta = float(getattr(config, "soft_power_beta", 0.9))  # EMA 衰减系数
                if ema_power is None:
                    ema_power = total_power_now
                else:
                    ema_power = beta * ema_power + (1.0 - beta) * total_power_now

                # ---- 2) 自适应阈值：max(安全下限, EMA + 边际) ----
                floor = float(getattr(config, "soft_power_floor", 6.0))  # 安全下限
                margin = float(getattr(config, "soft_power_margin", 0.3))  # 在均线之上多少才开始罚
                th = max(floor, ema_power + margin)

                # ---- 3) 平滑/可调的二次-线性混合惩罚----
                apply_soft_penalty = bool(getattr(config, "use_soft_power_penalty", False))
                if apply_soft_penalty and (total_power_now > th):
                    over = total_power_now - th
                    c1 = float(getattr(config, "soft_power_coef_linear", 0.0))
                    c2 = float(getattr(config, "soft_power_coef_quad", 0.05))
                    cap = float(getattr(config, "soft_power_cap_per_step", 0.3))
                    penalty = min(c1 * over + c2 * (over ** 2), cap)
                    # 仅记录：不再改 learn_reward / env_global_reward
                    metrics = locals().get("metrics", {}) if isinstance(locals().get("metrics", None), dict) else {}
                    metrics['soft_power_penalty'] = float(penalty)

                # ---- 4) 记录总/分路功率 ----
                Power.append(total_power_now)

                _pw_arr = np.array(_pw)
                if _pw_arr.ndim == 2:
                    if _pw_arr.shape[0] == 2:
                        Power_offload.append(float(_pw_arr[0, :].sum()))
                        Power_local.append(float(_pw_arr[1, :].sum()))
                    elif _pw_arr.shape[1] == 2:
                        Power_offload.append(float(_pw_arr[:, 0].sum()))
                        Power_local.append(float(_pw_arr[:, 1].sum()))
                else:
                    # 其它形状：仅记录总功率
                    pass
            else:
                # 兼容：若环境未提供 last_power_W，则退回到旧口径（动作指令值），避免中断
                Power.append(np.sum(action_for_env))
                Power_offload.append(np.sum(action_for_env[0]))
                Power_local.append(np.sum(action_for_env[1]))

            record_global_reward_average.append(learn_reward)  # 与回放一致，学习口径=ENV

            for i in range(config.n_veh):
                record_reward_[i, i_episode] += per_user_reward_norm[i]

            marl_state_new_all = [marl_get_state(i) for i in range(config.n_veh)]

            if i_step == config.n_step_per_episode - 1:
                done = True

            # —— 动作写回放：概率(N) + 功率(2) ——
            power_np = np.asarray(marl_power_actions, dtype=np.float32)  # [N, 2]
            probs_np = intent_probs_mat.astype(np.float32)  # [N, N]

            # 存 [onehot, power] 到每个体：一行 N + 2 维；再拼成 1D 向量
            action_cat = [np.concatenate([probs_np[i], power_np[i]], axis=0) for i in range(config.n_veh)]
            action_to_store = np.concatenate(action_cat, axis=0).astype(np.float32)  # 形状: N*(N+2,)

            # —— 新增：把当步的可行性掩码（N×N）展平成 1D 存进回放 ——
            # 若当步未构造掩码（mask_mat 为 None），存全 1（除对角线在配对阶段会再屏蔽）
            if mask_mat is None:
                mask_to_store = np.ones((config.n_veh, config.n_veh), dtype=np.float32).reshape(-1)
            else:
                mask_to_store = mask_mat.astype(np.float32).reshape(-1)

            memory.store_transition(
                np.asarray(marl_state_old_all, dtype=np.float32).flatten(),
                action_to_store.astype(np.float32),
                float(learn_reward),  # ← 学习信号=ENV
                per_user_reward_norm.astype(np.float32),
                np.asarray(marl_state_new_all, dtype=np.float32).flatten(),
                done,
                mask_to_store
            )

            if memory.mem_cntr >= config.batch_size:
                states, actions, rewards_g, rewards_l, states_, dones, masks = memory.sample_buffer(config.batch_size)
                global_agent.global_learn(agents, states, actions, rewards_g, rewards_l, states_, dones, masks)
                # ---- Stage-1: 线性退火 tau （从 start 降到 end）----
                ls = global_agent.learn_step_counter  # 全局学习步
                T0 = float(config.gumbel_tau_start)
                T1 = float(config.gumbel_tau_end)
                K = max(1, int(config.gumbel_anneal_steps))
                frac = min(1.0, ls / K)
                cur_tau = T0 + (T1 - T0) * frac
                import torch

                for ag in agents:
                    with torch.no_grad():
                        ag.policy.tau.fill_(float(cur_tau))
                    # 由 YAML 的 gumbel_hard_train 控制；若开启，再按更稳健的阈值 0.3
                    want_hard = bool(getattr(config, "gumbel_hard", False))
                    ag.policy.gumbel_hard = bool(want_hard and (cur_tau <= 0.3))

            marl_state_old_all = marl_state_new_all

        # Log rewards per episode
        for i in range(config.n_veh):
            record_reward_[i, i_episode] /= config.n_step_per_episode
            print(f'user {i}: {record_reward_[i, i_episode]:.4f}', end='   ')
        print()

        # —— 用 _safe_mean 统一统计，并避免空数组产生的 NaN/告警 ——
        global_loss_ep = _safe_mean(global_agent.Global_Loss, default=0.0)
        record_critics_loss_[0, i_episode] = global_loss_ep
        global_agent.Global_Loss.clear()

        for i in range(config.n_veh):
            local_loss_ep = _safe_mean(agents[i].local_critic_loss, default=0.0)
            record_critics_loss_[i + 1, i_episode] = local_loss_ep
            agents[i].local_critic_loss.clear()

        average_global_reward = np.mean(record_global_reward_average[-config.n_step_per_episode:])
        average_raw_global_reward = np.mean(record_global_reward_raw_average[-config.n_step_per_episode:])
        record_global_reward_episode.append(float(average_global_reward))  # 新增
        Power_episode = np.mean(Power)
        Power_local_episode = np.mean(Power_local)
        Power_offload_episode = np.mean(Power_offload)

        Sum_Power.append(Power_episode)
        Sum_Power_local.append(Power_local_episode)
        Sum_Power_offload.append(Power_offload_episode)
        # === (1) 先计算本集的“绝对口径”均值（延迟/能耗） ===
        if ep_steps > 0:
            delay_mean_ep = ep_delay_sum / ep_steps
            energy_mean_ep = ep_energy_sum / ep_steps
        else:
            delay_mean_ep = float("nan")
            energy_mean_ep = float("nan")

        # （同口径）分项时延/状态均值 —— 后面日志打印和 TensorBoard 会用到
        if ep_steps > 0:
            delay_local_ep = float(ep_delay_local_sum / ep_steps)
            delay_edge_q_ep = float(ep_delay_edge_q_sum / ep_steps)
            delay_edge_c_ep = float(ep_delay_edge_c_sum / ep_steps)
            delay_tx_ep = float(ep_delay_tx_sum / ep_steps)
            backlog_kbit_ep = float(ep_backlog_kbit_sum / ep_steps)
            mec_util_ep = float(ep_mec_util_sum / ep_steps)
            local_util_ep = float(ep_local_util_sum / ep_steps)
            qos_viol_ep = float(ep_qos_viol_sum / ep_steps)
        else:
            delay_local_ep = delay_edge_q_ep = delay_edge_c_ep = delay_tx_ep = float("nan")
            backlog_kbit_ep = mec_util_ep = local_util_ep = qos_viol_ep = float("nan")

        # === Three-tier metrics (unified) ===
        # [LEARN] 真正被优化的学习信号：含公平/方差等 re-shaping
        print(f'[LEARN] Global reward (with fairness/variance): {average_global_reward:.4f}')

        # [ENV] 环境层回报：来自 env.step()，含 QoS 违约等环境惩罚，但不含 re-shaping
        print(f'[ENV]   Env global (with QoS, no re-shaping):   {average_raw_global_reward:.4f}')

        # [PHYS] 绝对口径（研究目标）：物理代价与按权重折算的物理回报
        # 统一用“本 episode 均值”，单位：delay=秒(下行显示为ms), energy=焦耳
        w_d_log = float(getattr(env, "w_d", 1.0))
        w_e_log = float(getattr(env, "w_e", 1.0))
        phys_delay_s = float(delay_mean_ep)
        phys_energy_J = float(energy_mean_ep)
        phys_cost = phys_delay_s + phys_energy_J  # 代价: 越小越好
        phys_reward = -(w_d_log * phys_delay_s + w_e_log * phys_energy_J)  # 回报: 越大越好

        print(f'[PHYS]  Delay+Energy: cost={phys_cost:.6f} '
              f'(delay={phys_delay_s * 1000:.3f} ms, energy={phys_energy_J:.6f} J); '
              f'weighted_reward={phys_reward:.6f} (w_d={w_d_log:g}, w_e={w_e_log:g})')

        # 功率（单位：W）
        print(f'Average Total Power: {Power_episode:.4f}')
        print(f'Average local power: {Power_local_episode:.4f},   Average offload power: {Power_offload_episode:.4f}')
        # === A4：记录/打印 Gumbel 温度与功率到回合级 ===
        # 1) 读取当前策略的 Gumbel 温度与直通(gumbel_hard)状态（各 agent 相同，取第0个）
        try:
            tau_now = float(agents[0].policy.tau.item())
            hard_now = int(getattr(agents[0].policy, "gumbel_hard", False))
        except Exception:
            tau_now, hard_now = float("nan"), -1

        # 2) 追加到回合级数组
        episode_gumbel_tau.append(tau_now)
        episode_gumbel_hard.append(hard_now)
        episode_power_total.append(float(Power_episode))
        episode_power_local.append(float(Power_local_episode))
        episode_power_offload.append(float(Power_offload_episode))

        # 3) 记录奖励权重（若开启采样；否则写 NaN 以便 CSV 对齐）
        _wd = float(getattr(env, "w_d", float("nan")))
        _we = float(getattr(env, "w_e", float("nan")))
        episode_w_d.append(_wd)
        episode_w_e.append(_we)

        # 4) 控制台打印：把 τ / hard 与权重一并可见化
        print(
            f"[GUMBEL] tau={tau_now:.3f}  hard={hard_now}  | w_d={_wd if not np.isnan(_wd) else float('nan'):.4f}  w_e={_we if not np.isnan(_we) else float('nan'):.4f}")

        # “绝对口径”的新目标值（便于日志可读）：delay + energy
        new_reward_ep = delay_mean_ep + energy_mean_ep
        print(f'New reward (delay+energy): {new_reward_ep:.4f}')
        print(f"[Delay breakdown] local={delay_local_ep * 1000:.3f} ms, "
              f"edge_queue={delay_edge_q_ep * 1000:.3f} ms, "
              f"edge_compute={delay_edge_c_ep * 1000:.3f} ms, "
              f"tx={delay_tx_ep * 1000:.3f} ms")

        # 5) TensorBoard：按回合写绝对口径指标（此时 delay_mean_ep/energy_mean_ep 已就绪）
        if writer is not None:
            writer.add_scalar("gumbel/tau", tau_now, i_episode)
            writer.add_scalar("gumbel/hard", hard_now, i_episode)
            writer.add_scalar("power/total_W", Power_episode, i_episode)
            writer.add_scalar("power/local_W", Power_local_episode, i_episode)
            writer.add_scalar("power/offload_W", Power_offload_episode, i_episode)
            writer.add_scalar("abs/delay_ms", delay_mean_ep * 1000.0, i_episode)
            writer.add_scalar("abs/energy_J", energy_mean_ep, i_episode)
            writer.add_scalar("reward_weight/w_d", _wd, i_episode)
            writer.add_scalar("reward_weight/w_e", _we, i_episode)

        # ---- Stage-0 four metrics (per episode) ----
        min_user = float(np.min(record_reward_[:, i_episode]))
        var_user = float(np.var(record_reward_[:, i_episode]))
        jain = _jain_index(record_reward_[:, i_episode])
        elapsed = time.time() - t0

        episode_min_user.append(min_user)
        episode_var_user.append(var_user)
        episode_jain.append(jain)
        episode_elapsed_sec.append(elapsed)
        print(f"[E{i_episode:03d}] min_user={min_user:.4f}  var={var_user:.4f}  jain={jain:.4f}  time={elapsed:.2f}s")
        min_user = float(np.min(record_reward_[:, i_episode]))
        var_user = float(np.var(record_reward_[:, i_episode]))
        jain = _jain_index(record_reward_[:, i_episode])
        elapsed = time.time() - t0

        # === [B-4 精简] 仅保存“每集 delay/energy 均值”到数组/TB（均值已在前面计算）===
        episode_delay_mean.append(float(delay_mean_ep))
        episode_energy_mean.append(float(energy_mean_ep))

        if args.log == "tb":
            try:
                writer.add_scalar("delay/episode_mean", float(delay_mean_ep), i_episode)
                writer.add_scalar("energy/episode_mean", float(energy_mean_ep), i_episode)

                # 直接用已算好的分项均值
                writer.add_scalar("delay/local_ep_mean", float(delay_local_ep), i_episode)
                writer.add_scalar("delay/edge_queue_ep_mean", float(delay_edge_q_ep), i_episode)
                writer.add_scalar("delay/edge_compute_ep_mean", float(delay_edge_c_ep), i_episode)
                writer.add_scalar("delay/tx_ep_mean", float(delay_tx_ep), i_episode)

                writer.add_scalar("queue/backlog_kbit_ep_mean", float(backlog_kbit_ep), i_episode)
                writer.add_scalar("queue/mec_util_ep_mean", float(mec_util_ep), i_episode)
                writer.add_scalar("cpu/local_util_ep_mean", float(local_util_ep), i_episode)
                writer.add_scalar("qos/violation_rate_ep_mean", float(qos_viol_ep), i_episode)

                writer.add_scalar("queue/mec_cycles",
                                  float(getattr(env, "last_mec_queue_cycles", 0.0)), i_episode)
            except Exception:
                pass

                # === 新增：写入 TensorBoard，多条关键曲线 ===
                writer.add_scalar("delay/local_ep_mean", delay_local_ep, i_episode)
                writer.add_scalar("delay/edge_queue_ep_mean", delay_edge_q_ep, i_episode)
                writer.add_scalar("delay/edge_compute_ep_mean", delay_edge_c_ep, i_episode)
                writer.add_scalar("delay/tx_ep_mean", delay_tx_ep, i_episode)

                writer.add_scalar("queue/backlog_kbit_ep_mean", backlog_kbit_ep, i_episode)
                writer.add_scalar("queue/mec_util_ep_mean", mec_util_ep, i_episode)
                writer.add_scalar("cpu/local_util_ep_mean", local_util_ep, i_episode)

                # （可选）若你关心 QoS 违约/超时比
                writer.add_scalar("qos/violation_rate_ep_mean", qos_viol_ep, i_episode)

                # （推荐）也把 MEC 当前队列长度写出来，便于和排队时延对比
                writer.add_scalar("queue/mec_cycles",
                                  float(getattr(env, "last_mec_queue_cycles", 0.0)), i_episode)

            except Exception:
                pass

        # —— 计算本回合的配对比例与平均阈值 ——
        if ep_pair_calls > 0:
            pair_ratio = ep_pairs / max(1, (ep_pairs + ep_singles))
            tau_mean = ep_tau_sum / ep_pair_calls
        else:
            pair_ratio, tau_mean = 0.0, 0.0

        # —— 写入 TensorBoard（若启用） ——
        if writer is not None:
            writer.add_scalar("reward/global_avg", float(average_global_reward), i_episode)
            writer.add_scalar("power/total_avg", float(Power_episode), i_episode)
            writer.add_scalar("power/local_avg", float(Power_local_episode), i_episode)
            writer.add_scalar("power/offload_avg", float(Power_offload_episode), i_episode)
            # critic 损失：global + local(均值)
            writer.add_scalar("loss/global_critic", float(record_critics_loss_[0, i_episode]), i_episode)
            if config.n_veh > 0:
                local_mean_ep = _safe_mean(record_critics_loss_[1:, i_episode], default=0.0)
                writer.add_scalar("loss/local_critic_mean", float(local_mean_ep), i_episode)

            # alpha（兼容单α/双α）
            if hasattr(global_agent, "alpha_cont") and hasattr(global_agent, "alpha_disc"):
                try:
                    writer.add_scalar("sac/alpha_cont", float(global_agent.alpha_cont.detach().cpu().item()), i_episode)
                    writer.add_scalar("sac/alpha_disc", float(global_agent.alpha_disc.detach().cpu().item()), i_episode)
                except Exception:
                    pass
            elif hasattr(global_agent, "alpha"):
                try:
                    writer.add_scalar("sac/alpha", float(global_agent.alpha.detach().cpu().item()), i_episode)
                except Exception:
                    pass
            # 熵与actor损失（来自 centralized_actor_update 的返回）
            if hasattr(global_agent, "entropy_ema") and global_agent.entropy_ema is not None:
                writer.add_scalar("sac/entropy_ema", float(global_agent.entropy_ema), i_episode)
            if hasattr(global_agent, "last_actor_stats") and isinstance(global_agent.last_actor_stats, dict):
                if "actor_global_loss" in global_agent.last_actor_stats:
                    writer.add_scalar("sac/actor_loss", float(global_agent.last_actor_stats["actor_global_loss"]), i_episode)
                if "q_min_mean" in global_agent.last_actor_stats:
                    writer.add_scalar("sac/q_min_mean", float(global_agent.last_actor_stats["q_min_mean"]), i_episode)

            mask_zero_ratio_mean = (ep_mask_zero_ratio_sum / max(1, ep_pair_calls)) if ep_pair_calls > 0 else 0.0
            # 配对统计 & 阈值
            writer.add_scalar("pairing/pair_ratio", float(pair_ratio), i_episode)
            writer.add_scalar("pairing/tau_mean", float(tau_mean), i_episode)
            # 掩码屏蔽比例（需要你在step内累计 ep_mask_zero_ratio_sum/ep_pair_calls）
            writer.add_scalar("mask/zero_ratio", float(mask_zero_ratio_mean), i_episode)
            # === NEW: MEC 队列与本集流量 ===
            writer.add_scalar("queue/mec_cycles", float(getattr(env, "mec_queue_cycles", 0.0)), i_episode)
            writer.add_scalar("traffic/offload_kbit_ep", float(ep_off_kbit_sum), i_episode)
            writer.add_scalar("traffic/local_kbit_ep", float(ep_local_kbit_sum), i_episode)


        if i_episode % 50 == 0 and i_episode != 0:

            print('Saving models...')
            global_agent.save_models()
            for i in range(config.n_veh):
                agents[i].save_models()
        # ---- 每个 episode 采样 w_d, w_e（若开启） ----
        if config.use_weight_sampling:
            w_d = np.random.uniform(*config.w_d_range)
            w_e = np.random.uniform(*config.w_e_range)
            env.w_d = float(w_d)
            env.w_e = float(w_e)
            if i_episode % 20 == 0:  # 偶尔打印确认
                print(f"[Reward Weights] ep={i_episode}  w_d={env.w_d:.4f}, w_e={env.w_e:.4f}")

    # ################## SAVE DATA & PLOT RESULTS ######################
    print('Training Done.')
    print(
        f'Sum average local power: {np.mean(Sum_Power_local):.4f},   Sum Average offload power: {np.mean(Sum_Power_offload):.4f}')

    # Create directory for results if it doesn't exist
    results_dir = f'Data2/{config.n_veh}_users_NOMA_SAC'
    os.makedirs(results_dir, exist_ok=True)

    for i in range(config.n_veh):
        np.save(os.path.join(results_dir, f'User{i}_Reward.npy'), record_reward_[i, :])
    np.save(os.path.join(results_dir, 'Global_Reward.npy'), record_global_reward_average)
    np.save(os.path.join(results_dir, 'Global_Reward_Raw.npy'),
            np.array(record_global_reward_raw_average))
    np.save(os.path.join(results_dir, 'Global_Reward_per_episode.npy'),
            np.array(record_global_reward_episode))  # 新增
    # —— 同步保存到本次 run 目录（便于复现与汇总） ——
    try:
        art = run_dir / "artifacts"
        art.mkdir(parents=True, exist_ok=True)

        # 1) NPY：统一指标（阶段0）
        np.save(
            art / "episode_metrics.npy",
            {
                "global_reward": np.array(record_global_reward_episode, dtype=np.float32),
                "min_user": np.array(episode_min_user, dtype=np.float32),
                "var_user": np.array(episode_var_user, dtype=np.float32),

                # === A4：绝对口径 ===
                "delay_mean": np.array(episode_delay_mean, dtype=np.float32),  # [s] 单集均值
                "energy_mean": np.array(episode_energy_mean, dtype=np.float32),  # [J] 单集均值
                "power_total": np.array(episode_power_total, dtype=np.float32),  # [W]
                "power_local": np.array(episode_power_local, dtype=np.float32),  # [W]
                "power_offload": np.array(episode_power_offload, dtype=np.float32),  # [W]

                # === A4：Gumbel 可见性 ===
                "gumbel_tau": np.array(episode_gumbel_tau, dtype=np.float32),
                "gumbel_hard": np.array(episode_gumbel_hard, dtype=np.int32),

                # === A4：回合权重（若未启用采样则为 NaN）===
                "w_d": np.array(episode_w_d, dtype=np.float32),
                "w_e": np.array(episode_w_e, dtype=np.float32),

                # === 原有 ===
                "jain": np.array(episode_jain, dtype=np.float32),
                "elapsed_sec": np.array(episode_elapsed_sec, dtype=np.float32),
            },
            allow_pickle=True
        )

        # 2) CSV：方便快速查看
        # 2) CSV：方便快速查看
        delay_arr = np.array(episode_delay_mean, dtype=np.float32)
        energy_arr = np.array(episode_energy_mean, dtype=np.float32)
        pow_tot = np.array(episode_power_total, dtype=np.float32)
        pow_loc = np.array(episode_power_local, dtype=np.float32)
        pow_off = np.array(episode_power_offload, dtype=np.float32)
        tau_arr = np.array(episode_gumbel_tau, dtype=np.float32)
        hard_arr = np.array(episode_gumbel_hard, dtype=np.float32)
        wd_arr = np.array(episode_w_d, dtype=np.float32)
        we_arr = np.array(episode_w_e, dtype=np.float32)

        L = min(
            len(record_global_reward_episode),
            delay_arr.size, energy_arr.size,
            pow_tot.size, pow_loc.size, pow_off.size,
            tau_arr.size, hard_arr.size,
            wd_arr.size, we_arr.size
        )
        new_reward = (delay_arr[:L] + energy_arr[:L]).astype(np.float32)
        ep_idx = np.arange(L, dtype=int)

        csv_arr = np.stack([
            ep_idx,
            np.array(record_global_reward_episode[:L], dtype=np.float32),
            delay_arr[:L], energy_arr[:L], new_reward,
            pow_tot[:L], pow_loc[:L], pow_off[:L],
            tau_arr[:L], hard_arr[:L],
            wd_arr[:L], we_arr[:L],
            np.array(episode_min_user[:L], dtype=np.float32),
            np.array(episode_var_user[:L], dtype=np.float32),
            np.array(episode_jain[:L], dtype=np.float32),
            np.array(episode_elapsed_sec[:L], dtype=np.float32),
        ], axis=1)

        np.savetxt(
            art / "metrics.csv",
            csv_arr, delimiter=",",
            header="episode,global_reward,delay_mean,energy_mean,new_reward,"
                   "power_total,power_local,power_offload,"
                   "gumbel_tau,gumbel_hard,w_d,w_e,"
                   "min_user,var_user,jain,elapsed_sec",
            comments=""
        )

        print("[SAVE] metrics saved under:", art)
    except Exception as _e:
        print("[WARN] save metrics to run_dir failed:", _e)

    # —— 关闭 TensorBoard ——
    if writer is not None:
        try:
            writer.close()
        except Exception:
            pass


    # Plotting
    plt.figure(1)
    plt.title("Vehicle Trajectories")
    # Assuming BS and RIS are static, defined elsewhere if needed
    # plt.plot(BS_x, BS_y, 'o', markersize=5, color='black', label='BS')
    # plt.plot(RIS_x, RIS_y, 'o', markersize=5, color='brown', label='RIS')
    for i in range(config.n_veh):
        plt.plot(vehicle_positions[i]['x'], vehicle_positions[i]['y'], label=f'Vehicle {i}')
    plt.legend()
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")

    plt.figure(2)
    plt.title("Global Reward per Episode")  # 改标题
    episodes = np.arange(1, len(record_global_reward_episode) + 1)  # 新增x轴为1..N
    plt.plot(episodes, record_global_reward_episode, linewidth=1.5)  # 改成按episode的序列
    plt.xlabel('Episode')  # 改标签
    plt.ylabel('Global Reward')
    # （可选）让横轴正好到 n_episode，纵轴顶部靠近0更像你的样板图：
    # plt.xlim(1, config.n_episode)
    # plt.ylim(min(record_global_reward_episode) - 5, 5)
    # === NEW: plot RAW global (no-penalty) ===
    # === NEW: plot "new reward" = delay_mean + energy_mean, same style as Global ===
    try:
        # 1) 组装“新奖励”：逐集 delay_mean + energy_mean
        delay_arr = np.asarray(episode_delay_mean, dtype=np.float64)
        energy_arr = np.asarray(episode_energy_mean, dtype=np.float64)
        # 与 global 的 episodes 对齐（防御：长度不一致时截短到最短）
        L = min(len(episodes), delay_arr.size, energy_arr.size)
        if L > 0:
            new_reward = (delay_arr[:L] + energy_arr[:L]).astype(np.float64)

            # 2) 画图：完全沿用 Global 的画法（折线 + linewidth=1.5）
            plt.figure()
            plt.title("New Reward (Delay + Energy) per Episode")
            plt.plot(episodes[:L], new_reward, linewidth=1.5)
            plt.xlabel("Episode")
            plt.ylabel("Delay + Energy")  # 注意：这是“代价”的量纲和符号，越小越好

            # 3) （可选）同步保存到当前 run 的 artifacts 目录
            try:
                import numpy as _np, os as _os

                _np.save(_os.path.join(str(run_dir / "artifacts"), "new_reward_delay_plus_energy.npy"), new_reward)
            except Exception as _e:
                print("[WARN] save new_reward .npy failed:", _e)
        else:
            print("[WARN] cannot plot new_reward: empty arrays")
    except Exception as _e:
        print("[WARN] cannot plot Delay+Energy new reward:", _e)

    # === Raw Global Reward（按“回合”聚合到 300 个点）===
    try:
        raw_path = os.path.join(f'Data2/{config.n_veh}_users_NOMA_SAC', 'Global_Reward_Raw.npy')
        raw_global = np.load(raw_path)

        # 将“每步”序列聚合为“每回合”的均值（也可以改成 sum，看你评估口径）
        steps_per_ep = int(getattr(config, "n_step_per_episode", 100))
        total_steps = int(len(raw_global))
        num_eps = total_steps // steps_per_ep
        if num_eps > 0:
            raw_trim = raw_global[: num_eps * steps_per_ep].reshape(num_eps, steps_per_ep)
            raw_per_ep = raw_trim.mean(axis=1)  # ← 若想要“回合总和”就用 .sum(axis=1)

            plt.figure()
            plt.title("Raw Global Reward (no penalties)")
            plt.plot(np.arange(1, num_eps + 1), raw_per_ep, linewidth=1.5)
            plt.xlabel("Episode")
            plt.ylabel("Raw Global Reward")

            # 存一份回合级数组，便于后续复现/比对
            try:
                np.save(os.path.join(str(run_dir / "artifacts"), "raw_global_per_episode.npy"), raw_per_ep)
            except Exception as _e:
                print("[WARN] save raw_global_per_episode failed:", _e)
        else:
            print("[WARN] raw_global length < steps_per_ep; skip raw-per-episode plot.")
    except Exception as _e:
        print("[WARN] cannot plot Raw Global:", _e)

    plt.show()

else:
    # ################## TESTING ######################
    print("Loading models for testing...")
    global_agent.load_models()
    for i in range(config.n_veh):
        agents[i].load_models()

    # Test for a single episode
    env.renew_positions()
    env.compute_parms()
    marl_state_old_all = [marl_get_state(i) for i in range(config.n_veh)]
    # ---- Stage-6（test）：为公平性指标做统计容器 ----
    test_user_reward_sum = np.zeros(config.n_veh, dtype=np.float64)
    test_steps = 0
    test_pairs_total = 0
    test_singles_total = 0
    total_reward = 0
    # 阶段六：稳定性指标（测试版）
    test_min_user = []
    test_jain = []
    for _ in range(config.n_step_per_episode):
        # === Stage-6（test）：与训练一致的可行性掩码（采用课程“收敛后”的参数） ===
        feasible_mask_gpu = None
        if config.TEST_USE_MASK and config.mask_enable:
            K_now = int(config.mask_topk_end)
            q_now = float(config.mask_tau_q_end)
            tau_now = _adaptive_threshold_from_delta_g(env.get_channel_gains(), q_now)
            mask_mat = _build_feasible_mask_from_delta_g(env.get_channel_gains(), tau_now, K_now)


            import torch

            feasible_mask_gpu = torch.tensor(mask_mat, dtype=torch.float32, device=agents[0].policy.device)

        # 1) 采样：功率 + 意图“概率”（把和训练一致的 mask_row 传入）
        test_power_actions = []
        test_intent_probs_list = []
        for i in range(config.n_veh):
            mask_row = feasible_mask_gpu[i] if feasible_mask_gpu is not None else None
            power_action, intent_probs, _ = agents[i].choose_action(marl_state_old_all[i], mask=mask_row)
            test_power_actions.append(power_action)
            test_intent_probs_list.append(intent_probs)

        # 2) 组 N×N 概率矩阵（对角置 0，禁止自配）
        intent_probs_mat = np.stack(test_intent_probs_list, axis=0)
        np.fill_diagonal(intent_probs_mat, 0.0)

        # 3) 计算用于打分的“卸载功率”（[-1,1] → [0,1]）
        current_channel_gains = env.get_channel_gains()
        offload_power_for_pairing = np.zeros(config.n_veh, dtype=float)
        for i in range(config.n_veh):
            clipped = np.clip(test_power_actions[i], -0.999, 0.999)
            offload_power_for_pairing[i] = (clipped[0] + 1) / 2.0

        # === 阶段六：测试也用“同款 mask + 同款配对策略”（MWM 主配 + 软阈值；不硬配）===

        # 1) 构造测试期的“硬可行性”掩码（课程末端：K=mask_topk_end，q=mask_tau_q_end）
        if config.mask_enable:
            K_now = _anneal_topk(
                config.mask_warmup_episodes,  # 用 T_ep 作为“最大episode”，等价课程完成
                config.n_veh,
                config.mask_topk_start,
                config.mask_topk_end,
                config.mask_warmup_episodes
            )
            q_now = float(config.mask_tau_q_end)
            tau_now = _adaptive_threshold_from_delta_g(current_channel_gains, q_now)
            mask_mat = _build_feasible_mask_from_delta_g(current_channel_gains, tau_now, K_now)
            feasible_mask = mask_mat.astype(np.uint8)
        else:
            feasible_mask = np.ones((config.n_veh, config.n_veh), dtype=np.uint8) - np.eye(config.n_veh, dtype=np.uint8)

        # 2) 可选：QoS “软掩码”（仅用于评分扣分，不改变硬候选集合）
        qos_soft_mask = None
        if getattr(config, "qos_enable", False):
            qos_soft_mask = np.zeros_like(feasible_mask, dtype=np.uint8)
            g_lin = current_channel_gains
            Rmin = float(getattr(config, "qos_R_min_bpsHz", 0.0))
            for i in range(config.n_veh):
                for j in range(config.n_veh):
                    if i == j:
                        continue
                    ok = _qos_pair_feasible(
                        i, j,
                        g_lin, offload_power_for_pairing,
                        noise_power=env.noise_power, P_max=env.P_max, R_min=Rmin
                    )
                    qos_soft_mask[i, j] = 1 if ok else 0

        # 3) 评分矩阵（与训练一致：|Δg|_dB + 历史热度 + QoS 软惩罚；测试期历史可设 0 矩阵）
        S0 = _score_matrix_from_gain_and_history(
            gain_linear=current_channel_gains,
            feasible_mask=feasible_mask,
            hist_affinity=np.zeros((config.n_veh, config.n_veh), dtype=np.float32),  # test 不带历史
            w_delta_db=float(getattr(config, "score_w_delta_db", 1.0)),
            w_hist=float(getattr(config, "score_w_history", 0.3)),
            abs_gain_min_db=float(getattr(config, "abs_gain_min_db", -math.inf)),
            qos_soft_mask=qos_soft_mask,
            qos_soft_penalty=float(getattr(config, "qos_soft_penalty_dbscore", 6.0)),
        )

        # 4) 主配对：最大权匹配（MWM）+ 软阈值（按分位数过滤候选边，避免“硬配对”）
        accept_q = float(getattr(config, "mwm_accept_quantile", 0.20))
        pairs = _mwm_primary(
            S=S0,
            feasible=feasible_mask,
            accept_quantile=accept_q,
            allow_singles=bool(getattr(config, "mwm_allow_singles", True)),
        )
        # [PATCH][completion-test-main] 保底补配
        min_pairs_target = max(1, getattr(config, "min_pair_target", config.n_veh // 4))
        if len(pairs) < min_pairs_target:
            pairs = _mwm_completion(S0, feasible_mask, pairs, min_pairs_target)

        # 5) 受限回退（Backoff）：若对数不足，小步放宽（降分位/放宽Top-K/降低τ），每轮重算评分并重跑 MWM；仍不强配
        round_id = 0
        q_step = float(getattr(config, "relax_q_step", 0.02))  # 每轮分位数下降
        K_step = int(getattr(config, "relax_topk_step", 1))  # 每轮 Top-K +1
        tau_factor = float(getattr(config, "relax_tau_factor_per_round", 0.95))  # 每轮 τ×0.95
        min_pairs_target = max(1, getattr(config, "min_pair_target", config.n_veh // 4))


        # 回退起点：用课程末端参数作为初值
        K_back = _anneal_topk(config.mask_warmup_episodes, config.n_veh,
                              config.mask_topk_start, config.mask_topk_end,
                              config.mask_warmup_episodes)
        q_back = float(config.mask_tau_q_end)
        tau_back = _adaptive_threshold_from_delta_g(current_channel_gains, q_back)

        while (len(pairs) < min_pairs_target) and (round_id < int(getattr(config, "mwm_backoff_rounds", 3))):
            round_id += 1
            # 放宽硬掩码
            q_back = max(0.05, q_back - q_step)
            K_back = min(config.n_veh - 1, K_back + K_step)
            tau_back = tau_back * tau_factor
            feasible_mask = _relax_mask_once(feasible_mask, current_channel_gains, tau_back, K_back)

            # 重算评分（含 QoS 软惩罚），并放宽软阈值
            S_back = _score_matrix_from_gain_and_history(
                gain_linear=current_channel_gains,
                feasible_mask=feasible_mask,
                hist_affinity=np.zeros((config.n_veh, config.n_veh), dtype=np.float32),
                w_delta_db=float(getattr(config, "score_w_delta_db", 1.0)),
                w_hist=float(getattr(config, "score_w_history", 0.3)),
                abs_gain_min_db=float(getattr(config, "abs_gain_min_db", -math.inf)),
                qos_soft_mask=qos_soft_mask,
                qos_soft_penalty=float(getattr(config, "qos_soft_penalty_dbscore", 6.0)),
            )
            accept_q = max(0.05, accept_q - float(getattr(config, "mwm_accept_q_step", 0.05)))

            # 重跑 MWM（仍允许单播；不硬配）
            pairs = _mwm_primary(
                S=S_back,
                feasible=feasible_mask,
                accept_quantile=accept_q,
                allow_singles=bool(getattr(config, "mwm_allow_singles", True)),
            )
            # [PATCH][completion-test-backoff] 保底补配
            min_pairs_target = max(1, getattr(config, "min_pair_target", config.n_veh // 4))
            if len(pairs) < min_pairs_target:
                pairs = _mwm_completion(S_back, feasible_mask, pairs, min_pairs_target)

        # 6) 组装分组（与训练一致）：二人组 + 单播
        used_now = set([u for ab in pairs for u in ab])
        singles = [u for u in range(config.n_veh) if u not in used_now]
        noma_groups = [[i, j] for (i, j) in pairs] + [[k] for k in singles]
        # === 统计（测试）：当步的二人组/单播，并累加到全局计数 ===
        pairs_count_step = sum(1 for g in noma_groups if len(g) == 2)
        singles_count_step = sum(1 for g in noma_groups if len(g) == 1)
        test_pairs_total += pairs_count_step
        test_singles_total += singles_count_step

        # ——（可选）统计：你下方已有统计代码，这里不重复

        # 5) 仅把功率传给环境（与训练一致）
        action_for_env = np.zeros([2, config.n_veh], dtype=float)
        for i in range(config.n_veh):
            clipped_power = np.clip(test_power_actions[i], -0.999, 0.999)
            action_for_env[0, i] = (clipped_power[0] + 1) / 2
            action_for_env[1, i] = (clipped_power[1] + 1) / 2

        per_user_reward, global_reward, _, _, _, _, _ = env.step(action_for_env, noma_groups)
        # 收集稳定性指标（按step）
        test_min_user.append(float(np.min(per_user_reward)))
        test_jain.append(float(_jain_index(per_user_reward)))
        total_reward += global_reward
        marl_state_old_all = [marl_get_state(i) for i in range(config.n_veh)]
        test_user_reward_sum += np.asarray(per_user_reward, dtype=np.float64)
        test_steps += 1

    # ===== Stage-6（test）：计算公平性与协同指标，并输出 PASS/FAIL =====
    avg_user_reward = test_user_reward_sum / max(1, test_steps)


    def _jain_index_np(x):
        x = np.asarray(x, dtype=np.float64);
        n = x.size
        if n == 0: return 0.0
        s = x.sum();
        den = n * (x * x).sum() + 1e-12
        return float((s * s) / den)


    jain = _jain_index_np(avg_user_reward)
    min_user = float(np.min(avg_user_reward))
    pair_ratio = test_pairs_total / max(1, (test_pairs_total + test_singles_total))

    print(
        f"[TEST] total_reward={total_reward:.4f}  jain={jain:.4f}  min_user={min_user:.4f}  pair_ratio={pair_ratio:.3f}")

    # —— 简易 PASS/FAIL（若你有“训练基线均值”，可把它带入这里比较 ≤3%）——
    # 真实加载训练基线（取末尾20个episode的均值，你也可改成别的窗口）
    base = _load_training_baseline(args.exp, tail=20)

    pass_jain = (jain >= config.test_target_jain)

    # 计算测试期的“全局回报均值”（按step求均值，与训练记录口径对应）
    test_mean = float(total_reward / max(1, config.n_step_per_episode))

    if base is not None and ("global_mean" in base):
        baseline_mean = float(base["global_mean"])
        drop = max(0.0, (baseline_mean - test_mean)) / (abs(baseline_mean) + 1e-12)
        pass_global_drop = (drop <= config.test_max_global_drop)
    else:
        baseline_mean = None
        drop = None
        pass_global_drop = True  # 没有基线文件时宽松放过

    # “最小用户不下降”：若训练基线存在则做对比；否则沿用不为负的兜底
    if base is not None and ("min_user_mean" in base) and bool(config.test_check_min_user_non_drop):
        baseline_min_user = float(base["min_user_mean"])
        pass_min_user = (min_user >= baseline_min_user - 1e-12)
    else:
        pass_min_user = (min_user >= 0.0)

    ok = (pass_jain and pass_global_drop and pass_min_user)
    print(
        "Stage-6 Test:", "✅ PASS" if ok else "❌ FAIL",
        f"(Jain≥{config.test_target_jain}: {pass_jain}, "
        f"min_user{'≥baseline' if base is not None and ('min_user_mean' in base) else '≥0'}: {pass_min_user}, "
        f"global_drop≤{config.test_max_global_drop}: {pass_global_drop}, "
        f"baseline_mean={baseline_mean}, test_mean={test_mean}, drop={drop})"
    )
    # 把关键指标与判定写到本次 run 的 artifacts 目录中，便于 run_multi 或后续脚本汇总
    try:
        report = {
            "exp": args.exp,
            "seed": int(args.seed),
            "jain": float(jain),
            "min_user": float(min_user),
            "pair_ratio": float(pair_ratio),
            "test_mean_reward": float(test_mean),
            "baseline_mean_reward": (None if baseline_mean is None else float(baseline_mean)),
            "drop": (None if drop is None else float(drop)),
            "max_allowed_drop": float(config.test_max_global_drop),
            "pass": bool(ok),
            "pass_jain": bool(pass_jain),
            "pass_min_user": bool(pass_min_user),
            "pass_global_drop": bool(pass_global_drop),
        }
        art = run_dir / "artifacts"
        art.mkdir(parents=True, exist_ok=True)
        import json

        with open(art / "stage6_test_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print("[SAVE] Stage-6 test report ->", art / "stage6_test_report.json")
    except Exception as e:
        print("[WARN] save stage6_test_report.json failed:", e)
