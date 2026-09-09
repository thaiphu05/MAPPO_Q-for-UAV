import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class multiUAV(gym.Env):
    def __init__(self, users=250, uavs=3, size=2000, v_0=40, tau=1,
                 UAV_coverage=3000, mBS_coverage=3000,
                 psi_L=1, psi_M=1, K=50, d=1, lambda_c=0.05, h=120,
                 alpha=2.7, P_UAV=1, sigma_square=1e-12,
                 P_mBS=46, D_hb=205, f_c=2, sigma_logF=2,
                 W=20e6, r_th=20e6, grid_num=10, max_step=100,
                 reward_alpha=0.5, render_mode=None):
        super().__init__()

        self.users = users
        self.uavs = uavs
        self.size = size
        self.v_0 = v_0
        self.tau = tau
        self.UAV_coverage = UAV_coverage
        self.mBS_coverage = mBS_coverage
        self.psi_L = psi_L
        self.psi_M = psi_M
        self.K = K
        self.d = d
        self.lambda_c = lambda_c
        self.h = h
        self.alpha = alpha
        self.P_UAV = P_UAV
        self.sigma_square = sigma_square
        self.P_mBS = P_mBS
        self.D_hb = D_hb
        self.f_c = f_c
        self.sigma_logF = sigma_logF
        self.W = W
        self.r_th = r_th
        self.grid_num = grid_num
        self.max_step = max_step
        self.grid_size = self.size / self.grid_num
        self.render_mode = render_mode
        self.reward_alpha = reward_alpha

        # Action encoded as base-5 number: each UAV gets 5 actions (0=stay, 1=up, 2=left, 3=down, 4=right)
        # UAV0 = 5^0 digit, UAV1 = 5^1 digit, UAV2 = 5^2 digit, ...
        self.action_space = spaces.Discrete(5 ** self.uavs)
        self._action_to_direction = {
            0: np.array([0, 0]),
            1: np.array([0, self.v_0 * self.tau]),
            2: np.array([-self.v_0 * self.tau, 0]),
            3: np.array([0, -self.v_0 * self.tau]),
            4: np.array([self.v_0 * self.tau, 0]),
        }
        self._user_action_to_direction = {
            0: np.array([0, 0]),
            1: np.array([0, 5]),
            2: np.array([5, 0]),
            3: np.array([0, -5]),
            4: np.array([-5, 0]),
        }

        # Observation: all UAV positions (2*uavs) + heatmap per UAV (uavs * grid_num^2) + unsatisfied heatmap (grid_num^2)
        self.obs_dim = 2 * self.uavs + (self.uavs + 1) * self.grid_num ** 2
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(self.obs_dim,), dtype=np.float32)

        # Internal state
        self.mBS = np.zeros((2, 1)) + self.size / 2
        self._reset_state()

    def _reset_state(self):
        self.uavs_location = np.zeros((2, self.uavs)) + self.size / 2
        self.users_location = np.random.uniform(-self.size / 2, self.size / 2, (2, self.users))
        self.connect = np.zeros((self.uavs + 1, self.users))
        self.satisfied_users = np.zeros(self.users)
        self.data_rate_max_index = np.zeros(self.users) - 1000
        self.uav_behavior = np.zeros((self.uavs, 2, self.max_step)) + self.size / 2
        self.heatmaps = np.zeros((self.uavs + 1, self.grid_num, self.grid_num))
        self.step_ = 0
        self.prev_satisfied_total = None
        self.prev_N_UAV = None

    def _get_obs(self):
        obs_parts = []
        for i in range(self.uavs):
            obs_parts.append(self.uavs_location[:, i])
        for i in range(self.uavs + 1):
            obs_parts.append(self.heatmaps[i].flatten())
        return np.concatenate(obs_parts).astype(np.float32)

    def get_num_agents(self):
        return self.uavs

    def get_agent_obs_dim(self):
        return 2 + self.grid_num ** 2 + self.grid_num ** 2

    def get_agent_obs(self, agent_idx):
        return np.concatenate([
            self.uavs_location[:, agent_idx],
            self.heatmaps[agent_idx].flatten(),
            self.heatmaps[self.uavs].flatten(),
        ]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        return self._get_obs(), {}

    def step(self, action):
        self.connect = np.zeros((self.uavs + 1, self.users))
        self.satisfied_users = np.zeros(self.users)
        self.data_rate_max_index = np.zeros(self.users) - 1000
        self.heatmaps = np.zeros((self.uavs + 1, self.grid_num, self.grid_num))

        # Move each UAV according to the base-5 encoded action
        for i in range(self.uavs):
            uav_action = (action // (5 ** i)) % 5
            tem = self.uavs_location[:, i] + self._action_to_direction[uav_action]
            if not (-self.size / 2 > tem[0] or tem[0] > self.size / 2 or
                    -self.size / 2 > tem[1] or tem[1] > self.size / 2):
                self.uavs_location[:, i] = tem

        # Users move randomly
        for k in range(self.users):
            user_action = np.random.randint(0, 5)
            tem = self.users_location[:, k] + self._user_action_to_direction[user_action]
            if not (-self.size / 2 > tem[0] or tem[0] > self.size / 2 or
                    -self.size / 2 > tem[1] or tem[1] > self.size / 2):
                self.users_location[:, k] = tem

        # Connection coverage: each UAV + mBS
        connect_tem = np.zeros((self.uavs + 1, self.users))
        for i in range(self.uavs):
            d_UAV = np.linalg.norm(self.users_location - self.uavs_location[:, i:i+1], axis=0)
            connect_tem[i, d_UAV <= self.UAV_coverage] = 1

        d_mBS = np.linalg.norm(self.users_location - self.mBS[:, 0:1], axis=0)
        connect_tem[self.uavs, d_mBS <= self.mBS_coverage] = 1

        # Channel parameters
        psi_UAV = math.sqrt(self.K / (1 + self.K)) * self.psi_L + math.sqrt(1 / (1 + self.K)) * self.psi_M
        theta = 10 ** (-20 * math.log10(4 * 3.14 * self.d / self.lambda_c) / 10)

        # mBS data rate
        L_mBS = 40 * (1 - 0.004 * self.D_hb) * np.log10(d_mBS / 1000) - 18 * np.log10(self.D_hb) + 21 * np.log10(self.f_c) + 80
        gamma_mBS_linear = 10 ** ((self.P_mBS - L_mBS) / 10)
        rate_mBS = self.W * np.log2(1 + gamma_mBS_linear)

        # Per-UAV data rate
        rate_all = np.zeros((self.uavs + 1, self.users))
        for i in range(self.uavs):
            d_UAV = np.linalg.norm(self.users_location - self.uavs_location[:, i:i+1], axis=0)
            g_UAV = (abs(psi_UAV) ** 2) * theta * (np.sqrt(d_UAV ** 2 + self.h ** 2) / self.d) ** (-self.alpha)
            gamma_UAV = self.P_UAV * g_UAV / self.sigma_square
            rate_all[i] = self.W * np.log2(1 + gamma_UAV)

        rate_all[self.uavs] = rate_mBS
        rate_all *= connect_tem

        # Connection selection: each user picks the best available link
        self.data_rate_max_index = np.argmax(rate_all, axis=0)
        no_connect = np.all(connect_tem == 0, axis=0)
        self.data_rate_max_index[no_connect] = -1000

        for i in range(self.users):
            if self.data_rate_max_index[i] >= 0:
                self.connect[int(self.data_rate_max_index[i]), i] = 1

        # Satisfied users
        data_rate = np.sum(rate_all * self.connect, axis=0)
        self.satisfied_users = (data_rate >= self.r_th).astype(float)
        S_total = float(np.sum(self.satisfied_users))

        # Per-UAV served user count
        N_UAV_counts = np.array([float(np.sum(self.connect[i] * self.satisfied_users)) for i in range(self.uavs)])

        # Global reward (paper): sign of change in total satisfied users
        if self.prev_satisfied_total is None:
            gt = 0.0
        else:
            delta = S_total - self.prev_satisfied_total
            gt = 1.0 if delta > 0 else (-1.0 if delta < 0 else 0.0)

        # Local reward (paper): sign of change in per-UAV served users
        if self.prev_N_UAV is None:
            lt = np.zeros(self.uavs)
        else:
            lt = np.zeros(self.uavs)
            for k in range(self.uavs):
                delta_k = N_UAV_counts[k] - self.prev_N_UAV[k]
                lt[k] = 1.0 if delta_k > 0 else (-1.0 if delta_k < 0 else 0.0)

        # Paper eq 6: Rt(k) = wl * lt(k) + (1 - wl) * gt
        wl = 0.5
        agent_rewards = wl * lt + (1 - wl) * gt

        # Store for next step comparison
        self.prev_satisfied_total = S_total
        self.prev_N_UAV = N_UAV_counts.copy()

        # Gym scalar = global teamwork reward (critic estimates this)
        S = gt

        # Heatmaps
        for i in range(self.users):
            x = int((self.users_location[0, i] + self.size / 2) // self.grid_size)
            y = int((self.users_location[1, i] + self.size / 2) // self.grid_size)
            x = min(x, self.grid_num - 1)
            y = min(y, self.grid_num - 1)
            # Heatmap of users connected to each UAV
            if self.data_rate_max_index[i] >= 0 and self.data_rate_max_index[i] < self.uavs:
                self.heatmaps[int(self.data_rate_max_index[i]), x, y] += 1
            # Heatmap of unsatisfied users
            if self.satisfied_users[i] == 0:
                self.heatmaps[self.uavs, x, y] += 1

        # Track behavior for all UAVs
        for i in range(self.uavs):
            self.uav_behavior[i, :, self.step_] = self.uavs_location[:, i]
        self.step_ += 1

        terminated = False
        truncated = self.step_ >= self.max_step

        return self._get_obs(), S, terminated, truncated, {"agent_rewards": agent_rewards, "satisfied_total": S_total}

    def render(self):
        if self.render_mode == "rgb_array":
            fig, ax = plt.subplots(figsize=(6, 6))
            half = self.size / 2
            ax.set_xlim(-half, half)
            ax.set_ylim(-half, half)
            ax.set_aspect('equal')

            colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']

            # Users
            for i in range(self.users):
                uav_idx = int(self.data_rate_max_index[i])
                satisfied = self.satisfied_users[i] == 1

                if 0 <= uav_idx < self.uavs:
                    marker = 'o' if satisfied else '^'
                    color = colors[uav_idx] if satisfied else 'red'
                    ax.scatter(self.users_location[0, i], self.users_location[1, i],
                               marker=marker, color=color, s=8)
                elif satisfied:
                    ax.scatter(self.users_location[0, i], self.users_location[1, i],
                               marker='o', color='purple', s=8)
                else:
                    ax.scatter(self.users_location[0, i], self.users_location[1, i],
                               marker='x', color='gray', s=8)

            # mBS
            ax.scatter(self.mBS[0], self.mBS[1], marker='s', color='purple', s=80, zorder=5)

            # UAV trails and current positions
            for i in range(self.uavs):
                if self.step_ > 0:
                    ax.scatter(self.uav_behavior[i, 0, :self.step_], self.uav_behavior[i, 1, :self.step_],
                               color=colors[i], s=2, alpha=0.5, zorder=4)
                ax.scatter(self.uavs_location[0, i], self.uavs_location[1, i],
                           marker='s', color=colors[i], s=150, zorder=5)

            ax.set_xlabel('x(m)')
            ax.set_ylabel('y(m)')
            ax.set_title(f'Step {self.step_}')

            fig.canvas.draw()
            img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
            plt.close(fig)
            return img
        return None

    def close(self):
        pass
