import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class UAV_Environment(gym.Env):
    def __init__(self, users=250, uavs=1, size=2000, v_0=30, tau=1,
                 UAV_coverage=3000, mBS_coverage=3000,
                 psi_L=1, psi_M=1, K=50, d=1, lambda_c=0.05,  h=120,
                 alpha=2.7, P_UAV=1.0, sigma_square=1e-12,
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

        # Action: 0=stay, 1=up, 2=left, 3=down, 4=right
        self.action_space = spaces.Discrete(5)
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

        # Observation: uav_pos(2) + heatmap_UAV0(grid_num^2) + heatmap_satisfied(grid_num^2)
        self.obs_dim = 2 + 2 * self.grid_num ** 2
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(self.obs_dim,), dtype=np.float32)

        # Internal state
        self.mBS = np.zeros((2, 1)) + self.size / 2
        self._reset_state()

    def _reset_state(self):
        self.uavs_location = np.zeros((2, self.uavs)) + self.size / 2
        self.users_location = np.random.uniform(-self.size / 2, self.size / 2, (2, self.users))
        self.connect = np.zeros((self.uavs + 1, self.users))
        self.unsatisfied_users = np.zeros(self.users)
        self.data_rate_max_index = np.zeros(self.users) - 1000
        self.UAV0_behavior = np.zeros((2, self.max_step)) + self.size / 2
        self.heatmap_UAV0 = np.zeros((self.grid_num, self.grid_num))
        self.heatmap_satisfied = np.zeros((self.grid_num, self.grid_num))
        self.heatmap_users = np.zeros((self.grid_num, self.grid_num))
        self.step_ = 0

    def _get_obs(self):
        return np.concatenate([
            self.uavs_location[:, 0],
            self.heatmap_UAV0.flatten(),
            self.heatmap_satisfied.flatten(),
        ]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        return self._get_obs(), {}

    def step(self, action):
        # Reset per-step buffers
        self.connect = np.zeros((self.uavs + 1, self.users))
        self.unsatisfied_users = np.zeros(self.users)
        self.data_rate_max_index = np.zeros(self.users) - 1000
        self.heatmap_UAV0 = np.zeros((self.grid_num, self.grid_num))
        self.heatmap_satisfied = np.zeros((self.grid_num, self.grid_num))
        self.heatmap_users = np.zeros((self.grid_num, self.grid_num))

        # UAV action
        tem = self.uavs_location[:, 0] + self._action_to_direction[action]
        if not (-self.size / 2 > tem[0] or tem[0] > self.size / 2 or
                -self.size / 2 > tem[1] or tem[1] > self.size / 2):
            self.uavs_location[:, 0] = tem

        # Users move randomly
        for k in range(self.users):
            user_action = np.random.randint(0, 5)
            tem = self.users_location[:, k] + self._user_action_to_direction[user_action]
            if not (-self.size / 2 > tem[0] or tem[0] > self.size / 2 or
                    -self.size / 2 > tem[1] or tem[1] > self.size / 2):
                self.users_location[:, k] = tem

        # Distances
        d_UAV = np.linalg.norm(self.users_location - self.uavs_location[:, 0:1], axis=0)
        d_mBS = np.linalg.norm(self.users_location - self.mBS[:, 0:1], axis=0)

        # Connection
        connect_tem = np.zeros((self.uavs + 1, self.users))
        connect_tem[0, d_UAV <= self.UAV_coverage] = 1
        connect_tem[1, d_mBS <= self.mBS_coverage] = 1

        # SNR UAV
        psi_UAV = math.sqrt(self.K / (1 + self.K)) * self.psi_L + math.sqrt(1 / (1 + self.K)) * self.psi_M
        theta = 10 ** (-20 * math.log10(4 * 3.14 * self.d / self.lambda_c) / 10)
        g_UAV = (abs(psi_UAV) ** 2) * theta * (np.sqrt(d_UAV ** 2 + self.h ** 2) / self.d) ** (-self.alpha)
        gamma_UAV = self.P_UAV * g_UAV / self.sigma_square

        # SNR mBS
        L_mBS = 40 * (1 - 0.004 * self.D_hb) * np.log10(d_mBS / 1000) - 18 * np.log10(self.D_hb) + 21 * np.log10(self.f_c) + 80
        gamma_mBS = self.P_mBS - L_mBS
        gamma_mBS = 10 ** (gamma_mBS / 10)

        # Data rate
        rate_UAV = self.W * np.log2(1 + gamma_UAV)
        rate_mBS = self.W * np.log2(1 + gamma_mBS)

        # Connection selection
        rate_all = np.vstack([rate_UAV[np.newaxis, :], rate_mBS[np.newaxis, :]]) * connect_tem
        self.data_rate_max_index = np.argmax(rate_all, axis=0)
        no_connect = np.all(connect_tem == 0, axis=0)
        self.data_rate_max_index[no_connect] = -1000

        for i in range(self.users):
            if self.data_rate_max_index[i] >= 0:
                self.connect[int(self.data_rate_max_index[i]), i] = 1

        # Satisfied users
        data_rate = np.sum(rate_all * self.connect, axis=0)
        self.unsatisfied_users = (data_rate >= self.r_th).astype(float)
        S_total = float(np.sum(self.unsatisfied_users))
        uav_satisfied = float(np.sum(self.connect[0] * self.unsatisfied_users))
        S = self.reward_alpha * S_total + (1 - self.reward_alpha) * uav_satisfied
        N_UAV0 = int(np.sum(self.connect[0]))

        # Heatmap of users connected to UAV0
        for i in range(self.users):
            if self.connect[0, i] == 1:
                x = int((self.users_location[0, i] + self.size / 2) // self.grid_size)
                y = int((self.users_location[1, i] + self.size / 2) // self.grid_size)
                self.heatmap_UAV0[x, y] += 1

        # Heatmap of unsatisfied users
        for i in range(self.users):
            if self.unsatisfied_users[i] == 0:
                x = int((self.users_location[0, i] + self.size / 2) // self.grid_size)
                y = int((self.users_location[1, i] + self.size / 2) // self.grid_size)
                self.heatmap_satisfied[x, y] += 1

        # Heatmap of all users
        for i in range(self.users):
            x = int((self.users_location[0, i] + self.size / 2) // self.grid_size)
            y = int((self.users_location[1, i] + self.size / 2) // self.grid_size)
            self.heatmap_users[x, y] += 1

        # Track behavior
        self.UAV0_behavior[:, self.step_] = self.uavs_location[:, 0]
        self.step_ += 1

        terminated = False
        truncated = self.step_ >= self.max_step

        return self._get_obs(), S, terminated, truncated, {"N_UAV0": N_UAV0}

    def render(self):
        if self.render_mode == "rgb_array":
            fig, ax = plt.subplots(figsize=(6, 6))
            half = self.size / 2
            ax.set_xlim(-half, half)
            ax.set_ylim(-half, half)
            ax.set_aspect('equal')

            # Users
            for i in range(self.users):
                if self.data_rate_max_index[i] == 0:
                    ax.scatter(self.users_location[0, i], self.users_location[1, i],
                               marker='o' if self.unsatisfied_users[i] == 1 else '^', color='r', s=8)
                elif self.unsatisfied_users[i] == 1:
                    ax.scatter(self.users_location[0, i], self.users_location[1, i],
                               marker='o', color='green', s=8)
                else:
                    ax.scatter(self.users_location[0, i], self.users_location[1, i],
                               marker='x', color='gray', s=8)

            # mBS
            ax.scatter(self.mBS[0], self.mBS[1], marker='s', color='purple', s=80, zorder=5)

            # UAV trail
            if self.step_ > 0:
                ax.scatter(self.UAV0_behavior[0, :self.step_], self.UAV0_behavior[1, :self.step_],
                           color='blue', s=2, alpha=0.5, zorder=4)

            # UAV current position
            ax.scatter(self.uavs_location[0, 0], self.uavs_location[1, 0],
                       marker='s', color='blue', s=150, zorder=5)

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
