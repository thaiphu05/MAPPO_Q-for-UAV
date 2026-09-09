# PPO-Q — Hướng Dẫn Toàn Diện Cho Người Dùng

> **Proximal Policy Optimization với Policy/Value Lượng Tử Tham Số Hóa**
> Tài liệu tổng hợp kiến thức dự án cho người đọc là con người — từ cài đặt, hiểu ý tưởng, đến huấn luyện và trực quan hóa kết quả.

---

## Mục Lục

1. [Dự án là gì?](#1-dự-án-là-gì)
2. [Kiến trúc tổng quan](#2-kiến-trúc-tổng-quan)
3. [Cấu trúc thư mục](#3-cấu-trúc-thư-mục)
4. [Yêu cầu & Cài đặt](#4-yêu-cầu--cài-đặt)
5. [Hai chế độ huấn luyện](#5-hai-chế-độ-huấn-luyện)
6. [Tham số cấu hình (config YAML)](#6-tham-số-cấu-hình-config-yaml)
7. [Môi trường (Environments)](#7-môi-trường-environments)
8. [Mô hình Lượng Tử - Cổ Điển](#8-mô-hình-lượng-tử---cổ-điển)
9. [Quy trình huấn luyện PPO / MAPPO](#9-quy-trình-huấn-luyện-ppo--mappo)
10. [Theo dõi, lưu trọng số & render](#10-theo-dõi-lưu-trọng-số--render)
11. [Kết quả thực nghiệm](#11-kết-quả-thực-nghiệm)
12. [Ví dụ lệnh nhanh](#12-ví-dụ-lệnh-nhanh)
13. [Hạn chế & Hướng mở rộng](#13-hạn-chế--hướng-mở-rộng)
14. [FAQ](#14-faq)
15. [Tài liệu tham khảo nội bộ](#15-tài-liệu-tham-khao-nội-bộ)

---

## 1. Dự án là gì?

**PPO-Q** thay thế mạng Actor (và tùy chọn Critic) trong thuật toán **PPO** bằng **mạch lượng tử tham số hóa (PQC - Parametrized Quantum Circuit)** chạy trên mô phỏng `torchquantum` hoặc phần cứng lượng tử đám mây **Quafu**.

Mục tiêu:

- Kiểm chứng khả năng giải các bài toán RL cổ điển (CartPole, LunarLander, BipedalWalker...) bằng policy lượng tử lai.
- Giải bài toán **điều khiển đội UAV** (Unmanned Aerial Vehicle) phục vụ người dùng mặt đất — một bài toán tối ưu vị trí, phủ sóng và tốc độ dữ liệu.

Dự án cung cấp **hai pipeline tách biệt**:

| Pipeline | Entry point | Thuật toán | Actor | Critic | Môi trường chính |
|----------|-------------|------------|-------|--------|-----------------|
| Single-Agent | `main.py` | PPO | Lượng tử (`DiscreteActor`/`ContinuousActor`) | MLP cổ điển | Gym + `UAV_Environment` |
| Multi-Agent | `main_mappo.py` | MAPPO (CTDE) | Lượng tử `MAPPOActor` (mỗi agent 1 PQC) | Centralized Critic MLP | `multiUAV` (3 UAV) |

> **CTDE** = Centralized Training, Decentralized Execution.

---

## 2. Kiến trúc tổng quan

```
                  ┌─────────────────────────────────────────┐
                  │           File Cấu Hình YAML             │
                  │  config/CartPole.yaml, UAV.yaml,         │
                  │  multiUAV_MAPPO.yaml ...                 │
                  └──────────────┬──────────────────────────┘
                                 │ load
              ┌──────────────────▼──────────────────┐
              │  main.py / main_mappo.py             │
              │  setup_training() / setup_mappo_     │
              │  training()  (model/utils.py)        │
              └──────────┬───────────────────────────┘
                         │
          ┌──────────────▼──────────────┐
          │  Vectorized Env (AsyncVectorEnv) │
          │  num_envs song song (8~32)       │
          │  Gym / UAV_env / multiUAV_env    │
          └──────┬──────────────────┬────────┘
                 │ obs              │ reward
     ┌───────────▼──────┐  ┌────────▼────────┐
     │  Actor (Quantum) │  │ Critic (MLP)    │
     │  pre Linear ─►   │  │  [256,128] Tanh │
     │  PQCLayer  ─►    │  │  hoặc [64,64]   │
     │  post Linear ─►  │  │  → V(s)         │
     │  Softmax/Beta    │  └────────┬────────┘
     └───────────┬──────┘           │
                 │ action           │ value
                 └────────┬─────────┘
                          │ GAE + PPO Clip Loss
                 ┌────────▼────────┐
                 │ Adam Optimizer  │
                 │ TensorBoard log │
                 │ weights/*.pt    │
                 └─────────────────┘
```

**Luồng dữ liệu chi tiết:**

1. YAML → `argparse.Namespace` (`batch_size = n_steps * num_envs`).
2. `AsyncVectorEnv` thu thập `batch_step = batch_size / num_envs` bước.
3. Actor sinh `action` + `log_prob`, Critic sinh `V(s)`.
4. Tính **GAE advantage** và **value target**, chuẩn hóa advantage theo mini-batch.
5. Tối ưu `K_epochs` vòng qua `BatchSampler`, loss = `actor_loss - entropy_coef*entropy + 0.5*critic_loss`, clip grad 0.5.
6. Log `episodic_return`, `avg_satisfied_users`, `approx_kl`, `clipfrac`, `v_explained` ra TensorBoard.

---

## 3. Cấu trúc thư mục

```
PPO-Q/
├── main.py                 # Entry single-agent PPO
├── main_mappo.py           # Entry multi-agent MAPPO
├── UAV_env.py              # Env 1 UAV (Discrete 5, obs 202)
├── multiUAV_env.py         # Env 3 UAV (Discrete 125, obs 306)
├── render.py               # Render GIF + quỹ đạo 1 UAV
├── multiuav_render.py      # Render GIF + quỹ đạo 3 UAV
├── plot_rewards.py         # Vẽ reward curve từ TensorBoard
├── config/                 # 11 file YAML mẫu
│   ├── CartPole.yaml, Acrobot.yaml, LunarLander.yaml ...
│   ├── UAV.yaml
│   ├── multiUAV_MAPPO.yaml          # Quantum
│   └── multiUAV_MAPPO_normal.yaml   # MLP thuần (baseline)
├── model/
│   ├── models.py           # DiscreteActor, ContinuousActor, Critic, PQCLayer
│   ├── mappo_models.py     # MAPPOActor (quantum) + CentralizedCritic
│   ├── mappo_models_normal.py # MAPPOActor (MLP ReLU) — baseline
│   ├── trainer.py          # Lớp PPO + hàm trainer() single-agent
│   ├── mappo_trainer.py    # Lớp MAPPOTrainer + trainer() multi-agent
│   └── utils.py            # make_env, setup_training, init helpers
├── third_party/torchquantum/ # Thư viện mô phỏng lượng tử
├── runs/                   # Log TensorBoard + GIF render
├── weights/                # Trọng số đã huấn luyện
├── img/                    # GIF demo CartPole, Pendulum ...
└── docs/
    ├── PPO-Q-GUIDE.md      # ← File này (human)
    └── PPO-Q-AI.md         # File cho AI agent
```

---

## 4. Yêu cầu & Cài đặt

**Yêu cầu:** Python 3.10.20, CUDA 13 tùy chọn (fallback CPU). Đã kiểm chứng trên conda env `Paper_2`.

**Cách A — Conda (khuyến nghị):**
```bash
conda create -n Paper_2 python=3.10.20 -y && conda activate Paper_2
conda env update -f environment.yml --prune   # hoặc conda env create -f environment.yml
# Hoặc cài tay:
pip install torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cu130
pip install --editable ./third_party/torchquantum
pip install "quarkstudio==7.1.8" "qiskit==1.4.5" "qiskit-aer==0.13.3" "qiskit-ibm-runtime==0.20.0"
pip install "gymnasium[box2d]==0.29.1" box2d-py==2.3.5 pygame==2.6.1
pip install numpy==1.26.4 scipy==1.15.3 matplotlib==3.10.9 pyyaml==6.0.3 tqdm==4.67.3 tensorboard==2.20.0 imageio==2.37.3
```

**Cách B — pip thuần:**
```bash
pip install torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cu130  # CPU: .../whl/cpu
pip install --editable ./third_party/torchquantum
pip install "quarkstudio==7.1.8" "qiskit==1.4.5" "qiskit-aer==0.13.3" "qiskit-ibm-runtime==0.20.0"
pip install "gymnasium[box2d]==0.29.1" box2d-py==2.3.5 pygame==2.6.1
pip install numpy==1.26.4 scipy matplotlib pyyaml tqdm tensorboard imageio
```

> `quarkstudio` + `qiskit` chỉ cần khi `use_quafu: true`. `environment.yml` ở root export từ `Paper_2`.

Kiểm tra cài đặt:

```bash
python -c "import torchquantum; import quark; import qiskit; import gymnasium; print('OK', torch.__version__)"
tensorboard --logdir=./runs  # mở http://localhost:6006
```

---

## 5. Hai chế độ huấn luyện

### 5.1 Single-Agent PPO (`main.py`)

```bash
python main.py CartPole              # CartPole-v1, config/CartPole.yaml
python main.py LunarLander           # LunarLander-v2
python main.py BipedalWalker         # BipedalWalker-v3 (liên tục)
python main.py UAV                   # UAV đơn
python main.py MountainCar           # v.v.
```

Logic `main.py:6-13`: đọc `sys.argv[1]` (mặc định `CartPole`), gọi `get_config_path` → `setup_training` → `trainer(args)`.

### 5.2 Multi-Agent MAPPO (`main_mappo.py`)

```bash
python main_mappo.py multiUAV_MAPPO          # Quantum (mặc định)
python main_mappo.py multiUAV_MAPPO_normal   # MLP baseline (use_quantum: false)
```

Logic `main_mappo.py:6-13`: tương tự, gọi `setup_mappo_training`.

### 5.3 So sánh

| Tiêu chí | Single PPO | MAPPO Quantum | MAPPO Normal |
|----------|-----------|---------------|--------------|
| File config | `CartPole.yaml` ... `UAV.yaml` | `multiUAV_MAPPO.yaml` | `multiUAV_MAPPO_normal.yaml` |
| Số agent | 1 | 3 | 3 |
| Actor | 1 PQC | 3 PQC (ModuleList) | 3 MLP `[1024,512]` |
| Critic | MLP `[64,64]` Tanh | Centralized `[256,128]` | `[128,128]` |
| Action | `Discrete(n)` hoặc `Box` | `Discrete(5)` mỗi agent, encode base-5 | giống quantum |
| Reward | env gốc | `Rt(k)=0.5*lt+0.5*gt` (sign delta) | giống quantum |

---

## 6. Tham số cấu hình (config YAML)

Ví dụ `config/multiUAV_MAPPO.yaml`:

```yaml
env_name: 'multiUAV'
n_steps: 1024            # số bước mỗi env mỗi update
mini_batch_size: 64
max_train_steps: 1000000
lr_a: 0.001              # learning rate actor
lr_c: 0.001              # learning rate critic
gamma: 0.99              # discount
lamda: 0.98              # GAE lambda
epsilon: 0.1             # PPO clip
K_epochs: 4
entropy_coef: 0.01
num_envs: 32
normalize_state: False
normalize_reward: False
is_continuous: False
clip_decay: False
lr_decay: False
ini_method: 9            # index 0..11 → cặp [pre, post] init
seed: 42
n_blocks: 2              # số block lượng tử
n_wires: 4               # số qubit
critic_hidden: [64, 64]
```

Bảng đầy đủ:

| Tham số | Ý nghĩa | Giá trị điển hình |
|---------|---------|-------------------|
| `env_name` | Tên env (`CartPole-v1`, `UAV_Environment`, `multiUAV`...) | `multiUAV` |
| `n_steps` | Bước thu thập mỗi env mỗi vòng update | 32 (CartPole) / 1024 (UAV) |
| `mini_batch_size` | Kích thước mini-batch cho SGD | 64 ~ 256 |
| `max_train_steps` | Tổng số bước huấn luyện | 100K ~ 4M |
| `lr_a` / `lr_c` | LR actor / critic | 0.0001 ~ 0.01 |
| `gamma` | Hệ số chiết khấu | 0.98 ~ 0.99 |
| `lamda` | Tham số GAE | 0.8 ~ 0.98 |
| `epsilon` | Ngưỡng clip PPO | 0.1 ~ 0.2 |
| `K_epochs` | Số epoch PPO mỗi batch | 4 ~ 20 |
| `entropy_coef` | Hệ số khuyến khích khám phá | 0.0 ~ 0.02 |
| `num_envs` | Số env chạy song song | 8 ~ 32 |
| `normalize_state/reward` | Chuẩn hóa obs/reward (clip ±10) | False/True |
| `clip_decay` / `lr_decay` | Giảm epsilon/LR tuyến tính theo tiến trình | True/False |
| `ini_method` | 0..11 ánh xạ tới `[['NOT','I'], ['NOT','O'], ... ['X','X']]` | 0, 7, 9 |
| `n_blocks` / `n_wires` | Số block / số qubit PQC | 1~2 / 4 |
| `critic_hidden` / `actor_hidden` | Kích thước hidden MLP (chỉ MAPPO) | `[64,64]` / `[1024,512]` |
| `use_quantum` | `false` → dùng MLP thay PQC (chỉ MAPPO normal) | true/false |
| `seed` | Seed ngẫu nhiên | 42 |

> `ini_method` quyết định cách khởi tạo `pre_encoding_net` và `post_processing_net`: `NOT`=Identity, `I`=ones, `O`=orthogonal (gain 0.01 cho lớp cuối), `X`=xavier.

---

## 7. Môi trường (Environments)

### 7.1 Gym cổ điển (8 env)

| Env | Quan sát | Hành động | Ghi chú |
|-----|----------|-----------|---------|
| CartPole-v1 | 4 | Discrete 2 | `n_steps=32` |
| MountainCar-v0 | 2 | Discrete 3 | có reward shaping `+ sin(3x)` |
| Acrobot-v1 | 6 | Discrete 3 | |
| LunarLander-v2 | 8 | Discrete 4 | |
| MountainCarContinuous | 2 | Box 1 | `is_continuous: true`, Beta policy |
| Pendulum-v1 | 3 | Box 1 | |
| LunarLanderContinuous | 8 | Box 2 | |
| BipedalWalker-v3 | 24 | Box 4 | `normalize_state/reward: true` |

### 7.2 UAV đơn (`UAV_env.py`)

- **Mục tiêu:** 1 UAV bay trong bản đồ 2000×2000 m, phục vụ 250 user ngẫu nhiên, cạnh tranh với 1 trạm gốc mBS ở trung tâm.
- **Action:** `Discrete(5)` — 0=đứng yên, 1=lên, 2=trái, 3=xuống, 4=phải (bước `v0*tau = 30m`).
- **Quan sát (202 chiều):** `uav_pos(2) + heatmap_UAV0(100) + heatmap_unsatisfied(100)` với `grid 10×10`.
- **Mô hình kênh:**
  - UAV: Rician fading `K=50`, path loss `alpha=2.7`, độ cao `h=120m`.
  - mBS: `L = 40(1-0.004Dh)log10(d/1000) -18log10(Dh)+21log10(fc)+80`.
  - Tốc độ `W log2(1+SNR)`, ngưỡng `r_th=20 Mbps`.
- **Reward:** `S = 0.5*S_total + 0.5*S_uav` (tổng user thỏa + user do UAV phục vụ), episode 100 bước.

### 7.3 Multi-UAV (`multiUAV_env.py`) — Trọng tâm MAPPO

- **Cấu hình:** 3 UAV, 250 user, cùng bản đồ 2000m, mỗi UAV bước 40m.
- **Action:** `Discrete(125)` = `5^3`, giải mã base-5: `UAV_i = (action // 5^i) % 5`.
- **Quan sát toàn cục (306 chiều):** `2*3 (vị trí UAV) + 3*100 (heatmap mỗi UAV) + 100 (heatmap unsatisfied)`.
- **Quan sát cục bộ mỗi agent (202 chiều):** `pos(2) + heatmap_riêng(100) + heatmap_unsatisfied(100)` — trích qua `agent_slices`.
- **Reward (quan trọng):**
  ```
  gt = sign(S_total(t) - S_total(t-1))          # global teamwork
  lt(k) = sign(N_k(t) - N_k(t-1))               # local per-UAV
  Rt(k) = 0.5*lt(k) + 0.5*gt                     # cho actor
  S = gt                                         # cho critic (team reward)
  ```
  Giá trị chỉ là -1/0/+1, khuyến khích tăng số user thỏa.
- **User di chuyển ngẫu nhiên** mỗi bước 5m, `max_step=100`, `truncated` khi hết bước.

---

## 8. Mô hình Lượng Tử - Cổ Điển

### 8.1 Actor lượng tử (`model/models.py:25`)

```
Input (state_dim) → pre_encoding Linear(n_wires) → PQCLayer → post Linear(output_dim) → Softmax/Beta
                    (hoặc Identity nếu ini_method[0]=="NOT" và n_wires==input_dim)
```

- **Khởi tạo:** `pre` theo `ini_method[0]`, `post` theo `ini_method[1]` (orthogonal gain 0.01 nếu `O`).
- **ContinuousActor** (`models.py:124`): thay vì Softmax, ra `alpha, beta = softplus(Linear)+1` → `Beta(alpha,beta)` distribution.

### 8.2 PQCLayer (`models.py:276`) — Trái tim lượng tử

```
H (Hadamard toàn bộ qubit)
repeat n_blocks:
    RzRyVariational (RZ trainable + RY trainable + CZ entangle)
    RyRzScaleEncoder (RY(λ·tanh(λ·x)) + RZ(λ·tanh(λ·x)))
RzRyVariational cuối (không entangle)
MeasureAll PauliZ → vector n_wires chiều (expectation -1..1)
```

- `ScaleEncoder` (`models.py:374`): `x_scaled = tanh(λ * x)` với `λ` là tham số học được, rồi encode qua `RY`/`RZ`.
- Entanglement: `CZ` vòng tròn nếu `n_wires>2`, tuyến tính nếu `n_wires==2`.
- **Chế độ Quafu:** `use_quafu=true` → chuyển mạch sang QASM qua `tq2qiskit`, gửi lên `quark.Task`, dùng gradient mô phỏng để bù `delta` (straight-through estimator, `models.py:371`).

### 8.3 Critic cổ điển

- Single PPO: `Critic([64,64], Tanh)` — 2 hidden Tanh, orthogonal init gain `sqrt(2)`.
- MAPPO: `CentralizedCritic` — nhận `global_obs` (306), ra `V(s)` vô hướng, hidden `[256,128]` hoặc `[64,64]`.

### 8.4 MAPPOActor

- **Quantum** (`mappo_models.py:10`): `ModuleList` 3 `DiscreteActor` độc lập, mỗi agent quan sát riêng.
- **Normal** (`mappo_models_normal.py:24`): 3 MLP `ReLU [1024,512] → Softmax`, orthogonal init.

---

## 9. Quy trình huấn luyện PPO / MAPPO

### 9.1 Single PPO (`model/trainer.py:94`)

1. Tạo `AsyncVectorEnv` với `num_envs` env, tùy chọn `NormalizeObservation/Reward`.
2. Vòng `all_step = max_train_steps / batch_size`:
   - Thu thập `batch_step` bước: `interact_with_env` → `env.step` → lưu `b_s, b_a, b_logprob, b_r, b_vs, b_done`.
   - Xử lý `truncated` bằng bootstrapping `r += gamma * V(final_obs)`.
   - Tính GAE ngược: `delta = r + gamma*(1-done)*V' - V`, `adv = delta + gamma*lamda*adv_next`.
3. Tối ưu `K_epochs` × mini-batch:
   - `ratio = exp(logprob_now - logprob_old)`, clip `[1-eps, 1+eps]`.
   - `actor_loss = -min(ratio*adv_norm, clipped*adv_norm).mean()`
   - `critic_loss = MSE(V_target, V(s))`, `entropy` từ `Categorical`/`Beta`.
   - `loss = actor_loss - entropy_coef*entropy + 0.5*critic_loss`, clip grad 0.5.

### 9.2 MAPPO (`model/mappo_trainer.py:131`)

Khác biệt chính:

- Quan sát toàn cục được tách thành `agent_obs` qua `agent_slices` (`mappo_trainer.py:87`).
- Thu thập thêm `b_agent_r` (per-agent, -1/0/1) và `b_satisfied` từ `info["agent_rewards"]` / `info["satisfied_total"]`.
- **Hai advantage:** `b_adv` (team, cho critic) và `b_adv_agent` (per-agent, cho actor), cả hai đều dùng GAE với `V(s)` chung.
- Actor loss dùng `b_adv_agent` chuẩn hóa, critic loss dùng `b_adv`.

---

## 10. Theo dõi, lưu trọng số & render

### TensorBoard

```bash
tensorboard --logdir=./runs
```

Tags quan trọng: `charts/episodic_return`, `charts/episodic_length`, `charts/avg_reward`, `charts/avg_satisfied_users`, `losses/actor_loss`, `losses/critic_loss`, `losses/entropy`, `losses/approx_kl`, `losses/clipfrac`, `losses/v_explained`.

### Lưu trọng số

- Single: `weights/env_{env}_{timestamp}/PPO_actor.pt`, `PPO_critic.pt` (+ `PPO_norm_stats.txt` nếu normalize).
- MAPPO: `weights/env_multiUAV_MAPPO{Quantum|Normal}_{timestamp}/MAPPO_actor.pt`, `MAPPO_critic.pt`.

### Render

```bash
python render.py UAV                    # 1 UAV → runs/UAV_render.gif + trajectory.png
python multiuav_render.py multiUAV_MAPPO        # 3 UAV quantum
python multiuav_render.py multiUAV_MAPPO_normal # 3 UAV MLP
```

- `render.py:47-59` load `PPO_actor.pt` mới nhất, chạy 1 episode, lưu GIF 10fps.
- `multiuav_render.py:55-59` load `MAPPO_actor.pt`, giải mã base-5, vẽ quỹ đạo 3 UAV + mBS + user (màu theo UAV, `o`=thỏa, `^`/`x`=không).

### So sánh reward

```bash
python plot_rewards.py --compare-mappoq          # overlay Quantum vs Normal
python plot_rewards.py --both                    # cả single + MAPPO
python plot_rewards.py runs/env_multiUAV_MAPPO_* --window 50 --output runs/compare.png
```

---

## 11. Kết quả thực nghiệm

Bảng env đã giải (từ `README.md`):

| Env | State | Action | Trạng thái |
|-----|-------|--------|------------|
| CartPole-v1 | 4 | 2 | ✅ |
| MountainCar-v0 | 2 | 3 | ✅ |
| Acrobot-v1 | 6 | 3 | ✅ |
| LunarLander-v2 | 8 | 4 | ✅ |
| MountainCarContinuous | 2 | 1 | ✅ |
| Pendulum-v1 | 3 | 1 | ✅ |
| LunarLanderContinuous | 8 | 2 | ✅ |
| BipedalWalker-v3 | 24 | 4 | ✅ |

Ảnh GIF demo trong `img/` (CartPole, Acrobot, LunarLander, Pendulum, BipedalWalker) và `multiUAV_MAPPO_trajectory.png` cho thấy UAV học cách phủ vùng đông user, tăng số user thỏa từ ~50 lên ~150/250 sau huấn luyện.

---

## 12. Ví dụ lệnh nhanh

```bash
# Huấn luyện nhanh CartPole (100K bước, 8 env)
python main.py CartPole

# Huấn luyện UAV đơn (1M bước, 16 env)
python main.py UAV

# Huấn luyện MAPPO quantum 3 UAV (1M bước, 32 env)
python main_mappo.py multiUAV_MAPPO

# Baseline MLP để so sánh
python main_mappo.py multiUAV_MAPPO_normal

# Xem log
tensorboard --logdir=./runs

# Render sau huấn luyện
python multiuav_render.py multiUAV_MAPPO
python plot_rewards.py --compare-mappoq --window 50
```

---

## 13. Hạn chế & Hướng mở rộng

**Hạn chế hiện tại:**

- Mô phỏng lượng tử `torchquantum` chậm khi `n_wires>6`, chưa tối ưu batch lớn.
- Reward multiUAV dạng sign (-1/0/1) thưa, có thể gây dao động.
- Chưa có kiểm thử tự động, chưa có CI.

**Hướng mở rộng:**

- Thử `n_blocks=3`, `n_wires=6` hoặc ansatz khác (`RxRyRzVariationalLayer`).
- Thay reward bằng tỉ lệ user thỏa liên tục hoặc thêm phạt di chuyển.
- Thêm env `multiUAV` với số UAV thay đổi, chướng ngại vật.
- Tích hợp `use_quafu` với key thật để chạy trên phần cứng.

---

## 14. FAQ

**Q: Tại sao `ini_method` là số 0..11?**
A: Ánh xạ tới cặp `[pre_init, post_init]` trong `utils.py:120`. Ví dụ `9` → `['X','O']` (Xavier cho pre, Orthogonal gain 0.01 cho post).

**Q: Khác nhau giữa `normalize_state` và `normalize_reward`?**
A: `NormalizeObservation` duy trì mean/var chạy, clip obs ±10; `NormalizeReward` chuẩn hóa reward theo gamma, clip ±10. BipedalWalker bật cả hai.

**Q: Muốn thêm env mới thì sửa đâu?**
A: Thêm YAML trong `config/`, thêm nhánh trong `utils.py:57 make_env` và `trainer.py:85 make_env_direct` / `mappo_trainer.py:22`.

**Q: File trọng số nào được load khi render?**
A: Thư mục `weights/env_{name}_*` mới nhất theo `sorted(glob)`.

**Q: Làm sao biết quantum hay classical đang chạy?**
A: Kiểm tra `args.use_quantum` trong log `MAPPOTrainer:50`; tag thư mục `MAPPOQuantum` vs `MAPPONormal`.

---

## 15. Tài liệu tham khảo nội bộ

- `README.md` — hướng dẫn gốc
- `model/models.py` — định nghĩa PQC chi tiết
- `UAV_env.py` / `multiUAV_env.py` — công thức kênh, heatmap, reward
- `docs/PPO-Q-AI.md` — phiên bản cho AI agent (cấu trúc, contracts, invariants)

---

*Tài liệu được tổng hợp tự động từ mã nguồn tại thời điểm 2026. Khi thêm tính năng, hãy cập nhật cả hai file `PPO-Q-GUIDE.md` và `PPO-Q-AI.md`.*
