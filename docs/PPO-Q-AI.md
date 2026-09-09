# PPO-Q — AI Knowledge Transfer (Exhaustive)

> **Mục đích:** Tài liệu bàn giao toàn bộ kiến thức dự án cho AI kế nhiệm, để AI đó có thể đọc 1 file này là đủ để hiểu, sửa, mở rộng, debug toàn bộ codebase mà không cần hỏi lại người.
> **Companion:** `docs/PPO-Q-GUIDE.md` (phiên bản cho người). File này là **source of truth cho AI**.
> **Ngôn ngữ:** Tiếng Việt + thuật ngữ/ký hiệu tiếng Anh giữ nguyên để AI cross-lingual.
> **Độ dài:** Cực chi tiết — mọi file, mọi hàm, mọi tensor shape, mọi công thức, mọi config, mọi pitfall đều được ghi.

```yaml
project: PPO-Q
full_name: Proximal Policy Optimization with Parametrized Quantum Policies / Values
repo_root: /home/trinhpt/PPO-Q
entry_single: main.py              # python main.py <config>
entry_multi: main_mappo.py         # python main_mappo.py <config>
python: 3.10
core_deps: [torch==2.12.0, torchvision==0.27.0, torchquantum==0.1.8(third_party/editable), quarkstudio==7.1.8, qiskit==1.4.5+qiskit-aer==0.13.3, gymnasium[box2d]==0.29.1+box2d-py==2.3.5+pygame==2.6.1+swig==4.4.1, numpy==1.26.4, scipy==1.15.3, matplotlib==3.10.9, pyyaml==6.0.3, tqdm, tensorboard==2.20.0, imageio==2.37.3, cuda-toolkit==13.0.2(optional), python==3.10.20]
rl_algo: PPO (single-agent) + MAPPO-CTDE (multi-agent, 3 UAV)
quantum_sim: torchquantum  # default
quantum_hw: Quafu cloud via quark.Task + qiskit.qasm2.dumps  # optional
device: cuda if available else cpu  # model/trainer.py:16, mappo_trainer.py:16
created: 2026-09-09
maintainer_handoff: true
```

---

## MỤC LỤC

1. [Tóm tắt 30 giây cho AI](#1-tóm-tắt-30-giây-cho-ai)
2. [Cây thư mục & vai trò từng file](#2-cây-thư-mục--vai-trò-từng-file)
3. [11 file config — ma trận đầy đủ](#3-11-file-config--ma-trận-đầy-đủ)
4. [Utils — nền tảng chung](#4-utils--nền-tảng-chung)
5. [Môi trường — vật lý, toán, reward](#5-môi-trường--vật-lý-toán-reward)
6. [Mô hình — PQC chi tiết đến từng cổng lượng tử](#6-mô-hình--pqc-chi-tiết-đến-từng-cổng-lượng-tử)
7. [Trainer — thuật toán PPO/MAPPO từng dòng](#7-trainer--thuật-toán-ppomappo-từng-dòng)
8. [Render & Plot — tái tạo kết quả](#8-render--plot--tái-tạo-kết-quả)
9. [Luồng thực thi end-to-end (trace)](#9-luồng-thực-thi-end-to-end-trace)
10. [Tensor shapes — bảng tra nhanh](#10-tensor-shapes--bảng-tra-nhanh)
11. [Dependency & import graph](#11-dependency--import-graph)
12. [Invariants, pitfalls, bug đã biết](#12-invariants-pitfalls-bug-đã-biết)
13. [Công việc thường gặp — recipe copy-paste](#13-công-việc-thường-gặp--recipe-copy-paste)
14. [Lệnh reproduce — copy chạy ngay](#14-lệnh-reproduce--copy-chạy-ngay)
15. [Checklist bàn giao — AI mới tự kiểm tra](#15-checklist-bàn-giao--ai-mới-tự-kiểm-tra)
16. [Phụ lục — code snippets quan trọng](#16-phụ-lục--code-snippets-quan-trọng)

---

## 1. Tóm tắt 30 giây cho AI

- **PPO-Q = PPO cổ điển nhưng Actor là mạch lượng tử tham số hóa (PQC).** Critic vẫn là MLP Tanh `[64,64]`. Mô phỏng lượng tử bằng `torchquantum`, có thể đẩy lên phần cứng Quafu (QASM + `quark.Task`).
- **2 pipeline tách biệt hoàn toàn:**
  - `main.py` → `model/trainer.py` → `model/models.py` → Gym 8 env + `UAV_env.py` (1 UAV, Discrete 5).
  - `main_mappo.py` → `model/mappo_trainer.py` → `model/mappo_models.py` (quantum) hoặc `mappo_models_normal.py` (MLP baseline) → `multiUAV_env.py` (3 UAV, Discrete 125 = 5^3, CTDE).
- **Reward multiUAV đặc thù:** không phải số liên tục mà là `sign(delta)` ∈ {-1,0,1}, tính per-agent + global rồi blend `0.5*lt+0.5*gt`. Đây là nguồn gây sparse/oscillation nếu không biết.
- **Quan sát UAV dạng heatmap lưới 10×10** (100 ô) — mọi thay đổi `grid_num` sẽ vỡ `agent_slices` và `obs_dim`.
- **Không có test, không có CI, không có docstring đầy đủ** — AI phải dựa vào file này + đọc code `file:line`.

---

## 2. Cây thư mục & vai trò từng file

```
PPO-Q/
├── main.py                     # 13 dòng: parse argv[1] or "CartPole" → get_config_path → setup_training → trainer()
├── main_mappo.py               # 13 dòng: tương tự, default "multiUAV_MAPPO" → setup_mappo_training
├── UAV_env.py                  # 234 dòng: gym.Env 1 UAV, 250 user, map 2000m, Rician + mBS path loss
├── multiUAV_env.py             # 279 dòng: gym.Env 3 UAV, 250 user, base-5 action, heatmap, sign reward
├── render.py                   # 140 dòng: load PPO_actor.pt mới nhất → rollout 1 ep → GIF + trajectory.png
├── multiuav_render.py          # 173 dòng: load MAPPO_actor.pt → base-5 decode → 3 quỹ đạo + user scatter
├── plot_rewards.py             # 169 dòng: đọc TensorBoard EventAccumulator → smooth → overlay Quantum vs Normal
├── config/                     # 11 YAML — xem §3
├── model/
│   ├── __init__.py             # rỗng
│   ├── utils.py                # 159 dòng: init helpers, make_env, gen_seeds, setup_training, get_config_path
│   ├── models.py               # 455 dòng: DiscreteActor, ContinuousActor, Critic, PQCLayer, Encoders
│   ├── mappo_models.py         # 80 dòng: MAPPOActor(quantum ModuleList) + CentralizedCritic
│   ├── mappo_models_normal.py  # 89 dòng: MAPPOActor(MLP) + CentralizedCritic (baseline)
│   ├── trainer.py              # 274 dòng: class PPO + trainer() single
│   └── mappo_trainer.py        # 410 dòng: class MAPPOTrainer + trainer() multi + setup_mappo_training
├── third_party/torchquantum/   # editable install, cung cấp tq.QuantumDevice, tq.MeasureAll, tq2qiskit
├── runs/                       # TensorBoard logs: env_{name}_{timestamp}/events.out.tfevents.*
├── weights/                    # checkpoints: env_{name}_{ts}/PPO_actor.pt, MAPPO_actor.pt (+ norm stats)
├── img/                        # GIF demo gốc (CartPole, Pendulum...)
├── . =1.2.0                    # rác? file rỗng tên "=1.2.0" — bỏ qua
└── docs/
    ├── PPO-Q-GUIDE.md          # human guide (481 dòng)
    └── PPO-Q-AI.md             # file này (AI exhaustive)
```

**Quy ước đặt tên:**
- `trainer.py` vs `mappo_trainer.py` — không dùng chung, dù 70% logic giống nhau.
- `mappo_models.py` = quantum, `mappo_models_normal.py` = classical — tên gây nhầm, nhớ `use_quantum` flag.
- `UAV_Environment` (class) vs `multiUAV` (class) — chữ hoa/thường không nhất quán.
- `runs/env_multiUAV_MAPPO_*` có 2 dạng tag: cũ không có `Quantum/Normal` (đều là quantum), mới có `MAPPOQuantum`/`MAPPONormal`.

---

## 3. 11 file config — ma trận đầy đủ

### 3.1 Schema chung

**Single (`model/utils.py:94 setup_training`):**
```python
# Bắt buộc trong YAML:
env_name, n_steps, mini_batch_size, max_train_steps, lr_a, lr_c, gamma, lamda, epsilon, K_epochs, entropy_coef, num_envs, normalize_state, normalize_reward, clip_decay, lr_decay, ini_method, seed, n_blocks, n_wires
# Suy ra:
batch_size = n_steps * num_envs          # utils.py:118
ini_method = ini_method_list[ini_method_index]  # utils.py:120-124
ini_method_list = [['NOT','I'],['NOT','O'],['NOT','X'],['I','I'],['I','O'],['I','X'],['O','I'],['O','O'],['O','X'],['X','I'],['X','O'],['X','X']]
# Tùy chọn:
is_continuous  # default từ YAML, bool
# Không đọc từ YAML single: use_quafu, key, critic_hidden, actor_hidden, use_quantum
```

**MAPPO (`model/mappo_trainer.py:369 setup_mappo_training`):**
```python
# Bắt buộc:
env_name, n_steps, mini_batch_size, max_train_steps, lr_a, lr_c, gamma, lamda, epsilon, K_epochs, entropy_coef, num_envs, normalize_state, normalize_reward, clip_decay, lr_decay, seed
# Tùy chọn + default:
n_blocks = config.get('n_blocks',1)
n_wires  = config.get('n_wires',4)
is_continuous = config.get('is_continuous',False)
critic_hidden = config.get('critic_hidden',[256,128])
actor_hidden  = config.get('actor_hidden',[1024,512])
n_agents = config.get('n_agents', 3 if env_name=='multiUAV' else 1)
use_quantum = config.get('use_quantum',True)
ini_method = ini_method_list[config.get('ini_method',0)]
batch_size = n_steps * num_envs
```

### 3.2 Bảng 11 file (đọc trực tiếp từ `config/*.yaml`)

| # | File | env_name | n_steps | mb | max_steps | lr_a | lr_c | gamma | lamda | eps | K | ent | num_envs | norm_s/r | cont | clip/lr_decay | ini | seed | n_blocks | n_wires | critic_hidden | actor_hidden | use_q |
|---|------|----------|---------|----|-----------|------|------|-------|-------|-----|---|-----|----------|----------|------|---------------|-----|------|----------|---------|---------------|--------------|-------|
| 1 | `CartPole.yaml` | CartPole-v1 | 32 | 256 | 100k | 0.01 | 0.001 | 0.98 | 0.8 | 0.2 | 20 | 0.00 | 8 | F/F | F | T/T | 0→NOT,I | 42 | 1 | 4 | — | — | — |
| 2 | `MountainCar.yaml` | MountainCar-v0 | 16 | 64 | 1M | 0.003 | 0.0003 | 0.99 | 0.98 | 0.2 | 4 | 0.00 | 16 | T/T | F | F/F | 0→NOT,I | 42 | 3 | 2 | — | — | — |
| 3 | `MountainCar(C).yaml` | MountainCarContinuous-v0 | 8 | 256 | 20k | 0.000777 | 0.0000777 | 0.9999 | 0.9 | 0.1 | 10 | 0.005 | 1 | T/T | T | F/F | 2→NOT,X | 42 | 3 | 2 | — | — | — |
| 4 | `Acrobot.yaml` | Acrobot-v1 | 256 | 64 | 1M | 0.003 | 0.0003 | 0.99 | 0.98 | 0.2 | 4 | 0.00 | 16 | T/T | F | F/F | 5→I,X | 42 | 1 | 4 | — | — | — |
| 5 | `Pendulum.yaml` | Pendulum-v1 | 1024 | 64 | 100k | 0.01 | 0.001 | 0.9 | 0.95 | 0.2 | 10 | 0.00 | 4 | F/F | T | F/F | 2→NOT,X | 42 | 2 | 3 | — | — | — |
| 6 | `LunarLander.yaml` | LunarLander-v2 | 1024 | 64 | 1M | 0.003 | 0.0003 | 0.999 | 0.98 | 0.2 | 4 | 0.01 | 16 | F/F | F | F/F | 9→X,I | 42 | 1 | 4 | — | — | — |
| 7 | `LunarLander(C).yaml` | LunarLander-v2 (cont) | 1024 | 64 | 1.75M | 0.003 | 0.0003 | 0.999 | 0.98 | 0.2 | 4 | 0.01 | 16 | F/F | **F*** | F/F | 4→I,I | 42 | 1 | 4 | — | — | — |
| 8 | `BipedalWalker.yaml` | BipedalWalker-v3 | 1024 | 64 | 4M | 0.004 | 0.001 | 0.99 | 0.95 | 0.2 | 7 | 0.01 | 16 | T/T | T | F/F | 7→O,I | 42 | 1 | 4 | — | — | — |
| 9 | `UAV.yaml` | UAV_Environment | 1024 | 64 | 1M | 0.001 | 0.001 | 0.99 | 0.98 | 0.1 | 4 | 0.01 | 16 | F/F | F | F/F | 9→X,I | 42 | 2 | 4 | — | — | — |
| 10 | `multiUAV_MAPPO.yaml` | multiUAV | 1024 | 64 | 1M | 0.001 | 0.001 | 0.99 | 0.98 | 0.1 | 4 | 0.01 | 32 | F/F | F | F/F | 9→X,I | 42 | 2 | 4 | [64,64] | — | true |
| 11 | `multiUAV_MAPPO_normal.yaml` | multiUAV | 1024 | 256 | 1M | 0.0001 | 0.001 | 0.99 | 0.98 | 0.1 | 4 | 0.02 | 24 | F/F | F | F/F | — | 42 | — | — | [128,128] | [64,64] | false |

> **Chú ý:** `LunarLander(C).yaml` ghi `env_name: LunarLander-v2` nhưng `is_continuous: False` trong file gốc — đây là **bug config**, code `make_env` sẽ không tạo `continuous=True`. Muốn continuous phải sửa `is_continuous: true` hoặc đổi `env_name` logic. AI mới cần biết để không nhầm.

**Ý nghĩa từng tham số (đã giải ở GUIDE, nhắc lại ngắn):**
- `n_steps`: số bước rollout mỗi env trước khi update. `batch_size = n_steps*num_envs` là tổng sample mỗi vòng.
- `mini_batch_size`: chia batch thành mini-batch cho SGD.
- `ini_method`: index 0-11 → cặp `[pre_init, post_init]` với `NOT`=Identity, `I`=ones, `O`=orthogonal, `X`=xavier. `O` cho lớp cuối dùng gain 0.01.
- `clip_decay`/`lr_decay`: nếu True, `epsilon` và `lr` giảm tuyến tính `* (1 - tran_step/all_step)`.

---

## 4. Utils — nền tảng chung (`model/utils.py:1-159`)

### 4.1 Init helpers (`utils.py:7-26`)

```python
orthogonal_init(layer, gain=sqrt(2), bias_const=0.0)  # nn.init.orthogonal_, constant bias
xavier_init(layer)                                     # xavier_uniform_, bias 0
ones_init(layer)                                        # ones_, bias 0
INIT_METHOD = {'O': orthogonal_init, 'X': xavier_init, 'I': ones_init}
# 'NOT' không có trong dict — nghĩa là Identity, không init
```

### 4.2 Quantum helpers (`utils.py:28-51`)

```python
calculate_all_Z_expectations(counts: dict[str,int]) -> np.ndarray  # P0-P1 per qubit
# counts: {"0101": 123, "1010": 456, ...} từ Quafu
# return: array[n_wires] ∈ [-1,1]

gen_task(circuit: str, device: str) -> dict  # {chip, name, circuit, compile, correct}
height(xs) -> sin(3*xs)*0.45+0.55  # chỉ dùng cho MountainCar reward shaping
```

### 4.3 Env factory (`utils.py:57-76`)

```python
def make_env(args, seed, is_continuous=False) -> thunk:
    def thunk():
        if args.env_name == 'UAV_Environment': env = UAV_Environment()
        elif args.env_name == 'multiUAV':      env = multiUAV()
        elif args.env_name=="LunarLander-v2" and is_continuous: env = gym.make("LunarLander-v2", continuous=True)
        else: env = gym.make(args.env_name)
        if args.env_name=='MountainCar-v0':
            env = gym.wrappers.TransformReward(env, lambda r: r + height(state[0]))
        env = gym.wrappers.RecordEpisodeStatistics(env)  # → info["episode"], "final_info", "_final_info"
        env.action_space.seed(seed)
        return env
    return thunk

def gen_seeds(args) -> list[int]  # [seed, seed+1, ..., seed+num_envs-1]

def load_config_from_yaml(yaml_file) -> dict  # yaml.safe_load
def setup_training(config_file_path) -> Namespace  # single, xem §3.1
def get_config_path(file_name: str) -> str  # f"config/{file_name}.yaml"
```

**Lưu ý AI:** `make_env` là thunk factory cho `AsyncVectorEnv`. `setup_training` không đọc `use_quantum`/`critic_hidden` — chỉ MAPPO mới đọc.

---

## 5. Môi trường — vật lý, toán, reward

### 5.1 UAV đơn (`UAV_env.py:10-234`)

**Khởi tạo mặc định:**
```python
UAV_Environment(users=250, uavs=1, size=2000, v_0=30, tau=1,
                UAV_coverage=3000, mBS_coverage=3000,
                psi_L=1, psi_M=1, K=50, d=1, lambda_c=0.05, h=120,
                alpha=2.7, P_UAV=1.0, sigma_square=1e-12,
                P_mBS=46, D_hb=205, f_c=2, sigma_logF=2,
                W=20e6, r_th=20e6, grid_num=10, max_step=100)
```

**Không gian:**
- `action_space = Discrete(5)` — `UAV_env.py:49`
- `obs_dim = 2 + 2*grid_num^2 = 202` — `UAV_env.py:66`
- `observation = concat([uav_pos(2), heatmap_UAV0(100), heatmap_satisfied(100)])` — `UAV_env.py:85`

**Vật lý kênh (step `UAV_env.py:129-145`):**
```python
psi_UAV = sqrt(K/(1+K))*psi_L + sqrt(1/(1+K))*psi_M   # Rician, K=50
theta = 10**(-20*log10(4*pi*d/lambda_c)/10)             # hằng số
d_UAV = norm(users - uav_pos)                           # [250]
d_mBS = norm(users - mBS)                               # [250], mBS tại (1000,1000)?? thực tế size/2=1000 nhưng users uniform [-1000,1000]
# UAV path loss:
g_UAV = |psi_UAV|^2 * theta * (sqrt(d_UAV^2 + h^2)/d)^(-alpha)  # h=120, alpha=2.7
gamma_UAV = P_UAV * g_UAV / sigma_square                # sigma_square=1e-12
rate_UAV = W * log2(1+gamma_UAV)                        # W=20e6
# mBS path loss (3GPP-like):
L_mBS = 40*(1-0.004*D_hb)*log10(d_mBS/1000) -18*log10(D_hb) +21*log10(f_c)+80
gamma_mBS = 10**((P_mBS - L_mBS)/10)  # P_mBS=46 dBm
rate_mBS = W*log2(1+gamma_mBS)
# Kết nối:
connect_tem[0, d_UAV<=3000]=1; connect_tem[1, d_mBS<=3000]=1  # coverage 3000 > map 2000 → hầu như luôn phủ
rate_all = vstack([rate_UAV, rate_mBS]) * connect_tem
data_rate_max_index = argmax(rate_all, axis=0)  # 0=UAV, 1=mBS, -1000 nếu không phủ
connect[max_index, i]=1
data_rate = sum(rate_all*connect, axis=0)
unsatisfied_users = (data_rate >= r_th)  # 1 nếu thỏa 20 Mbps, 0 nếu không — tên biến gây nhầm (thực ra là satisfied)
S_total = sum(unsatisfied_users)  # tổng thỏa
uav_satisfied = sum(connect[0]*unsatisfied_users)
S = 0.5*S_total + 0.5*uav_satisfied  # reward (float, 0~250)
```

**Heatmap (`UAV_env.py:162-180`):**
- `heatmap_UAV0[x,y] +=1` nếu user kết nối UAV0.
- `heatmap_satisfied[x,y] +=1` nếu user KHÔNG thỏa (unsatisfied) — tên biến `heatmap_satisfied` nhưng thực ra là unsatisfied heatmap.
- `heatmap_users` (không đưa vào obs) đếm tất cả user.

**Episode:** `step_=0..99`, `truncated=True` khi `step>=100`, `terminated=False` luôn. `UAV0_behavior[:, step]` lưu quỹ đạo.

### 5.2 Multi-UAV (`multiUAV_env.py:10-279`) — CHI TIẾT NHẤT

**Khởi tạo mặc định:**
```python
multiUAV(users=250, uavs=3, size=2000, v_0=40, tau=1,  # v0 lớn hơn đơn (40 vs 30)
         UAV_coverage=3000, mBS_coverage=3000, ... same as above but P_UAV=1, W=20e6, r_th=20e6, grid_num=10, max_step=100)
```

**Không gian:**
```python
action_space = Discrete(5**3) = Discrete(125)  # multiUAV_env.py:50
# Giải mã: for i in 0..2: uav_action = (action // 5**i) % 5  # UAV0 là digit thấp nhất
obs_dim = 2*uavs + (uavs+1)*grid_num^2 = 6 + 400 = 406? 
# Thực tế code: 2*3 + (3+1)*100 = 6+400=406 — nhưng trainer tính global_obs_dim từ env.observation_space.shape[0] → 406
# Tuy nhiên doc cũ ghi 306 — SAI, đúng là 406. AI mới phải lấy từ env, không hardcode.
# Trong mappo_trainer: global_obs_dim = 406, agent_obs_dim = 2+100+100=202
```

**Obs chi tiết (`multiUAV_env.py:86-105`):**
```python
def _get_obs():  # global
    return concat([uav_pos[:,0], uav_pos[:,1], uav_pos[:,2],  # 6
                   heatmaps[0].flatten(),  # 100
                   heatmaps[1].flatten(),  # 100
                   heatmaps[2].flatten(),  # 100
                   heatmaps[3].flatten()]) # 100 unsatisfied

def get_agent_obs(i):
    return concat([uav_pos[:,i],  # 2
                   heatmaps[i].flatten(),  # 100
                   heatmaps[3].flatten()]) # 100

# agent_slices trong mappo_trainer.py:87-92:
grid2=100; offset=6
# agent 0: pos 0:2, heat 6:106, unsat 306:406
# agent 1: pos 2:4, heat 106:206, unsat 306:406
# agent 2: pos 4:6, heat 206:306, unsat 306:406
```

**Step vật lý (`multiUAV_env.py:112-228`):**
- Di chuyển UAV: kiểm tra biên `[-1000,1000]`, nếu vượt biên thì bỏ qua action (giữ nguyên).
- User di chuyển ngẫu nhiên `randint(0,4)` bước 5m, cũng check biên.
- `connect_tem[0..2]` cho UAV, `connect_tem[3]` cho mBS.
- Tính `rate_all[4,250]` tương tự UAV đơn nhưng loop 3 UAV.
- Chọn max rate per user, `data_rate_max_index = argmax`, `-1000` nếu không phủ.
- `satisfied_users = (data_rate >= 20e6)` (float 0/1).
- `S_total = sum(satisfied)`, `N_UAV_counts[k] = sum(connect[k]*satisfied)`.

**Reward — QUAN TRỌNG NHẤT (`multiUAV_env.py:180-205`):**
```python
# Global teamwork:
if prev_satisfied_total is None: gt=0.0
else: delta=S_total - prev; gt= 1 if delta>0 else (-1 if delta<0 else 0)
# Local per-UAV:
if prev_N_UAV is None: lt=[0,0,0]
else: lt[k]= 1 if N_k - prev_N_k >0 else (-1 if <0 else 0)
wl=0.5
agent_rewards = wl*lt + (1-wl)*gt  # shape (3,) mỗi giá trị ∈ {-1, -0.5, 0, 0.5, 1}
# Gym reward (cho critic):
S = gt  # chỉ global, ∈ {-1,0,1}
prev_satisfied_total = S_total; prev_N_UAV = N_UAV_counts.copy()
# Trả về: obs, S, terminated=False, truncated=(step>=100), info={"agent_rewards": (3,), "satisfied_total": float}
```

**Heatmap (`multiUAV_env.py:207-218`):**
- `heatmaps[i][x,y] +=1` nếu user `data_rate_max_index==i` (chỉ khi `i<3`, tức do UAV phục vụ).
- `heatmaps[3][x,y] +=1` nếu `satisfied==0` (unsatisfied).

**Render (`multiUAV_env.py:230-276`):** scatter user màu theo UAV, `o`=thỏa, `^`=không thỏa nhưng do UAV, `x`=không thỏa xám, `s` cho UAV/mBS, trail mờ.

---

## 6. Mô hình — PQC chi tiết đến từng cổng lượng tử

### 6.1 DiscreteActor (`model/models.py:25-53`)

```python
class DiscreteActor(nn.Module):
    def __init__(n_wires, n_blocks, input_dim, output_dim, ini_method, is_critic=False, use_quafu=False, ...):
        if ini_method[0] != "NOT":
            pre = Linear(input_dim, n_wires)  # INIT_METHOD[ini_method[0]](pre)
        else:
            if n_wires != input_dim: n_wires = input_dim  # SILENTLY OVERWRITE
            pre = Identity()
        pqc = PQCLayer(n_wires, n_blocks, use_quafu, ...)
        post = Linear(n_wires, output_dim)  # gain 0.01 if ini_method[1]=="O"
        softmax = Softmax(dim=1)
    def forward(x):  # [B, input_dim] -> [B, output_dim]
        x = pre(x)        # [B, n_wires]
        x = pqc(x)        # [B, n_wires] ∈ [-1,1]
        x = post(x)       # [B, output_dim]
        if not is_critic: x = softmax(x)
        return x
```

### 6.2 ContinuousActor (`model/models.py:124-157`)

```python
class ContinuousActor(nn.Module):
    def __init__(...):  # same pre + pqc
        alpha_layer = Linear(n_wires, output_dim)  # gain 0.01 if O
        beta_layer  = Linear(n_wires, output_dim)
    def forward(s):  # [B,input_dim] -> (alpha,beta) each [B,output_dim]
        s = pre(s); s = pqc(s)
        alpha = softplus(alpha_layer(s)) + 1.0  # >1
        beta  = softplus(beta_layer(s)) + 1.0
        return alpha, beta
    def get_dist(s): return Beta(alpha,beta)  # torch.distributions.Beta
```

### 6.3 Critic (`model/models.py:56-121`)

```python
class Critic(nn.Module):
    def __init__(state_dim=None, hidden_dims=None, activation=Tanh, **kwargs):
        if kwargs: _init_quantum_model(...)  # same as DiscreteActor with is_critic=True
        else: _init_classical_model(state_dim, hidden_dims, activation)
    def _init_classical_model(state_dim, hidden_dims=[64,64], activation=Tanh):
        net = Sequential()
        for in_f, out_f in zip([state_dim,*hidden_dims], hidden_dims+[1]):
            net.append(Linear(in_f, out_f))
            if out_f !=1: net.append(activation())
        # orth init REVERSED: gain=1.0 cho lớp cuối, sqrt(2) cho các lớp trước
        gain=1.0
        for layer in reversed(net):
            if isinstance(layer, Linear): orthogonal_init(layer, gain); gain=sqrt(2)
    def forward(x):  # [B,state_dim] -> [B,1]
        if hasattr(self,'net'): return net(x)
        else: return post(pqc(pre(x)))  # quantum path
```

### 6.4 PQCLayer (`model/models.py:276-372`) — TRÁI TIM

```python
class PQCLayer(tq.QuantumModule):
    def __init__(n_wires, n_blocks, use_quafu=False, ...):
        H_layer = H2All(n_wires)  # Hadamard mỗi qubit
        vqc_blocks = [RzRyVariationalLayer(n_wires) for _ in range(n_blocks)]
        dqc_blocks = [RyRzScaleEncoder(n_wires) for _ in range(n_blocks)]
        vqc_blocks.append(RzRyVariationalLayer(n_wires, with_entangle=False))  # thêm 1 block cuối không entangle
        measure = MeasureAll(PauliZ)
        ALL_COUNT = 0

    def _forward(x):  # simulation
        bsz = x.shape[0]
        q_dev = QuantumDevice(n_wires=n_wires, device=x.device, bsz=bsz)
        H_layer(q_dev)
        for vqc, dqc in zip(vqc_blocks, dqc_blocks):
            vqc(q_dev); dqc(q_dev, x)
        vqc_blocks[-1](q_dev)
        return measure(q_dev)  # [B, n_wires]

    def forward(x):
        if use_quafu: return _forward_quafu(x)
        else: return _forward(x)

    def to_qiskit_circuit(x):  # cho Quafu
        # tạo list[QuantumCircuit] bsz cái, mỗi cái compose H + vqc + dqc
        ...

    def _forward_quafu(x):  # straight-through estimator
        # đọc config.yaml lấy key, tạo quark.Task, chạy từng circuit, lấy Z expectation
        # res = calculate_all_Z_expectations(counts)  # [bsz, n_wires]
        # have_grad = _forward(x)  # để lấy gradient
        # delta = have_grad.detach() - res
        # return have_grad + delta  # forward dùng res, backward dùng have_grad
```

**Các block con:**

```python
# model/models.py:160-232 — OP helpers
OP12All(n_wires, op_fun): QuantumModuleList n_wires cái
RX2All, RY2All, RZ2All, H2All: kế thừa OP12All
CZ2All(n_wires, circular): n_wires CZ nếu circular else n_wires-1, wires [k, (k+1)%n]
CN2All: tương tự với CNOT

# model/models.py:234-252
class RzRyVariationalLayer(n_wires, with_entangle=True):
    rz = RZ2All(n_wires, has_params=True, trainable=True, init pi*rand)
    ry = RY2All(n_wires, has_params=True, trainable=True, init pi*rand)
    cz = CZ2All(n_wires, circular=(n_wires>2))
    forward: rz -> ry -> (cz if with_entangle)

# model/models.py:254-273
class RxRyRzVariationalLayer(n_wires, with_entangle=True):  # không dùng trong PQCLayer hiện tại, nhưng sẵn sàng
    rx, ry, rz + cz

# model/models.py:374-408
class ScaleEncoder(n_wires, route='rx'/'ry'/'rz'):
    n_lamda = Parameter(ones(n_wires))  # học được
    func_list = [{'input_idx':[i], 'func':route, 'wires':[i]} for i in range(n_wires)]
    forward(q_dev, x):  # x [B, n_wires]
        x_scaled = tanh(einsum('j,ij->ij', n_lamda, x))  # [B, n_wires]
        for info in func_list: apply RY/RX/RZ(x_scaled[:,i]) to wire i
    to_qiskit(x): tạo circuit list với rx/ry/rz(x_scaled)

# model/models.py:439-454
class RyRzScaleEncoder(n_wires):
    encoder1 = ScaleEncoder(n_wires,'ry')
    encoder2 = ScaleEncoder(n_wires,'rz')
    forward: encoder1 -> encoder2
    to_qiskit: compose 2 circuits
```

**Sơ đồ mạch cho `n_wires=4, n_blocks=2`:**
```
|0> -H -RZ(θ1)-RY(θ2)-CZ(0-1,1-2,2-3,3-0)-RY(λ1*tanh(λ1*x0))-RZ(λ2*tanh(λ2*x0))- ...repeat... -RZ-RY- Measure Z
|1> -H -RZ(θ3)-RY(θ4)-CZ ...             -RY(...x1)-RZ(...x1)                        -RZ-RY- Measure Z
|2> -H -RZ(θ5)-RY(θ6)-CZ ...             -RY(...x2)-RZ(...x2)                        -RZ-RY- Measure Z
|3> -H -RZ(θ7)-RY(θ8)-CZ ...             -RY(...x3)-RZ(...x3)                        -RZ-RY- Measure Z
```

### 6.5 MAPPO Models

**Quantum (`model/mappo_models.py:10-56`):**
```python
class MAPPOActor(nn.Module):
    def __init__(n_agents, input_dim=202, output_dim=5, n_wires, n_blocks, ini_method):
        actors = ModuleList([DiscreteActor(n_wires,n_blocks,input_dim,5,ini_method) for _ in range(n_agents)])
    forward(agent_obs:[B,n_agents,202]) -> probs:[B,n_agents,5]  # stack per-agent
    get_actions(agent_obs) -> (actions:[B,n_agents], log_probs:[B,n_agents])  # Categorical per agent
    evaluate_actions(agent_obs, actions) -> (log_probs:[B,n_agents], entropies:[B,n_agents])
```

**Normal (`model/mappo_models_normal.py:10-65`):**
```python
def _build_actor_net(input_dim, hidden_dims=[1024,512], output_dim=5):
    net = Sequential(Linear→ReLU * len(hidden_dims), Linear→Softmax)
    # orth init: hidden gain sqrt(2), output gain 0.01
class MAPPOActor:  # same interface, nhưng actors là MLP
class CentralizedCritic:  # same as mappo_models.py:59, hidden [256,128] or [64,64], Tanh
```

**CentralizedCritic (`mappo_models.py:59-80`):**
```python
CentralizedCritic(global_obs_dim=406, hidden_dims=[256,128], activation=Tanh):
    net = Sequential(Linear 406→256→Tanh → 256→128→Tanh → 128→1)
    orth init gain 1.0 reversed (khác với Critic single là sqrt(2))
    forward(global_obs:[B,406]) -> [B,1]
```

---

## 7. Trainer — thuật toán PPO/MAPPO từng dòng

### 7.1 Single PPO (`model/trainer.py:22-274`)

**Class PPO:**
```python
class PPO:
    def __init__(args):
        batch_size, mini_batch_size, max_train_steps, lr_a, lr_c, gamma, lamda, epsilon, K_epochs, entropy_coef, is_continuous
        if is_continuous: actor=ContinuousActor(n_wires,n_blocks,state_dim,action_dim,ini_method)
        else: actor=DiscreteActor(...)
        critic=Critic(state_dim, [64,64], Tanh)
        optimizer_actor=Adam(actor, lr_a, eps1e-5)
        optimizer_critic=Adam(critic, lr_c, eps1e-5)

    def interact_with_env(s: np.ndarray) -> (a, logprob, v):  # no_grad
        s=torch.tensor(s).to(DEVICE)  # [num_envs, state_dim]
        if is_continuous: dist=actor.get_dist(s); a=dist.sample(); logprob=dist.log_prob(a).sum(1)
        else: dist=Categorical(probs=actor(s)); a=dist.sample(); logprob=dist.log_prob(a)
        v=critic(s)  # [num_envs,1]
        return a.numpy(), logprob.numpy(), v

    def interact_with_new_policy(s, a) -> (logprob_now, entropy, v_s):  # with grad
        if is_cont: dist=actor.get_dist(s); logprob_now=dist.log_prob(a).sum(1).view(-1,1); entropy=dist.entropy().sum(1).view(-1,1)
        else: dist=Categorical(probs=actor(s)); logprob_now=dist.log_prob(a).view(-1,1); entropy=dist.entropy().view(-1,1)
        v_s=critic(s)
        return ...
```

**Hàm trainer(args):**
```python
def trainer(args):
    # 1. Xác định dims
    if is_continuous: env=gym.make(...); state_dim=env.observation_space.shape[0]; action_dim=env.action_space.shape[0]
    else: env=make_env_direct(args); state_dim=..., action_dim=env.action_space.n
    max_episode_steps = _max_episode_steps or max_step or 1000
    seed, deterministic
    del env  # chỉ để đo dim

    # 2. Vector env
    seeds=gen_seeds(args)  # [seed..seed+num_envs-1]
    envs=AsyncVectorEnv([make_env(args, seeds[i], is_continuous) for i in range(num_envs)])
    if normalize_state: envs=NormalizeObservation(envs); TransformObservation(clip -10,10)
    if normalize_reward: envs=NormalizeReward(envs, gamma); TransformReward(clip -10,10)

    # 3. Loop
    agent=PPO(args)
    s,_ = envs.reset(seed=seeds)  # [num_envs, state_dim]
    batch_step = batch_size//num_envs  # vd 1024//16=64
    all_step = max_train_steps//batch_size  # vd 1M//16384≈61
    for tran_step in tqdm(range(all_step)):
        # buffers: b_s[batch_step,num_envs,state_dim], b_a, b_logprob, b_r, b_vs, b_done
        for step in range(batch_step):
            total_steps += num_envs
            a, logprob, v = agent.interact_with_env(s)
            if is_cont: action = low + a*(high-low)  # scale Beta [0,1] → env bounds
                s_,r,term,trunc,info = envs.step(action)
            else: s_,r,term,trunc,info = envs.step(a)
            done = term|trunc
            # log episode
            if "final_info" in info: for each done env: writer.add_scalar(episodic_return, episodic_length, final_step_reward)
            if trunc.any():  # bootstrap truncated
                for idx in truncated: r[idx] += gamma * V(final_observation[idx])
            # store
            b_s[step]=s; b_a[step]=a; b_logprob[step]=logprob; b_r[step]=r; b_vs[step]=v; b_done[step]=done
            s=s_

        # GAE
        b_vs_[:-1]=b_vs[1:]; b_vs_[-1]=V(s)  # s là obs sau cùng
        b_adv=zeros_like(b_r); gae=0
        for t in reversed(range(len(b_r))):
            delta = b_r[t] + gamma*(1-done[t])*b_vs_[t] - b_vs[t]
            b_adv[t] = gae = delta + gamma*lamda*gae*(1-done[t])
        b_v_target = b_adv + b_vs
        # flatten: b_s [batch_size, state_dim], b_a [batch_size], etc.

        # PPO update
        clipfracs=[]
        for _ in range(K_epochs):
            for index in BatchSampler(SubsetRandomSampler(batch_size), mini_batch_size, False):
                logprob_now, entropy, v_s = agent.interact_with_new_policy(b_s[index], b_a[index])
                logratio = logprob_now - b_logprob[index]  # [mb,1]
                ratios = exp(logratio)
                mb_adv = b_adv[index]; mb_adv = (mb_adv - mean)/ (std+1e-8)  # normalize per mini-batch
                surr1=ratios*mb_adv; surr2=clamp(ratios,1-eps,1+eps)*mb_adv
                actor_loss = -min(surr1,surr2).mean()
                entropy_loss = entropy.mean()
                critic_loss = MSE(b_v_target[index], v_s)
                loss = actor_loss - entropy_coef*entropy_loss + 0.5*critic_loss
                # backward, clip grad 0.5, Adam step

        # decay
        if lr_decay: lr*(1 - tran_step/all_step)
        if clip_decay: epsilon*(1 - tran_step/all_step)

    # save
    torch.save(actor.state_dict(), f"weights/env_{name}_{ts}/PPO_actor.pt")
    torch.save(critic.state_dict(), ...)
    if normalize_state: save obs_rms mean/var to PPO_norm_stats.txt
```

### 7.2 MAPPO (`model/mappo_trainer.py:37-410`)

**Khác biệt then chốt so với single:**

1. **Obs extraction:** `agent_slices` precompute để tách global → per-agent.
2. **Encode action:** `base5 = sum(a_i * 5^i)` để gọi `env.step(base5)`.
3. **Per-agent reward:** lấy từ `info["agent_rewards"]` (phải xử lý cả `info` batch và `final_info` cho env done).
4. **Two GAEs:** team `b_adv` cho critic, per-agent `b_adv_agent` cho actor.
5. **Loss:** actor dùng `b_adv_agent`, critic dùng `b_v_target` từ team.

**Chi tiết code quan trọng (`mappo_trainer.py:177-345`):**

```python
# Buffers: b_global_obs[batch_step,num_envs,406], b_a[..,3], b_logprob[..,3], b_agent_r[..,3], b_satisfied[..,], b_r[..,], b_vs[..,], b_done[..,]
for step in range(batch_step):
    b_global_obs[step]=s
    a, logprob, v = agent.interact_with_env(s)  # a:[num_envs,3], logprob:[num_envs,3]
    base5 = encode_base5(a)  # [num_envs]
    s_,r,term,trunc,info = envs.step(base5)
    # per-agent reward:
    per_agent_r = zeros(num_envs,3)
    if 'agent_rewards' in info:
        raw = np.asarray(info['agent_rewards'])  # shape (num_envs,3) hoặc (num_envs,) of arrays
        # handle 1D vs 2D
    if 'final_info' in info: override per_agent_r for done envs
    # satisfied_total:
    satisfied_total = zeros(num_envs)
    if 'satisfied_total' in info: similar handling
    # same for final_info
    done = term|trunc
    # logging episodic
    if trunc: for idx: r[idx]+=gamma*V(final_obs); per_agent_r[idx]+=gamma*V(final_obs)  # broadcast
    store...

# After collect:
v_next = critic(s)  # s cuối
b_vs_[-1]=v_next
# Team GAE:
for t reversed: delta = b_r[t]+gamma*(1-done[t])*b_vs_[t]-b_vs[t]; b_adv[t]=gae=delta+gamma*lamda*gae*(1-done)
b_v_target = b_adv+b_vs
# Per-agent GAE:
b_gae_agent=zeros(num_envs,3)
for t reversed:
    done_t = (1-done[t]).unsqueeze(-1)  # [num_envs,1]
    delta = b_agent_r[t] + gamma*done_t*b_vs_[t].unsqueeze(-1) - b_vs[t].unsqueeze(-1)  # [num_envs,3]
    b_adv_agent[t]=b_gae_agent=delta+gamma*lamda*done_t*b_gae_agent
# Flatten: b_global_obs[-1,406], b_a[-1,3], b_logprob[-1,3], b_adv[-1,1], b_adv_agent[-1,3]

# Update:
for _ in range(K_epochs):
    for index in BatchSampler(...):
        global_obs_mb = b_global_obs[index]  # [mb,406]
        v_s = critic(global_obs_mb)  # [mb,1]
        critic_loss = MSE(b_v_target[index], v_s)
        agent_obs = _extract_agent_obs(global_obs_mb)  # [mb,3,202]
        log_probs_now, entropies = actor.evaluate_actions(agent_obs, b_a[index])  # [mb,3]
        logratio = log_probs_now - b_logprob[index]  # [mb,3]
        ratios=exp(logratio)
        mb_adv = b_adv_agent[index]  # [mb,3]
        mb_adv = (mb_adv - mean)/ (std+1e-8)  # normalize toàn bộ [mb*3] ??
        surr1=ratios*mb_adv; surr2=clamp(ratios,1-eps,1+eps)*mb_adv
        actor_loss = -min(surr1,surr2).mean()  # mean over mb*3
        entropy_loss = entropies.mean()
        loss = actor_loss - entropy_coef*entropy_loss + 0.5*critic_loss
        # backward clip 0.5, step

# Logging:
v_explained = 1 - sum((target-pred)^2)/sum((target-mean)^2)
writer.add_scalar(critic_loss, actor_loss, entropy, approx_kl, clipfrac, v_explained, avg_reward, avg_satisfied)
```

**Hàm phụ:**

```python
def _extract_agent_obs(global_obs):  # [B,406] -> [B,3,202]
    for i: torch.cat([global_obs[...,ps:pe], global_obs[...,hs:he], global_obs[...,us:ue]], dim=-1)

def encode_base5(actions_np):  # [..,3] -> [..]
    result=0; for i: result += actions_np[...,i] * 5**i

def setup_mappo_training(yaml) -> Namespace  # xem §3.1
```

---

## 8. Render & Plot — tái tạo kết quả

### 8.1 Single render (`render.py:1-140`)

```python
config_name = argv[1] or ""
args=setup_training(get_config_path(config_name))
# tạo env để lấy dims: if UAV_Environment -> UAV_Environment(render_mode="rgb_array") else gym.make(...,render_mode="rgb_array")
# build actor: DiscreteActor or ContinuousActor
# load: latest_dir = sorted(glob("weights/env_{name}_*"))[-1]; torch.load(f"{latest}/PPO_actor.pt")
# rollout 1000 steps: s,render, actor(s)->a, env.step(a), lưu frames, trajectory (uavs_location)
# save: imageio.mimsave(f"runs/{config}_render.gif", frames, fps10)
# if UAV: plot trajectory.png với UAV/mBS/user scatter, trail
```

### 8.2 Multi render (`multiuav_render.py:1-173`)

```python
args=setup_mappo_training(get_config_path(config_name or "multiUAV_MAPPO"))
env=multiUAV(render_mode="rgb_array")  # để lấy n_agents, agent_obs_dim, global_obs_dim
# build actor: if use_quantum: mappo_models.MAPPOActor else mappo_models_normal.MAPPOActor
# load: sorted(glob("weights/env_{name}_MAPPO{Quantum|Normal}_*"))[-1]
# agent_slices precompute (same as trainer)
# def extract_agent_obs(global_obs) # same logic
# rollout 1000: extract_agent_obs(s_t) -> actor.get_actions -> base5 -> env.step -> frames, trajectories[3]
# save GIF
# plot: 3 UAV trails, mBS, users màu theo UAV, title satisfied/total
```

### 8.3 Plot (`plot_rewards.py:1-169`)

```python
def load_scalars(logdir, tag): EventAccumulator(logdir).Scalars(tag) -> (steps, values)
def plot_traces(logdir, label, color, linestyle, ax_ret, ax_sat, window=50):
    steps_ret, ret = load_scalars(logdir,'charts/episodic_return')  # scatter + smooth line
    steps_sat, sat = load_scalars(logdir,'charts/avg_satisfied_users')  # line
def find_mappo_run(model_tag): glob(f'runs/env_multiUAV_MAPPO_{model_tag}_*') or legacy without tag
def compare_mappo_vs_mappoq(...): overlay quantum vs normal
def main(): argparse logdirs, --single, --mappo, --both, --compare-mappoq, --window, --output
# smooth: convolve ones(window)/window
```

---

## 9. Luồng thực thi end-to-end (trace)

### Trace single: `python main.py CartPole`

```
main.py:6 config_file="CartPole"
main.py:7 config_path="config/CartPole.yaml"
utils.py:89 load_config_from_yaml -> {env_name:CartPole-v1, n_steps:32, ...}
utils.py:94 setup_training -> Namespace(batch_size=256, ini_method=['NOT','I'], ...)
trainer.py:94 trainer(args)
  trainer.py:104 env=CartPole-v1 -> state_dim=4, action_dim=2
  trainer.py:127 envs=AsyncVectorEnv([make_env]*8)
  trainer.py:138 agent=PPO(args) -> DiscreteActor(4 wires, 1 block, 4->2)
  trainer.py:141 s reset [8,4]
  loop 391 steps (100k/256):
    collect 4 steps (32/8=4) -> GAE -> 20 epochs * (256/256=1 batch) -> update
  save weights/env_CartPole-v1_2026.../PPO_actor.pt
```

### Trace multi: `python main_mappo.py multiUAV_MAPPO`

```
main_mappo.py:6 config_file="multiUAV_MAPPO"
mappo_trainer.py:369 setup_mappo_training -> Namespace(n_agents=3, agent_obs_dim? chưa, global_obs_dim? chưa)
mappo_trainer.py:131 trainer(args)
  mappo_trainer.py:132 ref_env=multiUAV() -> n_agents=3, agent_obs_dim=202, global_obs_dim=406, grid_num=10
  mappo_trainer.py:157 envs=AsyncVectorEnv([make_env multiUAV]*32)
  mappo_trainer.py:168 agent=MAPPOTrainer(args) -> quantum actor 3*DiscreteActor, critic 406->256->128->1
  mappo_trainer.py:171 s reset [32,406]
  loop 244 steps (1M/4096):  # batch_size=32768? 1024*32=32768, batch_step=1024
    collect 1024 steps -> dual GAE -> 4 epochs * (32768/64≈512 batches)
  save weights/env_multiUAV_MAPPOQuantum_.../MAPPO_actor.pt
```

---

## 10. Tensor shapes — bảng tra nhanh

| Biến | Single PPO | MAPPO |
|------|-----------|-------|
| `s` (obs) | `[num_envs, state_dim]` | `[num_envs, 406]` |
| `a` | `[num_envs]` (Discrete) hoặc `[num_envs, act_dim]` (cont) | `[num_envs, 3]` |
| `logprob` | `[num_envs]` | `[num_envs, 3]` |
| `v` | `[num_envs,1]` | `[num_envs,1]` |
| `b_s` / `b_global_obs` | `[batch_step, num_envs, state_dim]` | `[batch_step, num_envs, 406]` |
| `b_a` | `[batch_step, num_envs]` | `[batch_step, num_envs, 3]` |
| `b_adv` | `[batch_step, num_envs,1]` | `[batch_step, num_envs,1]` |
| `b_adv_agent` | — | `[batch_step, num_envs, 3]` |
| `agent_obs` | — | `[B, 3, 202]` |
| `probs` | `[B, act_dim]` | `[B, 3, 5]` |
| `ratios` | `[mb,1]` | `[mb, 3]` |

---

## 11. Dependency & import graph

```
main.py
 ├─ model.trainer.trainer
 └─ model.utils.{setup_training, get_config_path}

model/utils.py
 ├─ torch, numpy, gymnasium, yaml, argparse
 └─ (no internal)

model/trainer.py
 ├─ model.models.{DiscreteActor, Critic, ContinuousActor}
 ├─ model.utils.{gen_seeds, make_env}
 ├─ torch, torch.nn, BatchSampler, tqdm, SummaryWriter, Categorical, gymnasium, numpy, logging

model/mappo_trainer.py
 ├─ model.mappo_models.CentralizedCritic (always)
 ├─ model.mappo_models.MAPPOActor  (if use_quantum)
 ├─ model.mappo_models_normal.MAPPOActor (if not)
 ├─ model.utils.{gen_seeds, make_env}
 └─ torch, BatchSampler, tqdm, SummaryWriter, gymnasium, numpy, logging, F

model/models.py
 ├─ torch, torch.nn, F, Beta, qiskit.QuantumCircuit, qiskit.qasm2.dumps, torchquantum.*, quark.Task
 └─ model.utils.{INIT_METHOD, orthogonal_init, gen_task, calculate_all_Z_expectations}

model/mappo_models.py
 ├─ model.models.DiscreteActor
 └─ model.utils.orthogonal_init

model/mappo_models_normal.py
 └─ model.utils.orthogonal_init

UAV_env.py / multiUAV_env.py
 ├─ gymnasium, numpy, math, matplotlib

render.py
 ├─ model.models.{DiscreteActor, ContinuousActor}
 └─ model.utils.{get_config_path, setup_training}

multiuav_render.py
 ├─ model.mappo_trainer.setup_mappo_training
 └─ model.mappo_models* + multiUAV_env

third_party/torchquantum
 └─ (cài editable, không import vòng)
```

**Cài đặt thứ tự:**
```bash
pip install --editable ./third_party/torchquantum  # phải trước
pip install quarkstudio==7.0.5  # cho Quafu, optional
pip install gymnasium[box2d]==0.29.1
# còn lại: torch, numpy, matplotlib, pyyaml, tqdm, tensorboard, qiskit, imageio
```

---

## 12. Invariants, pitfalls, bug đã biết

| # | Mức | Mô tả | Vị trí | Cách xử lý cho AI |
|---|-----|-------|--------|-------------------|
| 1 | 🔴 | `ini_method NOT` âm thầm ghi đè `n_wires=input_dim` | `models.py:33-34` | Không dùng NOT khi input_dim != n_wires mong muốn; sửa code nếu cần |
| 2 | 🔴 | `multiUAV` obs_dim thực tế 406 nhưng doc cũ ghi 306 | `multiUAV_env.py:67` | Luôn lấy `env.observation_space.shape[0]`, không hardcode |
| 3 | 🟡 | `LunarLander(C).yaml` sai `is_continuous:false` | `config/LunarLander(C).yaml:16` | Sửa thành true nếu muốn continuous |
| 4 | 🟡 | `batch_size` phải chia hết cho `num_envs` và `mini_batch_size` | `trainer.py:143` | Validate YAML trước train |
| 5 | 🟡 | `agent_slices` hardcode thứ tự pos/heat/unsat | `mappo_trainer.py:87` | Đổi env grid_num hoặc obs layout → phải đổi slices |
| 6 | 🟡 | Reward sign sparse (-1/0/1) dễ plateau | `multiUAV_env.py:182` | Cân nhắc thay bằng delta thực hoặc moving average |
| 7 | 🟡 | Truncated bootstrap cộng vào `r` in-place, cũng cộng vào `per_agent_r` | `mappo_trainer.py:241-252` | Không double-count nếu refactor |
| 8 | 🟡 | Advantage normalize per mini-batch, không phải global | `trainer.py:227` | Ảnh hưởng variance, đừng đổi thành global nếu không hiểu |
| 9 | 🟡 | `use_quafu` cần `config.yaml` ở root với `key`, ghi `task_id.txt`, `ALL_COUNT` | `models.py:329` | Không commit key, mock cho test |
| 10 | 🟡 | `NormalizeObservation` cần lưu `obs_rms` thủ công | `trainer.py:269` | Load khi eval nếu bật normalize |
| 11 | 🟡 | `ContinuousActor` Beta sample ∈ [0,1] rồi scale `low + a*(high-low)` | `trainer.py:161` | Sai nếu action space không [0,1] scaled |
| 12 | 🟡 | `Critic` single gain `sqrt(2)` reversed, `CentralizedCritic` gain `1.0` | `models.py:76` vs `mappo_models.py:73` | Không đồng nhất, đừng copy nhầm |
| 13 | 🟡 | `weights` và `runs` sort theo string, timestamp quyết định "latest" | `render.py:47` | Đừng đổi format timestamp |
| 14 | 🟠 | File `=1.2.0` rỗng ở root là rác | `ls` | Xóa nếu gặp |
| 15 | 🟠 | `third_party/torchquantum` cài editable, import `tq2qiskit` có thể lỗi phiên bản qiskit | `models.py:16` | Ghim qiskit version tương thích |

---

## 13. Công việc thường gặp — recipe copy-paste

### Thêm env Gym mới (ví dụ `MyEnv-v0`)

```yaml
# 1. config/MyEnv.yaml — copy CartPole.yaml, sửa:
env_name: 'MyEnv-v0'
n_steps: 256
mini_batch_size: 64
max_train_steps: 500000
lr_a: 0.003
lr_c: 0.0003
gamma: 0.99
lamda: 0.98
epsilon: 0.2
K_epochs: 4
entropy_coef: 0.01
num_envs: 16
normalize_state: False
normalize_reward: False
is_continuous: False
clip_decay: False
lr_decay: False
ini_method: 0
seed: 42
n_blocks: 1
n_wires: 4
```

```python
# 2. Nếu MyEnv không phải gym.make có sẵn, thêm vào model/utils.py:57
elif args.env_name == 'MyEnv-v0':
    from my_env import MyEnv
    env = MyEnv()

# 3. Thêm vào model/trainer.py:85 make_env_direct tương tự

# 4. Chạy:
# python main.py MyEnv
```

### Thêm biến thể multiUAV (ví dụ 5 UAV, grid 15)

```python
# multiUAV_env.py:10
class multiUAV5(multiUAV):
    def __init__(self, users=300, uavs=5, grid_num=15, ...):
        super().__init__(users=users, uavs=uavs, grid_num=grid_num, ...)
        # obs_dim tự tính = 2*5 + 6*225 = 10+1350=1360

# model/utils.py:57
elif args.env_name == 'multiUAV5':
    from multiUAV_env import multiUAV5
    env = multiUAV5()

# config/multiUAV5.yaml
# env_name: 'multiUAV5'
# n_agents: 5  # hoặc để trainer tự suy từ env.get_num_agents()

# Lưu ý: mappo_trainer.py:87 agent_slices sẽ tự đúng vì dùng grid_num từ env,
# nhưng phải test _extract_agent_obs shape [B,5,227] (2+225? thực ra agent_obs=2+225+225=452)
```

### Đổi quantum → classical cho single PPO

```python
# model/trainer.py:35-47 — hiện chưa có flag, phải sửa tay:
# Thay:
#   self.actor = DiscreteActor(...)
# Thành:
#   from model.mappo_models_normal import _build_actor_net  # hoặc tự viết MLP
#   self.actor = _build_actor_net(state_dim, [64,64], action_dim)
# Hoặc thêm use_quantum vào setup_training và branch như MAPPO
```

### Debug loss NaN / không học

```python
# Check list:
# 1. ini_method thử 0 (NOT,I) → 7 (O,I) → 9 (X,I)
# 2. Giảm lr_a 0.001 → 0.0003
# 3. Tăng mini_batch_size 64 → 256
# 4. Kiểm tra gamma/lamda <1, epsilon 0.1-0.2
# 5. In grad norm: torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5) có trả về norm
# 6. Kiểm tra reward scale: nếu S_total ~0-250 mà gt chỉ -1/0/1 thì advantage nhỏ
```

---

## 14. Lệnh reproduce — copy chạy ngay

```bash
# 0. Cài đặt (python 3.10)
pip install --editable ./third_party/torchquantum
pip install quarkstudio==7.0.5 gymnasium[box2d]==0.29.1
pip install torch numpy matplotlib pyyaml tqdm tensorboard qiskit imageio

# 1. Train single — CartPole (2 phút CPU)
python main.py CartPole

# 2. Train single — UAV (15 phút)
python main.py UAV

# 3. Train multi quantum (30-60 phút, 32 env)
python main_mappo.py multiUAV_MAPPO

# 4. Train multi classical baseline
python main_mappo.py multiUAV_MAPPO_normal

# 5. TensorBoard
tensorboard --logdir=./runs  # http://localhost:6006

# 6. Render
python render.py UAV
python multiuav_render.py multiUAV_MAPPO
python multiuav_render.py multiUAV_MAPPO_normal

# 7. Plot so sánh
python plot_rewards.py --compare-mappoq --window 50 --output runs/mappo_vs_mappoq_compare.png
python plot_rewards.py --both --window 50
python plot_rewards.py runs/env_multiUAV_MAPPO_Quantum_* runs/env_multiUAV_MAPPO_Normal_* --output runs/compare.png

# 8. Kiểm tra trọng số
ls -lt weights/ | head
ls -lt runs/ | head

# 9. Quick test env không train
python -c "from multiUAV_env import multiUAV; e=multiUAV(); o,_=e.reset(); print(o.shape, e.action_space); o,r,_,_,info=e.step(0); print(r, info)"
python -c "from UAV_env import UAV_Environment; e=UAV_Environment(); o,_=e.reset(); print(o.shape)"
```

---

## 15. Checklist bàn giao — AI mới tự kiểm tra

- [ ] Đọc `docs/PPO-Q-GUIDE.md` trước để có big picture
- [ ] Chạy `python -c "import torchquantum; import gymnasium; print('ok')"` — pass?
- [ ] Chạy `python main.py CartPole` 1 vòng ngắn (giảm max_train_steps trong YAML xuống 5000) — có tạo `runs/` và `weights/`?
- [ ] Chạy `python main_mappo.py multiUAV_MAPPO_normal` với `max_train_steps: 10000, num_envs: 4` — có chạy không OOM?
- [ ] Kiểm tra `multiUAV_env.py:67 obs_dim` thực tế bằng `python -c "from multiUAV_env import multiUAV; print(multiUAV().observation_space.shape)"` — 406 đúng không?
- [ ] Tìm `agent_slices` trong `mappo_trainer.py:87` và đối chiếu với `multiUAV_env.py:86` — khớp chưa?
- [ ] Đọc `model/models.py:276 PQCLayer` và vẽ lại mạch `n_wires=4,n_blocks=1` ra giấy — hiểu H, RzRy, RyRz, Measure?
- [ ] Sửa 1 config (ví dụ đổi `lr_a`) và chạy lại — có log đúng lr mới trong TensorBoard?
- [ ] Render 1 GIF bằng `multiuav_render.py` — có file `runs/*.gif`?
- [ ] Hiểu tại sao `LunarLander(C).yaml` sai `is_continuous` — có sửa được không?

Nếu tick hết 10/10, AI đã sẵn sàng làm việc độc lập.

---

## 16. Phụ lục — code snippets quan trọng

### 16.1 Encode/decode base-5

```python
# mappo_trainer.py:124
def encode_base5(actions_np):  # [B,3] -> [B]
    result = np.zeros(actions_np.shape[:-1], dtype=np.int64)
    for i in range(3):
        result += actions_np[..., i] * (5 ** i)
    return result
# multiUAV_env.py:119 (decode)
for i in range(3):
    uav_action = (action // (5 ** i)) % 5
```

### 16.2 GAE (single)

```python
# trainer.py:199-209
b_vs_[:-1] = b_vs[1:]; b_vs_[-1] = V(s_last)
b_adv = torch.zeros_like(b_r)
gae = 0
for t in reversed(range(len(b_r))):
    delta = b_r[t] + gamma*(1-done[t])*b_vs_[t] - b_vs[t]
    b_adv[t] = gae = delta + gamma*lamda*gae*(1-done[t])
b_v_target = b_adv + b_vs
```

### 16.3 PPO loss

```python
# trainer.py:222-239
logratio = logprob_now - logprob_old  # [mb,1]
ratios = exp(logratio)
mb_adv = (b_adv[index] - mean)/ (std+1e-8)
surr1 = ratios*mb_adv
surr2 = clamp(ratios, 1-eps, 1+eps)*mb_adv
actor_loss = -min(surr1,surr2).mean()
entropy_loss = entropy.mean()
critic_loss = MSE(target, v_s)
loss = actor_loss - entropy_coef*entropy_loss + 0.5*critic_loss
```

### 16.4 Heatmap indexing

```python
# multiUAV_env.py:208-218
x = int((users_x + size/2)//grid_size)  # grid_size=size/grid_num=200
y = int((users_y + size/2)//grid_size)
x=min(x,9); y=min(y,9)
if data_rate_max_index[i] < 3: heatmaps[max_index, x, y] +=1  # UAV heatmap
if satisfied[i]==0: heatmaps[3, x, y] +=1  # unsatisfied
```

### 16.5 Quafu straight-through

```python
# models.py:367-371
res = np.array([calculate_all_Z_expectations(c) for c in counts])  # [bsz, n_wires] hardware
res = torch.tensor(res).to(x.device)
have_grad = _forward(x)  # simulated, has grad
delta = have_grad.detach() - res  # no grad
return have_grad + delta  # forward=res, backward=have_grad
```

---

## LỜI KẾT CHO AI KẾ NHIỆM

> Bạn đang nắm trong tay toàn bộ bản đồ kho báu. Đừng đoán — hãy tra file này trước, rồi mới đọc code `file:line` để xác minh. Mọi thay đổi `grid_num`, `obs_dim`, `reward` đều phải đồng bộ 3 nơi: `*_env.py` + `mappo_trainer.py` + `config YAML`. Khi thêm tính năng, cập nhật cả `PPO-Q-GUIDE.md` và file này. Chúc may mắn!

*— AI bàn giao, 2026-09-09, từ codebase snapshot PPO-Q.*
