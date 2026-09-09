import sys
import os
import glob
import torch
import gymnasium as gym
import imageio
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model.mappo_trainer import setup_mappo_training
from model.utils import get_config_path

config_name = sys.argv[1] if len(sys.argv) > 1 else "multiUAV_MAPPO"
config_path = get_config_path(config_name)
args = setup_mappo_training(config_path)

# Create env to read dimensions
from multiUAV_env import multiUAV
env = multiUAV(render_mode="rgb_array")
args.n_agents = env.get_num_agents()
args.agent_obs_dim = env.get_agent_obs_dim()
args.global_obs_dim = env.observation_space.shape[0]
args.grid_num = env.grid_num
env.close()

# Build actor
use_quantum = getattr(args, 'use_quantum', True)
if use_quantum:
    from model.mappo_models import MAPPOActor as ActorClass
    actor = ActorClass(
        n_agents=args.n_agents,
        input_dim=args.agent_obs_dim,
        output_dim=5,
        n_wires=args.n_wires,
        n_blocks=args.n_blocks,
        ini_method=args.ini_method,
    )
else:
    from model.mappo_models_normal import MAPPOActor as ActorClass
    actor = ActorClass(
        n_agents=args.n_agents,
        input_dim=args.agent_obs_dim,
        output_dim=5,
        hidden_dims=getattr(args, 'actor_hidden', [1024, 512]),
    )

# Load weights
model_tag = 'Quantum' if use_quantum else 'Normal'
weight_dirs = sorted(glob.glob(f"./weights/env_{args.env_name}_MAPPO{model_tag}_*"))
if not weight_dirs:
    print(f"No trained weights found for {model_tag} model. Train first: python main_mappo.py {config_name}")
    sys.exit(1)
latest_dir = weight_dirs[-1]
state_dict = torch.load(f"{latest_dir}/MAPPO_actor.pt",
                        map_location="cpu", weights_only=True)
actor.load_state_dict(state_dict)
actor.eval()

# Precompute agent slices (same as MAPPOTrainer._extract_agent_obs)
grid2 = args.grid_num ** 2
heatmap_offset = 2 * args.n_agents
agent_slices = []
for i in range(args.n_agents):
    ps = 2 * i
    hs = heatmap_offset + i * grid2
    us = heatmap_offset + args.n_agents * grid2
    agent_slices.append((ps, ps + 2, hs, hs + grid2, us, us + grid2))


def extract_agent_obs(global_obs):
    agent_obs_list = []
    for i in range(args.n_agents):
        ps, pe, hs, he, us, ue = agent_slices[i]
        agent_obs_list.append(
            torch.cat([global_obs[..., ps:pe], global_obs[..., hs:he], global_obs[..., us:ue]], dim=-1)
        )
    return torch.stack(agent_obs_list, dim=-2)


# Create env and run episode
env = multiUAV(render_mode="rgb_array")
s, _ = env.reset()
frames = []
trajectories = [[], [], []]
rewards = []

for step in range(1000):
    for i in range(args.n_agents):
        trajectories[i].append(env.uavs_location[:, i].copy())
    frames.append(env.render())

    s_t = torch.tensor(s, dtype=torch.float).unsqueeze(0)
    agent_obs = extract_agent_obs(s_t)
    with torch.no_grad():
        actions, _ = actor.get_actions(agent_obs)  # (1, 3)
    a0, a1, a2 = actions[0]
    base5_action = int(a0 + a1 * 5 + a2 * 25)

    s, r, terminated, truncated, info = env.step(base5_action)
    rewards.append(r)
    if terminated or truncated:
        for i in range(args.n_agents):
            trajectories[i].append(env.uavs_location[:, i].copy())
        frames.append(env.render())
        break

env.close()

# Save GIF
gif_path = f"runs/{config_name}_render.gif"
imageio.mimsave(gif_path, frames, fps=10)
print(f"Saved GIF: {gif_path}")

# Plot trajectories
colors = ['blue', 'orange', 'green']
plt.figure(figsize=(10, 10))
half = env.size / 2
plt.xlim(-half, half)
plt.ylim(-half, half)
plt.gca().set_aspect('equal')

# UAV trails
for i in range(args.n_agents):
    traj = np.array(trajectories[i])
    plt.scatter(traj[0], traj[1],
                marker='s', color=colors[i], s=150, zorder=5,
                label=f'UAV{i} (end)')
    plt.scatter(traj[:-1, 0], traj[:-1, 1],
                color=colors[i], s=2, alpha=0.5, zorder=4)

# mBS
plt.scatter(env.mBS[0], env.mBS[1],
            marker='s', color='purple', s=80, zorder=5, label='mBS')

# Users
for i in range(env.users):
    uav_idx = int(env.data_rate_max_index[i])
    satisfied = env.satisfied_users[i] == 1
    if 0 <= uav_idx < args.n_agents:
        marker = 'o' if satisfied else '^'
        plt.scatter(env.users_location[0, i], env.users_location[1, i],
                    marker=marker, color=colors[uav_idx], s=8)
    elif satisfied:
        plt.scatter(env.users_location[0, i], env.users_location[1, i],
                    marker='o', color='purple', s=8)
    else:
        plt.scatter(env.users_location[0, i], env.users_location[1, i],
                    marker='x', color='gray', s=8)

plt.xlabel('x(m)')
plt.ylabel('y(m)')
plt.title(f'Multi-UAV Trajectory — Steps: {env.step_}, '
          f'Satisfied: {int(np.sum(env.satisfied_users))}/{env.users}')
plt.legend()

plot_path = f"runs/{config_name}_trajectory.png"
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved trajectory plot: {plot_path}")
print(f"Total steps: {env.step_}")
print(f"Users: total={env.users}, satisfied={int(np.sum(env.satisfied_users))}, unsatisfied={env.users - int(np.sum(env.satisfied_users))}")
print(f"Avg reward per step: {np.mean(rewards):.1f}")

# Per-UAV and mBS connected & satisfied users
for i in range(args.n_agents):
    connected = int(np.sum(env.connect[i]))
    satisfied = int(np.sum(env.connect[i] * env.satisfied_users))
    print(f"  UAV{i}: {connected} connected ({satisfied} satisfied)")
mbs_connected = int(np.sum(env.connect[args.n_agents]))
mbs_satisfied = int(np.sum(env.connect[args.n_agents] * env.satisfied_users))
print(f"  mBS: {mbs_connected} connected ({mbs_satisfied} satisfied)")
