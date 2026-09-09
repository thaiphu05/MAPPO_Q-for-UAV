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

from model.models import DiscreteActor, ContinuousActor
from model.utils import get_config_path, setup_training

config_name = sys.argv[1] if len(sys.argv) > 1 else ""
config_path = get_config_path(config_name)
args = setup_training(config_path)

# Create env for state/action dim
if args.env_name == 'UAV_Environment':
    from UAV_env import UAV_Environment
    EnvClass = UAV_Environment
    env = EnvClass(render_mode="rgb_array")
else:
    env = gym.make(args.env_name, render_mode="rgb_array")
    EnvClass = None

args.state_dim = env.observation_space.shape[0]
args.action_dim = env.action_space.n if hasattr(env.action_space, 'n') else env.action_space.shape[0]
env.close()

# Build actor
if args.is_continuous:
    actor = ContinuousActor(
        n_wires=args.n_wires, n_blocks=args.n_blocks,
        input_dim=args.state_dim, output_dim=args.action_dim,
        ini_method=args.ini_method
    )
else:
    actor = DiscreteActor(
        n_wires=args.n_wires, n_blocks=args.n_blocks,
        input_dim=args.state_dim, output_dim=args.action_dim,
        ini_method=args.ini_method
    )

# Load weights
weight_dirs = sorted(glob.glob(f"./weights/env_{args.env_name}_*"))
if not weight_dirs:
    print("No trained weights found. Train first: python main.py", config_name)
    sys.exit(1)
latest_dir = weight_dirs[-1]
state_dict = torch.load(f"{latest_dir}/PPO_actor.pt",
                        map_location="cpu", weights_only=True)
actor.load_state_dict(state_dict)
actor.eval()

# Create env and run
if EnvClass:
    env = EnvClass(render_mode="rgb_array")
else:
    env = gym.make(args.env_name, render_mode="rgb_array")

s, _ = env.reset()
frames = []
trajectory = []  # (x, y) positions
rewards = []

# Run episode
for _ in range(1000):
    if hasattr(env, 'uavs_location'):
        trajectory.append(env.uavs_location[:, 0].copy())
    frames.append(env.render())
    s_t = torch.tensor(s, dtype=torch.float).unsqueeze(0)
    with torch.no_grad():
        if args.is_continuous:
            dist = actor.get_dist(s_t)
            a = dist.sample().squeeze(0).numpy()
        else:
            probs = actor(s_t)
            a = torch.distributions.Categorical(probs=probs).sample().item()
    s, r, terminated, truncated, info = env.step(a)
    rewards.append(r)
    if terminated or truncated:
        if hasattr(env, 'uavs_location'):
            trajectory.append(env.uavs_location[:, 0].copy())
        frames.append(env.render())
        break

env.close()

# Save GIF
gif_path = f"runs/{config_name}_render.gif"
imageio.mimsave(gif_path, frames, fps=10)
print(f"Saved GIF: {gif_path}")

# Plot UAV trajectory (only for UAV env)
if EnvClass is not None and len(trajectory) > 0:
    traj = np.array(trajectory)  # (steps, 2)

    plt.figure(figsize=(10, 10))
    half = env.size / 2

    # UAV
    plt.scatter(env.uavs_location[0, 0], env.uavs_location[1, 0],
                label='UAV0', marker='s', color='blue', s=150, zorder=5)
    # mBS
    plt.scatter(env.mBS[0], env.mBS[1],
                label='mBS', marker='s', color='purple', s=80, zorder=5)

    # Users colored by connection and satisfaction
    for i in range(env.users):
        if env.data_rate_max_index[i] == 0:
            plt.scatter(env.users_location[0, i], env.users_location[1, i],
                        marker='o' if env.unsatisfied_users[i] == 1 else '^', color='r')
        elif env.unsatisfied_users[i] == 1:
            plt.scatter(env.users_location[0, i], env.users_location[1, i],
                        marker='o', color='green')
        else:
            plt.scatter(env.users_location[0, i], env.users_location[1, i],
                        marker='x', color='gray')

    # UAV behavior trail
    plt.scatter(env.UAV0_behavior[0, :env.step_], env.UAV0_behavior[1, :env.step_],
                color='r', s=1)

    plt.xlabel('x(m)')
    plt.ylabel('y(m)')
    plt.title(f'UAV Trajectory — Total steps: {len(traj)}, '
              f'Satisfied: {int(np.sum(env.unsatisfied_users))}/{env.users}')
    plt.legend()

    plot_path = f"runs/{config_name}_trajectory.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved trajectory plot: {plot_path}")
    print(f"UAV steps: {len(traj)}")
    print(f"Final step — Reward (blended): {rewards[-1]:.1f} | Satisfied users: {int(np.sum(env.unsatisfied_users))}/{env.users}")
    print(f"Avg reward per step: {np.mean(rewards):.1f}")
else:
    print(f"Done. Frames: {len(frames)}")
